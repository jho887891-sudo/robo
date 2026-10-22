#!/usr/bin/env python3
"""R15 Lane T / P0-RESUME-1: true cumulative resume from a full-state checkpoint.

The audit found that `run_train()` rebuilt the model, the LoRA, the projector, AdamW and the
scheduler from scratch on every invocation and only ever saved LoRA + projector + metadata.
`--steps 300`, `--steps 1000`, `--steps 3000` therefore produced three unrelated models and
4300 formal steps.  This module provides what a real continuation needs:

  full state  = model weights (LoRA) + 42D projector + AdamW moments + LR scheduler +
                python/numpy/torch-CPU/torch-CUDA RNG + resumable sampler position +
                global_step + the data/normalisation/codec contract that produced it.

Properties:
  * every checkpoint is written into a NEW directory (`<name>.tmp-<pid>`), fsynced, then renamed;
    an existing target directory is never overwritten;
  * a MANIFEST.json carries the sha256 of every artefact plus schema/optimizer format versions,
    the base-model identity and the data contract;
  * loading verifies every hash, the base model identity, the schema, the global_step and the
    full data contract, and fails closed on any mismatch (missing file, edited q01/q99, resumed
    from another task, legacy 14D/28D assets, tampered manifest);
  * the sampler is an explicit permutation + position, so "the next sample" is reproducible
    without depending on how many times the RNG was drawn.
"""
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import tempfile

import numpy as np

CHECKPOINT_SCHEMA = "r15-full-state-v1"
OPTIMIZER_FORMAT_VERSION = 1
REQUIRED_FILES = ("MANIFEST.json", "trainer_state.json", "sampler.json", "rng.pt",
                  "optimizer_state.pt", "scheduler_state.pt", "proprio_projector_42.pt")


class ResumeError(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_hashes(root):
    out = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            out[os.path.relpath(p, root)] = sha256_file(p)
    return out


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass


# ------------------------------------------------------------------ base model identity
def attest_base_model(model_dir):
    """Hash the immutable parts of the pristine base: config + weight index + module sources."""
    if not os.path.isdir(model_dir):
        raise ResumeError("base model directory missing: %s" % model_dir)
    picks = ("config.json", "model.safetensors.index.json", "modeling_prismatic.py",
             "configuration_prismatic.py", "preprocessor_config.json", "processor_config.json")
    out = {}
    for name in picks:
        p = os.path.join(model_dir, name)
        out[name] = sha256_file(p) if os.path.isfile(p) else None
    out["_dir"] = os.path.abspath(model_dir)
    blobs = sorted(g for g in os.listdir(model_dir) if g.endswith(".safetensors"))
    out["_weight_shards"] = blobs
    return out


# ------------------------------------------------------------------------------- sampler
class FrameSampler:
    """Deterministic, resumable frame sampler (permutation + explicit position)."""

    def __init__(self, frames, seed, order_seed=None, position=0, perm=None):
        if not frames:
            raise ResumeError("sampler needs at least one frame")
        self.frames = [(str(e), int(t)) for e, t in frames]
        self.seed = int(seed)
        self.order_seed = int(seed if order_seed is None else order_seed)
        if perm is None:
            rs = np.random.RandomState(self.order_seed)
            perm = rs.permutation(len(self.frames)).tolist()
            self._fresh_perm = True
        else:
            self._fresh_perm = False
        if sorted(perm) != list(range(len(self.frames))):
            raise ResumeError("sampler permutation is not a permutation of the frame list")
        self.perm = [int(x) for x in perm]
        self.position = int(position)
        if self.position < 0:
            raise ResumeError("sampler position must be >= 0")

    def order_hash(self):
        h = hashlib.sha256()
        h.update(json.dumps({"seed": self.order_seed,
                             "frames": self.frames,
                             "perm": self.perm}, sort_keys=True).encode())
        return h.hexdigest()

    def peek(self, k=0):
        """The frame next() WOULD return, without advancing.

        Reading the sampler for a log line must never consume a sample: doing exactly that is how the
        first R15 resume silently skipped one frame (issue found by audit PR#21). Logging therefore
        uses peek(); next() is the only method allowed to advance the position.
        """
        idx = self.perm[(self.position + int(k)) % len(self.perm)]
        return self.frames[idx]

    def next(self):
        idx = self.perm[self.position % len(self.perm)]
        self.position += 1
        return self.frames[idx]

    def state(self):
        return {"seed": self.seed, "order_seed": self.order_seed, "perm": self.perm,
                "position": self.position, "n_frames": len(self.frames),
                "order_hash": self.order_hash(),
                "frames_sha256": hashlib.sha256(
                    json.dumps(self.frames, sort_keys=True).encode()).hexdigest()}

    @classmethod
    def from_state(cls, frames, state):
        if int(state["n_frames"]) != len(frames):
            raise ResumeError("sampler frame count %s != current %d"
                              % (state["n_frames"], len(frames)))
        got = hashlib.sha256(json.dumps([[str(e), int(t)] for e, t in frames],
                                        sort_keys=True).encode()).hexdigest()
        if got != state.get("frames_sha256"):
            raise ResumeError("sampler frame list differs from the checkpoint "
                              "(data contract changed); refusing to resume")
        s = cls(frames, state["seed"], order_seed=state["order_seed"], position=state["position"],
                perm=state["perm"])
        if s.order_hash() != state.get("order_hash"):
            raise ResumeError("sampler order hash mismatch")
        return s


# ----------------------------------------------------------------------------------- RNG
def capture_rng():
    import torch
    state = {"python": random.getstate(),
             "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    import torch
    if "python" not in state or "numpy" not in state or "torch_cpu" not in state:
        raise ResumeError("rng state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# --------------------------------------------------------------------------- saving
def save_full_state(target_dir, *, model, proj, opt, sched, sampler, global_step, run_id,
                    data_contract, model_kind="torch", extra=None, expect_new=True):
    """Write a complete, hash-manifested checkpoint atomically. Never overwrites."""
    import torch
    target_dir = os.path.abspath(target_dir)
    if expect_new and os.path.exists(target_dir):
        raise ResumeError("checkpoint directory already exists (never overwrite): %s" % target_dir)
    parent = os.path.dirname(target_dir)
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=parent, prefix=".tmp-%d-" % os.getpid())
    try:
        files = {}
        if model_kind == "peft":
            model_dir = os.path.join(tmp, "lora_adapter")
            os.makedirs(model_dir)
            model.save_pretrained(model_dir)
            files.update({"lora_adapter/" + k: v for k, v in _tree_hashes(model_dir).items()})
        elif model_kind == "torch":
            p = os.path.join(tmp, "model_state.pt")
            torch.save(model.state_dict(), p)
            files["model_state.pt"] = sha256_file(p)
        else:
            raise ResumeError("unknown model_kind %r" % (model_kind,))

        torch.save(proj.state_dict(), os.path.join(tmp, "proprio_projector_42.pt"))
        torch.save(opt.state_dict(), os.path.join(tmp, "optimizer_state.pt"))
        # torch.save, NOT json: MultiStepLR keeps `milestones` as a Counter and a json/default=str
        # round trip silently degrades it to a string, so the resumed run never decays the LR.
        torch.save(sched.state_dict(), os.path.join(tmp, "scheduler_state.pt"))
        torch.save(capture_rng(), os.path.join(tmp, "rng.pt"))
        with open(os.path.join(tmp, "sampler.json"), "w") as fh:
            json.dump(sampler.state(), fh, indent=1, sort_keys=True)
        trainer_state = {"run_id": run_id, "global_step": int(global_step),
                         "created_utc": __import__("datetime").datetime.now(
                             __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "data_contract": data_contract,
                         "model_kind": model_kind,
                         "optimizer_format_version": OPTIMIZER_FORMAT_VERSION,
                         "checkpoint_schema": CHECKPOINT_SCHEMA,
                         "env": {"python": platform.python_version(),
                                 "torch": torch.__version__,
                                 "numpy": np.__version__,
                                 "platform": platform.platform()}}
        trainer_state.update(extra or {})
        with open(os.path.join(tmp, "trainer_state.json"), "w") as fh:
            json.dump(trainer_state, fh, indent=1, sort_keys=True, default=str)

        for name in ("proprio_projector_42.pt", "optimizer_state.pt", "rng.pt",
                     "scheduler_state.pt", "sampler.json", "trainer_state.json"):
            files[name] = sha256_file(os.path.join(tmp, name))
        manifest = {"schema": CHECKPOINT_SCHEMA, "global_step": int(global_step),
                    "run_id": run_id, "files": files,
                    "created_utc": trainer_state["created_utc"]}
        with open(os.path.join(tmp, "MANIFEST.json"), "w") as fh:
            json.dump(manifest, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        _fsync_dir(tmp)
        os.rename(tmp, target_dir)          # atomic; raises if the target appeared meanwhile
        _fsync_dir(parent)
        return {"dir": target_dir, "manifest": manifest}
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


# --------------------------------------------------------------------------- loading
def verify_and_load_manifest(ckpt_dir):
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        raise ResumeError("checkpoint directory missing: %s" % ckpt_dir)
    man_path = os.path.join(ckpt_dir, "MANIFEST.json")
    if not os.path.isfile(man_path):
        raise ResumeError("checkpoint has no MANIFEST.json: %s" % ckpt_dir)
    man = json.load(open(man_path))
    if man.get("schema") != CHECKPOINT_SCHEMA:
        raise ResumeError("checkpoint schema %r != %r (legacy/14D checkpoint refused)"
                          % (man.get("schema"), CHECKPOINT_SCHEMA))
    for name in REQUIRED_FILES:
        if name == "MANIFEST.json":
            continue
        if name not in man.get("files", {}) and not any(
                k.startswith("lora_adapter/") for k in man.get("files", {})):
            raise ResumeError("manifest is missing %s" % name)
    missing, bad = [], []
    for rel, want in man["files"].items():
        p = os.path.join(ckpt_dir, rel)
        if not os.path.isfile(p):
            missing.append(rel)
        elif sha256_file(p) != want:
            bad.append(rel)
    if missing or bad:
        raise ResumeError("checkpoint is corrupt/incomplete: missing=%s hash_mismatch=%s"
                          % (missing, bad))
    ts = json.load(open(os.path.join(ckpt_dir, "trainer_state.json")))
    if ts.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ResumeError("trainer_state schema mismatch")
    if int(ts.get("global_step", -1)) != int(man.get("global_step", -2)):
        raise ResumeError("global_step disagrees between manifest and trainer_state")
    if int(ts.get("optimizer_format_version", -1)) != OPTIMIZER_FORMAT_VERSION:
        raise ResumeError("optimizer format version %r unsupported"
                          % (ts.get("optimizer_format_version"),))
    return man, ts


def load_full_state(ckpt_dir, *, model, proj, opt, sched, frames, model_kind="torch",
                    expected_contract=None, expected_base=None, expected_global_step=None):
    """Verify then restore everything; fail closed on any contract mismatch."""
    import torch
    man, ts = verify_and_load_manifest(ckpt_dir)
    if expected_global_step is not None and int(ts["global_step"]) != int(expected_global_step):
        raise ResumeError("checkpoint global_step %s != expected parent %s (wrong stage/order)"
                          % (ts["global_step"], expected_global_step))
    if expected_contract is not None:
        if ts.get("data_contract") != expected_contract:
            diff = sorted(set(json.dumps(ts.get("data_contract"), sort_keys=True).split(",")) ^
                          set(json.dumps(expected_contract, sort_keys=True).split(",")))
            raise ResumeError("data/normalisation/codec contract differs from the checkpoint; "
                              "refusing to resume (first differences: %s)" % diff[:8])
    if expected_base is not None and ts.get("base_model") != expected_base:
        raise ResumeError("base model identity differs from the checkpoint")
    if ts.get("model_kind") != model_kind:
        raise ResumeError("model_kind %r != %r" % (ts.get("model_kind"), model_kind))

    if model_kind == "peft":
        from safetensors.torch import load_file
        from peft.utils.save_and_load import set_peft_model_state_dict
        adapter = os.path.join(ckpt_dir, "lora_adapter", "adapter_model.safetensors")
        if not os.path.isfile(adapter):
            raise ResumeError("adapter_model.safetensors missing")
        set_peft_model_state_dict(model, load_file(adapter))
    else:
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, "model_state.pt"),
                                         map_location="cpu"))
    proj.load_state_dict(torch.load(os.path.join(ckpt_dir, "proprio_projector_42.pt"),
                                    map_location="cpu"))
    opt.load_state_dict(torch.load(os.path.join(ckpt_dir, "optimizer_state.pt"),
                                   map_location="cpu"))
    sched.load_state_dict(torch.load(os.path.join(ckpt_dir, "scheduler_state.pt"),
                                     map_location="cpu"))
    restore_rng(torch.load(os.path.join(ckpt_dir, "rng.pt"), map_location="cpu"))
    sampler = FrameSampler.from_state(frames, json.load(open(os.path.join(ckpt_dir,
                                                                       "sampler.json"))))
    return {"manifest": man, "trainer_state": ts, "sampler": sampler,
            "global_step": int(ts["global_step"])}


# ------------------------------------------------------------------ seed discipline
def seed_everything(seed):
    """Seed BEFORE the model/LoRA/projector are constructed (the audit's ordering fix)."""
    import torch
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
