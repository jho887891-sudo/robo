#!/usr/bin/env python3
"""R16 (G4): build r42_metadata.json for an R15 stage checkpoint, byte-identically.

Why this exists
---------------
r15_resume.save_full_state() writes the *full* training state (lora_adapter/, proprio_projector_42.pt,
optimizer_state.pt, sampler.json, rng.pt, scheduler_state.pt, trainer_state.json) plus a MANIFEST.json
of sha256s, but it never writes r42_metadata.json - only run_smoke()/run_train() do.  Both the deploy
server (r42_deploy_server.py:44-48) and the RoboDojo adapter (r42_robodojo_adapter.py:334-335) refuse
to run without that 21-field file, so an R15 stage checkpoint is currently undeployable.

This bridge reconstructs the file from evidence that already exists inside the checkpoint plus the
frozen artefacts, and proves itself by reproducing an existing r42_metadata.json byte for byte:

    MANIFEST.json                          -> lora_sha256, projector_sha256, global_step
      (or the sha256 of those two files, if the checkpoint predates MANIFEST.json)
    trainer_state.json (verified vs MANIFEST) -> base_model["config.json"] (base_model_sha256),
                                                data_contract (cross-checks), global_step
    evidence/r9_gate_20261020/frozen_contract_64train.json -> normalizer payload, gripper_policy
    data/push_T_bounds.npz                 -> safety_limits.amin / .amax
    data/split80.json                      -> split_sha256 (sha256 of the file)

Everything is cross-checked and fail-closed: a manifest/file hash mismatch, a contract that is not
42D signed_motion velocity_residual fixed-codec, a bounds file whose sha disagrees with the
trainer state, a normalizer sha that disagrees with the trainer state, or a missing base-model
attestation all abort instead of producing a plausible-looking metadata file.

CPU only; numpy at import time, nothing else.  No model, no CUDA, no simulator.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PURE = os.path.abspath(os.path.join(_HERE, "..", "OpenVLA_Pure"))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
for _p in (_HERE, _PURE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r42_adapter as R  # noqa: E402

DEFAULT_CONTRACT = os.path.join(_REPO, "evidence", "r9_gate_20261020",
                                "frozen_contract_64train.json")
DEFAULT_BOUNDS = os.path.join(_REPO, "data", "push_T_bounds.npz")
DEFAULT_SPLIT = os.path.join(_REPO, "data", "split80.json")
DEFAULT_BASE_MODEL_CONFIG = ("/home/dgut/pusht_runtime/model_cache/openvla-7b-pristine/"
                             "config.json")
MANIFEST_NAME = "MANIFEST.json"
TRAINER_STATE_NAME = "trainer_state.json"
METADATA_NAME = "r42_metadata.json"
LORA_REL = os.path.join("lora_adapter", "adapter_model.safetensors")
PROJECTOR_REL = "proprio_projector_42.pt"
TASK_ID = "push_T"
DEFAULT_SEED = 0                 # run_smoke/run_train/run_r15 --seed default
DEFAULT_MAX_RATE = 0.5           # run_r15: limits max_rate = [0.5] * 14
REQUIRED_MANIFEST_FILES = (LORA_REL.replace(os.sep, "/"), PROJECTOR_REL)


class R42BridgeError(R.R42Error):
    """Any violation of the checkpoint -> metadata reconstruction contract."""


# --------------------------------------------------------------------------------- hashing
def sha256_file(path, chunk=1 << 20):
    if not os.path.isfile(path):
        raise R42BridgeError("missing file: %s" % path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _sha_ok(value):
    try:
        value = str(value)
    except Exception:
        return False
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def _json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise R42BridgeError("cannot read json %s: %s" % (path, exc))


# --------------------------------------------------------------------------------- manifest
def read_manifest(checkpoint_dir, manifest_path=None):
    """MANIFEST.json if present, else the same structure derived from the files themselves.

    Returns {'source': 'manifest'|'derived', 'files': {rel: sha256}, 'global_step': int|None,
             'verified': [rel, ...], 'schema': str}
    """
    if not os.path.isdir(checkpoint_dir):
        raise R42BridgeError("checkpoint dir does not exist: %s" % checkpoint_dir)
    path = manifest_path or os.path.join(checkpoint_dir, MANIFEST_NAME)
    source, schema, files, global_step = "derived", "derived-from-files-v1", {}, None
    if os.path.isfile(path):
        raw = _json(path)
        if not isinstance(raw, dict) or not isinstance(raw.get("files"), dict):
            raise R42BridgeError("MANIFEST.json is malformed: %s" % path)
        files = {str(k): str(v) for k, v in raw["files"].items()}
        global_step = raw.get("global_step")
        schema = str(raw.get("schema", "unknown"))
        source = "manifest"
    for rel in REQUIRED_MANIFEST_FILES:
        local = os.path.join(checkpoint_dir, rel.replace("/", os.sep))
        if not os.path.isfile(local):
            raise R42BridgeError("checkpoint is missing %s (needed for a deploy metadata)" % rel)
        actual = sha256_file(local)
        if source == "manifest":
            if rel not in files:
                raise R42BridgeError("MANIFEST.json does not list %s" % rel)
            if files[rel] != actual:
                raise R42BridgeError(
                    "MANIFEST/file disagreement for %s: manifest %s, file %s (the checkpoint "
                    "changed after it was sealed - refusing to describe it)"
                    % (rel, files[rel], actual))
        files[rel] = actual
    if source == "derived":
        # Fill in whatever else is there so the metadata is traceable, never required.
        for rel in ("trainer_state.json", "sampler.json", "scheduler_state.pt", "rng.pt"):
            local = os.path.join(checkpoint_dir, rel)
            if os.path.isfile(local):
                files[rel] = sha256_file(local)
    return {"source": source, "schema": schema, "files": files,
            "global_step": None if global_step is None else int(global_step),
            "verified": list(REQUIRED_MANIFEST_FILES)}


def read_trainer_state(checkpoint_dir, manifest):
    """trainer_state.json, verified against the manifest when the manifest lists it."""
    path = os.path.join(checkpoint_dir, TRAINER_STATE_NAME)
    if not os.path.isfile(path):
        return None
    state = _json(path)
    if "trainer_state.json" in manifest["files"]:
        actual = sha256_file(path)
        if manifest["files"]["trainer_state.json"] != actual:
            raise R42BridgeError(
                "trainer_state.json does not match the manifest (%s vs %s); refusing to read "
                "the base-model attestation from a modified file"
                % (manifest["files"]["trainer_state.json"], actual))
    return state


# ---------------------------------------------------------------------------- frozen inputs
def load_contract(path=DEFAULT_CONTRACT):
    """The frozen 64-TRAIN contract, checked for the 42D signed_motion fixed-codec identity."""
    c = _json(path)
    if not isinstance(c, dict):
        raise R42BridgeError("frozen contract must be a dict: %s" % path)
    checks = [("proprio_dim", c.get("proprio_dim"), R.N42),
              ("proprio_mode", c.get("proprio_mode"), "svacc"),
              ("normalization_mode", c.get("normalization_mode"), "signed_motion"),
              ("target_representation", c.get("target_representation"), R.TARGET),
              ("action_codec_mode", c.get("action_codec_mode"), "fixed")]
    for name, got, want in checks:
        if got != want:
            raise R42BridgeError("frozen contract %s = %r, the R42 deploy contract requires %r"
                                 % (name, got, want))
    for key in ("pmin", "pmax", "q01", "q99", "v_scale", "a_scale"):
        if key not in c:
            raise R42BridgeError("frozen contract is missing %s" % key)
    gp = c.get("gripper_policy") or {}
    gdims = tuple(int(x) for x in gp.get("gripper_dims", ()))
    if gdims != tuple(R.GRIPPER_DIMS):
        raise R42BridgeError("frozen contract gripper_dims %s != %s (the 14D joint order would "
                             "change meaning)" % (gdims, tuple(R.GRIPPER_DIMS)))
    src = c.get("frozen_from") or {}
    return {"contract": c, "gripper_policy": gp, "frozen_from": src,
            "sha256": sha256_file(path), "path": path}


def load_bounds(path=DEFAULT_BOUNDS):
    if not os.path.isfile(path):
        raise R42BridgeError("bounds file not found: %s" % path)
    nb = np.load(path)
    for key in ("amin", "amax", "pmin", "pmax"):
        if key not in nb.files:
            raise R42BridgeError("bounds file lacks %r (has %s)" % (key, nb.files))
    amin = np.asarray(nb["amin"], np.float32).reshape(-1)
    amax = np.asarray(nb["amax"], np.float32).reshape(-1)
    if amin.shape != (R.BASE_DIM,) or amax.shape != (R.BASE_DIM,):
        raise R42BridgeError("bounds must be %dD, got amin %s / amax %s"
                             % (R.BASE_DIM, amin.shape, amax.shape))
    if not (amax > amin).all():
        raise R42BridgeError("bounds amax must exceed amin on every dim")
    return {"amin": amin, "amax": amax, "sha256": sha256_file(path), "path": path}


def build_normalizer(frozen):
    """The frozen signed_motion normalizer payload, from the contract only."""
    c, src = frozen["contract"], frozen["frozen_from"]
    if "n_train_episodes_for_quantiles" not in src or "n_dev_episodes_excluded" not in src:
        raise R42BridgeError("frozen_from must declare n_train_episodes_for_quantiles and "
                             "n_dev_episodes_excluded (got %s)" % sorted(src))
    return R.normalizer_payload(
        normalization_mode=c["normalization_mode"], pmin=c["pmin"], pmax=c["pmax"],
        q01=c["q01"], q99=c["q99"], v_scale=c["v_scale"], a_scale=c["a_scale"],
        motion_quantile=c.get("motion_quantile"),
        n_train_episodes=src["n_train_episodes_for_quantiles"],
        n_dev_episodes_excluded=src["n_dev_episodes_excluded"])


# ------------------------------------------------------------------------------- the bridge
def build_stage_metadata(checkpoint_dir, contract_path=DEFAULT_CONTRACT,
                         bounds_path=DEFAULT_BOUNDS, split_path=DEFAULT_SPLIT,
                         task_id=TASK_ID, seed=DEFAULT_SEED, max_rate=DEFAULT_MAX_RATE,
                         train_steps=None, manifest_path=None, base_model_sha256=None,
                         base_model_config=DEFAULT_BASE_MODEL_CONFIG):
    """Reconstruct the 21-field R42 metadata of one stage checkpoint.

    Raises R42BridgeError on any inconsistency; returns (metadata, report).
    """
    manifest = read_manifest(checkpoint_dir, manifest_path=manifest_path)
    trainer = read_trainer_state(checkpoint_dir, manifest)

    # ---- base model attestation: explicit override, else the checkpoint's own trainer state
    base_sha = None
    base_src = None
    if base_model_sha256 is not None:
        if not _sha_ok(base_model_sha256):
            raise R42BridgeError("base_model_sha256 must be a sha256 hex string, got %r"
                                 % (base_model_sha256,))
        base_sha, base_src = str(base_model_sha256), "explicit"
    if trainer is not None:
        attested = ((trainer.get("base_model") or {}).get("config.json"))
        if attested is not None:
            if not _sha_ok(attested):
                raise R42BridgeError("trainer_state base_model config.json is not a sha256: %r"
                                     % (attested,))
            if base_sha is not None and str(attested) != base_sha:
                raise R42BridgeError(
                    "base model attestation mismatch: trainer_state says %s, caller says %s"
                    % (attested, base_sha))
            base_sha, base_src = str(attested), "trainer_state.base_model.config.json"
    if base_sha is None:
        if not base_model_config or not os.path.isfile(base_model_config):
            raise R42BridgeError(
                "no base-model attestation: the checkpoint has no trainer_state.json and no "
                "--base-model-config was given; refusing to invent base_model_sha256")
        base_sha, base_src = sha256_file(base_model_config), \
            "sha256(%s)" % os.path.basename(base_model_config)

    # ---- train steps: manifest / trainer state must agree, an override must match both
    candidates = []
    if manifest["global_step"] is not None:
        candidates.append(("manifest.global_step", int(manifest["global_step"])))
    if trainer is not None and trainer.get("global_step") is not None:
        candidates.append(("trainer_state.global_step", int(trainer["global_step"])))
    if train_steps is not None:
        candidates.append(("caller", int(train_steps)))
    if not candidates:
        raise R42BridgeError("cannot determine train_steps: pass train_steps explicitly")
    steps = candidates[0][1]
    for name, value in candidates[1:]:
        if value != steps:
            raise R42BridgeError("train_steps disagreement: %s = %d vs %d"
                                 % (name, value, steps))
    if steps < 0:
        raise R42BridgeError("train_steps must be >= 0, got %d" % steps)

    # ---- frozen inputs
    frozen = load_contract(contract_path)
    bounds = load_bounds(bounds_path)
    norm = build_normalizer(frozen)
    norm_sha = R.normalizer_sha256(norm)

    # ---- cross-checks against the checkpoint's own data contract
    cross = {}
    if trainer is not None:
        dc = trainer.get("data_contract") or {}
        if dc.get("bounds_sha256") and str(dc["bounds_sha256"]) != bounds["sha256"]:
            raise R42BridgeError("bounds sha256 mismatch: trainer_state says %s, file is %s"
                                 % (dc["bounds_sha256"], bounds["sha256"]))
        if dc.get("frozen_contract_sha256") and \
                str(dc["frozen_contract_sha256"]) != frozen["sha256"]:
            raise R42BridgeError("frozen contract sha256 mismatch: trainer_state says %s, "
                                 "file is %s" % (dc["frozen_contract_sha256"], frozen["sha256"]))
        if dc.get("normalizer_sha256") and str(dc["normalizer_sha256"]) != norm_sha:
            raise R42BridgeError("normalizer sha256 mismatch: trainer_state says %s, the "
                                 "contract reconstructs to %s"
                                 % (dc["normalizer_sha256"], norm_sha))
        if dc.get("split_sha256") and os.path.isfile(split_path):
            if str(dc["split_sha256"]) != sha256_file(split_path):
                raise R42BridgeError("split sha256 mismatch: trainer_state says %s, file is %s"
                                     % (dc["split_sha256"], sha256_file(split_path)))
        if dc.get("codec") and dc["codec"].get("mode") not in (None, "fixed"):
            raise R42BridgeError("trainer_state codec mode %r is not fixed"
                                 % (dc["codec"].get("mode"),))
        cross = {k: dc.get(k) for k in ("bounds_sha256", "frozen_contract_sha256",
                                        "normalizer_sha256", "split_sha256", "codec",
                                        "camera_order", "gripper_dims") if k in dc}

    limits = {"amin": bounds["amin"].astype(np.float32).tolist(),
              "amax": bounds["amax"].astype(np.float32).tolist()}
    if max_rate is not None:
        rate = np.asarray(max_rate, np.float32)
        if rate.shape == ():
            rate = np.full(R.BASE_DIM, float(max_rate), np.float32)
        rate = rate.reshape(-1)
        if rate.shape != (R.BASE_DIM,) or (rate < 0).any() or not np.isfinite(rate).all():
            raise R42BridgeError("max_rate must be a non-negative 14D vector, got %r"
                                 % (max_rate,))
        limits["max_rate"] = rate.astype(np.float32).tolist()

    meta = R.build_metadata(
        task_id, base_sha, manifest["files"][REQUIRED_MANIFEST_FILES[0]],
        manifest["files"][REQUIRED_MANIFEST_FILES[1]], frozen["sha256"],
        sha256_file(split_path) if os.path.isfile(split_path) else "0" * 64,
        train_steps=steps, seed=seed, gripper_policy=frozen["gripper_policy"],
        safety_limits=limits, normalizer=norm)
    R.validate_metadata(meta, expect_task=task_id)
    report = {"checkpoint_dir": os.path.abspath(checkpoint_dir),
              "manifest_source": manifest["source"], "manifest_schema": manifest["schema"],
              "verified_files": manifest["verified"],
              "train_steps": steps, "train_steps_sources": dict(candidates),
              "base_model_sha256": base_sha, "base_model_source": base_src,
              "frozen_contract": frozen["path"], "frozen_contract_sha256": frozen["sha256"],
              "bounds": bounds["path"], "bounds_sha256": bounds["sha256"],
              "split_sha256": meta["split_sha256"], "normalizer_sha256": norm_sha,
              "lora_sha256": meta["lora_sha256"],
              "projector_sha256": meta["projector_sha256"],
              "codec_mode": meta["codec_mode"], "task_id": meta["task_id"],
              "cross_checks": cross,
              "metadata_sha256": hashlib.sha256(
                  R.canonical_json(meta).encode("utf-8")).hexdigest()}
    return meta, report


def write_metadata(meta, out_path):
    """Write with the exact writer the training scripts use (byte-identical formatting)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sha = R.save_metadata(out_path, meta)
    return {"path": os.path.abspath(out_path), "file_sha256": sha,
            "bytes": os.path.getsize(out_path)}


def build_and_write(checkpoint_dir, out_path=None, **kwargs):
    meta, report = build_stage_metadata(checkpoint_dir, **kwargs)
    out = out_path or os.path.join(checkpoint_dir, METADATA_NAME)
    if os.path.abspath(out) == os.path.abspath(os.path.join(checkpoint_dir, METADATA_NAME)) \
            and os.path.isfile(out):
        raise R42BridgeError("refusing to overwrite an existing %s; pass --out" % out)
    report.update(write_metadata(meta, out))
    return meta, report


def assert_byte_parity(produced_path, reference_path):
    """Byte-for-byte comparison of two r42_metadata.json files."""
    with open(produced_path, "rb") as fh:
        got = fh.read()
    with open(reference_path, "rb") as fh:
        want = fh.read()
    if got != want:
        for i, (a, b) in enumerate(zip(got, want)):
            if a != b:
                raise R42BridgeError(
                    "byte parity FAILED at offset %d: produced %r, reference %r"
                    % (i, got[max(0, i - 40):i + 40], want[max(0, i - 40):i + 40]))
        raise R42BridgeError("byte parity FAILED: produced %d bytes, reference %d bytes "
                             "(one is a prefix of the other)" % (len(got), len(want)))
    return {"parity": True, "bytes": len(got),
            "sha256": hashlib.sha256(got).hexdigest()}


# ------------------------------------------------------------------------------------- CLI
def _parse(argv=None):
    p = argparse.ArgumentParser(
        description="Build r42_metadata.json for an R15 stage checkpoint (byte-identical to "
                    "the training writers).")
    p.add_argument("--checkpoint", required=True, help="stage checkpoint directory")
    p.add_argument("--out", default=None,
                   help="output path (default: <checkpoint>/r42_metadata.json, never overwritten)")
    p.add_argument("--contract", default=DEFAULT_CONTRACT)
    p.add_argument("--bounds", default=DEFAULT_BOUNDS)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--task", default=TASK_ID)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-rate", type=float, default=DEFAULT_MAX_RATE)
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--base-model-sha256", default=None)
    p.add_argument("--base-model-config", default=DEFAULT_BASE_MODEL_CONFIG)
    p.add_argument("--json", action="store_true", help="print the full report as json")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    try:
        _meta, report = build_and_write(
            args.checkpoint, out_path=args.out, contract_path=args.contract,
            bounds_path=args.bounds, split_path=args.split, task_id=args.task,
            seed=args.seed, max_rate=args.max_rate, train_steps=args.train_steps,
            manifest_path=args.manifest, base_model_sha256=args.base_model_sha256,
            base_model_config=args.base_model_config)
    except R42BridgeError as exc:
        print("[r42_metadata_bridge] REFUSED: %s" % exc)
        return 2
    print("[r42_metadata_bridge] wrote %s (%d bytes, sha256=%s)"
          % (report["path"], report["bytes"], report["file_sha256"]))
    print("[r42_metadata_bridge] train_steps=%d lora=%s projector=%s base=%s (%s)"
          % (report["train_steps"], report["lora_sha256"][:12],
             report["projector_sha256"][:12], report["base_model_sha256"][:12],
             report["base_model_source"]))
    print("[r42_metadata_bridge] normalizer_sha256=%s codec_mode=%s task=%s"
          % (report["normalizer_sha256"], report["codec_mode"], report["task_id"]))
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
