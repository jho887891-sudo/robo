#!/usr/bin/env python3
"""R44 formal full-dataset trainer for OpenVLA-7B + LoRA + 42D projector.

Safety properties:
- No model/CUDA initialization until data contracts, data hashes, budget and explicit
  authorization have passed.
- Every epoch is a deterministic permutation of every TRAIN frame exactly once.
- Supervision is joint_delta_v1: Delta[t] = A[t] - S[t], never the legacy velocity residual.
- Checkpoints include trainable parameters, optimizer/scheduler, RNG and the exact epoch order/cursor.
- SIGTERM/SIGINT/STOP and low global-free-memory requests stop at a safe optimizer boundary.
- No automatic restart, deployment, simulator launch, or main-branch mutation.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import random
import shutil
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
EXPECTED_GPU_UUID = "GPU-73599ded-b934-3bb7-53fe-4fb3121db213"
TASKS = ("push_T", "stack_bowls", "sweep_blocks")
N_ACTIONS = 14
LORA_R = 32
LORA_ALPHA = 16
LORA_SCALE = LORA_ALPHA / LORA_R
LORA_TARGET_MODULES = ("q_proj", "v_proj")
STOP_TOKEN_ID = 2
IGNORE_INDEX = -100
LEGAL_TOKEN_MIN = 31744
LEGAL_TOKEN_MAX = 31999
DEFAULT_TRAIN_SAMPLES = 108780
DEFAULT_DEV_SAMPLES = 28268


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: os.PathLike[str] | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def task_episode_names(task: str, qualified_names: list[str], M2) -> list[str]:
    """Preserve the task prefix: filenames alone are ambiguous across M2 task directories."""
    out = []
    for key in qualified_names:
        key_task, episode = M2.split_qualified(key)
        if key_task != task:
            raise ValueError(f"{task}: split entry {key!r} belongs to {key_task!r}")
        out.append(episode)
    return out


def acquire_gpu_lane_lock(args):
    """Fail fast if another repo-managed heavy GPU job already owns the single-GPU lane."""
    import r15_ledger as L15

    old_wait = os.environ.get("R15_GPU_LOCK_WAIT_S")
    os.environ["R15_GPU_LOCK_WAIT_S"] = str(float(args.gpu_lock_wait_seconds))
    lock = L15.GpuJobLock(args.ledger_dir)
    try:
        lock.__enter__()
    finally:
        if old_wait is None:
            os.environ.pop("R15_GPU_LOCK_WAIT_S", None)
        else:
            os.environ["R15_GPU_LOCK_WAIT_S"] = old_wait
    atexit.register(lock.__exit__, None, None, None)
    return lock


def current_git_sha() -> str:
    import subprocess
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                           capture_output=True, text=True, timeout=5)
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception:
        pass
    return os.environ.get("GIT_COMMIT_SHA", "UNKNOWN_RUNTIME_SHA")


def validate_resume_compatibility(run_meta: dict, args, plan: dict, base_identity: dict,
                                  model_info: dict, scheduler_info: dict, source_sha: str) -> None:
    """Reject resumes that would silently change data, base weights, schedule, or target semantics."""
    if run_meta.get("schema") != "r44-run-meta/v1":
        raise ValueError("run_meta.json has an unknown schema")
    expected = {
        "source_sha": source_sha,
        "epochs_max": int(args.epochs),
        "min_epochs": int(args.min_epochs),
        "seed": int(args.seed),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "target_representation": "joint_delta_v1",
        "train_samples_sha256": plan["train_samples_sha256"],
        "dev_samples_sha256": plan["dev_samples_sha256"],
        "split_sha256": plan["split_sha256"],
        "feature_normalizer_sha256": plan["feature_normalizer_sha256"],
        "delta_normalizer_sha256": plan["delta_normalizer_sha256"],
        "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
        "base_model_identity": base_identity,
        "scheduler": scheduler_info,
        "early_stopping_patience": int(args.early_stopping_patience),
        "checkpoint_every_steps": int(args.checkpoint_every_steps),
    }
    for key, value in expected.items():
        if run_meta.get(key) != value:
            raise ValueError(f"resume contract mismatch for run_meta.{key}")
    old_model = run_meta.get("model_info") or {}
    for key in ("modeling_sha256", "lora_config", "target_module_count",
                "trainable_lora_parameter_count", "projector_parameter_count",
                "prompt_text_sha256", "prompt_token_ids_sha256"):
        if old_model.get(key) != model_info.get(key):
            raise ValueError(f"resume model contract mismatch for model_info.{key}")


def atomic_json(path: os.PathLike[str] | str, obj: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, allow_nan=False, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def build_label_block(prompt_ids, action_token_ids, stop_token_id: int = STOP_TOKEN_ID) -> dict:
    """Pure-Python label contract: ignored prompt, 14 legal action tokens, then STOP."""
    prompt = [int(x) for x in prompt_ids]
    tokens = [int(x) for x in action_token_ids]
    if not prompt:
        raise ValueError("prompt_ids must not be empty")
    if len(tokens) != N_ACTIONS:
        raise ValueError(f"expected {N_ACTIONS} action tokens, got {len(tokens)}")
    if any(x < LEGAL_TOKEN_MIN or x > LEGAL_TOKEN_MAX for x in tokens):
        raise ValueError("an action target is outside the frozen legal token window")
    ids = prompt + tokens + [int(stop_token_id)]
    labels = [IGNORE_INDEX] * len(prompt) + tokens + [int(stop_token_id)]
    if len(ids) != len(labels) or labels[:len(prompt)] != [IGNORE_INDEX] * len(prompt):
        raise AssertionError("prompt/action label alignment failed")
    return {"input_ids": ids, "labels": labels, "n_prompt": len(prompt), "action_tokens": tokens}


def estimate_budget_hours(
    n_train: int,
    n_dev: int,
    epochs: int,
    seconds_per_train_sample: float,
    seconds_per_dev_sample: float,
    model_load_seconds: float,
    margin_fraction: float,
) -> dict:
    if n_train <= 0 or n_dev <= 0 or epochs <= 0:
        raise ValueError("sample counts and epochs must be positive")
    time_values = (seconds_per_train_sample, seconds_per_dev_sample, model_load_seconds, margin_fraction)
    if not all(math.isfinite(float(x)) for x in time_values):
        raise ValueError("time estimates and margin must be finite numbers")
    if min(seconds_per_train_sample, seconds_per_dev_sample, model_load_seconds) < 0:
        raise ValueError("time estimates cannot be negative")
    if margin_fraction < 0:
        raise ValueError("margin_fraction cannot be negative")
    train_s = n_train * epochs * seconds_per_train_sample
    dev_s = n_dev * epochs * seconds_per_dev_sample
    base_s = train_s + dev_s + model_load_seconds
    required_s = base_s * (1.0 + margin_fraction)
    return {
        "train_samples_per_epoch": int(n_train),
        "dev_samples_per_epoch": int(n_dev),
        "epochs_max": int(epochs),
        "optimizer_steps_max": int(n_train * epochs),
        "train_hours_estimate": train_s / 3600.0,
        "dev_hours_estimate": dev_s / 3600.0,
        "model_load_hours_estimate": model_load_seconds / 3600.0,
        "base_total_hours_estimate": base_s / 3600.0,
        "margin_fraction": float(margin_fraction),
        "required_budget_hours": required_s / 3600.0,
        "is_estimate_not_measurement": True,
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, allow_nan=False, default=str) + "\n")
        fh.flush()


def stat_identity(path: Path) -> tuple:
    st = path.stat()
    return (int(st.st_size), int(st.st_mtime_ns), int(st.st_ino), int(st.st_dev))


def load_runtime_modules(args):
    """Import repository/runtime modules only after CLI parsing; no CUDA context is created here."""
    # Keep the ARX marker present for the entire process lifetime before any Prismatic import.
    if not any(("arx" in str(x).lower() or "goai" in str(x).lower()) for x in sys.argv):
        sys.argv.append("arx_x5_push_T")
    for p in (REPO / "training", REPO / "diagnostics",
              REPO / "code/xpolicylab/policy/OpenVLA_Pure",
              REPO / "code/xpolicylab/policy/OpenVLA_R42"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    oft = Path(args.oft_root).expanduser().resolve()
    if not oft.is_dir():
        raise FileNotFoundError(f"OpenVLA/OFT source directory not found: {oft}")
    for p in (oft, oft / "vla-scripts"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import r11_train_pushT_r42 as R11
    import r42_adapter as R
    import r42_train_contract as TC
    import m2_multitask_data as M2
    import m2_task_normalizer as TN
    import r44_delta_v1 as D
    return R11, R, TC, M2, TN, D


def load_and_validate_cpu_plan(args, modules) -> dict:
    """Validate split, both normalizers, all HDF5 identities and complete TRAIN/DEV sample lists."""
    R11, R, TC, M2, TN, D = modules
    import numpy as np

    split_path = Path(args.splits).expanduser().resolve()
    feature_path = Path(args.feature_normalizer).expanduser().resolve()
    delta_path = Path(args.delta_normalizer).expanduser().resolve()
    manifest_path = Path(args.dataset_manifest).expanduser().resolve()
    for path in (split_path, feature_path, delta_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"required R44 contract file is missing: {path}")

    split_sha = sha256_file(split_path)
    feature_sha = sha256_file(feature_path)
    delta_sha = sha256_file(delta_path)
    manifest_sha = sha256_file(manifest_path)
    with split_path.open(encoding="utf-8") as fh:
        split_doc = json.load(fh)
    split_map = split_doc.get("splits") or {}
    if set(split_map) != set(TASKS):
        raise ValueError(f"frozen split tasks {sorted(split_map)} != required {list(TASKS)}")

    dev_lists = {}
    train_episodes = {}
    dev_episodes = {}
    for task in TASKS:
        sp = split_map[task]
        tr = task_episode_names(task, sp.get("train", []), M2)
        dv = task_episode_names(task, sp.get("dev", []), M2)
        if not tr or not dv or len(set(tr)) != len(tr) or len(set(dv)) != len(dv):
            raise ValueError(f"{task}: empty or duplicate TRAIN/DEV episode list")
        if set(tr) & set(dv):
            raise ValueError(f"{task}: TRAIN and DEV episodes overlap")
        train_episodes[task], dev_episodes[task] = tr, dv
        dev_lists[task] = dv

    feature_payload = TN.load_payload(str(feature_path), tasks=list(TASKS), dev_lists=dev_lists)
    delta_payload = json.loads(delta_path.read_text(encoding="utf-8"))
    D.validate_normalizer(delta_payload)
    if delta_payload.get("fit", {}).get("split_sha256") != split_sha:
        raise ValueError("joint_delta_v1 normalizer was fitted against a different split SHA")
    if manifest_doc := json.loads(manifest_path.read_text(encoding="utf-8")):
        if manifest_doc.get("delta_normalizer_payload_sha256") != delta_payload.get("sha256"):
            raise ValueError("dataset manifest delta-normalizer payload SHA mismatch")
    if set(delta_payload.get("tasks", {})) != set(TASKS):
        raise ValueError("joint_delta_v1 normalizer does not contain exactly the three frozen tasks")

    hdf5_root = Path(args.hdf5_root).expanduser().resolve()
    indexes = {}
    episode_paths = {}
    train_samples, dev_samples = [], []
    for task in TASKS:
        idx = M2.load_task_index(task, str(hdf5_root), verify_targets=True)
        if not idx.get("prompt_matches_constant", False):
            raise ValueError(f"{task}: actual /instruction differs from the frozen task prompt")
        indexes[task] = idx
        hdf5_dir = Path(idx["hdf5_dir"]).resolve()
        for ep in train_episodes[task] + dev_episodes[task]:
            if ep not in idx["episodes"]:
                raise ValueError(f"{task}/{ep}: split names an episode absent from HDF5 index")
            episode_paths[f"{task}/{ep}"] = (hdf5_dir / ep).resolve()
        if set(train_episodes[task] + dev_episodes[task]) != set(idx["episodes"]):
            raise ValueError(f"{task}: frozen split does not cover exactly the indexed episodes")
        for ep in train_episodes[task]:
            T = int(idx["frames"][ep])
            train_samples.extend((task, ep, t) for t in range(M2.MIN_HISTORY_FRAMES, T - 1))
        for ep in dev_episodes[task]:
            T = int(idx["frames"][ep])
            dev_samples.extend((task, ep, t) for t in range(M2.MIN_HISTORY_FRAMES, T - 1))

    if len(train_samples) != args.expected_train_samples:
        raise ValueError(f"TRAIN sample count {len(train_samples)} != frozen {args.expected_train_samples}")
    if len(dev_samples) != args.expected_dev_samples:
        raise ValueError(f"DEV sample count {len(dev_samples)} != frozen {args.expected_dev_samples}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "r44-dataset-manifest/v1":
        raise ValueError("dataset manifest schema must be r44-dataset-manifest/v1; legacy size/mtime manifests are refused")
    if manifest.get("split_sha256") != split_sha:
        raise ValueError("dataset manifest split SHA mismatch")
    if manifest.get("feature_normalizer_sha256") != feature_sha:
        raise ValueError("dataset manifest feature-normalizer SHA mismatch")
    if manifest.get("delta_normalizer_sha256") != delta_sha:
        raise ValueError("dataset manifest delta-normalizer SHA mismatch")
    records = manifest.get("files") or {}
    if set(records) != set(episode_paths):
        raise ValueError("dataset manifest episode set differs from the frozen TRAIN+DEV episode set")
    instruction_records = manifest.get("instructions") or {}
    input_id = {}
    for task in TASKS:
        actual = str(indexes[task]["instruction"])
        rec = instruction_records.get(task) or {}
        expected_sha = hashlib.sha256(actual.encode("utf-8")).hexdigest()
        if rec.get("text") != actual or rec.get("sha256") != expected_sha:
            raise ValueError(f"{task}: instruction text/hash missing or inconsistent in dataset manifest")

    file_stats_before = {}
    for key, path in sorted(episode_paths.items()):
        rec = records[key]
        if Path(str(rec.get("path", ""))).expanduser().resolve() != path:
            raise ValueError(f"{key}: dataset manifest path differs from the indexed file")
        before = stat_identity(path)
        actual_sha = sha256_file(path)
        after = stat_identity(path)
        if before != after:
            raise RuntimeError(f"{key}: file changed while its SHA256 was being computed")
        if actual_sha != rec.get("sha256") or before[0] != int(rec.get("size_bytes", -1)):
            raise ValueError(f"{key}: HDF5 content SHA256/size mismatch against manifest")
        file_stats_before[key] = before

    train_rows_by_task = {t: sum(1 for sample in train_samples if sample[0] == t) for t in TASKS}
    for task in TASKS:
        got = int(delta_payload["tasks"][task]["n_rows"])
        if got != train_rows_by_task[task]:
            raise ValueError(f"{task}: delta normalizer TRAIN row count {got} != full sample index {train_rows_by_task[task]}")
    train_hash = sha256_json([[t, ep, int(i)] for t, ep, i in train_samples])
    dev_hash = sha256_json([[t, ep, int(i)] for t, ep, i in dev_samples])
    return {
        "split_path": str(split_path), "split_sha256": split_sha,
        "feature_normalizer_path": str(feature_path), "feature_normalizer_sha256": feature_sha,
        "delta_normalizer_path": str(delta_path), "delta_normalizer_sha256": delta_sha,
        "dataset_manifest_path": str(manifest_path), "dataset_manifest_sha256": manifest_sha,
        "hdf5_root": str(hdf5_root), "indexes": indexes, "episode_paths": episode_paths,
        "manifest_files": records, "file_stats_before": file_stats_before, "feature_payload": feature_payload,
        "delta_payload": delta_payload, "train_samples": train_samples, "dev_samples": dev_samples,
        "train_samples_sha256": train_hash, "dev_samples_sha256": dev_hash,
        "n_train": len(train_samples), "n_dev": len(dev_samples),
        "instruction_sha256": {t: hashlib.sha256(indexes[t]["instruction"].encode("utf-8")).hexdigest()
                               for t in TASKS},
    }


def verify_files_unchanged(plan: dict, *, verify_sha256: bool = True) -> None:
    for key, path in plan["episode_paths"].items():
        if stat_identity(path) != plan["file_stats_before"][key]:
            raise RuntimeError(f"{key}: HDF5 file identity changed during the run")
        if verify_sha256:
            expected = plan["manifest_files"][key]["sha256"]
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"{key}: HDF5 content SHA256 changed during the run")


def gpu_snapshot(args, *, enforce_headroom: bool = True) -> dict:
    import subprocess
    cmd = [
        "nvidia-smi",
        "--query-gpu=uuid,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    if p.returncode != 0:
        raise RuntimeError(f"nvidia-smi query failed (exit {p.returncode}): {p.stderr.strip()}")
    rows = []
    for line in p.stdout.splitlines():
        cols = [x.strip() for x in line.split(",")]
        if len(cols) != 4:
            continue
        rows.append({"uuid": cols[0], "memory_total_mib": float(cols[1]),
                     "memory_free_mib": float(cols[2]), "utilization_pct": float(cols[3])})
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    wanted = args.expected_gpu_uuid
    matches = [r for r in rows if r["uuid"] == wanted]
    if len(matches) != 1:
        raise RuntimeError(f"expected GPU UUID {wanted} not uniquely present; observed {[r['uuid'] for r in rows]}")
    row = matches[0]
    row["memory_free_gb"] = row["memory_free_mib"] / 1024.0
    row["memory_total_gb"] = row["memory_total_mib"] / 1024.0
    if enforce_headroom and row["memory_free_gb"] < args.expected_peak_gpu_memory_gb + args.min_free_gpu_gb:
        raise RuntimeError(
            f"insufficient headroom: free={row['memory_free_gb']:.2f} GiB, "
            f"need peak {args.expected_peak_gpu_memory_gb:.2f} + reserve {args.min_free_gpu_gb:.2f} GiB"
        )
    return row


def build_model(args, modules):
    import torch
    import inspect
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    R11, R, _TC, _M2, _TN, _D = modules

    R11.force_arx_platform()
    runtime_contract = R.assert_runtime_contract()
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)

    base = Path(args.base_model).expanduser().resolve()
    proc = AutoProcessor.from_pretrained(str(base), trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        str(base), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    reference = REPO / "code/openvla_dynamic_modules/modeling_prismatic.py"
    active = Path(inspect.getfile(type(vla))).resolve()
    if not reference.is_file():
        raise FileNotFoundError(f"pristine dynamic modeling reference missing: {reference}")
    active_sha = sha256_file(active)
    reference_sha = sha256_file(reference)
    if active_sha != reference_sha:
        raise RuntimeError(f"active modeling_prismatic.py SHA {active_sha} != frozen reference {reference_sha}")
    vla.vision_backbone.set_num_images_in_input(int(args.cams))
    from pure_autoreg_mode import enable_pure_autoreg
    enable_pure_autoreg(vla)

    target_modules = [
        name for name, _module in vla.named_modules()
        if name.rsplit(".", 1)[-1] in LORA_TARGET_MODULES
    ]
    if not target_modules or not any(n.rsplit(".", 1)[-1] == "q_proj" for n in target_modules) \
            or not any(n.rsplit(".", 1)[-1] == "v_proj" for n in target_modules):
        raise RuntimeError("base model does not expose both q_proj and v_proj target modules")
    if args.gradient_checkpointing:
        if not hasattr(vla, "gradient_checkpointing_enable"):
            raise RuntimeError("gradient checkpointing requested but model has no enable method")
        vla.config.use_cache = False
        vla.gradient_checkpointing_enable()
        if hasattr(vla, "enable_input_require_grads"):
            vla.enable_input_require_grads()

    cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
        target_modules=list(LORA_TARGET_MODULES), init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, cfg)
    from r8_vla_adapter import build_projector
    projector = build_projector(int(vla.llm_dim), device="cuda")
    R.assert_projector_width(projector.fc1.weight.shape, expected=42)

    lora_params = [p for p in vla.parameters() if p.requires_grad]
    if not lora_params:
        raise RuntimeError("no trainable LoRA parameters found")
    bad_trainables = [n for n, p in vla.named_parameters() if p.requires_grad and "lora_" not in n]
    if bad_trainables:
        raise RuntimeError(f"non-LoRA base parameters unexpectedly trainable: {bad_trainables[:10]}")
    names = [n for n, p in vla.named_parameters() if p.requires_grad]
    if not any("q_proj" in n and "lora_" in n for n in names) \
            or not any("v_proj" in n and "lora_" in n for n in names):
        raise RuntimeError("LoRA trainables do not include both q_proj and v_proj")
    vla = vla.to("cuda")
    projector = projector.to("cuda")
    return proc, vla, projector, {
        "runtime_contract": runtime_contract,
        "modeling_sha256": active_sha,
        "target_module_count": len(target_modules),
        "trainable_lora_parameter_count": sum(p.numel() for p in lora_params),
        "projector_parameter_count": sum(p.numel() for p in projector.parameters()),
        "lora_config": {"r": LORA_R, "lora_alpha": LORA_ALPHA, "scale": LORA_SCALE,
                        "target_modules": list(LORA_TARGET_MODULES), "dropout": 0.0},
    }


def build_prompt_ids(proc, task: str, modules, include_task_token: bool) -> list[int]:
    _R11, R, _TC, M2, _TN, _D = modules
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    instruction = M2.build_prompt(task, include_token=include_task_token)["prompt"]
    text = f"What action should the robot take to {instruction}?"
    builder = PurePromptBuilder("openvla")
    builder.add_turn("human", text)
    ids = proc.tokenizer(builder.get_prompt(), return_tensors="np").input_ids.reshape(-1).tolist()
    return [int(x) for x in R.normalize_prompt_ids(ids)]


def build_runtime_labels(prompt_ids, state, action, task, delta_payload, modules):
    _R11, _R, _TC, _M2, _TN, D = modules
    delta = D.delta_target(state, action)
    _bins, token_ids = D.encode_delta(delta, task, delta_payload)
    lab = build_label_block(prompt_ids, [int(x) for x in token_ids], STOP_TOKEN_ID)
    return lab, delta


def episode_images(proc, episode_path: Path, t: int, cams: int, modules):
    """Load one task-qualified HDF5 frame using the same camera order and transform as R11."""
    import cv2
    import h5py
    import numpy as np
    import torch
    from PIL import Image

    M2 = modules[3]
    if cams != 3:
        raise ValueError("R44 image contract requires exactly 3 cameras")
    tiles = []
    with h5py.File(episode_path, "r") as h:
        for cam in M2.CAMERA_ORDER[:cams]:
            dataset = h["vision/" + cam + "/colors"]
            if t < 0 or t >= len(dataset):
                raise IndexError(f"{episode_path}: frame {t} outside camera dataset length {len(dataset)}")
            buf = dataset[t]
            img = cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"{episode_path}: failed to decode JPEG {cam} frame {t}")
            img = img[..., ::-1]
            tiles.append(proc.image_processor.apply_transform(Image.fromarray(np.ascontiguousarray(img))))
    pixel_values = torch.cat(tiles, dim=0).unsqueeze(0).to(torch.bfloat16).to("cuda")
    if pixel_values.shape[1] != cams:
        raise RuntimeError(f"camera transform produced {tuple(pixel_values.shape)} for cams={cams}")
    return pixel_values


def read_state_action(path: Path, TC):
    import h5py
    with h5py.File(path, "r") as h:
        return TC.read14(h, "state"), TC.read14(h, "action")


def load_state_cache(plan: dict, TC) -> dict:
    cache = {}
    for key, path in sorted(plan["episode_paths"].items()):
        task, episode = key.split("/", 1)
        cache[(task, episode)] = read_state_action(path, TC)
    return cache


def make_optimizer_and_scheduler(vla, projector, args, total_steps: int):
    import torch
    lora_params = [p for p in vla.parameters() if p.requires_grad]
    projector_params = list(projector.parameters())
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lr},
         {"params": projector_params, "lr": args.lr * args.projector_lr_multiplier}],
        weight_decay=0.0,
    )
    warmup_steps = max(1, int(total_steps * args.warmup_fraction))
    def lr_lambda(step):
        step = int(step)
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        remaining = max(0, total_steps - step)
        return float(remaining) / float(max(1, total_steps - warmup_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    return opt, sched, {"total_steps": int(total_steps), "warmup_steps": warmup_steps,
                        "decay": "linear_to_zero", "warmup_fraction": args.warmup_fraction,
                        "lr_lora": args.lr, "lr_projector": args.lr * args.projector_lr_multiplier}


def checkpoint_state(
    torch, vla, projector, optimizer, scheduler, *, epoch, cursor, order, global_step,
    plan, base_identity, model_info, prompts, prompt_ids, run_state, reason=None,
):
    from peft.utils.save_and_load import get_peft_model_state_dict
    return {
        "schema": "r44-full-trainer-checkpoint/v1",
        "created_utc": utcnow(),
        "global_step": int(global_step), "epoch": int(epoch), "cursor": int(cursor),
        "order": [int(x) for x in order],
        "order_sha256": hashlib.sha256(__import__("numpy").asarray(order, dtype="int64").tobytes()).hexdigest(),
        "adapter_state": get_peft_model_state_dict(vla),
        "projector_state": projector.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": __import__("numpy").random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        },
        "contracts": {
            "split_sha256": plan["split_sha256"],
            "feature_normalizer_sha256": plan["feature_normalizer_sha256"],
            "delta_normalizer_sha256": plan["delta_normalizer_sha256"],
            "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
            "train_samples_sha256": plan["train_samples_sha256"],
            "dev_samples_sha256": plan["dev_samples_sha256"],
        },
        "base_model_identity": base_identity,
        "model_info": model_info,
        "prompts": prompts,
        "prompt_token_ids": prompt_ids,
        "run_state": run_state,
        "stop_reason": reason,
    }


def save_checkpoint(out_dir: Path, state: dict, *, suffix: str = "") -> Path:
    from training.r44_safe_stop import atomic_save_checkpoint
    root = out_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    name = f"epoch{int(state['epoch']) + 1:02d}_cursor{int(state['cursor']):06d}_step{int(state['global_step']):06d}"
    if suffix:
        name += "_" + suffix
    target = root / name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite an existing checkpoint: {target}")
    tmp = Path(tempfile.mkdtemp(prefix=".r44-ckpt-", dir=str(root)))
    try:
        state_path = tmp / "state.pt"
        atomic_save_checkpoint(state_path, lambda fh: __import__("torch").save(state, fh))
        manifest = {
            "schema": "r44-checkpoint-manifest/v1",
            "created_utc": utcnow(),
            "state_file": "state.pt",
            "state_sha256": sha256_file(state_path),
            "global_step": int(state["global_step"]),
            "epoch": int(state["epoch"]),
            "cursor": int(state["cursor"]),
            "contracts": state["contracts"],
            "stop_reason": state.get("stop_reason"),
        }
        atomic_json(tmp / "checkpoint_manifest.json", manifest)
        os.replace(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


def load_checkpoint(path: Path, torch, plan: dict) -> dict:
    manifest_path = path / "checkpoint_manifest.json"
    state_path = path / "state.pt"
    if not manifest_path.is_file() or not state_path.is_file():
        raise ValueError(f"checkpoint is incomplete: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "r44-checkpoint-manifest/v1":
        raise ValueError("checkpoint manifest has an unknown schema")
    if sha256_file(state_path) != manifest.get("state_sha256"):
        raise ValueError("checkpoint state SHA256 mismatch")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("schema") != "r44-full-trainer-checkpoint/v1":
        raise ValueError("unknown checkpoint schema")
    for key in ("global_step", "epoch", "cursor", "contracts"):
        if manifest.get(key) != state.get(key):
            raise ValueError(f"checkpoint manifest/state mismatch for {key}")
    expected = {
        "split_sha256": plan["split_sha256"],
        "feature_normalizer_sha256": plan["feature_normalizer_sha256"],
        "delta_normalizer_sha256": plan["delta_normalizer_sha256"],
        "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
        "train_samples_sha256": plan["train_samples_sha256"],
        "dev_samples_sha256": plan["dev_samples_sha256"],
    }
    if state.get("contracts") != expected:
        raise ValueError("checkpoint data/normalizer contract differs from this run")
    return state


def save_run_state(out_dir: Path, state: dict) -> None:
    atomic_json(out_dir / "run_state.json", state)


def load_run_state(out_dir: Path) -> dict:
    p = out_dir / "run_state.json"
    if not p.is_file():
        return {"schema": "r44-run-state/v1", "dev_history": [], "best_dev_mae": None,
                "stale_epochs": 0, "completed_epochs": 0, "global_step": 0}
    return json.loads(p.read_text(encoding="utf-8"))


def evaluate_dev(
    torch, proc, vla, projector, plan, state_cache, prompts, prompt_ids, args, modules, stop_controller,
    *, runtime_deadline: float | None = None,
):
    import numpy as np
    _R11, _R, _TC, M2, TN, D = modules
    metrics = {}
    vla.eval()
    projector.eval()
    stop_early = False
    by_task = {t: {
        "n": 0, "abs_error_sum": np.zeros(14, dtype=np.float64),
        "noop_abs_error_sum": np.zeros(14, dtype=np.float64),
        "constant_velocity_abs_error_sum": np.zeros(14, dtype=np.float64),
        "pred_delta_abs_sum": 0.0, "gt_delta_abs_sum": 0.0,
        "pred_delta_sq_sum": 0.0, "gt_delta_sq_sum": 0.0,
        "zero_delta_count": 0, "direction_cosine_sum": 0.0, "direction_cosine_n": 0,
    } for t in TASKS}
    with torch.no_grad():
        for i, (task, episode, t) in enumerate(plan["dev_samples"]):
            if i % 100 == 0:
                if runtime_deadline is not None and time.monotonic() >= runtime_deadline:
                    stop_controller.request_stop("approved_gpu_budget_deadline")
                stop_controller.observe_cuda_free_memory()
                if stop_controller.should_stop():
                    stop_early = True
                    break
            states, actions = state_cache[(task, episode)]
            row = M2.build_row(states, actions, int(t), TN.contract_for(plan["feature_payload"], task),
                               task, episode=episode)
            state = np.asarray(states[int(t)], dtype=np.float32)
            action = np.asarray(actions[int(t)], dtype=np.float32)
            velocity = np.asarray(row["velocity"], dtype=np.float32)
            delta = D.delta_target(state, action)
            _bins, token_ids = D.encode_delta(delta, task, plan["delta_payload"])
            lab = build_label_block(prompt_ids[task], [int(x) for x in token_ids], STOP_TOKEN_ID)
            inp = torch.tensor([lab["input_ids"]], dtype=torch.long, device="cuda")
            tgt = torch.tensor([lab["labels"]], dtype=torch.long, device="cuda")
            pv = episode_images(proc, plan["episode_paths"][f"{task}/{episode}"], int(t), args.cams, modules)
            pr = torch.from_numpy(np.asarray(row["features"], dtype=np.float32)).reshape(1, 42).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                          labels=tgt, proprio=pr, proprio_projector=projector)
            logits = out.logits[0].float()
            pad = int(logits.shape[0]) - len(lab["input_ids"])
            rows = pad + int(lab["n_prompt"]) - 1 + np.arange(14)
            if pad < 0 or rows[-1] >= logits.shape[0]:
                raise RuntimeError("DEV logits/labels alignment failed")
            window = logits[torch.as_tensor(rows, device=logits.device),
                            D.LEGAL_TOKEN_MIN:D.LEGAL_TOKEN_MAX + 1]
            pred_bins = torch.argmax(window, dim=-1).detach().cpu().numpy().astype(np.int64)
            pred_delta = D.decode_delta(pred_bins, task, plan["delta_payload"])
            pred_action = state + pred_delta
            noop_action = state
            cv_action = state + velocity
            acc = by_task[task]
            acc["n"] += 1
            acc["abs_error_sum"] += np.abs(pred_action - action)
            acc["noop_abs_error_sum"] += np.abs(noop_action - action)
            acc["constant_velocity_abs_error_sum"] += np.abs(cv_action - action)
            acc["pred_delta_abs_sum"] += float(np.abs(pred_delta).mean())
            acc["gt_delta_abs_sum"] += float(np.abs(delta).mean())
            acc["pred_delta_sq_sum"] += float(np.square(pred_delta).sum())
            acc["gt_delta_sq_sum"] += float(np.square(delta).sum())
            if float(np.linalg.norm(pred_delta)) <= 1e-6:
                acc["zero_delta_count"] += 1
            pn, gn = float(np.linalg.norm(pred_delta)), float(np.linalg.norm(delta))
            if pn > 1e-8 and gn > 1e-8:
                acc["direction_cosine_sum"] += float(np.dot(pred_delta, delta) / (pn * gn))
                acc["direction_cosine_n"] += 1

    for task, acc in by_task.items():
        n = int(acc["n"])
        if n == 0:
            metrics[task] = {"n": 0, "status": "INCOMPLETE"}
            continue
        metrics[task] = {
            "n": n,
            "action_mae": float(acc["abs_error_sum"].sum() / (n * 14)),
            "action_mae_per_dim": (acc["abs_error_sum"] / n).tolist(),
            "noop_baseline_mae": float(acc["noop_abs_error_sum"].sum() / (n * 14)),
            "constant_velocity_baseline_mae": float(acc["constant_velocity_abs_error_sum"].sum() / (n * 14)),
            "mean_abs_pred_delta": float(acc["pred_delta_abs_sum"] / n),
            "mean_abs_gt_delta": float(acc["gt_delta_abs_sum"] / n),
            "pred_gt_delta_rms_ratio": float(math.sqrt(acc["pred_delta_sq_sum"] / (n * 14)) /
                                               max(1e-12, math.sqrt(acc["gt_delta_sq_sum"] / (n * 14)))),
            "zero_delta_fraction": float(acc["zero_delta_count"] / n),
            "mean_direction_cosine": (float(acc["direction_cosine_sum"] / acc["direction_cosine_n"])
                                       if acc["direction_cosine_n"] else None),
            "status": "INCOMPLETE" if stop_early else "COMPLETE",
        }
    vla.train()
    projector.train()
    return {"complete": not stop_early and sum(v["n"] for v in metrics.values()) == len(plan["dev_samples"]),
            "n_dev_expected": len(plan["dev_samples"]),
            "n_dev_evaluated": sum(v["n"] for v in metrics.values()),
            "by_task": metrics}


def export_final(out_dir: Path, vla, projector, plan: dict, model_info: dict, prompts: dict,
                 prompt_ids: dict, global_step: int, epoch_count: int) -> Path:
    import torch
    target = out_dir / "final"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite final export: {target}")
    tmp = Path(tempfile.mkdtemp(prefix=".r44-final-", dir=str(out_dir)))
    try:
        adapter_dir = tmp / "lora_adapter"
        vla.save_pretrained(str(adapter_dir), safe_serialization=True)
        torch.save(projector.state_dict(), tmp / "proprio_projector_42.pt")
        adapter_path = adapter_dir / "adapter_model.safetensors"
        projector_path = tmp / "proprio_projector_42.pt"
        meta = {
            "schema": "r44-joint-delta-checkpoint/v1",
            "target_representation": "joint_delta_v1",
            "target_formula": "Delta_t = A_t - S_t; decode A_hat_t = S_t + Delta_hat_t",
            "lora": model_info["lora_config"],
            "global_step": int(global_step), "completed_epochs": int(epoch_count),
            "split_sha256": plan["split_sha256"],
            "feature_normalizer_sha256": plan["feature_normalizer_sha256"],
            "delta_normalizer_sha256": plan["delta_normalizer_sha256"],
            "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
            "train_samples_sha256": plan["train_samples_sha256"],
            "dev_samples_sha256": plan["dev_samples_sha256"],
            "instruction_sha256": plan["instruction_sha256"],
            "prompts": prompts,
            "prompt_token_ids_sha256": {t: sha256_json(prompt_ids[t]) for t in TASKS},
            "model_info": model_info,
            "adapter_sha256": sha256_file(adapter_path),
            "projector_sha256": sha256_file(projector_path),
            "created_utc": utcnow(),
        }
        atomic_json(tmp / "r44_metadata.json", meta)
        files = {}
        for p in sorted(tmp.rglob("*")):
            if p.is_file():
                files[str(p.relative_to(tmp))] = sha256_file(p)
        atomic_json(tmp / "SHA256SUMS.json", files)
        os.replace(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="R44 formal full-data joint_delta_v1 trainer")
    ap.add_argument("--splits", default=str(REPO / "data/m2_splits.json"))
    ap.add_argument("--hdf5-root", default=os.environ.get("M2_HDF5_ROOT", ""))
    ap.add_argument("--feature-normalizer", default=os.environ.get("M2_NORMALIZER_JSON", ""))
    ap.add_argument("--delta-normalizer", default="")
    ap.add_argument("--dataset-manifest", default="")
    ap.add_argument("--base-model", default="")
    ap.add_argument("--oft-root", default=os.environ.get(
        "PUSHT_OFT", os.path.join(os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime"),
                                  "openvla_oft_src")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume-from", default="")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--min-epochs", type=int, default=2)
    ap.add_argument("--early-stopping-patience", type=int, default=0,
                    help="0 disables early stopping; formal default is all 5 full epochs")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--projector-lr-multiplier", type=float, default=0.1)
    ap.add_argument("--warmup-fraction", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cams", type=int, default=3)
    ap.add_argument("--include-task-token", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--expected-train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    ap.add_argument("--expected-dev-samples", type=int, default=DEFAULT_DEV_SAMPLES)
    ap.add_argument("--seconds-per-train-sample", type=float, default=3.9135)
    ap.add_argument("--seconds-per-dev-sample", type=float, default=0.36)
    ap.add_argument("--model-load-seconds", type=float, default=116.257)
    ap.add_argument("--budget-margin-fraction", type=float, default=0.05)
    ap.add_argument("--runtime-stop-margin-seconds", type=float, default=600.0)
    ap.add_argument("--gpu-budget-hours", type=float, default=0.0)
    ap.add_argument("--formal-training-authorized", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--resource-mode", choices=("time-window", "shared-memory"), default="time-window")
    ap.add_argument("--ledger-dir", default=os.environ.get("R15_LEDGER_DIR", None))
    ap.add_argument("--gpu-lock-wait-seconds", type=float, default=0.0,
                    help="fail fast by default if another repo-managed GPU job holds the lane")
    ap.add_argument("--gpu-window-confirmed", action="store_true")
    ap.add_argument("--accept-shared-gpu-risk", action="store_true")
    ap.add_argument("--max-process-memory-gb", type=float, default=None)
    ap.add_argument("--min-free-gpu-gb", type=float, default=12.0)
    ap.add_argument("--expected-peak-gpu-memory-gb", type=float, default=24.0)
    ap.add_argument("--resource-check-every", type=int, default=10)
    ap.add_argument("--checkpoint-every-steps", type=int, default=5000,
                    help="durable full-state checkpoint cadence; epoch boundaries are always saved")
    ap.add_argument("--expected-gpu-uuid", default=EXPECTED_GPU_UUID)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.epochs < 1 or args.epochs > 5 or args.min_epochs < 1 or args.min_epochs > args.epochs:
        raise SystemExit("epochs must be 1..5 and min-epochs must be within 1..epochs")
    float_args = {
        "gpu-budget-hours": args.gpu_budget_hours,
        "runtime-stop-margin-seconds": args.runtime_stop_margin_seconds,
        "gpu-lock-wait-seconds": args.gpu_lock_wait_seconds,
        "min-free-gpu-gb": args.min_free_gpu_gb,
        "expected-peak-gpu-memory-gb": args.expected_peak_gpu_memory_gb,
        "budget-margin-fraction": args.budget_margin_fraction,
        "seconds-per-train-sample": args.seconds_per_train_sample,
        "seconds-per-dev-sample": args.seconds_per_dev_sample,
        "model-load-seconds": args.model_load_seconds,
    }
    for name, value in float_args.items():
        if not math.isfinite(float(value)):
            raise SystemExit(f"--{name} must be finite")
    if args.gpu_budget_hours < 0 or args.runtime_stop_margin_seconds < 0 or args.gpu_lock_wait_seconds < 0:
        raise SystemExit("GPU budget, runtime stop margin, and lock wait must be >= 0")
    if args.min_free_gpu_gb <= 0 or args.expected_peak_gpu_memory_gb <= 0:
        raise SystemExit("GPU memory thresholds must be > 0")
    if args.max_process_memory_gb is not None and (
            not math.isfinite(args.max_process_memory_gb) or args.max_process_memory_gb <= 0):
        raise SystemExit("--max-process-memory-gb must be finite and > 0")
    if args.early_stopping_patience < 0:
        raise SystemExit("early-stopping-patience must be >= 0; use 0 to disable early stopping")
    if args.cams != 3:
        raise SystemExit("R44 frozen contract requires exactly 3 cameras")
    base_model_path = Path(args.base_model).expanduser().resolve() if args.base_model else None
    if base_model_path is None or not base_model_path.is_dir() or not (base_model_path / "config.json").is_file():
        raise SystemExit("--base-model must point to the exact local OpenVLA-7B directory containing config.json")
    if args.resource_check_every < 1:
        raise SystemExit("resource-check-every must be >= 1")
    if args.checkpoint_every_steps < 1:
        raise SystemExit("checkpoint-every-steps must be >= 1")
    for name in ("hdf5_root", "feature_normalizer", "delta_normalizer", "dataset_manifest", "base_model"):
        if not getattr(args, name):
            raise SystemExit(f"--{name.replace('_', '-')} is required; do not guess the server path")

    modules = load_runtime_modules(args)
    R11, R, TC, M2, TN, D = modules
    plan = load_and_validate_cpu_plan(args, modules)
    budget = estimate_budget_hours(
        plan["n_train"], plan["n_dev"], args.epochs,
        args.seconds_per_train_sample, args.seconds_per_dev_sample,
        args.model_load_seconds, args.budget_margin_fraction,
    )
    cpu_report = {
        "schema": "r44-preflight/v1", "utc": utcnow(),
        "status": "CPU_CONTRACTS_PASS",
        "train_samples": plan["n_train"], "dev_samples": plan["n_dev"],
        "train_samples_sha256": plan["train_samples_sha256"],
        "dev_samples_sha256": plan["dev_samples_sha256"],
        "split_sha256": plan["split_sha256"],
        "feature_normalizer_sha256": plan["feature_normalizer_sha256"],
        "delta_normalizer_sha256": plan["delta_normalizer_sha256"],
        "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
        "instruction_sha256": plan["instruction_sha256"],
        "budget": budget,
        "epochs_max": args.epochs, "min_epochs": args.min_epochs,
        "resource_mode": args.resource_mode,
        "training_started": False, "cuda_context_created": False,
    }
    print(json.dumps(cpu_report, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        return 0

    if args.epochs != 5:
        print("BLOCKED_EPOCH_COUNT: formal R44 requires exactly 5 epochs; smaller runs are not formal training", flush=True)
        return 2

    # All explicit approvals and budget checks happen before any torch CUDA call/model load.
    if not args.formal_training_authorized:
        print("BLOCKED_AUTHORIZATION: pass --formal-training-authorized only after approval", flush=True)
        return 2
    if args.gpu_budget_hours < budget["required_budget_hours"]:
        print(
            "BLOCKED_BUDGET: approved %.3f GPU-hours < required estimate %.3f GPU-hours"
            % (args.gpu_budget_hours, budget["required_budget_hours"]), flush=True
        )
        return 2
    if args.resource_mode == "time-window" and not args.gpu_window_confirmed:
        print("BLOCKED_GPU_WINDOW: require an explicitly agreed GPU time window", flush=True)
        return 2
    source_sha = current_git_sha()
    if source_sha == "UNKNOWN_RUNTIME_SHA":
        print("BLOCKED_SOURCE_SHA: formal training requires a verifiable Git commit SHA", flush=True)
        return 2

    if args.resource_mode == "shared-memory":
        if not args.accept_shared_gpu_risk or args.max_process_memory_gb is None:
            print("BLOCKED_SHARED_MODE: require --accept-shared-gpu-risk and --max-process-memory-gb", flush=True)
            return 2
        if args.max_process_memory_gb < args.expected_peak_gpu_memory_gb:
            print(
                "BLOCKED_SHARED_MODE: allocator cap %.2f GiB is below configured expected peak %.2f GiB; "
                "do not run without a measured lower peak (e.g. verified checkpointing)."
                % (args.max_process_memory_gb, args.expected_peak_gpu_memory_gb), flush=True
            )
            return 2
    elif args.max_process_memory_gb is not None:
        print("BLOCKED_CONFIG: memory cap is only accepted in shared-memory mode", flush=True)
        return 2

    out_dir = Path(args.out).expanduser().resolve()
    if args.resume_from:
        if not out_dir.is_dir():
            raise SystemExit("--resume-from requires the existing --out run directory")
        if (out_dir / "STOP").exists():
            raise SystemExit("STOP file still exists; remove it explicitly before requesting resume")
    elif out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Serialize repo-managed heavy GPU jobs before hashing/loading the base model.
    gpu_lock = acquire_gpu_lane_lock(args)
    # Base-model identity is CPU/file based. Do it before creating a CUDA context.
    import r15_resume as RS
    base_identity = RS.attest_base_model(str(Path(args.base_model).expanduser().resolve()))
    # The legacy identity helper records shard names but not shard bytes. R44 pins each weight
    # shard's content so a changed local base cannot silently share a checkpoint identity.
    base_path = Path(args.base_model).expanduser().resolve()
    shard_hashes = {}
    for shard in sorted(base_path.glob("*.safetensors")):
        before = stat_identity(shard)
        digest = sha256_file(shard)
        if stat_identity(shard) != before:
            raise RuntimeError(f"base-model shard changed while hashing: {shard}")
        shard_hashes[shard.name] = digest
    if not shard_hashes:
        raise RuntimeError(f"no .safetensors base-model shards found under {base_path}")
    base_identity = {**base_identity, "_weight_shard_sha256": shard_hashes}
    gpu_before = gpu_snapshot(args)
    job_start_monotonic = time.monotonic()
    runtime_deadline = job_start_monotonic + max(
        0.0, args.gpu_budget_hours * 3600.0 - args.runtime_stop_margin_seconds
    )

    import torch
    from training.r44_safe_stop import (
        R44StopController, configure_torch_memory_fraction, atomic_save_checkpoint,
    )
    memory_cap_info = configure_torch_memory_fraction(args.max_process_memory_gb, device=0)
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    proc, vla, projector, model_info = build_model(args, modules)
    prompts = {t: M2.TASK_PROMPTS[t] for t in TASKS}
    prompt_ids = {t: build_prompt_ids(proc, t, modules, args.include_task_token) for t in TASKS}
    prompt_texts = {t: f"What action should the robot take to {M2.build_prompt(t, include_token=args.include_task_token)['prompt']}?"
                    for t in TASKS}
    model_info["prompt_text_sha256"] = {t: hashlib.sha256(prompt_texts[t].encode()).hexdigest() for t in TASKS}
    model_info["prompt_token_ids_sha256"] = {t: sha256_json(prompt_ids[t]) for t in TASKS}
    model_info["memory_cap"] = memory_cap_info
    total_steps = plan["n_train"] * args.epochs
    optimizer, scheduler, scheduler_info = make_optimizer_and_scheduler(vla, projector, args, total_steps)
    state_cache = load_state_cache(plan, TC)
    stop_controller = R44StopController(
        out_dir / "STOP", min_free_gpu_gb=args.min_free_gpu_gb,
        consecutive_low_memory_samples=2,
    )
    stop_controller.install_signal_handlers()
    run_state = load_run_state(out_dir)
    global_step, epoch, cursor = 0, 0, 0
    order = D.EpochSampler(plan["n_train"], args.seed, epoch).order
    last_checkpoint = None
    resumed = False

    if args.resume_from:
        run_meta_path = out_dir / "run_meta.json"
        if not run_meta_path.is_file():
            raise RuntimeError("resume requires the original run_meta.json")
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        validate_resume_compatibility(
            run_meta, args, plan, base_identity, model_info, scheduler_info, source_sha
        )
        ckpt_path = Path(args.resume_from).expanduser().resolve()
        state = load_checkpoint(ckpt_path, torch, plan)
        disk_run_state = run_state
        if int(disk_run_state.get("global_step", 0)) > int(state["global_step"]):
            raise RuntimeError("run_state.json is newer than selected checkpoint; refusing inconsistent resume")
        if state.get("base_model_identity") != base_identity:
            raise RuntimeError("checkpoint base-model identity differs from the selected local model")
        if state.get("model_info", {}).get("modeling_sha256") != model_info.get("modeling_sha256"):
            raise RuntimeError("checkpoint active modeling_prismatic.py identity differs from this runtime")
        if state.get("model_info", {}).get("lora_config") != model_info.get("lora_config"):
            raise RuntimeError("checkpoint LoRA configuration differs from this runtime")
        if state.get("prompts") != prompts or state.get("prompt_token_ids") != prompt_ids:
            raise RuntimeError("checkpoint task prompts/token IDs differ from this runtime")
        for key in ("prompt_text_sha256", "prompt_token_ids_sha256"):
            if state.get("model_info", {}).get(key) != model_info.get(key):
                raise RuntimeError(f"checkpoint {key} differs from this runtime")
        from peft.utils.save_and_load import set_peft_model_state_dict
        set_peft_model_state_dict(vla, state["adapter_state"])
        projector.load_state_dict(state["projector_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        global_step, epoch, cursor = int(state["global_step"]), int(state["epoch"]), int(state["cursor"])
        run_state = state.get("run_state", disk_run_state)
        run_state = {**run_state, "global_step": global_step, "last_checkpoint": str(ckpt_path)}
        order = np.asarray(state["order"], dtype=np.int64)
        expected_order = D.EpochSampler(plan["n_train"], args.seed, epoch).order
        if not np.array_equal(order, expected_order):
            raise RuntimeError("checkpoint epoch order differs from deterministic sampler; refusing resume")
        if hashlib.sha256(order.tobytes()).hexdigest() != state["order_sha256"]:
            raise RuntimeError("checkpoint sample order SHA mismatch")
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
        last_checkpoint = str(ckpt_path)
        resumed = True
    else:
        if (out_dir / "run_meta.json").exists():
            raise SystemExit("run_meta.json already exists; refusing to reuse an old output directory")
        atomic_json(out_dir / "run_meta.json", {
            "schema": "r44-run-meta/v1", "run_id": args.run_id or f"r44-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            "utc_start": utcnow(), "source_sha": source_sha,
            "tasks": list(TASKS), "epochs_max": args.epochs, "min_epochs": args.min_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "checkpoint_every_steps": args.checkpoint_every_steps,
            "target_representation": "joint_delta_v1", "train_samples_per_epoch": plan["n_train"],
            "dev_samples_per_epoch": plan["n_dev"], "train_samples_sha256": plan["train_samples_sha256"],
            "dev_samples_sha256": plan["dev_samples_sha256"], "split_sha256": plan["split_sha256"],
            "feature_normalizer_sha256": plan["feature_normalizer_sha256"],
            "delta_normalizer_sha256": plan["delta_normalizer_sha256"],
            "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
            "base_model_identity": base_identity, "model_info": model_info, "scheduler": scheduler_info,
            "budget": budget, "gpu_before": gpu_before, "resource_mode": args.resource_mode,
            "gpu_lock_path": gpu_lock.path,
            "memory_cap": memory_cap_info, "gpu_window_confirmed": args.gpu_window_confirmed,
            "gradient_checkpointing": args.gradient_checkpointing, "seed": args.seed,
        })
        run_state = {"schema": "r44-run-state/v1", "dev_history": [], "best_dev_mae": None,
                     "stale_epochs": 0, "completed_epochs": 0, "global_step": 0}
        save_run_state(out_dir, run_state)

    if stop_controller.should_stop():
        state = checkpoint_state(torch, vla, projector, optimizer, scheduler,
                                 epoch=epoch, cursor=cursor, order=order, global_step=global_step,
                                 plan=plan, base_identity=base_identity, model_info=model_info,
                                 prompts=prompts, prompt_ids=prompt_ids, run_state=run_state,
                                 reason=stop_controller.reason)
        last_checkpoint = str(save_checkpoint(out_dir, state, suffix="stop"))
        save_run_state(out_dir, {**run_state, "status": "STOPPED_SAFE",
                                 "stop_reason": stop_controller.reason, "last_checkpoint": last_checkpoint,
                                 "global_step": global_step})
        stop_controller.restore_signal_handlers()
        gpu_lock.__exit__(None, None, None)
        return 0

    train_path = out_dir / "train_metrics.jsonl"
    coverage_path = out_dir / "coverage.jsonl"
    dev_path = out_dir / "dev_metrics.jsonl"
    if resumed:
        # A crash may leave append-only logs newer than the last durable checkpoint.
        def _rewrite_jsonl(path: Path, keep_row) -> None:
            if not path.is_file():
                return
            kept = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if keep_row(row):
                    kept.append(json.dumps(row, sort_keys=True, allow_nan=False, default=str))
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for line in kept:
                    fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        _rewrite_jsonl(train_path, lambda row: int(row.get("global_step", -1)) <= global_step)
        _rewrite_jsonl(coverage_path, lambda row: int(row.get("global_step", -1)) <= global_step)
        valid_epochs = {int(x.get("epoch_completed", -1)) for x in run_state.get("dev_history", [])}
        _rewrite_jsonl(dev_path, lambda row: int(row.get("epoch_completed", -1)) in valid_epochs
                       and int(row.get("global_step", -1)) <= global_step)
    # Save a recoverable step-0 checkpoint before the first optimizer update.
    if not resumed:
        state = checkpoint_state(torch, vla, projector, optimizer, scheduler,
                                 epoch=epoch, cursor=cursor, order=order, global_step=global_step,
                                 plan=plan, base_identity=base_identity, model_info=model_info,
                                 prompts=prompts, prompt_ids=prompt_ids, run_state=run_state)
        last_checkpoint = str(save_checkpoint(out_dir, state, suffix="initial"))
        save_run_state(out_dir, {**run_state, "status": "RUNNING", "last_checkpoint": last_checkpoint})
    status = "RUNNING"
    try:
        vla.train()
        projector.train()
        while epoch < args.epochs:
            if cursor >= plan["n_train"]:
                # A checkpoint exists at the completed-epoch boundary. Evaluate if not already recorded.
                already_evaluated = int(run_state.get("completed_epochs", 0)) >= epoch + 1
                if (already_evaluated and args.early_stopping_patience > 0
                        and epoch + 1 >= args.min_epochs
                        and int(run_state.get("stale_epochs", 0)) >= args.early_stopping_patience):
                    status = "EARLY_STOPPED"
                    break
                if not already_evaluated:
                    eval_result = evaluate_dev(torch, proc, vla, projector, plan, state_cache,
                                               prompts, prompt_ids, args, modules, stop_controller,
                                               runtime_deadline=runtime_deadline)
                    if not eval_result["complete"]:
                        status = "STOPPED_SAFE"
                        save_run_state(out_dir, {**run_state, "status": status,
                                                 "stop_reason": stop_controller.reason,
                                                 "last_checkpoint": last_checkpoint,
                                                 "global_step": global_step})
                        break
                    total_abs = sum(eval_result["by_task"][t]["action_mae"] *
                                    eval_result["by_task"][t]["n"] * 14 for t in TASKS)
                    total_n = sum(eval_result["by_task"][t]["n"] for t in TASKS)
                    dev_mae = float(total_abs / max(1, total_n * 14))
                    improved = run_state.get("best_dev_mae") is None or dev_mae < float(run_state["best_dev_mae"])
                    stale = 0 if improved else int(run_state.get("stale_epochs", 0)) + 1
                    best = dev_mae if improved else float(run_state["best_dev_mae"])
                    rec = {"epoch_completed": epoch + 1, "global_step": global_step,
                           "dev_mae": dev_mae, "improved": improved, "stale_epochs": stale,
                           "metrics": eval_result, "utc": utcnow()}
                    append_jsonl(dev_path, rec)
                    run_state = {**run_state, "dev_history": list(run_state.get("dev_history", [])) + [rec],
                                 "best_dev_mae": best, "stale_epochs": stale,
                                 "completed_epochs": epoch + 1, "global_step": global_step,
                                 "last_evaluated_epoch": epoch + 1, "last_checkpoint": last_checkpoint}
                    save_run_state(out_dir, run_state)
                    print(f"[R44] epoch={epoch + 1} dev_action_mae={dev_mae:.6f} best={best:.6f} stale={stale}", flush=True)
                    if stop_controller.should_stop():
                        status = "STOPPED_SAFE"
                        save_run_state(out_dir, {**run_state, "status": status,
                                                 "stop_reason": stop_controller.reason})
                        break
                    if args.early_stopping_patience > 0 and epoch + 1 >= args.min_epochs and stale >= args.early_stopping_patience:
                        status = "EARLY_STOPPED"
                        break
                epoch += 1
                if epoch >= args.epochs:
                    status = "COMPLETED"
                    break
                cursor = 0
                order = D.EpochSampler(plan["n_train"], args.seed, epoch).order
                vla.train()
                projector.train()
                continue

            if time.monotonic() >= runtime_deadline:
                stop_controller.request_stop("approved_gpu_budget_deadline")
            if stop_controller.should_stop():
                state = checkpoint_state(torch, vla, projector, optimizer, scheduler,
                                         epoch=epoch, cursor=cursor, order=order, global_step=global_step,
                                         plan=plan, base_identity=base_identity, model_info=model_info,
                                         prompts=prompts, prompt_ids=prompt_ids, run_state=run_state,
                                         reason=stop_controller.reason)
                last_checkpoint = str(save_checkpoint(out_dir, state, suffix="stop"))
                status = "STOPPED_SAFE"
                save_run_state(out_dir, {**run_state, "status": status, "stop_reason": stop_controller.reason,
                                         "last_checkpoint": last_checkpoint, "global_step": global_step})
                break

            if time.monotonic() >= runtime_deadline:
                stop_controller.request_stop("approved_gpu_budget_deadline")
                state = checkpoint_state(torch, vla, projector, optimizer, scheduler,
                                         epoch=epoch, cursor=cursor, order=order, global_step=global_step,
                                         plan=plan, base_identity=base_identity, model_info=model_info,
                                         prompts=prompts, prompt_ids=prompt_ids, run_state=run_state,
                                         reason=stop_controller.reason)
                last_checkpoint = str(save_checkpoint(out_dir, state, suffix="budget-stop"))
                status = "STOPPED_SAFE"
                save_run_state(out_dir, {**run_state, "status": status, "stop_reason": stop_controller.reason,
                                         "last_checkpoint": last_checkpoint, "global_step": global_step})
                break
            sample_idx = int(order[cursor])
            task, episode, t = plan["train_samples"][sample_idx]
            states, actions = state_cache[(task, episode)]
            row = M2.build_row(states, actions, int(t), TN.contract_for(plan["feature_payload"], task),
                               task, episode=episode)
            state_vec = np.asarray(states[int(t)], dtype=np.float32)
            action_vec = np.asarray(actions[int(t)], dtype=np.float32)
            lab, _delta = build_runtime_labels(prompt_ids[task], state_vec, action_vec, task,
                                               plan["delta_payload"], modules)
            pv = episode_images(proc, plan["episode_paths"][f"{task}/{episode}"], int(t), args.cams, modules)
            pr = torch.from_numpy(np.asarray(row["features"], dtype=np.float32)).reshape(1, 42).to("cuda")
            inp = torch.tensor([lab["input_ids"]], dtype=torch.long, device="cuda")
            tgt = torch.tensor([lab["labels"]], dtype=torch.long, device="cuda")
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                              labels=tgt, proprio=pr, proprio_projector=projector)
                    loss = out.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {global_step + 1}")
                loss.backward()
                params = [p for p in vla.parameters() if p.requires_grad] + list(projector.parameters())
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm at step {global_step + 1}")
                optimizer.step()
                scheduler.step()
            except torch.cuda.OutOfMemoryError as exc:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                atomic_json(out_dir / "FAILURES.json", {
                    "status": "FAIL_RESOURCE_OOM", "utc": utcnow(), "global_step": global_step,
                    "epoch": epoch, "cursor": cursor, "detail": repr(exc),
                    "note": "Do not trust partially mutated in-memory state; resume from last valid checkpoint.",
                })
                status = "FAIL_RESOURCE_OOM"
                break
            except FloatingPointError as exc:
                optimizer.zero_grad(set_to_none=True)
                atomic_json(out_dir / "FAILURES.json", {
                    "status": "FAIL_NUMERICS", "utc": utcnow(), "global_step": global_step,
                    "epoch": epoch, "cursor": cursor, "detail": str(exc),
                })
                status = "FAIL_NUMERICS"
                break

            step_seconds = time.perf_counter() - step_start
            cursor += 1
            global_step += 1
            append_jsonl(train_path, {
                "global_step": global_step, "epoch": epoch + 1, "cursor": cursor,
                "sample_index": sample_idx, "task": task, "episode": episode, "t": int(t),
                "loss": float(loss.detach().float().cpu()), "grad_norm": grad_norm,
                "lr_lora": float(optimizer.param_groups[0]["lr"]),
                "lr_projector": float(optimizer.param_groups[1]["lr"]),
                "step_wall_seconds": step_seconds, "utc": utcnow(),
            })
            append_jsonl(coverage_path, {
                "global_step": global_step, "epoch": epoch + 1, "cursor": cursor, "sample_index": sample_idx,
                "task": task, "episode": episode, "t": int(t),
            })

            due_checkpoint = (
                global_step % args.checkpoint_every_steps == 0
                or cursor == plan["n_train"]
            )
            if due_checkpoint:
                state = checkpoint_state(torch, vla, projector, optimizer, scheduler,
                                         epoch=epoch, cursor=cursor, order=order, global_step=global_step,
                                         plan=plan, base_identity=base_identity, model_info=model_info,
                                         prompts=prompts, prompt_ids=prompt_ids, run_state=run_state)
                last_checkpoint = str(save_checkpoint(out_dir, state))
                save_run_state(out_dir, {**run_state, "status": "RUNNING",
                                         "last_checkpoint": last_checkpoint, "global_step": global_step,
                                         "epoch": epoch, "cursor": cursor})
                print(f"[R44] checkpoint saved: {last_checkpoint}", flush=True)

            if global_step % args.resource_check_every == 0:
                stop_controller.observe_cuda_free_memory()
            if stop_controller.should_stop():
                if last_checkpoint is None or int(state.get("global_step", -1)) != global_step:
                    state = checkpoint_state(torch, vla, projector, optimizer, scheduler,
                                             epoch=epoch, cursor=cursor, order=order, global_step=global_step,
                                             plan=plan, base_identity=base_identity, model_info=model_info,
                                             prompts=prompts, prompt_ids=prompt_ids, run_state=run_state,
                                             reason=stop_controller.reason)
                    last_checkpoint = str(save_checkpoint(out_dir, state, suffix="stop"))
                status = "STOPPED_SAFE"
                save_run_state(out_dir, {**run_state, "status": status, "stop_reason": stop_controller.reason,
                                         "last_checkpoint": last_checkpoint, "global_step": global_step})
                break

        verify_files_unchanged(plan, verify_sha256=True)
        if status == "COMPLETED" or status == "EARLY_STOPPED":
            run_state = {**run_state, "status": status, "global_step": global_step,
                         "completed_epochs": int(run_state.get("completed_epochs", epoch)),
                         "utc_end": utcnow(), "last_checkpoint": last_checkpoint}
            save_run_state(out_dir, run_state)
            final_path = export_final(out_dir, vla, projector, plan, model_info, prompts,
                                      prompt_ids, global_step, int(run_state["completed_epochs"]))
            print(f"[R44] final export: {final_path}", flush=True)
        else:
            save_run_state(out_dir, {**run_state, "status": status, "global_step": global_step,
                                     "last_checkpoint": last_checkpoint, "utc_end": utcnow()})
        atomic_json(out_dir / "exit_report.json", {
            "status": status, "utc_end": utcnow(), "global_step": global_step,
            "epoch": epoch + 1, "cursor": cursor, "last_checkpoint": last_checkpoint,
            "gpu_before": gpu_before, "gpu_after": gpu_snapshot(args, enforce_headroom=False),
            "peak_memory_gb": float(torch.cuda.max_memory_allocated()) / (1024**3),
            "training_started": True, "resource_mode": args.resource_mode,
            "gpu_lock_path": gpu_lock.path,
        })
        return 0 if status in ("COMPLETED", "EARLY_STOPPED", "STOPPED_SAFE") else 3
    finally:
        stop_controller.restore_signal_handlers()
        gpu_lock.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
