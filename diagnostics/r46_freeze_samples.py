#!/usr/bin/env python3
"""Freeze five deterministic real push_T TRAIN samples using CPU-only production readers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "training") not in sys.path:
    sys.path.insert(0, str(REPO / "training"))
from diagnostics.r46_smoke_contract import (  # noqa: E402
    CAMERA_ORDER, FROZEN_CONTRACT, SELECTION_SOURCE_SHA, canonical_plan_sha256,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ordered_train_episodes(train_episodes, seed=0):
    return sorted(train_episodes, key=lambda episode: hashlib.sha256(
        f"{seed}|push_T|{episode}".encode()).hexdigest())


def sample_time_index(episode: str, frame_count: int, seed=0) -> int:
    if frame_count < 4:
        raise ValueError("episode must have at least four state/action frames")
    digest = hashlib.sha256(f"{seed}|push_T|{episode}".encode()).hexdigest()
    return 2 + int(digest, 16) % (frame_count - 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--hdf5-root", required=True)
    ap.add_argument("--feature-normalizer", required=True)
    ap.add_argument("--delta-normalizer", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--oft-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing sample plan: {out}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import numpy as np
    import cv2
    import h5py
    import torch
    from PIL import Image
    torch.cuda._lazy_init = lambda: (_ for _ in ()).throw(RuntimeError("CUDA initialization forbidden in R46 sample freeze"))

    import r44_train as R44
    modules = R44.load_runtime_modules(argparse.Namespace(oft_root=args.oft_root))
    _R11, _R, TC, M2, TN, D = modules
    split_path, manifest_path = Path(args.splits).resolve(), Path(args.manifest).resolve()
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_sha, manifest_sha = sha256_file(split_path), sha256_file(manifest_path)
    delta_payload = json.loads(Path(args.delta_normalizer).read_text(encoding="utf-8"))
    D.validate_normalizer(delta_payload)
    if delta_payload.get("fit", {}).get("split_sha256") != split_sha:
        raise ValueError("delta normalizer split SHA does not match frozen split")
    split_map = split_doc["splits"]
    if set(split_map) != set(M2.TASKS):
        raise ValueError("frozen split must contain exactly the three R44 tasks")
    for task in M2.TASKS:
        train_part, dev_part = split_map[task]["train"], split_map[task]["dev"]
        train_ep = [M2.split_qualified(key)[1] for key in train_part]
        dev_ep = [M2.split_qualified(key)[1] for key in dev_part]
        if len(set(train_ep)) != len(train_ep) or len(set(dev_ep)) != len(dev_ep) \
                or set(train_ep) & set(dev_ep):
            raise ValueError(f"TRAIN/DEV has duplicates or overlap for task {task}")
    train_keys = split_map["push_T"]["train"]
    dev_keys = set(split_map["push_T"]["dev"])
    train_keys = sorted(train_keys)
    if set(train_keys) & dev_keys:
        raise ValueError("TRAIN/DEV overlap in push_T")
    train_eps = [M2.split_qualified(key)[1] for key in train_keys]
    dev_eps = [M2.split_qualified(key)[1] for key in dev_keys]
    feature = TN.load_payload(args.feature_normalizer, tasks=list(M2.TASKS), dev_lists={
        task: [M2.split_qualified(key)[1] for key in split_map[task]["dev"]]
        for task in M2.TASKS
    })
    base = Path(args.base_model).resolve()
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    proc = AutoProcessor.from_pretrained(str(base), trust_remote_code=True)
    prompt_ids = R44.build_prompt_ids(proc, "push_T", modules, False)
    contract = TN.contract_for(feature, "push_T")

    root = Path(args.hdf5_root).resolve()
    ordered_episodes = ordered_train_episodes(train_eps, seed=0)
    candidates = []
    for episode in ordered_episodes:
        key = f"push_T/{episode}"
        rec = (manifest.get("files") or {}).get(key)
        path = (root / "push_T" / "arx_x5" / "data" / episode).resolve()
        if root not in path.parents or not path.is_file() or not rec:
            raise ValueError(f"TRAIN HDF5 or manifest record missing for {key}")
        candidates.append((episode, path, rec))

    selected = []
    reader = R44.R45DATA.RuntimeReader(cache_bytes=0)
    errors = []
    for episode, path, rec in candidates:
        if len(selected) == 5 or len(errors) >= 5:
            break
        t = None
        try:
            with h5py.File(path, "r") as h5:
                if tuple(h5["vision"].keys()) != tuple(CAMERA_ORDER):
                    raise ValueError(f"camera order mismatch: push_T/{episode}")
                frames = int(h5["state/left_arm_joint_states"].shape[0])
            t = sample_time_index(episode, frames, seed=0)
            before = R44.stat_identity(path)
            actual_sha = sha256_file(path)
            if R44.stat_identity(path) != before:
                raise ValueError("HDF5 changed while hashing")
            if actual_sha != rec.get("sha256"):
                raise ValueError("HDF5 SHA differs from strict manifest")
            with h5py.File(path, "r") as h5:
                instr = h5["instruction"][()]
                instr = instr.decode("utf-8") if isinstance(instr, bytes) else str(instr)
                if instr != M2.TASK_PROMPTS["push_T"]:
                    raise ValueError("instruction differs from frozen push_T prompt")
                states, actions = TC.read14(h5, "state"), TC.read14(h5, "action")
                rgb_shapes = []
                for camera in CAMERA_ORDER:
                    buf = bytes(h5[f"vision/{camera}/colors"][t])
                    bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise ValueError(f"JPEG decode failed for {camera}")
                    rgb = np.ascontiguousarray(bgr[..., ::-1])
                    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                        raise ValueError(f"RGB contract failed for {camera}: {rgb.shape}/{rgb.dtype}")
                    rgb_shapes.append(list(rgb.shape))
            row = M2.build_row(states, actions, t, contract, "push_T", episode=episode)
            features = np.asarray(row["features"], dtype=np.float32)
            if features.shape != (42,) or not np.isfinite(features).all():
                raise ValueError("42D feature row invalid")
            labels, _target = R44.build_runtime_labels(
                prompt_ids, np.asarray(states[t], dtype=np.float32),
                np.asarray(actions[t], dtype=np.float32), "push_T", delta_payload, modules,
            )
            action_tokens = labels["labels"][-15:-1]
            if len(action_tokens) != 14 or any(x < 31744 or x > 31999 for x in action_tokens):
                raise ValueError("14 action-token labels invalid")
            # Run the production three-camera transform on CPU; this must never create CUDA.
            pixels, timing = reader.load_images(proc, path, t, 3, CAMERA_ORDER, device="cpu")
            if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[1] <= 0:
                raise ValueError(f"production camera transform returned invalid shape {tuple(pixels.shape)}")
            if list(pixels.shape) != FROZEN_CONTRACT["processor_output_shape"]:
                raise ValueError(f"production processor output {list(pixels.shape)} differs from frozen interface")
            if torch.cuda.is_initialized():
                raise RuntimeError("unexpected CUDA context during CPU sample freeze")
            selected.append({
                "order_index": len(selected), "role": "warmup" if not selected else "measured",
                "task": "push_T", "episode": episode, "t": t,
                "hdf5_sha256": actual_sha, "camera_order": list(CAMERA_ORDER),
                "image_shape": [480, 640, 3], "image_dtype": "uint8", "image_colorspace": "RGB",
                "proprio_shape": [1, 42], "action_token_ids": [int(x) for x in action_tokens],
                "stop_token_id": 2, "prompt_labels_ignored": labels["labels"][:labels["n_prompt"]] == [-100] * labels["n_prompt"],
                "cpu_decoded_rgb_shapes": rgb_shapes,
                "processor_output_shape": list(pixels.shape),
                "cpu_reader_output_shape": list(pixels.shape), "cpu_reader_timing": timing,
            })
        except Exception as exc:
            errors.append({"episode": episode, "t": t, "error": f"{type(exc).__name__}: {exc}"})
    reader.close()
    if len(selected) != 5:
        raise RuntimeError(f"BLOCKED_SAMPLE_COVERAGE: selected {len(selected)}/5; failures={errors!r}")

    plan = {
        "schema": "r46-smoke-samples/v1", "source_sha": SELECTION_SOURCE_SHA,
        "selection_tool_sha256": sha256_file(Path(__file__).resolve()),
        "seed": 0, "selection_algorithm": "sha256('0|push_T|episode') order; frame index from sha256('0|push_T|episode') modulo valid TRAIN time indices",
        "task": "push_T", "sample_count": 5,
        "split_sha256": split_sha, "manifest_sha256": manifest_sha,
        "contract": FROZEN_CONTRACT, "samples": selected,
    }
    plan["sample_plan_sha256"] = canonical_plan_sha256(plan)
    from diagnostics.r46_smoke_contract import validate_sample_plan
    validation = validate_sample_plan(
        plan, split_doc=split_doc, manifest=manifest, split_sha256=split_sha,
        manifest_sha256=manifest_sha, hdf5_root=root,
        source_sha=SELECTION_SOURCE_SHA, verify_files=False,
    )
    if validation["sample_count"] != 5:
        raise RuntimeError("central sample-plan validator did not accept five real TRAIN rows")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CPU_SAMPLE_PLAN_FROZEN", "plan": str(out),
                      "sample_plan_sha256": plan["sample_plan_sha256"],
                      "selected": [{"episode": x["episode"], "t": x["t"], "role": x["role"]}
                                  for x in selected],
                      "cuda_initialized": torch.cuda.is_initialized(),
                      "candidate_failures_before_five": errors}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
