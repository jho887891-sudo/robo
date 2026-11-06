#!/usr/bin/env python3
"""CPU-only R50 P02 A/B of tensor assembly after the real OpenVLA processor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summary(values):
    values = sorted(values)
    return {"n": len(values), "mean_ms": statistics.mean(values) * 1000,
            "median_ms": statistics.median(values) * 1000,
            "p95_ms": values[int(0.95 * (len(values) - 1))] * 1000}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--hdf5-root", type=Path, required=True)
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--oft-root", type=Path, required=True)
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if args.out.exists() or args.repeats < 3:
        raise ValueError("fresh output and three independent repeats required")
    import cv2
    import h5py
    import numpy as np
    import torch
    from PIL import Image
    from training.r45_data import CAMERAS
    from training.r50_data_pipeline import load_openvla_processor
    from training.r50_batch_builder import assemble_camera_tiles
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized")
    proc = load_openvla_processor(str(args.base_model), str(args.oft_root))
    selected = json.loads(args.selected.read_text())
    splits = json.loads(args.splits.read_text())["splits"]
    train_names = {task: {Path(x).name for x in splits[task]["train"]} for task in splits}
    dev_names = {task: {Path(x).name for x in splits[task]["dev"]} for task in splits}
    for task, episode, frame in selected:
        if episode not in train_names[task] or episode in dev_names[task] or int(frame) < 2:
            raise ValueError("selected sample violates frozen TRAIN split")
    args.out.mkdir(parents=True, mode=0o700)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    environment = {"code_sha": head, "selected_sha256": sha(args.selected),
                   "split_sha256": sha(args.splits),
                   "processor_sha256": sha(args.base_model / "preprocessor_config.json"),
                   "numpy": np.__version__, "torch": torch.__version__,
                   "camera_order": list(CAMERAS), "repeats": args.repeats,
                   "scope": "CPU tensor assembly; no GPU/model training"}
    (args.out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    records = []
    with (args.out / "samples.jsonl").open("w", buffering=1) as stream:
        for repeat in range(args.repeats):
            for task, episode, frame in selected:
                path = args.hdf5_root / task / "arx_x5" / "data" / episode
                start = time.perf_counter()
                tiles = []
                with h5py.File(path, "r") as h:
                    for camera in CAMERAS:
                        raw = h[f"vision/{camera}/colors"][int(frame)]
                        bgr = cv2.imdecode(np.frombuffer(bytes(raw), np.uint8),
                                           cv2.IMREAD_COLOR)
                        if bgr is None:
                            raise ValueError("JPEG decode failed")
                        tiles.append(proc.image_processor.apply_transform(
                            Image.fromarray(np.ascontiguousarray(bgr[..., ::-1]))))
                preprocess_s = time.perf_counter() - start
                # Alternate the order to reduce thermal/cache bias in the tiny assembly step.
                order = ("cat", "preallocate_bf16") if (repeat + int(frame)) % 2 else \
                    ("preallocate_bf16", "cat")
                results = {}
                for method in order:
                    t0 = time.perf_counter()
                    pixels = assemble_camera_tiles(tiles, method)
                    duration = time.perf_counter() - t0
                    results[method] = (pixels, duration)
                if not torch.equal(results["cat"][0], results["preallocate_bf16"][0]):
                    raise AssertionError(f"batch tensor mismatch at {task}/{episode} t={frame}")
                if tuple(results["cat"][0].shape) != (1, 18, 224, 224):
                    raise AssertionError("OpenVLA tensor shape drift")
                row = {"task": task, "episode": episode, "frame": int(frame),
                       "repeat": repeat, "preprocess_s": preprocess_s,
                       "cat_s": results["cat"][1],
                       "preallocate_bf16_s": results["preallocate_bf16"][1]}
                records.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    old = summary([r["cat_s"] for r in records])
    new = summary([r["preallocate_bf16_s"] for r in records])
    report = {"status": "PASS_CPU_PARITY", "code_sha": head,
              "samples": len(selected), "visits": len(records),
              "baseline_cat": old, "candidate_preallocate_bf16": new,
              "assembly_mean_gain_fraction": 1 - new["mean_ms"] / old["mean_ms"],
              "preprocess": summary([r["preprocess_s"] for r in records]),
              "limits": "Assembly only; processor and HDF5 still run on CPU; no 7B throughput"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out / "SHA256SUMS").open("w") as stream:
        for name in ("environment.json", "samples.jsonl", "report.json"):
            stream.write(f"{sha(args.out / name)}  {name}\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
