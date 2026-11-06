#!/usr/bin/env python3
"""CPU-only parity probe for the opt-in GPU tensor preprocessing candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for key in ("selected", "hdf5_root", "base_model", "oft_root", "out"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.out.exists() or not 1 <= args.limit <= 30:
        raise ValueError("fresh output and limit 1..30 required")
    import cv2
    import h5py
    import numpy as np
    import torch
    from training.r45_data import CAMERAS
    from training.r50_data_pipeline import load_openvla_processor
    from training.r51_gpu_preprocess_candidate import (compare_with_reference,
                                                      compare_hybrid_with_reference)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized")
    processor = load_openvla_processor(str(args.base_model), str(args.oft_root))
    selected = json.loads(args.selected.read_text())[:args.limit]
    rows = []
    for task, episode, frame in selected:
        if task not in ("push_T", "sweep_blocks"):
            raise ValueError("non-R51 task")
        path = args.hdf5_root / task / "arx_x5" / "data" / episode
        with h5py.File(path, "r") as h:
            rgb = []
            for camera in CAMERAS:
                raw = np.asarray(h[f"vision/{camera}/colors"][int(frame)])
                bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("JPEG decode failed")
                rgb.append(torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])))
        images = torch.stack(rgb).unsqueeze(0)
        direct = compare_with_reference(images, processor.image_processor)
        hybrid = compare_hybrid_with_reference(images, processor.image_processor)
        rows.append({"task": task, "episode": episode, "frame": frame,
                     "direct": direct, "hybrid": hybrid})
    args.out.mkdir(mode=0o700, parents=True)
    (args.out / "samples.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    report = {"schema": "r51-gpu-candidate-cpu-parity/v1",
              "status": "HYBRID_EXACT_CPU" if all(x["hybrid"]["exact"] for x in rows)
              else "HYBRID_MISMATCH_CPU",
              "samples": len(rows),
              "direct_exact_samples": sum(x["direct"]["exact"] for x in rows),
              "direct_max_abs_error": max(x["direct"]["max_abs_error"] for x in rows),
              "hybrid_exact_samples": sum(x["hybrid"]["exact"] for x in rows),
              "hybrid_max_abs_error": max(x["hybrid"]["max_abs_error"] for x in rows),
              "selected_sha256": digest(args.selected),
              "processor_sha256": digest(args.base_model / "preprocessor_config.json"),
              "limits": "CPU emulation only; no CUDA, GPU timing or 7B model training"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out / "SHA256SUMS").open("w") as stream:
        for name in ("report.json", "samples.jsonl"):
            stream.write(f"{digest(args.out / name)}  {name}\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
