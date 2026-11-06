#!/usr/bin/env python3
"""CPU-only P00: fixed real TRAIN samples with the production OpenVLA processor.

This measures data preparation only. It never loads 7B weights or initializes CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe(values: list[float]) -> dict:
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean_ms": 1000 * sum(values) / len(values),
            "median_ms": 1000 * values[len(values) // 2],
            "total_s": sum(values)}


def legacy_cpu_sample(path: Path, frame: int, proc, camera_order):
    import cv2
    import h5py
    import numpy as np
    import torch
    from PIL import Image

    phases = {"hdf5_s": 0.0, "jpeg_s": 0.0, "preprocess_s": 0.0,
              "tensor_s": 0.0}
    tiles = []
    with h5py.File(path, "r") as h:
        for camera in camera_order:
            start = time.perf_counter()
            raw = h[f"vision/{camera}/colors"][frame]
            phases["hdf5_s"] += time.perf_counter() - start
            start = time.perf_counter()
            bgr = cv2.imdecode(np.frombuffer(bytes(raw), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"JPEG decode failed: {path.name} t={frame} {camera}")
            rgb = np.ascontiguousarray(bgr[..., ::-1])
            phases["jpeg_s"] += time.perf_counter() - start
            start = time.perf_counter()
            tile = proc.image_processor.apply_transform(Image.fromarray(rgb))
            phases["preprocess_s"] += time.perf_counter() - start
            tiles.append(tile)
    start = time.perf_counter()
    pixels = torch.cat(tiles, dim=0).unsqueeze(0).to(torch.bfloat16)
    phases["tensor_s"] += time.perf_counter() - start
    return pixels, phases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--hdf5-root", type=Path, required=True)
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--oft-root", type=Path, required=True,
                    help="Local prismatic source path; the model is standard OpenVLA-7B")
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--batch-assembly", choices=("cat", "preallocate_bf16"),
                    default="cat")
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if not all(p.exists() for p in (args.splits, args.hdf5_root, args.base_model,
                                   args.oft_root, args.selected)):
        raise FileNotFoundError("one or more frozen inputs do not exist")
    import numpy as np
    import torch
    import transformers
    import peft
    import h5py
    import cv2
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before CPU profile")
    versions = {"python": sys.version.split()[0], "numpy": np.__version__,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "peft": peft.__version__, "h5py": h5py.__version__, "opencv": cv2.__version__}
    expected = {"python": "3.10.12", "numpy": "1.26.4", "torch": "2.5.1+cu124",
                "transformers": "4.46.3", "peft": "0.13.2", "h5py": "3.16.0"}
    if any(versions[key] != value for key, value in expected.items()):
        raise RuntimeError(f"frozen environment mismatch: {versions}")
    sys.path.insert(0, str(args.oft_root))
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    proc = AutoProcessor.from_pretrained(str(args.base_model), trust_remote_code=True,
                                         local_files_only=True)
    from training.r45_data import CAMERAS, RuntimeReader

    split_doc = json.loads(args.splits.read_text())['splits']
    selected = json.loads(args.selected.read_text())
    for task, episode, frame in selected:
        if episode not in {Path(s).name for s in split_doc[task]["train"]}:
            raise ValueError(f"not a frozen TRAIN episode: {task}/{episode}")
        if episode in {Path(s).name for s in split_doc[task]["dev"]}:
            raise ValueError(f"TRAIN/DEV overlap: {task}/{episode}")
        if not isinstance(frame, int) or frame < 2:
            raise ValueError("invalid frame index")
    # R51 replacement draws intentionally repeat some sample identities.
    args.out.mkdir(parents=True, mode=0o700)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = {"code_sha": head, "versions": versions, "selected_sha256": digest(args.selected),
              "split_sha256": digest(args.splits),
              "processor_sha256": digest(args.base_model / "preprocessor_config.json"),
              "camera_order": list(CAMERAS), "repeats": args.repeats,
              "batch_assembly": args.batch_assembly,
              "threads": {k: os.environ.get(k) for k in
                          ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
              "scope": "CPU production processor; no CUDA/model weights/training"}
    (args.out / "environment.json").write_text(json.dumps(config, indent=2) + "\n")
    reader = RuntimeReader(cache_bytes=0, batch_assembly=args.batch_assembly)
    records = []
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        for repeat in range(args.repeats):
            for task, episode, frame in selected:
                path = args.hdf5_root / task / "arx_x5" / "data" / episode
                if not path.is_file():
                    raise FileNotFoundError(path)
                start = time.perf_counter()
                old, phases = legacy_cpu_sample(path, frame, proc, CAMERAS)
                old_s = time.perf_counter() - start
                start = time.perf_counter()
                new, timings = reader.load_images(proc, path, frame, 3, CAMERAS,
                                                  device="cpu")
                new_s = time.perf_counter() - start
                if not torch.equal(old, new):
                    raise AssertionError(f"processor tensor mismatch: {task}/{episode} t={frame}")
                if tuple(new.shape) != (1, 18, 224, 224) or new.dtype != torch.bfloat16:
                    raise AssertionError(f"unexpected processor tensor {new.shape} {new.dtype}")
                records.append({"task": task, "episode": episode, "frame": frame,
                                "repeat": repeat, "legacy_wall_s": old_s,
                                "reader_wall_s": new_s, "legacy_phases": phases,
                                "reader_phases": timings, "shape": list(new.shape)})
    finally:
        reader.close()
    elapsed = time.perf_counter() - wall_start
    with (args.out / "samples.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    reader_total = sum(r["reader_wall_s"] for r in records)
    report = {"status": "PASS_CPU_ONLY", "code_sha": head,
              "samples": len(selected), "visits": len(records),
              "comparison_wall_s": elapsed,
              "comparison_pairs_per_s": len(records) / elapsed,
              "reader_serial_samples_per_s": len(records) / reader_total,
              "process_cpu_s": time.process_time() - cpu_start,
              "process_cpu_cores_used": (time.process_time() - cpu_start) / elapsed,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "cache_hits": sum(int(r["reader_phases"]["cache_hit"]) for r in records),
              "legacy_wall": describe([r["legacy_wall_s"] for r in records]),
              "reader_wall": describe([r["reader_wall_s"] for r in records]),
              "hdf5": describe([r["legacy_phases"]["hdf5_s"] for r in records]),
              "jpeg": describe([r["legacy_phases"]["jpeg_s"] for r in records]),
              "processor": describe([r["legacy_phases"]["preprocess_s"] for r in records]),
              "tensor": describe([r["legacy_phases"]["tensor_s"] for r in records]),
              "gpu_utilization": None, "vram": None,
              "limitations": "CPU only; serial old-then-new order can favor the latter via OS cache; no 7B throughput"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (args.out / "SHA256SUMS").open("w") as stream:
        for name in ("environment.json", "samples.jsonl", "report.json"):
            stream.write(f"{digest(args.out / name)}  {name}\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
