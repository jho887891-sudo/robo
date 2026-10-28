#!/usr/bin/env python3
"""CPU-only shadow profile of REAL TRAIN 3-camera HDF5/JPEG I/O.

This measures the exact stored JPEG bytes, decode path, and R45 RuntimeReader
with an identity CPU transform. It does NOT measure the production OpenVLA
image processor, CUDA transfer, GPU compute, or end-to-end train samples/sec.
No model weights are loaded. A fixed seed and source hashes make results auditable.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(b)
    return digest.hexdigest()


def select_samples(index, per_task, seed):
    groups = defaultdict(list)
    for task, path, t in index:
        groups[task].append((str(path), int(t)))
    chosen = []
    for task in sorted(groups):
        candidates = groups[task]
        if len(candidates) < per_task:
            raise ValueError(f"{task}: {len(candidates)} TRAIN samples < requested {per_task}")
        keyed = sorted(candidates, key=lambda pair:
            hashlib.sha256(f"{seed}|{task}|{pair[0]}|{pair[1]}".encode()).digest())
        chosen.extend((task, path, t) for path, t in keyed[:per_task])
    if not chosen:
        raise ValueError("empty TRAIN index")
    return chosen


def legacy_cpu_images(path, t, processor):
    import cv2
    import h5py
    import numpy as np
    import torch
    from PIL import Image
    from training.r45_data import CAMERAS
    tiles = []
    with h5py.File(path, "r") as h:
        for cam in CAMERAS:
            raw = h[f"vision/{cam}/colors"][t]
            decoded = cv2.imdecode(np.frombuffer(bytes(raw), np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"JPEG decode failed at {path} t={t} cam={cam}")
            tiles.append(processor.image_processor.apply_transform(
                Image.fromarray(np.ascontiguousarray(decoded[..., ::-1]))))
    return torch.cat(tiles, dim=0).unsqueeze(0).to(torch.bfloat16)


class CpuIdentityProcessor:
    """A controlled transform proxy: NOT the model's resize/normalize pipeline."""
    def __init__(self):
        self.image_processor = self

    def apply_transform(self, image):
        import numpy as np
        import torch
        arr = np.asarray(image, dtype=np.uint8)
        return torch.from_numpy(np.array(arr, copy=True)).permute(2, 0, 1).unsqueeze(0)


def worker_decode(entry):
    import cv2
    import numpy as np
    from training.r45_data import CAMERAS, _handle
    task, path, t = entry
    h = _handle(path)
    npx = 0
    shapes = []
    for cam in CAMERAS:
        raw = h[f"vision/{cam}/colors"][int(t)]
        bgr = cv2.imdecode(np.frombuffer(bytes(raw), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"worker failed JPEG: {path} {t} {cam}")
        shapes.append(tuple(int(v) for v in bgr.shape))
        npx += int(bgr.size)
    return (task, Path(path).name, int(t), npx, shapes)


def describe(series):
    if not series:
        return {"n": 0}
    a = sorted(series)
    return {"n": len(a), "mean_s": float(statistics.fmean(a)),
            "median_s": float(statistics.median(a)),
            "p95_s": float(a[max(0, (95 * len(a) + 99) // 100 - 1)]),
            "sum_s": float(sum(a))}


def run_profile(args):
    import numpy as np
    import torch
    from training import r45_data as R45

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA context unexpectedly initialized")
    if args.per_task < 1 or args.repeats < 1 or args.cache_bytes < 0:
        raise ValueError("per-task/repeats must be positive; cache-bytes nonnegative")
    if args.max_open_handles < 1:
        raise ValueError("max-open-handles must be >= 1")
    counts = tuple(int(x) for x in args.workers.split(","))
    if not counts or any(x < 1 or x > 8 for x in counts):
        raise ValueError("--workers must contain integers in [1,8]")
    source = Path(args.splits)
    root = Path(args.hdf5_root)
    out = Path(args.out)
    if not source.is_file() or not root.is_dir():
        raise FileNotFoundError("real splits JSON and HDF5 root are required")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {out}")

    splits = json.loads(source.read_text(encoding="utf-8"))["splits"]
    entries = R45.build_index(splits, str(root), which="train")
    chosen = select_samples(entries, args.per_task, args.seed)
    plan_hash = hashlib.sha256(json.dumps(
        [(task, Path(path).name, t) for task, path, t in chosen],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    proc = CpuIdentityProcessor()
    reader = R45.RuntimeReader(cache_bytes=args.cache_bytes,
                               max_open_handles=args.max_open_handles)
    records = []
    start = time.perf_counter()
    try:
        for repeat in range(args.repeats):
            for task, path, t in chosen:
                t0 = time.perf_counter()
                baseline = legacy_cpu_images(path, t, proc)
                baseline_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                optimized, timings = reader.load_images(
                    proc, path, t, 3, R45.CAMERAS, device="cpu")
                optimized_s = time.perf_counter() - t0
                if not torch.equal(baseline, optimized):
                    raise AssertionError(f"pixel parity failure: {task}/{Path(path).name} t={t}")
                records.append({"task": task, "episode": Path(path).name, "t": t,
                                "repeat": repeat, "legacy_cpu_s": baseline_s,
                                "r45_cpu_s": optimized_s, **timings})
        reader_stats = {"open_handles_at_end": len(reader._handles),
                        "image_cache_bytes": reader._cache_bytes,
                        "cache_cap_bytes": reader.cache_bytes}
    finally:
        reader.close()

    workers = []
    for count in counts:
        start_pool = time.perf_counter()
        # Explicit 'spawn' avoids inheriting a parent's HDF5 handles.
        import multiprocessing as mp
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=count, mp_context=mp.get_context("spawn"),
                initializer=R45.worker_init, initargs=(0,)) as executor:
            results = list(executor.map(worker_decode, chosen))
        workers.append({"workers": count, "samples": len(results),
                        "total_wall_s_including_startup": time.perf_counter() - start_pool,
                        "decoded_pixel_count": sum(x[3] for x in results)})
        if len(results) != len(chosen):
            raise AssertionError("worker sample count mismatch")

    report = {
        "schema": "r45-cpu-train-io-profile/v1",
        "status": "CPU_IO_PROXY_PASS",
        "scope": "real TRAIN HDF5, 3-camera JPEG, identity CPU transform, no model/GPU",
        "excluded": ["production image resize/normalize", "CPU->CUDA", "7B model",
                     "training optimizer", "training samples/sec"],
        "split_sha256": sha_file(source),
        "sample_manifest_sha256": plan_hash,
        "index_n": len(entries), "selected_n": len(chosen),
        "seed": args.seed, "per_task": args.per_task, "repeats": args.repeats,
        "phase": {k: describe([float(row[k]) for row in records])
                  for k in ("legacy_cpu_s", "r45_cpu_s", "hdf5_read_s",
                            "jpeg_preprocess_s", "cpu_to_gpu_s")},
        "cache_hits": int(sum(row["cache_hit"] for row in records)),
        "cache_total": len(records),
        "cache": reader_stats,
        "worker_runs": workers,
        "total_profile_s": time.perf_counter() - start,
        "code_sha256": sha_file(Path(__file__)),
    }
    # Only create output after all validation, parity and worker checks pass.
    out.mkdir(parents=True, exist_ok=False)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    (out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "selected_samples.json").write_text(
        json.dumps([(task, Path(path).name, t) for task, path, t in chosen],
                   indent=2) + "\n", encoding="utf-8")
    with (out / "SHA256SUMS").open("w") as f:
        for name in ("samples.jsonl", "report.json", "selected_samples.json"):
            f.write(f"{sha_file(out / name)}  {name}\n")
    print(json.dumps({"status": "CPU_IO_PROXY_PASS",
                      "evidence": str(out), "selected_n": len(chosen),
                      "sample_manifest_sha256": plan_hash}, sort_keys=True))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--hdf5-root", required=True)
    ap.add_argument("--out", required=True, help="new, non-existing evidence directory")
    ap.add_argument("--per-task", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=45)
    ap.add_argument("--cache-bytes", type=int, default=256 * 1024 * 1024)
    ap.add_argument("--max-open-handles", type=int, default=24)
    ap.add_argument("--workers", default="1,2,4",
                    help="process count for separate JPEG/HDF5 worker proxy")
    args = ap.parse_args()
    try:
        return run_profile(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"BLOCKED_CPU_IO_PROFILE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
