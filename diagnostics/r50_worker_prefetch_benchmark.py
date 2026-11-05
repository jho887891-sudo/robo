#!/usr/bin/env python3
"""CPU-only R50 P01 DataLoader matrix on frozen real TRAIN samples."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
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


def pixel_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(__import__("torch").uint8)
                          .numpy().tobytes()).hexdigest()


def integers(value, minimum, maximum):
    out = [int(x) for x in value.split(",")]
    if not out or len(set(out)) != len(out) or any(x < minimum or x > maximum for x in out):
        raise ValueError("invalid configuration list")
    return out


def rss_bytes():
    import psutil
    proc = psutil.Process()
    members = [proc] + proc.children(recursive=True)
    total = 0
    for member in members:
        try:
            total += member.memory_info().rss
        except psutil.Error:
            pass
    return total


def one_run(selected, roots, expected, workers, prefetch, pin, warmup):
    import torch
    from training.r50_data_pipeline import R50WorkerDataset, create_loader
    dataset = R50WorkerDataset(selected, *roots, cache_bytes=0)
    start = time.perf_counter()
    loader = create_loader(dataset, workers, prefetch, pin)
    startup = None
    epochs = []
    peak_rss = 0
    try:
        for epoch in range(2):
            iterator = iter(loader)
            waits = []
            hdf5 = 0.0
            jpeg_preprocess = 0.0
            pinned = 0
            since = time.perf_counter()
            steady_start = None
            for index in range(len(selected)):
                t0 = time.perf_counter()
                item = next(iterator)
                wait = time.perf_counter() - t0
                if startup is None:
                    startup = time.perf_counter() - start
                actual = int(item["index"].item())
                if actual != index:
                    raise AssertionError(f"order changed at epoch={epoch}: {actual} != {index}")
                pixels = item["pixels"]
                if tuple(pixels.shape) != (1, 18, 224, 224) or pixels.dtype != torch.bfloat16:
                    raise AssertionError(f"shape/dtype drift: {pixels.shape} {pixels.dtype}")
                if pixel_sha(pixels[0]) != expected[index]:
                    raise AssertionError(f"pixel mismatch at index={index} epoch={epoch}")
                waits.append(wait)
                hdf5 += float(item["hdf5_s"].item())
                jpeg_preprocess += float(item["jpeg_preprocess_s"].item())
                pinned += int(pixels.is_pinned())
                if (index + 1) == warmup:
                    steady_start = time.perf_counter()
                if index % 8 == 0:
                    peak_rss = max(peak_rss, rss_bytes())
                    if peak_rss > 12 * 1024**3:
                        raise MemoryError("R50 P01 worker RSS exceeded 12 GiB")
            end = time.perf_counter()
            epochs.append({"epoch": epoch, "n": len(selected),
                           "wall_s": end - since,
                           "warmup_samples": warmup,
                           "steady_samples_per_s": (len(selected) - warmup) /
                           (end - steady_start),
                           "batch_wait_mean_ms": 1000 * sum(waits) / len(waits),
                           "hdf5_worker_s": hdf5,
                           "jpeg_preprocess_worker_s": jpeg_preprocess,
                           "pinned_batches": pinned})
    finally:
        iterator_obj = getattr(loader, "_iterator", None)
        if iterator_obj is not None:
            iterator_obj._shutdown_workers()
        del loader
        dataset.close()
        gc.collect()
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA context initialized by CPU DataLoader")
    return {"workers": workers, "prefetch": prefetch, "pin_memory_requested": pin,
            "persistent_workers": True, "startup_first_batch_s": startup,
            "peak_parent_plus_workers_rss_bytes": peak_rss, "epochs": epochs}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--hdf5-root", type=Path, required=True)
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--oft-root", type=Path, required=True)
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", default="2,4,6,8")
    ap.add_argument("--prefetch", default="1,2,4")
    ap.add_argument("--pin", default="0,1")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=16)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    selected = json.loads(args.selected.read_text())
    if not (1 <= args.warmup < len(selected)) or args.repeats < 3:
        raise ValueError("warmup must fit fixed sample set; at least three independent repeats")
    split = json.loads(args.splits.read_text())["splits"]
    for task, episode, frame in selected:
        train = {Path(x).name for x in split[task]["train"]}
        dev = {Path(x).name for x in split[task]["dev"]}
        if episode not in train or episode in dev or int(frame) < 2:
            raise ValueError(f"not frozen TRAIN: {task}/{episode} t={frame}")
    if len(selected) != len(set(map(tuple, selected))):
        raise ValueError("duplicate fixed sample")
    import torch
    from training.r50_data_pipeline import R50WorkerDataset
    roots = (str(args.hdf5_root), str(args.base_model), str(args.oft_root))
    reference = R50WorkerDataset(selected, *roots, cache_bytes=0)
    try:
        expected = [pixel_sha(reference[i]["pixels"]) for i in range(len(selected))]
    finally:
        reference.close()
    args.out.mkdir(parents=True, mode=0o700)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    environment = {"code_sha": head, "selected_sha256": sha(args.selected),
                   "split_sha256": sha(args.splits), "processor_sha256": sha(args.base_model / "preprocessor_config.json"),
                   "torch": torch.__version__, "numpy": __import__("numpy").__version__,
                   "thread_settings": {key: os.environ.get(key) for key in
                                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                   "gpu": "disabled", "sample_count": len(selected),
                   "sample_digest_sha256": hashlib.sha256("".join(expected).encode()).hexdigest()}
    (args.out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    workers = integers(args.workers, 1, 8)
    prefetch = integers(args.prefetch, 1, 4)
    pins = integers(args.pin, 0, 1)
    results = []
    with (args.out / "runs.jsonl").open("w", buffering=1) as stream:
        for count in workers:
            for factor in prefetch:
                for pin in pins:
                    for repeat in range(args.repeats):
                        result = one_run(selected, roots, expected, count, factor,
                                         bool(pin), args.warmup)
                        result["repeat"] = repeat
                        results.append(result)
                        stream.write(json.dumps(result, sort_keys=True) + "\n")
                        print(f"workers={count} prefetch={factor} pin={pin} repeat={repeat} "
                              f"steady={result['epochs'][1]['steady_samples_per_s']:.2f}", flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA context initialized")
    report = {"status": "PASS_CPU_ONLY", "code_sha": head, "configurations": len(results),
              "repeats_per_config": args.repeats, "samples_per_epoch": len(selected),
              "epochs_per_repeat": 2,
              "limits": "DataLoader CPU preparation only; pin_memory may be unavailable with CUDA hidden"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out / "SHA256SUMS").open("w") as stream:
        for name in ("environment.json", "runs.jsonl", "report.json"):
            stream.write(f"{sha(args.out / name)}  {name}\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
