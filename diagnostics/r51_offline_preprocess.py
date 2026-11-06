#!/usr/bin/env python3
"""Bounded CPU prototype of lossless OpenVLA input materialization for R51."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--hdf5-root", type=Path, required=True)
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--oft-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()
    if args.out.exists() or not 1 <= args.limit <= 100:
        raise ValueError("fresh output and limit 1..100 required")
    import numpy as np
    import torch
    from training.r45_data import CAMERAS, RuntimeReader
    from training.r50_data_pipeline import load_openvla_processor
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA context initialized")
    selected = json.loads(args.selected.read_text())[:args.limit]
    if len(selected) != args.limit or any(t not in ("push_T", "sweep_blocks")
                                         for t, _, _ in selected):
        raise ValueError("selection must contain R51 TRAIN tasks")
    if len(set(map(tuple, selected))) != len(selected):
        raise ValueError("prototype requires unique sample identities")
    processor = load_openvla_processor(str(args.base_model), str(args.oft_root))
    args.out.mkdir(parents=True, mode=0o700)
    shard = args.out / "openvla_bf16.bin"
    reader = RuntimeReader(cache_bytes=0, cache_kind="none")
    records = []
    start = time.perf_counter()
    try:
        with shard.open("xb") as stream:
            for index, (task, episode, frame) in enumerate(selected):
                path = args.hdf5_root / task / "arx_x5" / "data" / episode
                pixels, timing = reader.load_images(processor, path, int(frame),
                                                     3, CAMERAS, device="cpu")
                if pixels.dtype != torch.bfloat16 or tuple(pixels.shape) != (1, 18, 224, 224):
                    raise AssertionError("OpenVLA input shape/dtype drift")
                raw = pixels.contiguous().view(torch.uint16).numpy().tobytes()
                offset = stream.tell()
                stream.write(raw)
                records.append({"index": index, "task": task, "episode": episode,
                                "frame": int(frame), "offset": offset, "bytes": len(raw),
                                "pixel_sha256": hashlib.sha256(raw).hexdigest(),
                                "hdf5_read_s": timing["hdf5_read_s"],
                                "jpeg_preprocess_s": timing["jpeg_preprocess_s"]})
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        reader.close()
    convert_s = time.perf_counter() - start
    mapped = np.memmap(shard, dtype=np.uint16, mode="r")
    read_start = time.perf_counter()
    for record in records:
        offset = record["offset"] // 2
        words = mapped[offset:offset + record["bytes"] // 2]
        if hashlib.sha256(words.tobytes()).hexdigest() != record["pixel_sha256"]:
            raise AssertionError("offline tensor bits changed")
    read_s = time.perf_counter() - read_start
    (args.out / "samples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    report = {"schema": "r51-offline-prototype/v1", "status": "PASS_CPU_BITS",
              "samples": len(records), "selected_sha256": sha(args.selected),
              "processor_sha256": sha(args.base_model / "preprocessor_config.json"),
              "shard_bytes": shard.stat().st_size, "shard_sha256": sha(shard),
              "conversion_wall_s": convert_s, "mapped_read_verify_wall_s": read_s,
              "projected_two_task_train_dev_bytes": shard.stat().st_size / len(records) * (72922 + 19552),
              "limits": "Prototype only; warm OS cache, no GPU/model training or full conversion"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out / "SHA256SUMS").open("w") as stream:
        for name in ("openvla_bf16.bin", "samples.jsonl", "report.json"):
            stream.write(f"{sha(args.out / name)}  {name}\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
