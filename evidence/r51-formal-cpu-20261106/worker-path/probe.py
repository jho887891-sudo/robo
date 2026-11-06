#!/usr/bin/env python3
"""Read five real R51 TRAIN draws through the formal worker loader, CPU only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "")


def digest(tensor):
    import torch
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--hdf5-root", type=Path, required=True)
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--oft-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    import torch
    from training.r45_data import CAMERAS, RuntimeReader, build_index
    from training.r50_data_pipeline import load_openvla_processor
    from training.r51_sampling import draw_epoch
    from training.r44_train import make_r51_image_loader, close_r51_image_loader
    splits = json.loads(args.splits.read_text())["splits"]
    index = build_index(splits, str(args.hdf5_root), which="train",
                        tasks={"push_T", "sweep_blocks"})
    train = [(task, Path(path).name, t) for task, path, t in index]
    if len(train) != 72922:
        raise AssertionError("frozen two-task TRAIN count changed")
    order, summary = draw_epoch(train, base_seed=51, epoch=0,
                                pool_size=500)
    plan = {"train_samples": train, "n_train": len(train),
            "hdf5_root": str(args.hdf5_root)}
    config = SimpleNamespace(r51_two_task=True, r51_workers=2, r51_prefetch=2,
                             r51_pin_memory=False, r51_cache_mib=0,
                             r51_cache_kind="none", base_model=str(args.base_model),
                             oft_root=str(args.oft_root))
    loader, iterator = make_r51_image_loader(config, plan, order, 731)
    parent_reader = RuntimeReader(cache_bytes=0, cache_kind="none")
    proc = load_openvla_processor(str(args.base_model), str(args.oft_root))
    records = []
    try:
        for local_index in range(5):
            item = next(iterator)
            if int(item["index"].item()) != local_index:
                raise AssertionError("worker order changed")
            task, episode, frame = train[order[731 + local_index]]
            if task not in ("push_T", "sweep_blocks"):
                raise AssertionError("non-R51 task")
            path = args.hdf5_root / task / "arx_x5" / "data" / episode
            reference, _ = parent_reader.load_images(proc, path, frame, 3, CAMERAS,
                                                     device="cpu")
            pixels = item["pixels"]
            if pixels.dtype != torch.bfloat16 or tuple(pixels.shape) != (1, 18, 224, 224):
                raise AssertionError("worker shape/dtype mismatch")
            if digest(pixels) != digest(reference):
                raise AssertionError("worker pixels differ from serial reader")
            worker_pid = int(item["worker_pid"].item())
            if worker_pid == os.getpid():
                raise AssertionError("worker did not run separately")
            records.append({"draw_position": 731 + local_index,
                            "task": task, "episode": episode, "frame": frame,
                            "pixel_sha256": digest(pixels), "worker_pid": worker_pid})
    finally:
        close_r51_image_loader(loader)
        parent_reader.close()
    if torch.cuda.is_initialized():
        raise AssertionError("CUDA was initialized")
    args.out.mkdir(parents=True, mode=0o700)
    report = {"status": "PASS_CPU_WORKER_PATH", "samples": len(records),
              "train_draws_per_epoch": summary["draws"], "resume_cursor": 731,
              "worker_pids": sorted({x["worker_pid"] for x in records}),
              "gpu": "disabled", "limits": "five real image draws only; no model training"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "samples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
