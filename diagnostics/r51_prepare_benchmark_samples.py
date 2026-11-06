#!/usr/bin/env python3
"""Freeze a bounded CPU benchmark window from one complete R51 TRAIN epoch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.r45_data import build_index
from training.r51_sampling import TASKS, draw_epoch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--hdf5-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--mode", choices=("replacement", "without_replacement"), default="replacement")
    ap.add_argument("--task-mix", choices=("proportional", "balanced"), default="proportional")
    ap.add_argument("--active-pool", type=int)
    ap.add_argument("--draw-limit", type=int, default=1000)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    splits = json.loads(args.splits.read_text())["splits"]
    if not set(TASKS).issubset(splits):
        raise ValueError("frozen split is missing an R51 task")
    index = build_index(splits, str(args.hdf5_root), which="train", tasks=set(TASKS))
    if len(index) != 72922:
        raise ValueError(f"R51 two-task TRAIN count drift: {len(index)}")
    draws, summary = draw_epoch(index, base_seed=args.seed, epoch=args.epoch,
                                mode=args.mode, task_mix=args.task_mix,
                                pool_size=args.active_pool)
    if not (1 <= args.draw_limit <= len(draws)):
        raise ValueError("draw-limit outside complete epoch")
    selected = [(index[i][0], Path(index[i][1]).name, int(index[i][2]))
                for i in draws[:args.draw_limit]]
    if any(task not in TASKS for task, _, _ in selected):
        raise AssertionError("non-R51 task in benchmark")
    window_unique = len(set(selected))
    args.out.mkdir(parents=True, mode=0o700)
    (args.out / "selected_samples.json").write_text(json.dumps(selected, indent=2) + "\n")
    report = {"schema": "r51-benchmark-selection/v1", "status": "PASS_CPU_SELECTION",
              "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "split_sha256": digest(args.splits), "tasks": list(TASKS),
              "full_epoch": summary, "benchmark_window_draws": len(selected),
              "benchmark_window_unique": window_unique,
              "benchmark_window_duplicate_ratio": 1 - window_unique / len(selected),
              "benchmark_task_draws": {task: sum(row[0] == task for row in selected)
                                       for task in TASKS},
              "scope": "CPU benchmark window from complete R51 draw plan; no training"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (args.out / "SHA256SUMS").open("w") as stream:
        for name in ("report.json", "selected_samples.json"):
            stream.write(f"{digest(args.out / name)}  {name}\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
