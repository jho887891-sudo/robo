"""CPU-only two-task view of a fully attested R44 training plan.

The full three-task Manifest is validated by R44 first; this module derives a
new immutable two-task identity without rewriting historical normalizers.
"""
from __future__ import annotations

import hashlib
import json

from training.r51_sampling import TASKS, draw_epoch

EXPECTED_TRAIN = 72922
EXPECTED_DEV = 19552


def _digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def two_task_plan(full):
    if full.get("n_train") != 108780 or full.get("n_dev") != 28268:
        raise ValueError("R51 requires the fully attested R44 source plan")
    if any(key not in full for key in ("split_sha256", "dataset_manifest_sha256",
                                        "feature_normalizer_sha256", "delta_normalizer_sha256",
                                        "feature_payload", "delta_payload")):
        raise ValueError("missing frozen R44 provenance")
    train = [tuple(row) for row in full["train_samples"] if row[0] in TASKS]
    dev = [tuple(row) for row in full["dev_samples"] if row[0] in TASKS]
    if len(train) != EXPECTED_TRAIN or len(dev) != EXPECTED_DEV:
        raise ValueError("two-task TRAIN/DEV count drift")
    if set(train) & set(dev):
        raise ValueError("two-task TRAIN/DEV overlap")
    by_task = {task: sum(row[0] == task for row in train) for task in TASKS}
    if by_task != {"push_T": 25313, "sweep_blocks": 47609}:
        raise ValueError("two-task TRAIN per-task count drift")
    for task in TASKS:
        if int(full["delta_payload"]["tasks"][task]["n_rows"]) != by_task[task]:
            raise ValueError(f"{task} frozen delta normalizer row count drift")
    paths = {key: value for key, value in full["episode_paths"].items()
             if key.split("/", 1)[0] in TASKS}
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("invalid two-task episode paths")
    result = dict(full)
    result.update({"tasks": TASKS, "train_samples": train, "dev_samples": dev,
                   "n_train": len(train), "n_dev": len(dev),
                   "episode_paths": paths,
                   "manifest_files": {k: full["manifest_files"][k] for k in paths},
                   "file_stats_before": {k: full["file_stats_before"][k] for k in paths},
                   "instruction_sha256": {k: full["instruction_sha256"][k] for k in TASKS},
                   "train_samples_sha256": _digest([list(x) for x in train]),
                   "dev_samples_sha256": _digest([list(x) for x in dev]),
                   "source_train_samples_sha256": full["train_samples_sha256"],
                   "source_dev_samples_sha256": full["dev_samples_sha256"]})
    return result


def epoch_draws(plan, *, seed, epoch, mode="replacement", task_mix="proportional",
                pool_size=None):
    if tuple(plan.get("tasks", ())) != TASKS or plan.get("n_train") != EXPECTED_TRAIN:
        raise ValueError("not an attested R51 two-task plan")
    return draw_epoch(plan["train_samples"], base_seed=seed, epoch=epoch,
                      mode=mode, task_mix=task_mix, pool_size=pool_size)


def validate_resume_cursor(plan, draws, *, cursor, committed_steps, epoch):
    """Check draw position against committed optimizer steps, not unique samples."""
    if len(draws) != plan["n_train"] or not 0 <= cursor <= len(draws):
        raise ValueError("invalid complete epoch draw plan or cursor")
    if committed_steps != epoch * plan["n_train"] + cursor:
        raise ValueError("optimizer step/cursor mismatch")
    if any(not 0 <= i < plan["n_train"] for i in draws):
        raise ValueError("draw index out of range")
    return _digest(draws)
