"""Deterministic R51 two-task draw plans. No training or CUDA side effects."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np

TASKS = ("push_T", "sweep_blocks")


def draw_epoch(entries, *, base_seed: int, epoch: int, mode: str = "replacement",
               task_mix: str = "proportional", pool_size: int | None = None):
    """Return global entry indices and an auditable per-epoch summary.

    An epoch always has len(two-task TRAIN) draws. Active pools affect eligible
    samples, never the number of optimizer input visits.
    """
    if epoch < 0 or base_seed < 0:
        raise ValueError("seed and epoch must be nonnegative")
    indexed = {task: [i for i, row in enumerate(entries) if row[0] == task]
               for task in TASKS}
    if any(not indexed[task] for task in TASKS):
        raise ValueError("both R51 TRAIN tasks require samples")
    if any(row[0] not in TASKS for row in entries):
        raise ValueError("R51 TRAIN entries must exclude stack_bowls and unknown tasks")
    if mode not in ("replacement", "without_replacement") or task_mix not in ("proportional", "balanced"):
        raise ValueError("unknown sampling mode or task mix")
    if pool_size is not None and (pool_size < 2 or pool_size > len(entries)):
        raise ValueError("active pool size outside two-task TRAIN")
    if mode == "without_replacement" and (pool_size is not None or task_mix != "proportional"):
        raise ValueError("full no-replacement baseline has fixed task proportions and no pool")
    rng = np.random.default_rng([base_seed, epoch])
    eligible = indexed
    pool_sha = None
    if pool_size is not None:
        pool_rng = np.random.default_rng([base_seed, 5101])
        counts = [len(indexed[t]) for t in TASKS]
        first = max(1, min(counts[0], round(pool_size * counts[0] / sum(counts))))
        second = pool_size - first
        if second < 1 or second > counts[1]:
            raise ValueError("active pool cannot represent both tasks")
        eligible = {TASKS[0]: pool_rng.choice(indexed[TASKS[0]], first, replace=False).tolist(),
                    TASKS[1]: pool_rng.choice(indexed[TASKS[1]], second, replace=False).tolist()}
        pool_sha = hashlib.sha256(json.dumps(
            {t: sorted(eligible[t]) for t in TASKS}, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
    n = len(entries)
    if mode == "without_replacement":
        draws = rng.permutation(n).astype(int).tolist()
    else:
        p_first = len(indexed[TASKS[0]]) / n if task_mix == "proportional" else 0.5
        choices = rng.random(n) < p_first
        draws_array = np.empty(n, dtype=np.int64)
        for task, mask in ((TASKS[0], choices), (TASKS[1], ~choices)):
            candidates = np.asarray(eligible[task], dtype=np.int64)
            draws_array[mask] = candidates[rng.integers(len(candidates), size=int(mask.sum()))]
        draws = draws_array.tolist()
    unique = len(set(draws))
    counts = Counter(entries[i][0] for i in draws)
    serialized = json.dumps(draws, separators=(",", ":")).encode()
    return draws, {"epoch": epoch, "seed": base_seed, "mode": mode,
                   "task_mix": task_mix, "pool_size": pool_size,
                   "pool_sha256": pool_sha,
                   "draws": n, "unique_samples": unique,
                   "duplicate_draws": n - unique, "duplicate_ratio": (n - unique) / n,
                   "task_draws": {task: counts[task] for task in TASKS},
                   "order_sha256": hashlib.sha256(serialized).hexdigest()}
