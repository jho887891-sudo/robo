#!/usr/bin/env python3
"""Success-rate statistics for the official FINAL report: Wilson interval + bootstrap macro CI.

Why not a normal approximation: the per-task sample size is 25 (the official eval_nums) and a task
that scores 0 or 25 has a degenerate normal interval. Wilson behaves at the boundary, which is
exactly where an honest robotics report lives.

Definitions are frozen in mission/R15_R23_FINAL_EVAL_CONTRACT.json:
  * per task: Wilson score interval at 95 percent over (successes, episodes);
  * macro: unweighted mean of the per-task rates, with a bootstrap interval that resamples BOTH
    tasks and episodes (tasks are the unit of generalisation, so resampling tasks is the honest part).
"""
import math

Z95 = 1.959963984540054


def wilson_ci(successes, n, z=Z95):
    """Wilson score interval for a binomial proportion. Returns (low, high)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    if not (0 <= successes <= n):
        raise ValueError("successes must be 0..n (%r of %r)" % (successes, n))
    p = successes / float(n)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def macro_mean(per_task):
    """Unweighted mean over tasks (NOT pooled episodes); None if empty."""
    vals = [v for v in per_task.values() if v is not None]
    return sum(vals) / len(vals) if vals else None


def bootstrap_macro_ci(per_task_counts, n_boot=10000, seed=0, z=Z95):
    """Bootstrap the macro mean by resampling tasks and, within a task, episodes.

    per_task_counts: {task: (successes, episodes)}. Returns (low, high, n_effective_tasks).
    Implemented without numpy so it can run on a bare python3.
    """
    import random
    tasks = [t for t, (k, n) in per_task_counts.items() if n > 0]
    if not tasks:
        return (float("nan"), float("nan"), 0)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        acc = []
        for _t in tasks:
            k, n = per_task_counts[rng.choice(tasks)]
            hits = sum(1 for _i in range(n) if rng.random() < (k / float(n)))
            acc.append(hits / float(n))
        draws.append(sum(acc) / len(acc))
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return (lo, hi, len(tasks))


def summarize(per_task_counts, n_boot=10000, seed=0):
    """Full machine-readable report block: per-task Wilson + macro + macro bootstrap CI."""
    out = {"per_task": {}, "n_tasks": len(per_task_counts)}
    rates = {}
    for task, (k, n) in sorted(per_task_counts.items()):
        lo, hi = wilson_ci(k, n)
        rates[task] = (k / float(n)) if n else None
        out["per_task"][task] = {"successes": k, "episodes": n, "success_rate": rates[task],
                                 "wilson_95_ci": [lo, hi]}
    out["macro_success_rate"] = macro_mean(rates)
    lo, hi, nt = bootstrap_macro_ci(per_task_counts, n_boot=n_boot, seed=seed)
    out["macro_bootstrap_95_ci"] = [lo, hi]
    out["macro_bootstrap_n_tasks"] = nt
    out["tasks_with_zero_success"] = sorted(t for t, (k, n) in per_task_counts.items()
                                            if k == 0)
    out["tasks_below_target_3_of_25"] = sorted(t for t, (k, n) in per_task_counts.items()
                                               if k < 3)
    return out
