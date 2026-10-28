"""Numerical fingerprint of q01/q99 on *the same* frozen TRAIN delta rows.

Pure diagnostic: never modifies a frozen normalizer, produces a manifest,
or authorizes training. Each candidate is a distinct floating-point evaluation
path for linear quantile interpolation, NOT a guarantee about historical NumPy.
"""
from __future__ import annotations

import math
import numpy as np


def quantile_candidates(deltas, q):
    d = np.asarray(deltas, dtype=np.float32)
    if d.ndim != 2 or d.shape[1] != 14 or not np.isfinite(d).all():
        raise ValueError("requires finite (N,14) float32 TRAIN delta rows")
    if len(d) < 2 or not (0.0 < q < 1.0):
        raise ValueError("invalid row count or quantile")
    ordered = np.sort(d, axis=0)
    h = (len(d) - 1) * float(q)
    lower = int(math.floor(h))
    upper = int(math.ceil(h))
    w64 = float(h - lower)
    w32 = np.float32(w64)
    lo, hi = ordered[lower], ordered[upper]
    lo64, hi64 = lo.astype(np.float64), hi.astype(np.float64)
    # Candidate arithmetic paths. The first is exactly the code currently
    # used by R44 fit_normalizer; others isolate interpolation/dtype effects.
    return {
        "numpy_current_float32_input": np.quantile(d, q, axis=0).astype(np.float32),
        "numpy_float64_input": np.quantile(d.astype(np.float64), q, axis=0).astype(np.float32),
        "manual_float32_lerp": (lo + (hi - lo) * w32).astype(np.float32),
        "manual_float32_weighted_sum": (
            lo * (np.float32(1.0) - w32) + hi * w32).astype(np.float32),
        "manual_float64_lerp": (
            lo64 + (hi64 - lo64) * w64).astype(np.float32),
    }


def fingerprint_quantiles(deltas_by_task, frozen):
    out = {}
    for task in sorted(deltas_by_task):
        if task not in frozen["tasks"]:
            raise ValueError(f"task missing from frozen payload: {task}")
        frozen_task = frozen["tasks"][task]
        task_out = {}
        for key, q in (("q01", 0.01), ("q99", 0.99)):
            target = np.asarray(frozen_task[key], dtype=np.float32)
            if target.shape != (14,):
                raise ValueError(f"{task}.{key}: frozen vector must be 14D")
            candidate_out = {}
            for name, values in quantile_candidates(deltas_by_task[task], q).items():
                differ = np.flatnonzero(values.view(np.uint32) != target.view(np.uint32))
                candidate_out[name] = {
                    "exact_float32_match_count": int(14 - len(differ)),
                    "exact_float32_all_equal": bool(not len(differ)),
                    "differing_indices": differ.tolist(),
                    "max_abs_diff": float(np.max(
                        np.abs(values.astype(np.float64) - target.astype(np.float64)))),
                    # Retain diagnostic computed quantile vectors for independent checks.
                    "candidate_values_float32": values.tolist(),
                }
            task_out[key] = {
                "frozen_values_float32": target.tolist(),
                "candidates": candidate_out,
            }
        out[task] = task_out
    return {
        "schema": "r45-quantile-arithmetic-fingerprint/v1",
        "tasks": out,
        "limitations": (
            "Candidate matches demonstrate numerical compatibility only. "
            "They do not establish the historical NumPy version, dataset identity, "
            "or permission to replace/bypass the frozen normalizer."
        ),
    }
