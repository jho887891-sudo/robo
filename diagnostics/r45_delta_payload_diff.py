"""Deterministic, read-only diff of two r44-delta-normalizer/v1 payloads.

Exact payload identity remains the acceptance rule. Differences are diagnostic,
never used as tolerance-based training approval.
"""
from __future__ import annotations

import math
from numbers import Real


def _number(x):
    return isinstance(x, Real) and not isinstance(x, bool)


def _single(path, before, after):
    result = {"field": path, "frozen": before, "recomputed": after,
              "kind": "metadata", "equal": before == after}
    if _number(before) and _number(after):
        if math.isfinite(float(before)) and math.isfinite(float(after)):
            result.update(kind="numeric", abs_diff=abs(float(after) - float(before)))
        return result
    if isinstance(before, list) and isinstance(after, list):
        if len(before) == len(after) and all(_number(x) for x in before + after):
            mismatched = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
            diffs = [abs(float(a) - float(b)) for a, b in zip(before, after)]
            result.update(kind="numeric_array", indices=mismatched,
                          n_mismatched=len(mismatched),
                          max_abs_diff=max(diffs, default=0.0))
    return result


def compare_delta_payloads(frozen, reconstructed):
    """Summarize differences without silently tolerating any mismatch."""
    fields = []
    for name in ("schema", "target", "codec", "n_bins", "token_window"):
        fields.append(_single(name, frozen.get(name), reconstructed.get(name)))
    f_fit = frozen.get("fit") or {}
    r_fit = reconstructed.get("fit") or {}
    for name in sorted(set(f_fit) | set(r_fit)):
        fields.append(_single(f"fit.{name}", f_fit.get(name), r_fit.get(name)))
    a_tasks = frozen.get("tasks") or {}
    b_tasks = reconstructed.get("tasks") or {}
    all_tasks = sorted(set(a_tasks) | set(b_tasks))
    fields.append(_single("task_names", sorted(a_tasks), sorted(b_tasks)))
    for task in all_tasks:
        before = a_tasks.get(task) or {}
        after = b_tasks.get(task) or {}
        for name in sorted(set(before) | set(after)):
            fields.append(_single(f"tasks.{task}.{name}", before.get(name),
                                  after.get(name)))
    changed = [x for x in fields if not x["equal"]]
    return {
        "schema": "r45-delta-payload-difference/v1",
        "status": "BLOCKED_DELTA_MISMATCH" if changed else "PAYLOAD_FIELDS_EQUAL",
        "frozen_payload_sha256": frozen.get("sha256"),
        "recomputed_payload_sha256": reconstructed.get("sha256"),
        "exact_payload_sha_equal": frozen.get("sha256") == reconstructed.get("sha256"),
        "changed_fields": changed,
        "changed_field_count": len(changed),
        "comparison_scope": "All top-level fields except sha256 and all per-task fields",
        "interpretation": (
            "Exact SHA256 identity is mandatory even for tiny numerical diffs. "
            "This report diagnoses differences; it never authorizes retraining."
        ),
    }
