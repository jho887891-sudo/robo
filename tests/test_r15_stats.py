#!/usr/bin/env python3
"""Regression for tools/r15_stats.py: the FINAL report's confidence bounds must be right at the
boundaries (0/25 and 25/25), where a normal approximation breaks."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import r15_stats as S  # noqa: E402


def test_wilson_known_values():
    lo, hi = S.wilson_ci(0, 25)
    assert lo == pytest.approx(0.0, abs=1e-12)   # clamped at 0, up to float noise
    assert 0.10 < hi < 0.14          # exact Wilson upper bound for 0/25 at 95% is ~0.1332
    lo, hi = S.wilson_ci(25, 25)
    assert hi == pytest.approx(1.0, abs=1e-9)    # clamped at 1, up to float noise
    assert 0.85 < lo < 0.88          # ~0.8668
    lo, hi = S.wilson_ci(12, 25)
    assert 0.30 < lo < 0.32 and 0.66 < hi < 0.68   # ~0.3089 .. 0.6690


def test_wilson_is_fail_closed():
    assert S.wilson_ci(0, 0)[0] != S.wilson_ci(0, 0)[0]      # nan, never a fake 0.0
    with pytest.raises(ValueError):
        S.wilson_ci(5, 4)
    with pytest.raises(ValueError):
        S.wilson_ci(-1, 4)


def test_macro_is_unweighted_over_tasks():
    counts = {"a": (0, 25), "b": (25, 25)}                    # pooled would be 0.5 but so is macro
    assert S.macro_mean({"a": 0.0, "b": 1.0}) == pytest.approx(0.5)
    counts = {"a": (1, 2), "b": (10, 100)}                    # pooled 0.108, macro 0.30
    rep = S.summarize(counts, n_boot=200)
    assert rep["macro_success_rate"] == pytest.approx((0.5 + 0.1) / 2)
    assert rep["tasks_with_zero_success"] == []
    # the research target is >=3 successes, so a=1 fails it and b=10 does not
    assert rep["tasks_below_target_3_of_25"] == ["a"]


def test_bootstrap_interval_brackets_the_point_estimate():
    counts = {("t%d" % i): (i, 25) for i in range(6)}         # rates 0, .04, ... .20
    point = S.macro_mean({t: k / 25.0 for t, (k, n) in counts.items()})
    lo, hi, nt = S.bootstrap_macro_ci(counts, n_boot=2000, seed=0)
    assert nt == 6
    assert lo <= point <= hi
    assert hi - lo > 0.0


def test_summarize_shape_is_frozen():
    rep = S.summarize({"push_T": (3, 25), "stack_bowls": (0, 25)}, n_boot=500)
    assert set(rep) >= {"per_task", "macro_success_rate", "macro_bootstrap_95_ci", "n_tasks",
                        "tasks_with_zero_success", "tasks_below_target_3_of_25"}
    assert rep["per_task"]["push_T"]["successes"] == 3
    assert len(rep["per_task"]["push_T"]["wilson_95_ci"]) == 2
    assert rep["tasks_with_zero_success"] == ["stack_bowls"]
