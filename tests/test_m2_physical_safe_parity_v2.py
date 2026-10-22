#!/usr/bin/env python3
"""M2 physical v2: matched SAFE decoder and legacy comparison negative cases."""
import os
import sys
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "code/xpolicylab/policy/OpenVLA_R42"))
import m2_physical_audit_v2 as V2
import r42_adapter as R


def config():
    return {"q01": [-1.0]*14, "q99": [1.0]*14,
            "pmin": [-1.0]*14, "pmax": [1.0]*14}


def test_matched_cv_is_clipped_not_unbounded():
    s = np.full(14, 0.9, np.float32)
    v = np.full(14, 0.5, np.float32)
    bins = np.full(14, 128, np.int64)
    model, cv, noop, mi, ci = V2.matched_commands(s, v, bins, config())
    assert np.all(cv <= 1.0)
    assert np.all(cv >= -1.0)
    assert ci["clipped_joint"] > 0
    assert np.allclose(noop, s)
    assert np.any(np.abs(cv - (s+v)) > 0.1), "legacy CV omitted bounds"


def test_model_gripper_residual_is_ignored_by_real_decoder():
    s = np.zeros(14, np.float32)
    v = np.full(14, 0.1, np.float32)
    bins = np.full(14, 255, np.int64)
    model, cv, noop, mi, ci = V2.matched_commands(s, v, bins, config())
    for d in R.GRIPPER_DIMS:
        assert model[d] == pytest.approx(s[d] + v[d], abs=1e-6)
        assert cv[d] == pytest.approx(s[d] + v[d], abs=1e-6)
    assert model[0] > cv[0] + 0.5


def test_noop_bounds_are_the_same_as_other_policies():
    s = np.full(14, 1.1, np.float32)
    v = np.zeros(14, np.float32)
    bins = np.full(14, 127, np.int64)
    _, _, noop, _, _ = V2.matched_commands(s, v, bins, config())
    assert np.all(noop == 1.0)
    assert np.any(np.abs(noop-s) > 0.01)


def test_numerical_comparison_demonstrates_legacy_gate_can_flip():
    # In a state beyond the joint bound, 'unbounded CV' can be close to GT
    # even though the physically deployable CV is much farther away.
    s = np.full(14, 0.99, np.float32)
    v = np.full(14, 0.06, np.float32)
    a_demo = np.full(14, 1.05, np.float32)
    bins = np.full(14, 127, np.int64)
    model, cv, *_ = V2.matched_commands(s, v, bins, config())
    assert V2.mean_l1(s+v, a_demo) < 1e-6
    assert V2.mean_l1(cv, a_demo) > 0.04
    assert V2.mean_l1(model, a_demo) >= 0.0
