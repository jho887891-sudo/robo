#!/usr/bin/env python3
"""The no-op baseline must command the CURRENT STATE, not zero joint positions (audit PR#27).

tools/m2_gate_check.py::baseline_bins() reported l1_noop = mean|(s[t]-s[t])-a[t]| = mean|a[t]|, which is
the error of commanding zero joint positions. That silently understated the no-op margin that P6 depends
on. These tests use a REAL DEV frame from the committed HDF5 so the physical numbers are the ones the gate
would see.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _s in ("training", "code/xpolicylab/policy/OpenVLA_Pure",
           "code/xpolicylab/policy/OpenVLA_R42", "tools"):
    sys.path.insert(0, os.path.join(REPO, _s))

import m2_multitask_data as MD          # noqa: E402
import m2_task_normalizer as TN         # noqa: E402
import m2_train_shared as MT            # noqa: E402
import m2_gate_check as G               # noqa: E402

HDF5 = "/home/T7/ojh/robot_actual/data/data/hdf5"
NORM = os.path.join(REPO, "data", "m2_task_normalizer_2task.json")
SPLITS = os.path.join(REPO, "data", "m2_splits.json")


@pytest.fixture(scope="module")
def frame():
    if not os.path.isdir(HDF5):
        pytest.skip("HDF5 dataset is not present on this host")
    payload = json.load(open(NORM))
    splits = json.load(open(SPLITS))
    idx = MD.load_task_index("push_T", HDF5)
    dev = MT.dev_frames_for_task(idx, (splits.get("splits") or {})["push_T"], "push_T", 1, 0)
    ep, t = dev[0]
    import h5py
    import r42_train_contract as TC
    with h5py.File(os.path.join(idx["hdf5_dir"], ep), "r") as h:
        state, action = TC.read14(h, "state"), TC.read14(h, "action")
    return payload, idx, ep, int(t), state, action


def test_noop_l1_is_the_documented_no_op(frame):
    payload, idx, ep, t, state, action = frame
    b = G.baseline_bins(payload, "push_T", state, action, t)
    s = np.asarray(state, np.float32)
    a = np.asarray(action, np.float32)
    t = int(t)
    assert b["l1_noop"] == pytest.approx(float(np.abs(s[t] - a[t]).mean()), rel=0, abs=1e-7), \
        "no-op must command the current state S[t]"
    # and it must NOT be the zero-position predictor the old code implemented
    assert b["l1_zero_position_LEGACY"] == pytest.approx(float(np.abs(a[t]).mean()), rel=0, abs=1e-7)
    assert b["l1_noop"] != pytest.approx(b["l1_zero_position_LEGACY"], rel=1e-9, abs=1e-9)


def test_constant_velocity_l1_is_unchanged(frame):
    payload, idx, ep, t, state, action = frame
    b = G.baseline_bins(payload, "push_T", state, action, t)
    s = np.asarray(state, np.float32)
    a = np.asarray(action, np.float32)
    tt = int(t)
    v = s[tt] - s[tt - 1]
    assert b["l1_cv"] == pytest.approx(float(np.abs((s[tt] + v) - a[tt]).mean()), rel=0, abs=1e-7)


def test_noop_is_a_plausible_predictor_and_zero_position_is_not(frame):
    """Sanity in PHYSICAL terms: holding still beats commanding all joints to zero."""
    payload, idx, ep, t, state, action = frame
    b = G.baseline_bins(payload, "push_T", state, action, t)
    assert b["l1_noop"] < b["l1_zero_position_LEGACY"], (
        "holding the current pose (%.4f) must be a better predictor than commanding zeros (%.4f)"
        % (b["l1_noop"], b["l1_zero_position_LEGACY"]))
