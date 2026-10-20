#!/usr/bin/env python3
"""R11 P0-5 (RED first): fail-closed assertions for the 42D/R training entry.

Everything here is checked against the real Push-T HDF5 (skipped when the data is absent):
  * frame index >= 2 so that S/Acc come from history,
  * A[t] is the next-step ABSOLUTE joint target (A[t] == S[t+1]),
  * S/V/Acc only from observed history, never from future rows,
  * [B,42] -> [B,14] -> [B,14,256] shapes, gripper dims 6/13,
  * per-env history isolation.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("training", "code/xpolicylab/policy/OpenVLA_R42", "code/xpolicylab/policy/OpenVLA_Pure"):
    sys.path.insert(0, os.path.join(REPO, sub))

import r42_train_contract as TC  # noqa: E402

HDF5_DIR = os.environ.get("PUSHT_HDF5",
                          "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
CONTRACT = os.path.join(REPO, "evidence", "r9_gate_20261020", "frozen_contract_64train.json")
SPLIT = os.path.join(REPO, "data", "split80.json")
BOUNDS = os.path.join(REPO, "data", "push_T_bounds.npz")
HAVE_DATA = os.path.isdir(HDF5_DIR) and os.path.isfile(SPLIT)


def test_error_type_and_api_surface():
    for name in ("assert_frame_index", "assert_next_step_target", "build_training_row",
                 "assert_no_future_leakage", "assert_shapes", "assert_gripper_dims",
                 "assert_env_isolation", "TrainContractError"):
        assert hasattr(TC, name), "missing %s" % name
    assert issubclass(TC.TrainContractError, Exception)


def test_frame_index_must_be_at_least_two():
    TC.assert_frame_index(2)
    TC.assert_frame_index(17)
    for bad in (-1, 0, 1):
        with pytest.raises(TC.TrainContractError):
            TC.assert_frame_index(bad)


def test_shapes_are_checked():
    ok = TC.assert_shapes(np.zeros((2, 42), np.float32), np.zeros((2, 14), np.float32),
                          np.zeros((2, 14, 256), np.float32))
    assert ok is True
    with pytest.raises(TC.TrainContractError):
        TC.assert_shapes(np.zeros((2, 14), np.float32), np.zeros((2, 14), np.float32),
                         np.zeros((2, 14, 256), np.float32))
    with pytest.raises(TC.TrainContractError):
        TC.assert_shapes(np.zeros((2, 42), np.float32), np.zeros((2, 42), np.float32),
                         np.zeros((2, 14, 256), np.float32))
    with pytest.raises(TC.TrainContractError):
        TC.assert_shapes(np.zeros((2, 42), np.float32), np.zeros((2, 14), np.float32),
                         np.zeros((2, 14, 255), np.float32))


def test_gripper_dims_are_six_and_thirteen():
    assert TC.assert_gripper_dims((6, 13)) == (6, 13)
    for bad in ((6,), (0, 1), (13, 6), (6, 13, 14)):
        with pytest.raises(TC.TrainContractError):
            TC.assert_gripper_dims(bad)


@pytest.mark.skipif(not HAVE_DATA, reason="Push-T HDF5 not available")
def test_real_episode_contract():
    import h5py
    import r42_adapter as R
    from r8_normalization import NormalizationContract
    meta = json.load(open(CONTRACT))
    contract = NormalizationContract.from_metadata(meta)
    split = json.load(open(SPLIT))
    name = sorted(split["train"])[0]
    with h5py.File(os.path.join(HDF5_DIR, name), "r") as h:
        state = TC.read14(h, "state")
        action = TC.read14(h, "action")
    T = len(state)
    assert T > 8
    for t in (2, 3, T - 2):
        row = TC.build_training_row(state, action, t, contract)
        assert row["features"].shape == (42,)
        assert row["residual"].shape == (14,)
        assert np.isfinite(row["features"]).all()
        assert TC.assert_next_step_target(state, action, t) <= 1e-6
    # history rows are the observed past, not the future
    TC.assert_no_future_leakage(state, action, 5, contract)
    with pytest.raises(TC.TrainContractError):
        TC.assert_next_step_target(state, action, T - 1)


@pytest.mark.skipif(not HAVE_DATA, reason="Push-T HDF5 not available")
def test_a_of_t_equals_state_of_t_plus_one():
    import h5py
    split = json.load(open(SPLIT))
    for name in sorted(split["train"])[:3]:
        with h5py.File(os.path.join(HDF5_DIR, name), "r") as h:
            state = TC.read14(h, "state")
            action = TC.read14(h, "action")
        TC.assert_a_equals_next_state(state, action)


@pytest.mark.skipif(not HAVE_DATA, reason="Push-T HDF5 not available")
def test_per_env_history_isolation():
    import r42_adapter as R
    from r8_normalization import NormalizationContract
    meta = json.load(open(CONTRACT))
    contract = NormalizationContract.from_metadata(meta)
    rng = np.random.RandomState(0)
    a = (rng.randn(4, 14) * 0.01).astype(np.float32)
    b = (rng.randn(4, 14) * 0.01 + 1.0).astype(np.float32)
    TC.assert_env_isolation(R.EpisodeHistory, contract, {0: a, 1: b})
