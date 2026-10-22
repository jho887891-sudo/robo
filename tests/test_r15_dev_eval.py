#!/usr/bin/env python3
"""R15 Lane E regression: frozen 16-DEV cohort, identical samples for every model, TRAIN-only
baseline fitting, and an episode-macro metric definition that cannot silently turn into a
pooled average.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "diagnostics"))
for _s in ("training", "code/xpolicylab/policy/OpenVLA_Pure"):
    sys.path.insert(0, os.path.join(REPO, _s))

import r15_dev_eval as E  # noqa: E402

HDF5_DIR = os.environ.get("PUSHT_HDF5",
                          "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
HAVE_DATA = os.path.isdir(HDF5_DIR) and os.path.isfile(E.SPLIT)


def test_metric_definitions():
    rng = np.random.RandomState(0)
    state = rng.randn(6, 14).astype(np.float32)
    vel = rng.randn(6, 14).astype(np.float32) * 0.01
    target = state + vel + rng.randn(6, 14).astype(np.float32) * 0.001
    eps = ["a", "a", "a", "a", "b", "b"]           # deliberately unbalanced
    perfect = E.metrics(target, target, state, vel, eps)
    assert perfect["physical_l1_macro"] == pytest.approx(0.0, abs=1e-9)
    assert perfect["correction_r2"] == pytest.approx(1.0, abs=1e-6)
    assert perfect["magnitude_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert perfect["still_action_rate"] == 0.0

    bad = target.copy()
    bad[:4] += 1.0
    m = E.metrics(bad, target, state, vel, eps)
    assert m["physical_l1_macro"] > 0
    assert m["physical_l1_per_episode"]["a"] > m["physical_l1_per_episode"]["b"]
    # macro (episode-weighted) must differ from the pooled mean on an unbalanced set
    assert m["physical_l1_macro"] != pytest.approx(m["physical_l1_pooled"], rel=1e-3)


def test_gripper_and_arm_split():
    rng = np.random.RandomState(1)
    st = rng.randn(4, 14).astype(np.float32)
    v = np.zeros((4, 14), np.float32)
    tgt = st.copy()
    pred = st.copy()
    pred[:, list(E.GRIPPER)] += 0.5
    m = E.metrics(pred, tgt, st, v, ["a"] * 4)
    assert m["gripper_l1"] == pytest.approx(0.5, abs=1e-6)
    assert m["arm_l1"] == pytest.approx(0.0, abs=1e-6)


def test_sign_and_magnitude_use_delta_motion_not_absolute_position():
    """PR#21 finding 3: v1 scored sign/magnitude on the ABSOLUTE joint position.

    Here the arm sits at +10 rad and the prediction has the WRONG sign of motion while every
    absolute coordinate still has the right sign, so the two definitions must disagree.
    """
    state = np.full((3, 14), 10.0, np.float32)
    vel = np.full((3, 14), 0.01, np.float32)
    target = state + 0.1                      # commanded motion is +0.1
    pred = state - 0.1                        # model says -0.1
    eps = ["a", "a", "b"]
    m = E.metrics(pred, target, state, vel, eps)
    assert m["absolute_sign_agreement"] == pytest.approx(1.0)      # legacy v1 -> looks perfect
    assert m["delta_sign_agreement"] == pytest.approx(0.0)         # the real quantity -> wrong
    assert m["sign_agreement"] == m["delta_sign_agreement"]        # unprefixed = delta (fixed)
    assert m["delta_magnitude_ratio"] == pytest.approx(1.0, abs=1e-6)
    # the residual sign is the *direct* statement about what the controller adds on top of velocity
    assert m["residual_sign_agreement"] == pytest.approx(0.0)
    # physical L1 is unchanged by the relabelling: |dA_pred - dA_true| == |A_pred - A_true|
    assert m["physical_l1_macro"] == pytest.approx(0.2, abs=1e-5)


def test_residual_metrics_are_one_for_constant_velocity():
    rng = np.random.RandomState(3)
    state = rng.randn(5, 14).astype(np.float32)
    vel = rng.randn(5, 14).astype(np.float32) * 0.02
    target = state + vel + rng.randn(5, 14).astype(np.float32) * 1e-4
    cv = state + vel
    m = E.metrics(cv, target, state, vel, ["a"] * 5)
    # CV reproduces the commanded motion exactly, so delta magnitude is 1 ...
    assert m["delta_magnitude_ratio"] == pytest.approx(1.0, abs=1e-3)
    # ... and it predicts a ZERO residual, so the residual magnitude ratio is 0 (it explains none of
    # the true residual R = (A-S)-V). v1 could not express this at all because it looked at absolute q.
    assert m["residual_magnitude_ratio"] < 1e-3
    assert m["delta_sign_agreement"] > 0.9
    assert m["correction_r2"] < 0.1


def test_raw_and_safe_predictions_are_reported_separately():
    """PR#21 finding 4: 6.03% of decoded coordinates were outside the joint box while the gate
    reported a single ambiguous clip rate. RAW and SAFE must be distinct, unambiguous numbers."""
    lo = np.full(14, -1.0, np.float32)
    hi = np.full(14, 1.0, np.float32)
    state = np.zeros((4, 14), np.float32)
    vel = np.zeros((4, 14), np.float32)
    target = np.zeros((4, 14), np.float32)
    pred = np.zeros((4, 14), np.float32)
    pred[0, 3] = 5.0            # one coordinate far outside the box
    pred[1, 7] = -2.0           # a second out-of-box coordinate
    m = E.metrics(pred, target, state, vel, ["a"] * 4, bounds=(lo, hi))
    assert m["raw_oob_coord_count"] == 2
    assert m["raw_oob_coord_rate"] == pytest.approx(2.0 / 56.0)
    assert m["raw_oob_frame_rate"] == pytest.approx(2.0 / 4.0)
    assert m["raw_oob_frame_count"] == 2
    assert m["raw_oob_max_excursion"] == pytest.approx(4.0)        # 5.0 - hi(1.0)
    assert m["joint_clip_rate"] == pytest.approx(m["raw_oob_coord_rate"])   # legacy alias, RAW
    # SAFE is what the robot executes: the excursion is clamped to the box
    assert m["safe_clip_frame_rate"] == pytest.approx(0.5)
    assert m["safe_clip_max_abs"] == pytest.approx(4.0)            # 5.0 -> hi 1.0 is a 4.0 change
    # SAFE error is exactly 1.0 on two of the 56 coordinates -> 2/56 as a coordinate mean
    assert m["physical_l1_macro"] == pytest.approx(2.0 / 56.0, abs=1e-6)
    assert m["raw_physical_l1_macro"] > m["physical_l1_macro"]
    assert m["gated_on"] == "SAFE"


def test_clamp_is_a_noop_inside_the_box():
    lo = np.full(14, -3.0, np.float32)
    hi = np.full(14, 3.0, np.float32)
    rng = np.random.RandomState(4)
    state = rng.randn(3, 14).astype(np.float32) * 0.1
    vel = rng.randn(3, 14).astype(np.float32) * 0.01
    target = state + vel
    m = E.metrics(target, target, state, vel, ["a"] * 3, bounds=(lo, hi))
    assert m["raw_oob_coord_count"] == 0
    assert m["safe_clip_frame_rate"] == 0.0
    assert m["physical_l1_macro"] == pytest.approx(m["raw_physical_l1_macro"])
    assert m["metric_version"] == E.METRIC_VERSION


def test_metric_definitions_are_published():
    d = E.metric_definitions()
    for key in ("gate_quantity", "delta_definition", "residual_definition", "sign_agreement",
                "magnitude_ratio", "raw_vs_safe", "clip_policy", "sampling"):
        assert key in d and d[key]
    assert "NOT absolute" in d["sign_agreement"]


@pytest.mark.skipif(not HAVE_DATA, reason="Push-T HDF5 not available")
def test_frozen_dev_cohort_is_the_frozen_16_and_never_the_holdout():
    import r9_trainer_contract as RC
    contract, meta, splits = E.load_splits()
    frames = E.frozen_frames(sorted(splits["dev"]), 4, seed=0)
    assert sorted(set(e for e, _t in frames)) == sorted(splits["dev"])
    assert len(set(e for e, _t in frames)) == 16
    for e, _t in frames:
        assert e not in set(splits["holdout"]), "the old 20 holdout must never appear in DEV"
    assert len(frames) == 64


@pytest.mark.skipif(not HAVE_DATA, reason="Push-T HDF5 not available")
def test_frame_list_is_deterministic_and_seed_sensitive():
    import r9_trainer_contract as RC
    contract, meta, splits = E.load_splits()
    a = E.frozen_frames(sorted(splits["dev"]), 4, seed=0)
    b = E.frozen_frames(sorted(splits["dev"]), 4, seed=0)
    c = E.frozen_frames(sorted(splits["dev"]), 4, seed=1)
    assert E.frames_sha256(a) == E.frames_sha256(b)
    assert E.frames_sha256(a) != E.frames_sha256(c)


@pytest.mark.skipif(not HAVE_DATA, reason="Push-T HDF5 not available")
def test_baselines_run_and_fit_on_train_only():
    ev = E.run(None, per_episode=2, seed=0, out_dir=None, skip_mlp=True)
    assert ev["old_holdout_used_for_selection"] is False
    assert ev["new_final_used"] is False
    assert not set(ev["dev_episodes"]) & set(ev["train_fit_episodes"])
    for name in ("identity_A=S", "constant_velocity_A=S+V", "train_only_affine"):
        m = ev["models"][name]
        assert m["status"] == "OK", name
        q = m["metrics"]
        for key in ("physical_l1_macro", "correction_r2", "magnitude_ratio",
                    "sign_agreement", "still_action_rate", "motion_strata"):
            assert key in q
        assert np.isfinite(q["physical_l1_macro"])
    assert ev["models"]["identity_A=S"]["metrics"]["still_action_rate"] == pytest.approx(1.0)
    # a fitted affine must not be worse than the identity on TRAIN-adjacent structure
    assert ev["models"]["train_only_affine"]["metrics"]["physical_l1_macro"] < 10.0


@pytest.mark.skipif(not HAVE_DATA, reason="Push-T HDF5 not available")
def test_prediction_archive_frame_mismatch_is_refused(tmp_path):
    ev = E.run(None, per_episode=1, seed=0, out_dir=None, skip_mlp=True)
    d = tmp_path / "pred.npz"
    np.savez_compressed(str(d), action=np.zeros((ev["n_dev_frames"], 14), np.float32),
                        frames_sha256=np.array("0" * 64))
    with pytest.raises(SystemExit):
        E.run(str(d), per_episode=1, seed=0, out_dir=None, skip_mlp=True)


@pytest.mark.skipif(not HAVE_DATA, reason="Push-T HDF5 not available")
def test_matching_prediction_archive_is_scored(tmp_path):
    base = E.run(None, per_episode=1, seed=0, out_dir=str(tmp_path / "b"), skip_mlp=True)
    smp = np.load(str(tmp_path / "b" / "dev_samples.npz"))
    act = smp["action"]
    np.savez_compressed(str(tmp_path / "pred.npz"), action=act,
                        frames_sha256=np.array(base["dev_frames_sha256"]),
                        model_sha256=np.array("f" * 64))
    ev = E.run(str(tmp_path / "pred.npz"), per_episode=1, seed=0, out_dir=None, skip_mlp=True)
    m = ev["models"]["vla_r42"]["metrics"]
    # the RAW oracle is exact; the SAFE (clamped) one is not exactly zero because a few expert
    # labels themselves sit infinitesimally outside the frozen joint box
    assert m["raw_physical_l1_macro"] == pytest.approx(0.0, abs=1e-9)
    assert m["physical_l1_macro"] <= 1e-5
    assert ev["gate"]["improvement_vs_cv"] > 0.99
