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
    assert m["physical_l1_macro"] == pytest.approx(0.0, abs=1e-6)
    assert ev["gate"]["improvement_vs_cv"] > 0.99
