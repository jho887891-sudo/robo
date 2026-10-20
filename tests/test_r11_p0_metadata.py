#!/usr/bin/env python3
"""R11 P0-3 (RED first): frozen normalizer + 21-field checkpoint metadata, SHA consistency and a
save/load round trip that reproduces the action bit-for-bit.

The R42 checkpoint metadata must carry the frozen normalizer itself (positional bounds for S plus
signed q99 motion scales for V/Acc and the residual quantiles q01/q99), their canonical SHA, and
the 64-TRAIN/16-DEV provenance.  Legacy 80-episode quantiles, legacy positional-only motion
normalisation and a 14D projector must all be refused.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_R42"))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))

import r42_adapter as R  # noqa: E402

CONTRACT = os.path.join(REPO, "evidence", "r9_gate_20261020", "frozen_contract_64train.json")
ORIGINAL_19 = ("task_id", "base_model_sha256", "lora_sha256", "projector_sha256", "proprio_dim",
               "proprio_layout", "history_policy", "camera_order", "prompt_source",
               "frozen_contract_sha256", "split_sha256", "codec_mode", "action_token_window",
               "target_representation", "train_steps", "seed", "gripper_policy", "safety_limits",
               "alignment_version")


@pytest.fixture(scope="module")
def frozen():
    return json.load(open(CONTRACT))


@pytest.fixture(scope="module")
def norm(frozen):
    return R.normalizer_payload(
        normalization_mode=frozen["normalization_mode"], pmin=frozen["pmin"], pmax=frozen["pmax"],
        q01=frozen["q01"], q99=frozen["q99"], v_scale=frozen["v_scale"],
        a_scale=frozen["a_scale"], motion_quantile=frozen["motion_quantile"],
        n_train_episodes=frozen["frozen_from"]["n_train_episodes_for_quantiles"],
        n_dev_episodes_excluded=frozen["frozen_from"]["n_dev_episodes_excluded"])


@pytest.fixture(scope="module")
def meta(frozen, norm):
    sha = "a" * 64
    nb = np.load(os.path.join(REPO, "data", "push_T_bounds.npz"))
    return R.build_metadata(
        "push_T", sha, sha, sha, sha, sha, train_steps=0, seed=0,
        gripper_policy=frozen["gripper_policy"],
        safety_limits={"amin": nb["amin"].astype(np.float32).tolist(),
                       "amax": nb["amax"].astype(np.float32).tolist(),
                       "max_rate": [0.2] * 14},
        normalizer=norm)


def test_original_metadata_fields_are_still_required():
    assert set(ORIGINAL_19).issubset(set(R.REQUIRED_METADATA))
    assert "normalizer" in R.REQUIRED_METADATA
    assert "normalizer_sha256" in R.REQUIRED_METADATA


def test_normalizer_payload_is_complete_and_sha_stable(norm, frozen):
    for key in ("normalization_mode", "pmin", "pmax", "q01", "q99", "v_scale", "a_scale",
                "motion_quantile", "fit_source"):
        assert key in norm
    assert norm["fit_source"]["n_train_episodes_for_quantiles"] == 64
    assert norm["fit_source"]["n_dev_episodes_excluded"] == 16
    assert norm["normalization_mode"] == "signed_motion"
    for key in ("pmin", "pmax", "q01", "q99", "v_scale", "a_scale"):
        assert len(norm[key]) == 14
    assert R.normalizer_sha256(R.normalizer_payload(**{
        "normalization_mode": frozen["normalization_mode"], "pmin": frozen["pmin"],
        "pmax": frozen["pmax"], "q01": frozen["q01"], "q99": frozen["q99"],
        "v_scale": frozen["v_scale"], "a_scale": frozen["a_scale"],
        "motion_quantile": frozen["motion_quantile"], "n_train_episodes": 64,
        "n_dev_episodes_excluded": 16})) == R.normalizer_sha256(norm)


def test_metadata_carries_normalizer_and_its_sha(meta, norm):
    assert R.validate_metadata(meta, expect_task="push_T") is True
    assert meta["normalizer"] == norm
    assert meta["normalizer_sha256"] == R.normalizer_sha256(norm)
    assert len(meta["normalizer_sha256"]) == 64
    # 19 original fields + normalizer + normalizer_sha256
    assert len(R.REQUIRED_METADATA) >= 21


def test_missing_normalizer_is_refused(frozen):
    sha = "b" * 64
    nb = np.load(os.path.join(REPO, "data", "push_T_bounds.npz"))
    with pytest.raises(R.R42Error):
        R.build_metadata("push_T", sha, sha, sha, sha, sha, train_steps=0, seed=0,
                         gripper_policy=frozen["gripper_policy"],
                         safety_limits={"amin": nb["amin"].tolist(), "amax": nb["amax"].tolist()},
                         normalizer=None)


def test_legacy_80_episode_quantiles_are_refused(frozen):
    with pytest.raises(R.R42Error):
        R.normalizer_payload(
            normalization_mode=frozen["normalization_mode"], pmin=frozen["pmin"],
            pmax=frozen["pmax"], q01=frozen["q01"], q99=frozen["q99"],
            v_scale=frozen["v_scale"], a_scale=frozen["a_scale"],
            motion_quantile=frozen["motion_quantile"], n_train_episodes=80,
            n_dev_episodes_excluded=0)


def test_legacy_positional_motion_normalisation_is_refused(frozen):
    with pytest.raises(R.R42Error):
        R.normalizer_payload(
            normalization_mode="legacy_position", pmin=frozen["pmin"], pmax=frozen["pmax"],
            q01=frozen["q01"], q99=frozen["q99"], v_scale=None, a_scale=None,
            motion_quantile=None, n_train_episodes=64, n_dev_episodes_excluded=16)


def test_tampered_normalizer_sha_is_refused(meta):
    bad = json.loads(json.dumps(meta))
    bad["normalizer"]["q99"] = [x + 1e-3 for x in bad["normalizer"]["q99"]]
    with pytest.raises(R.R42Error):
        R.validate_metadata(bad)


def test_roundtrip_decode_is_bit_identical(meta, norm):
    idx = np.arange(14, dtype=np.int64) * 17 % 256
    state = np.zeros(14, np.float32)
    velocity = np.zeros(14, np.float32)
    a1, r1, _i1 = R.decode_with_metadata(meta, idx, state, velocity)
    a2, r2, _i2 = R.decode_command(idx, np.asarray(norm["q01"], np.float32),
                                   np.asarray(norm["q99"], np.float32), state, velocity,
                                   limits=meta["safety_limits"])
    assert np.array_equal(a1, a2) and np.array_equal(r1, r2)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "meta.json")
        sha = R.save_metadata(p, meta)
        loaded = R.load_metadata(p, expect_task="push_T")
        assert loaded == meta
        assert sha == R.save_metadata(p, loaded)
        a3, r3, _i3 = R.decode_with_metadata(loaded, idx, state, velocity)
        assert np.array_equal(a1, a3) and np.array_equal(r1, r3)


def test_decode_with_metadata_rejects_bad_bins(meta):
    state = np.zeros(14, np.float32)
    velocity = np.zeros(14, np.float32)
    with pytest.raises(R.R42Error):
        R.decode_with_metadata(meta, np.zeros(13, np.int64), state, velocity)
    with pytest.raises(R.R42Error):
        R.decode_with_metadata(meta, np.full(14, 256, np.int64), state, velocity)


def test_q01_q99_must_be_ordered(frozen):
    bad = list(frozen["q01"])
    bad[0] = frozen["q99"][0] + 1.0
    with pytest.raises(R.R42Error):
        R.normalizer_payload(
            normalization_mode=frozen["normalization_mode"], pmin=frozen["pmin"],
            pmax=frozen["pmax"], q01=bad, q99=frozen["q99"], v_scale=frozen["v_scale"],
            a_scale=frozen["a_scale"], motion_quantile=frozen["motion_quantile"],
            n_train_episodes=64, n_dev_episodes_excluded=16)
