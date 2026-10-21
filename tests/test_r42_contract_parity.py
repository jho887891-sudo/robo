#!/usr/bin/env python3
"""R16 T4/T5/T6: 42D feature parity, codec parity, decode + safety parity.

T4  the 42D [S,V,Acc] features the deploy server feeds the projector must be the SAME row a
    training frame produces (training/r42_train_contract.py:114 build_training_row), for the same
    observed history - bit for bit, not approximately.

T5  the 256-bin codec used at deploy must be the single source of truth (OpenVLA_Pure/action_codec)
    in MODE_FIXED: same bins, same token ids, same legal window, no PAD/STOP leakage.

T6  decode is A_hat = S + V + R_hat, the gripper channels are S + V (DEV-selected), the physical
    box and the rate rule are both enforced, and the evaluation entry point
    (decode_with_metadata) is nothing but decode_command wired to the checkpoint metadata.

CPU only.  No 7B weights, no CUDA, no simulator.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_R42"))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))
sys.path.insert(0, os.path.join(REPO, "training"))

import action_codec as AC  # noqa: E402
import r42_adapter as R  # noqa: E402
import r42_deploy_server as S  # noqa: E402
import r42_train_contract as TC  # noqa: E402

CONTRACT = os.path.join(REPO, "evidence", "r9_gate_20261020", "frozen_contract_64train.json")
BOUNDS = os.path.join(REPO, "data", "push_T_bounds.npz")
PAD = 32000
STOP = 2


class StubBackend(object):
    """Deterministic stand-in with the real backend signature (see r42_deploy_server)."""

    def __init__(self, seed=0, vocab=32064):
        self.seed = seed
        self.vocab = vocab
        self.calls = 0

    def next_token_logits(self, input_ids, pixel_values=None, proprio=None):
        self.calls += 1
        sig = (self.seed * 1000003
               + sum((i + 1) * int(x) for i, x in enumerate(input_ids))) % (2 ** 31 - 1)
        if proprio is not None:
            sig = (sig + int(round(float(np.abs(proprio).sum()) * 1e6))) % (2 ** 31 - 1)
        logits = (np.random.RandomState(sig).randn(self.vocab) * 0.01).astype(np.float32)
        logits[PAD] = 500.0
        logits[STOP] = 400.0
        return logits


def _meta():
    frozen = json.load(open(CONTRACT))
    norm = R.normalizer_payload(
        normalization_mode=frozen["normalization_mode"], pmin=frozen["pmin"],
        pmax=frozen["pmax"], q01=frozen["q01"], q99=frozen["q99"],
        v_scale=frozen["v_scale"], a_scale=frozen["a_scale"],
        motion_quantile=frozen["motion_quantile"], n_train_episodes=64,
        n_dev_episodes_excluded=16)
    nb = np.load(BOUNDS)
    sha = "c" * 64
    return R.build_metadata(
        "push_T", sha, sha, sha, sha, sha, train_steps=0, seed=0,
        gripper_policy=frozen["gripper_policy"],
        safety_limits={"amin": nb["amin"].astype(np.float32).tolist(),
                       "amax": nb["amax"].astype(np.float32).tolist(),
                       "max_rate": [0.5] * 14},
        normalizer=norm)


def _server():
    return S.R42DeployServer(_meta(), backend=StubBackend(), device="cpu",
                             tokenizer_prompt_ids=[128000, 9906, 11, 5123, 29901])


def _images(shape=(8, 10, 3)):
    rs = np.random.RandomState(7)
    return {cam: rs.randint(0, 255, shape).astype(np.uint8) for cam in R.CAMERA_ORDER}


def _inside(alpha=0.5):
    nb = np.load(BOUNDS)
    lo = np.asarray(nb["amin"], np.float32)
    hi = np.asarray(nb["amax"], np.float32)
    return (lo + float(alpha) * (hi - lo)).astype(np.float32)


# ------------------------------------------------------------------ T4: 42D feature parity
def test_deploy_features_equal_the_training_row_features():
    """Observed history -> 42D features must be the training row's features, exactly."""
    server = _server()
    rs = np.random.RandomState(3)
    base = _inside(0.5)
    span = 1e-3
    states = np.stack([base + rs.uniform(-span, span, 14).astype(np.float32)
                       for _ in range(6)]).astype(np.float32)
    actions = np.stack([states[t + 1] for t in range(5)]).astype(np.float32)
    # feed the observed history t=0..4 exactly like an episode does; compare from t=2 on,
    # which is the first frame with a full three-frame window (V and Acc both defined).
    for t in range(5):
        out = server.act(_images(), states[t], env_idx=0, timestamp=float(t) * 0.04)
        if t < 2:
            continue
        row = TC.build_training_row(states, actions, t, server.contract)
        assert row["features"].shape == (42,)
        assert np.array_equal(out["features"], row["features"]), (
            "deploy 42D features differ from the training row at t=%d" % t)
        assert np.array_equal(out["velocity"], row["velocity"])
        assert out["features"].dtype == np.float32
        assert np.isfinite(out["features"]).all()


def test_deploy_features_use_three_observed_frames_and_no_prediction():
    """Boundary policy: V=0,Acc=0 at the first frame; Acc=0 at the second; then the real row."""
    server = _server()
    base = _inside(0.5)
    s0 = base.copy()
    s1 = (base + 1e-3).astype(np.float32)
    out0 = server.act(_images(), s0, env_idx=0, timestamp=0.0)
    out1 = server.act(_images(), s1, env_idx=0, timestamp=0.04)
    assert np.allclose(out0["velocity"], 0.0)
    assert np.array_equal(out1["velocity"], (s1 - s0).astype(np.float32))
    zeros = np.zeros(14, np.float32)
    assert np.array_equal(out0["features"], server.contract.transform(s0, zeros, zeros))
    assert np.array_equal(out1["features"],
                          server.contract.transform(s1, (s1 - s0).astype(np.float32), zeros))


def test_feature_parity_is_per_env_isolated():
    server = _server()
    a = _inside(0.4)
    b = _inside(0.6)
    server.act(_images(), a, env_idx=0, timestamp=0.0)
    server.act(_images(), b, env_idx=1, timestamp=0.0)
    server.act(_images(), a, env_idx=0, timestamp=0.04)
    f_env1 = server.history_features(1)
    fresh = _server()
    fresh.act(_images(), b, env_idx=1, timestamp=0.0)
    assert np.array_equal(f_env1, fresh.history_features(1))


# --------------------------------------------------------------------- T5: codec parity
def test_action_codec_is_the_single_source_of_truth():
    assert R.TOKEN_BEGIN_IDX == AC.ACTION_TOKEN_BEGIN_IDX == 31743
    assert tuple(R.CODEC_WINDOW) == (AC.LEGAL_TOKEN_MIN, AC.LEGAL_TOKEN_MAX) == (31744, 31999)
    assert R.STOP_ID == AC.STOP_INDEX == 2 and R.PAD_TOKEN_ID == AC.PAD_TOKEN_ID == 32000
    assert AC.MODE_FIXED == "fixed"


def test_decode_direction_is_exact_for_all_256_bins():
    """Deploy only decodes: bin -> token -> bin and bin -> residual must be exact everywhere."""
    meta = _meta()
    q01 = np.asarray(meta["normalizer"]["q01"], np.float32)
    q99 = np.asarray(meta["normalizer"]["q99"], np.float32)
    for bin_index in range(AC.N_BINS):
        ids = np.asarray(AC.token_ids(np.full(14, bin_index, np.int64)), np.int64)
        assert np.array_equal(AC.indices_from_token_ids(ids),
                              np.full(14, bin_index, np.int64))
        assert np.array_equal(ids, np.full(14, 31744 + bin_index))
        AC.assert_legal_token_ids(ids)
        residual = R.residual_from_indices(np.full(14, bin_index, np.int64), q01, q99)
        level = AC.decode_indices(np.full(14, bin_index, np.int64), mode=AC.MODE_FIXED)
        assert np.array_equal(residual, AC.denormalize(level, q01, q99))


def test_encode_of_the_decoded_level_is_stable_within_one_bin():
    """LEVELS sits on a np.digitize bin edge, so a float32 round trip can move one bin.

    That is a property of the signed normalisation, not of the token mapping (which is exact
    above), and it cannot affect deploy: deploy only decodes, it never re-encodes.  The bound is
    recorded so a future codec change shows up here as a bigger drift.
    """
    meta = _meta()
    q01 = np.asarray(meta["normalizer"]["q01"], np.float32)
    q99 = np.asarray(meta["normalizer"]["q99"], np.float32)
    total = AC.N_BINS * 14
    flipped = 0
    worst = 0
    for bin_index in range(AC.N_BINS):
        residual = R.residual_from_indices(np.full(14, bin_index, np.int64), q01, q99)
        got_bins, got_ids = R.token_ids_for_residual(residual, q01, q99)
        diff = np.abs(got_bins - np.full(14, bin_index, np.int64))
        worst = max(worst, int(diff.max()))
        flipped += int((diff > 0).sum())
        AC.assert_legal_token_ids(got_ids)
    print("[T5] float32 re-encode drift: %d/%d channels moved, max %d bin"
          % (flipped, total, worst))
    # The decoded level lands exactly on a np.digitize edge, so which side float32 rounds to is
    # a coin flip: a 0..1 bin drift on a minority of channels is expected, >1 is not.
    assert worst <= 1, "a >1 bin drift would be a real codec change, not float32 rounding"
    assert flipped > 0, "the digitize edge effect disappeared - re-check that LEVELS is unchanged"


def test_deploy_tokens_match_action_codec_arithmetic():
    meta = _meta()
    q01 = np.asarray(meta["normalizer"]["q01"], np.float32)
    q99 = np.asarray(meta["normalizer"]["q99"], np.float32)
    rs = np.random.RandomState(11)
    for _ in range(25):
        residual = rs.uniform(-0.02, 0.02, 14).astype(np.float32)
        bins, ids = R.token_ids_for_residual(residual, q01, q99)
        ref_bins = np.asarray(AC.encode_indices(AC.normalize(residual, q01, q99),
                                                mode=AC.MODE_FIXED), np.int64)
        ref_ids = np.asarray(AC.token_ids(ref_bins), np.int64)
        assert np.array_equal(bins, ref_bins)
        assert np.array_equal(ids, ref_ids)
        AC.assert_legal_token_ids(ref_ids)
        assert PAD not in set(int(x) for x in ids) and STOP not in set(int(x) for x in ids)


def test_illegal_token_ids_are_refused():
    for bad in (31743, 32000, 32064, 1, 2):
        with pytest.raises(ValueError):
            AC.assert_legal_token_ids(np.asarray([31744, bad], np.int64))


def test_out_of_range_bin_indices_are_refused():
    q01 = np.asarray(_meta()["normalizer"]["q01"], np.float32)
    q99 = np.asarray(_meta()["normalizer"]["q99"], np.float32)
    for bad in (-1, 256):
        with pytest.raises(R.R42Error):
            R.residual_from_indices(np.full(14, bad, np.int64), q01, q99)


# ------------------------------------------------------------- T6: decode and safety parity
def test_decode_with_metadata_is_decode_command_with_the_checkpoint_limits():
    meta = _meta()
    norm = meta["normalizer"]
    rs = np.random.RandomState(5)
    state = _inside(0.5)
    velocity = rs.uniform(-1e-3, 1e-3, 14).astype(np.float32)
    bins = rs.randint(0, 256, 14)
    a, r_a, info_a = R.decode_with_metadata(meta, bins, state, velocity)
    b, r_b, info_b = R.decode_command(bins, norm["q01"], norm["q99"], state, velocity,
                                      limits=meta["safety_limits"],
                                      gripper_dims=tuple(meta["gripper_policy"]["gripper_dims"]))
    assert np.array_equal(a, b) and np.array_equal(r_a, r_b)
    assert info_a == info_b


def test_gripper_channels_are_state_plus_velocity_only():
    meta = _meta()
    state = _inside(0.5)
    state[6] = 0.25
    state[13] = 0.75
    velocity = np.full(14, 1e-4, np.float32)
    for bins in (np.zeros(14, np.int64), np.full(14, 255, np.int64)):
        cmd, residual, info = R.decode_with_metadata(meta, bins, state, velocity)
        assert cmd[6] == pytest.approx(float(state[6] + velocity[6]), abs=1e-7)
        assert cmd[13] == pytest.approx(float(state[13] + velocity[13]), abs=1e-7)
        assert info["gripper_fallback_dims"] == [6, 13]
        assert residual[6] != 0.0        # the decoded residual exists, it is just not applied


def test_decoded_command_stays_inside_the_checkpoint_box():
    meta = _meta()
    lo = np.asarray(meta["safety_limits"]["amin"], np.float32)
    hi = np.asarray(meta["safety_limits"]["amax"], np.float32)
    rs = np.random.RandomState(9)
    for _ in range(50):
        state = _inside(float(rs.uniform(0.0, 1.0)))
        velocity = rs.uniform(-0.01, 0.01, 14).astype(np.float32)
        bins = rs.randint(0, 256, 14)
        cmd, _r, _info = R.decode_with_metadata(meta, bins, state, velocity)
        assert (cmd >= lo - 1e-6).all() and (cmd <= hi + 1e-6).all()
        assert cmd[6] >= 0.0 and cmd[6] <= 1.0
        assert cmd[13] >= 0.0 and cmd[13] <= 1.0


def test_rate_rule_caps_every_dim_at_max_rate_times_dt():
    meta = _meta()
    state = _inside(0.5)
    prev = _inside(0.5)
    velocity = np.zeros(14, np.float32)
    dt = 0.04
    rate = np.asarray(meta["safety_limits"]["max_rate"], np.float32)
    cmd, _r, info = R.decode_with_metadata(meta, np.full(14, 255, np.int64), state, velocity,
                                           prev_cmd=prev, dt=dt)
    step = (rate * dt).astype(np.float32)
    assert (np.abs(cmd - prev) <= step + 1e-6).all()
    assert info["rate_limited"] >= 0


def test_rate_rule_needs_prev_cmd_dt_and_max_rate_together():
    meta = _meta()
    state = _inside(0.5)
    velocity = np.zeros(14, np.float32)
    bins = np.zeros(14, np.int64)
    with pytest.raises(R.R42Error, match="rate limiting needs"):
        R.decode_with_metadata(meta, bins, state, velocity, prev_cmd=state)
    with pytest.raises(R.R42Error, match="rate limiting needs"):
        R.decode_with_metadata(meta, bins, state, velocity, dt=0.04)


def test_prev_command_outside_the_box_is_refused():
    meta = _meta()
    lo = np.asarray(meta["safety_limits"]["amin"], np.float32)
    prev = lo - 1e-3
    with pytest.raises(R.R42Error, match="prev_cmd is outside"):
        R.decode_with_metadata(meta, np.zeros(14, np.int64), _inside(0.5),
                               np.zeros(14, np.float32), prev_cmd=prev, dt=0.04)


def test_non_finite_state_is_refused_by_decode():
    meta = _meta()
    state = _inside(0.5)
    state[4] = np.inf
    with pytest.raises(R.R42Error):
        R.decode_with_metadata(meta, np.zeros(14, np.int64), state, np.zeros(14, np.float32))


def test_metadata_with_a_tampered_normalizer_is_refused():
    meta = _meta()
    meta["normalizer_sha256"] = "d" * 64
    with pytest.raises(R.R42Error, match="normalizer_sha256 mismatch"):
        R.validate_metadata(meta)
