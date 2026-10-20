#!/usr/bin/env python3
"""R11 P0-1 (RED first): the R42 hand-written bin/token/STOP codec must be BIT-EXACT with the
unified action_codec, and the causal label shift must match the legacy entrypoint construction.

Defect under test (pre-fix behaviour):
  * r42_adapter.decode_command() denormalised with (q99-q01) while action_codec.denormalize()
    uses (q99-q01+1e-8) -> encode/decode are not exact inverses and are not bit-equal to the
    codec that produced the frozen 64-TRAIN q01/q99.
  * r42_adapter.build_labels() did not apply the trailing <space> (id 29871) rule that
    train_pushT_pure2.make_ids() applies -> prompt length, label positions and therefore the
    HF causal shift were off by one token relative to the trained contract.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_R42"))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))

import action_codec as AC  # noqa: E402
import r42_adapter as R  # noqa: E402

CONTRACT = os.path.join(REPO, "evidence", "r9_gate_20261020", "frozen_contract_64train.json")
SPACE = 29871
BEGIN = 31743
STOP = 2
PAD = 32000


@pytest.fixture(scope="module")
def qq():
    meta = json.load(open(CONTRACT))
    q01 = np.asarray(meta["q01"], np.float32)
    q99 = np.asarray(meta["q99"], np.float32)
    assert q01.shape == (14,) and q99.shape == (14,)
    return q01, q99


def reference_make_ids(prompt_ids, bin_indices):
    """Pure-numpy replica of train_pushT_pure2.make_ids() (torch-free reference)."""
    pids = [int(x) for x in prompt_ids]
    if pids[-1] != SPACE:
        pids = pids + [SPACE]
    aid = [BEGIN + int(t) + 1 for t in bin_indices]
    inp = pids + aid + [STOP]
    lab = [-100] * len(pids) + aid + [STOP]
    return inp, lab


# ------------------------------------------------------------------ encode == unified codec
def test_encoder_is_bit_equal_to_action_codec_on_random_14d(qq):
    q01, q99 = qq
    rs = np.random.RandomState(20261021)
    for _ in range(500):
        residual = (rs.randn(14) * 0.02).astype(np.float32)
        idx_r, ids_r = R.token_ids_for_residual(residual, q01, q99)
        idx_ac = AC.encode_indices(AC.normalize(residual, q01, q99), mode=AC.MODE_FIXED)
        assert np.array_equal(idx_r, np.asarray(idx_ac, np.int64))
        assert np.array_equal(ids_r, np.asarray(AC.token_ids(idx_ac), np.int64))


def test_all_256_bins_are_reachable_and_bijective(qq):
    q01, q99 = qq
    sweep = np.linspace(-1.0, 1.0, 65536, dtype=np.float32)
    dim = 3
    seen = set()
    for v in sweep:
        residual = np.zeros(14, np.float32)
        residual[dim] = 0.0
        phys = np.zeros(14, np.float32)
        phys[dim] = AC.denormalize(np.asarray([v], np.float32), q01[dim:dim + 1],
                                   q99[dim:dim + 1])[0]
        residual = phys
        idx, _ids = R.token_ids_for_residual(residual, q01, q99)
        seen.add(int(idx[dim]))
    assert seen == set(range(256)), "reachable bins: %d/256" % len(seen)


def test_decode_is_bit_equal_to_action_codec_denormalize(qq):
    q01, q99 = qq
    idx = np.arange(256, dtype=np.int64)
    dim = 5
    tokens = np.zeros(14, np.int64)
    state = np.zeros(14, np.float32)
    velocity = np.zeros(14, np.float32)
    for k in idx:
        tokens[dim] = int(k)
        _cmd, r_hat, _info = R.decode_command(tokens, q01, q99, state, velocity)
        want = AC.denormalize(AC.decode_indices(np.asarray([k], np.int64), mode=AC.MODE_FIXED),
                              q01[dim:dim + 1], q99[dim:dim + 1])[0]
        assert r_hat[dim] == want, "bin %d: r42=%r codec=%r" % (k, r_hat[dim], want)


def test_encode_decode_roundtrip_is_bit_equal_to_the_codec(qq):
    """decode -> encode must reproduce the codec's own answer for every one of the 256 bins.

    (float32 is not an exact field, so the *reference* is the unified codec itself rather than the
    identity; the residual span is asserted to stay within one bin as a sanity bound.)
    """
    q01, q99 = qq
    tokens = np.zeros(14, np.int64)
    state = np.zeros(14, np.float32)
    velocity = np.zeros(14, np.float32)
    for dim in (0, 6, 9, 13):
        worst = 0
        for k in range(256):
            tokens[dim] = k
            _cmd, r_hat, _info = R.decode_command(tokens, q01, q99, state, velocity)
            back, _ids = R.token_ids_for_residual(r_hat, q01, q99)
            ref = AC.encode_indices(AC.normalize(r_hat, q01, q99), mode=AC.MODE_FIXED)
            assert int(back[dim]) == int(ref[dim]), \
                "dim %d bin %d: r42 -> %d but codec -> %d" % (dim, k, int(back[dim]), int(ref[dim]))
            worst = max(worst, abs(int(back[dim]) - k))
        assert worst <= 1, "dim %d round-trip drifts by %d bins" % (dim, worst)


def test_boundaries_have_the_documented_saturation_semantics(qq):
    """Measured property of the FROZEN codec (q01/q99 are the 1%/99% quantiles):

      * the [q01,q99] interval is mapped through (2(x-q01)/(q99-q01+1e-8))-1, so exactly q99 sits
        2*1e-8/(q99-q01) ~ 6.7e-7 BELOW +1.0 and therefore encodes to bin 254, not 255;
      * the closed top of the codec is q99+1e-8 (== denormalize(+1.0)), which encodes to 255.
    These are asserted here instead of an idealised "q99 -> 255" so that a future change of the
    frozen codec cannot pass silently.
    """
    q01, q99 = qq
    at_q01, _ = R.token_ids_for_residual(q01, q01, q99)
    at_q99, _ = R.token_ids_for_residual(q99, q01, q99)
    above, _ = R.token_ids_for_residual(q99 * 4.0, q01, q99)
    below, _ = R.token_ids_for_residual(q01 * 4.0, q01, q99)
    assert np.array_equal(at_q01, np.zeros(14, np.int64))
    assert np.array_equal(at_q99, np.full(14, 254, np.int64))
    assert np.array_equal(below, np.zeros(14, np.int64))
    assert np.array_equal(above, np.full(14, 255, np.int64))
    # the closed top of the lattice is exactly the top of the denormalisation range
    top = AC.denormalize(np.ones(14, np.float32), q01, q99)
    top_idx, _ = R.token_ids_for_residual(top, q01, q99)
    assert np.array_equal(top_idx, np.full(14, 255, np.int64))



def test_token_ids_are_always_inside_the_window(qq):
    q01, q99 = qq
    rs = np.random.RandomState(7)
    for _ in range(200):
        residual = (rs.randn(14) * 0.05).astype(np.float32)
        _idx, ids = R.token_ids_for_residual(residual, q01, q99)
        assert ids.min() >= 31744 and ids.max() <= 31999
        assert PAD not in set(int(x) for x in ids)
        assert STOP not in set(int(x) for x in ids)


def test_stop_and_pad_are_not_interchangeable():
    assert AC.STOP_INDEX == 2
    assert AC.PAD_TOKEN_ID == 32000
    assert not (AC.LEGAL_TOKEN_MIN <= AC.STOP_INDEX <= AC.LEGAL_TOKEN_MAX)
    assert not (AC.LEGAL_TOKEN_MIN <= AC.PAD_TOKEN_ID <= AC.LEGAL_TOKEN_MAX)
    with pytest.raises(ValueError):
        AC.validate_token_ids(np.asarray([PAD]))
    with pytest.raises(ValueError):
        AC.validate_token_ids(np.asarray([STOP]))


# ------------------------------------------------------- prompt / label / causal shift
def test_prompt_gets_the_trailing_space_token():
    pids = [128000, 9906, 11]
    out = R.normalize_prompt_ids(pids)
    assert out == pids + [SPACE]
    assert R.normalize_prompt_ids(pids + [SPACE]) == pids + [SPACE]


def test_build_labels_matches_the_legacy_entrypoint(qq):
    q01, q99 = qq
    rs = np.random.RandomState(3)
    residual = (rs.randn(14) * 0.01).astype(np.float32)
    pids = [128000, 9906, 11, 5123, 29901]
    got = R.build_labels(pids, residual, q01, q99)
    idx, ids = R.token_ids_for_residual(residual, q01, q99)
    want_inp, want_lab = reference_make_ids(pids, idx.tolist())
    assert got["input_ids"] == want_inp
    assert got["labels"] == want_lab
    assert got["n_prompt"] == len(pids) + 1
    assert got["n_supervised"] == 15


def test_causal_label_shift_positions(qq):
    """HF causal shift: token at position p is predicted by logits[:, p-1]; verify the 14 action
    tokens plus STOP cover exactly the intended 15 supervised predictions."""
    q01, q99 = qq
    rs = np.random.RandomState(11)
    residual = (rs.randn(14) * 0.01).astype(np.float32)
    pids = [128000, 9906, 11, 5123, 29901]
    got = R.build_labels(pids, residual, q01, q99)
    lab = got["labels"]
    inp = got["input_ids"]
    n_prompt = got["n_prompt"]
    assert len(lab) == len(inp) == n_prompt + 15
    assert all(x == -100 for x in lab[:n_prompt])
    # every supervised token is preceded by exactly one prediction slot
    for k in range(15):
        assert lab[n_prompt + k] == inp[n_prompt + k]
        assert lab[n_prompt + k] != -100
    # prediction used for action token 0 comes from the last prompt position
    assert max(n_prompt - 1, 0) == n_prompt - 1
    assert lab[n_prompt] == got["action_ids"][0]
    assert lab[-1] == STOP
    assert got["n_supervised"] == 15
