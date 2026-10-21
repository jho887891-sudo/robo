#!/usr/bin/env python3
"""Regression for the teacher-forced CE used as the per-task forgetting anchor.

The first real 2-task run died here: "labels length 51 != logits length 820". The model prepends the
projected vision tokens to the text embeddings, so the logits are longer than the labels by that
prefix; the CE must shift the supervised rows by it, not require equal lengths.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _s in ("training", "code/xpolicylab/policy/OpenVLA_Pure",
           "code/xpolicylab/policy/OpenVLA_R42"):
    sys.path.insert(0, os.path.join(REPO, _s))
import m2_train_shared as T  # noqa: E402

V = 64
N_SUP = T.N_SUPERVISED


def _case(n_prompt=8, pad=0, seed=0):
    rng = np.random.RandomState(seed)
    L = n_prompt + N_SUP
    labels = np.concatenate([np.full(n_prompt, -100), rng.randint(0, V, size=N_SUP)])
    logits = rng.randn(pad + L, V).astype(np.float32)
    return logits, labels, n_prompt, pad


def test_zero_prefix_matches_the_plain_case():
    logits, labels, n_prompt, _ = _case(pad=0)
    out = T.token_ce_from_logits(logits, labels, n_prompt)
    assert out["vision_prefix_tokens"] == 0
    assert np.isfinite(out["ce"]) and out["ce"] > 0
    assert len(out["per_position"]) == N_SUP


def test_vision_prefix_is_shifted_not_rejected():
    logits, labels, n_prompt, pad = _case(pad=769)
    assert logits.shape[0] == 769 + len(labels)
    out = T.token_ce_from_logits(logits, labels, n_prompt)
    assert out["vision_prefix_tokens"] == 769
    assert np.isfinite(out["ce"])


def test_the_prefix_cancels_out_exactly():
    """With the shift applied, a prefixed logits array must score identically to its text-only tail.

    That is the whole point: the vision tokens are context, they must not move which TEXT positions the
    supervised labels are read from. If the offset were ignored, the rows would land on different text
    positions and the CE would differ (or, before the fix, the call would raise on the length check).
    """
    labels = np.concatenate([np.full(8, -100), np.arange(N_SUP) % V])
    rng = np.random.RandomState(1)
    body = rng.randn(len(labels), V).astype(np.float32)          # the text rows
    prefixed = np.concatenate([rng.randn(5, V).astype(np.float32), body])
    a = T.token_ce_from_logits(prefixed, labels, 8)
    b = T.token_ce_from_logits(body, labels, 8)
    assert a["vision_prefix_tokens"] == 5
    assert b["vision_prefix_tokens"] == 0
    assert a["ce"] == pytest.approx(b["ce"], rel=1e-9, abs=1e-9)
    assert a["per_position"] == pytest.approx(b["per_position"], rel=1e-9, abs=1e-9)


def test_shorter_logits_than_labels_is_refused():
    logits, labels, n_prompt, _ = _case(pad=0)
    with pytest.raises(Exception):
        T.token_ce_from_logits(logits[:-3], labels, n_prompt)


def test_all_ignore_index_is_refused():
    logits, labels, n_prompt, _ = _case(pad=0)
    labels = np.full(len(labels), -100)
    with pytest.raises(Exception):
        T.token_ce_from_logits(logits, labels, n_prompt)


def test_action_window_probs_shift_by_visual_prefix():
    """P6 must score action logits, not the 769 visual prefix rows."""
    import r42_adapter as R
    n_prompt, n_actions, pad = 8, T.N_ACTIONS, 5
    vocab = R.TOKEN_BEGIN_IDX + 1 + 256
    rng = np.random.RandomState(23)
    body = rng.randn(n_prompt + N_SUP, vocab).astype(np.float32)
    # Make the true action bin 7 strongly preferred at its causal row.
    idx = R.TOKEN_BEGIN_IDX + 1 + 7
    for k in range(n_actions):
        body[n_prompt - 1 + k, idx] = 17.0
    visual = rng.randn(pad, vocab).astype(np.float32)
    prefixed = np.concatenate([visual, body], axis=0)
    base = T.action_window_probs(body, n_prompt, n_text_tokens=body.shape[0])
    got = T.action_window_probs(prefixed, n_prompt, n_text_tokens=body.shape[0])
    np.testing.assert_array_equal(got["pred_bins"], base["pred_bins"])
    np.testing.assert_allclose(got["pred_probs"], base["pred_probs"], rtol=0, atol=0)
    assert got["pred_bins"].tolist() == [7] * n_actions
    assert T.ce_window_from_probs(got["pred_probs"], np.full(n_actions, 7)) == pytest.approx(
        T.ce_window_from_probs(base["pred_probs"], np.full(n_actions, 7)), abs=1e-10)
    # Old code would pick prefix rows and silently return a different distribution.
    wrong = T.action_window_probs(prefixed, n_prompt)
    assert not np.array_equal(wrong["pred_bins"], got["pred_bins"])


def test_action_window_probs_reject_bad_text_length():
    import r42_adapter as R
    logits = np.zeros((22, R.TOKEN_BEGIN_IDX + 257), dtype=np.float32)
    with pytest.raises(T.M2TrainError):
        T.action_window_probs(logits, n_prompt=8, n_text_tokens=99)


def test_old_p6_archive_cannot_be_scored(tmp_path):
    """Old prediction archives remain intact but are invalid as Gate evidence."""
    p = tmp_path / "legacy.npz"
    np.savez_compressed(p, **{"push_T/episodes": np.array(["episode_0.hdf5"])})
    with pytest.raises(T.M2TrainError, match="vision-prefix-aware-v2"):
        T.load_eval_archive(str(p))
