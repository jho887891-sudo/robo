#!/usr/bin/env python3
"""RED->GREEN regression for the P6 action-window alignment bug (issue #26, audit PR #25).

The defect: token_ce_from_logits() correctly offset the 769-token multimodal vision prefix, but
action_window_probs() read the 14 action positions from n_prompt-1 WITHOUT that offset. Everything the
P6 gate compared - ce_window, pred_bins, the counterfactual task-routing distributions - was read from
vision-token rows, not from the legal action rows. These tests fail on the pre-fix code and must pass
after it; they are written against the contract, not against the implementation.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _s in ("training", "code/xpolicylab/policy/OpenVLA_Pure",
           "code/xpolicylab/policy/OpenVLA_R42"):
    sys.path.insert(0, os.path.join(REPO, _s))
import m2_train_shared as T  # noqa: E402

V = 32000
LO = 31744                       # TOKEN_BEGIN_IDX + 1
N_ACT = T.N_ACTIONS              # 14
N_BINS = 256
NORM = os.path.join(REPO, "data", "m2_task_normalizer_2task.json")


def _text_rows(seed=0, n_prompt=10):
    """Logits for [prompt + 15 supervised] text positions, with a peaked window per action slot."""
    rng = np.random.RandomState(seed)
    L = n_prompt + N_ACT + 1
    lg = rng.randn(L, V).astype(np.float32) * 0.01
    for k in range(N_ACT):
        lg[n_prompt - 1 + k, LO + (k * 7) % N_BINS] = 12.0     # distinct argmax per slot
    return lg, L


def _with_vision_prefix(lg, pad=769, seed=1):
    rng = np.random.RandomState(seed)
    pre = rng.randn(pad, V).astype(np.float32) * 0.01
    for r in range(0, pad, 97):                                 # make some vision rows peaky too
        pre[r, LO + 3] = 20.0
    return np.concatenate([pre, lg], axis=0)


def test_window_probs_are_identical_with_and_without_the_vision_prefix():
    lg, L = _text_rows()
    n_prompt = 10
    pref = _with_vision_prefix(lg)
    plain = T.action_window_probs(lg, n_prompt, n_text_tokens=L)
    offset = T.action_window_probs(pref, n_prompt, n_text_tokens=L)
    assert np.array_equal(plain["pred_bins"], offset["pred_bins"])
    assert np.allclose(plain["pred_probs"], offset["pred_probs"], atol=1e-6)
    # the expected bins are the peaked ones planted in the TEXT rows
    assert list(plain["pred_bins"]) == [(k * 7) % N_BINS for k in range(N_ACT)]
    assert offset["vision_prefix_tokens"] == 769


def test_window_probs_reject_a_negative_prefix():
    lg, L = _text_rows()
    with pytest.raises(Exception):
        T.action_window_probs(lg[:5], 10, n_text_tokens=L)


def test_window_probs_without_the_offset_would_read_the_wrong_rows():
    """Guard against 'fixing' this by ignoring the prefix: the offset must change the answer."""
    lg, L = _text_rows()
    pref = _with_vision_prefix(lg)
    correct = T.action_window_probs(pref, 10, n_text_tokens=L)
    # exactly the pre-fix behaviour: same n_prompt, no offset
    as_if_no_prefix = T.action_window_probs(pref, 10, n_text_tokens=None)
    assert correct["vision_prefix_tokens"] == 769
    assert as_if_no_prefix["vision_prefix_tokens"] == 0
    assert not np.array_equal(correct["pred_bins"], as_if_no_prefix["pred_bins"])


@pytest.mark.skipif(not os.path.isfile(NORM), reason="M2 normalizer artefact not present")
def test_evaluate_frame_uses_each_prompts_own_text_length():
    """A counterfactual prompt has a DIFFERENT length, so its pad differs; the frame banner must not
    silently reuse the actual task's offset."""
    payload = json.load(open(NORM))
    task, other = "push_T", "stack_bowls"
    residual = np.zeros(14, np.float32)          # any residual; we only test row selection
    seen = {}

    def predict(prompt_task, frame_task, episode, t):
        n_prompt = 10 if prompt_task == task else 13        # different instruction lengths
        lg, L = _text_rows(seed=0 if prompt_task == task else 2, n_prompt=n_prompt)
        pref = _with_vision_prefix(lg, pad=769)
        seen[prompt_task] = (n_prompt, L)
        return {"logits": pref, "labels": np.concatenate([np.full(n_prompt, -100),
                                                          np.zeros(N_ACT + 1, np.int64)]),
                "n_prompt": n_prompt, "residual": residual}

    out = T.evaluate_frame(predict, payload, task, "ep.hdf5", 5, [task, other])
    assert out["vision_prefix_tokens"] == 769
    for key in (task, other):
        expected = [((k * 7) % N_BINS) if key == task else ((k * 7) % N_BINS)
                    for k in range(N_ACT)]
        assert list(out["counterfactual"][key]["pred_bins"]) == expected
        assert out["counterfactual"][key]["vision_prefix_tokens"] == 769


@pytest.mark.skipif(not os.path.isfile(NORM), reason="M2 normalizer artefact not present")
def test_old_archive_without_the_marker_is_refused(tmp_path):
    payload = json.load(open(NORM))
    task = "push_T"
    entry = {"task": task, "episode": "ep.hdf5", "t": 5, "ce": 1.0, "ce_window": 1.0,
             "pred_bins": np.zeros(N_ACT, np.int64), "pred_probs": np.full((N_ACT, N_BINS), 1 / 256,
                                                                          np.float32),
             "bins_true": np.zeros(N_ACT, np.int64), "counterfactual": {}}
    blocks = {task: {"frames": [entry], "ce_mean": 1.0, "ce_full_mean": 1.0, "ce_frames": [1.0],
                     "frames_sha256": "0" * 64}}
    # a LEGACY archive: the arrays an old run wrote, WITHOUT the alignment marker
    legacy_arrays = {k: v for k, v in T.eval_archive_from(blocks).items()
                     if k not in ("action_window_alignment", "schema")}
    legacy = tmp_path / "legacy.npz"
    np.savez_compressed(str(legacy), **legacy_arrays)
    with pytest.raises(Exception) as exc:
        T.load_eval_archive(str(legacy))
    assert "LEGACY" in str(exc.value).upper() or "alignment" in str(exc.value).lower()

    # a correct archive carries the marker and loads
    good = tmp_path / "good.npz"
    T.write_eval_archive(str(good), blocks, data_contract={"probe": 1})
    loaded = T.load_eval_archive(str(good))
    assert loaded["action_window_alignment"] == "vision-prefix-aware-v2"
    assert loaded["tasks"] == [task]
