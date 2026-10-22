#!/usr/bin/env python3
"""The bug that invalidated three R16 parity reports, pinned.

r16_kv_equivalence.decode_full() appended the ARGMAX BIN (0..255) as the next token id while
decode_cached() correctly appended TOKEN_BEGIN_IDX+1+bin. So the "full reference" decoded from ordinary
text tokens after step 0 and every comparison concluded the cached path diverged. These tests use a stub
backend that records every call it receives, so a bin leaking into the sequence is caught directly.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _s in ("training", "code/xpolicylab/policy/OpenVLA_Pure",
           "code/xpolicylab/policy/OpenVLA_R42", "tools"):
    sys.path.insert(0, os.path.join(REPO, _s))
import r42_adapter as R              # noqa: E402
import r16_kv_equivalence as KV      # noqa: E402

LO = R.TOKEN_BEGIN_IDX + 1           # 31744
TOK_MAX = LO + 255                   # 31999, the last legal action token
WIN_HI = LO + 256                    # 32000, exclusive end of the 256-bin window
V = 32064
PROMPT = [1, 2, 3]


class StubBackend(object):
    """Records every call. The next bin depends on the sequence length, so the two loops can differ if
    they feed different tokens - which is exactly the failure mode under test."""

    def __init__(self):
        self.calls = []
        self.step_tokens = []

    def _logits(self, bin_idx):
        lg = np.zeros(V, np.float32)
        w = np.zeros(256, np.float32)
        w[int(bin_idx) % 256] = 20.0
        lg[LO:WIN_HI] = w
        return lg

    def next_token_logits(self, input_ids, pixel_values=None, proprio=None):
        self.calls.append([int(x) for x in input_ids])
        return self._logits((len(input_ids) * 7) % 256)

    def prefill_cached(self, input_ids, pixel_values=None, proprio=None):
        self.calls.append([int(x) for x in input_ids])
        return self._logits((len(input_ids) * 7) % 256), {"len": len(input_ids)}, len(input_ids)

    def step_cached(self, tok_id, cache, n_past):
        self.step_tokens.append(int(tok_id))
        n = n_past + 1
        return self._logits((n * 7) % 256), {"len": n}, n


def window(vec):
    return np.asarray(vec, np.float32).reshape(-1)[LO:WIN_HI]


def test_decode_full_only_ever_appends_legal_window_tokens():
    b = StubBackend()
    bins, logits = KV.decode_full(b, PROMPT, None, None, window)
    assert len(bins) == R.N_ACTIONS == len(logits)
    for call in b.calls:
        extra = call[len(PROMPT):]                      # everything the decoder appended
        assert all(LO <= t <= TOK_MAX for t in extra), \
            "a bin index leaked in as a token: %r" % (extra[:5],)


def test_decode_full_and_decode_cached_agree_on_a_stub():
    b = StubBackend()
    bins_f, _ = KV.decode_full(b, PROMPT, None, None, window)
    bins_c, _ = KV.decode_cached(b, PROMPT, None, None, window)
    assert bins_f == bins_c
    assert b.step_tokens and all(LO <= t <= TOK_MAX for t in b.step_tokens)


def test_the_old_bug_is_detectable():
    """Feeding bins as tokens must be caught, so the regression cannot silently return."""
    class Strict(StubBackend):
        def next_token_logits(self, input_ids, pixel_values=None, proprio=None):
            extra = [int(x) for x in input_ids][len(PROMPT):]
            if any(t < LO for t in extra):
                raise AssertionError("ILLEGAL TOKEN %r" % (extra[:3],))
            return StubBackend.next_token_logits(self, input_ids, pixel_values, proprio)

    strict = Strict()
    ids = []
    with pytest.raises(AssertionError):
        for _k in range(R.N_ACTIONS):
            w = window(strict.next_token_logits(PROMPT + ids))
            ids.append(int(np.argmax(w)))               # <-- the pre-fix behaviour
    KV.decode_full(Strict(), PROMPT, None, None, window)   # the fixed loop is clean


def test_the_guard_bounds_are_exact():
    KV.needs_legal_token_check(None, [LO])              # first legal token
    KV.needs_legal_token_check(None, [TOK_MAX])         # last legal token
    with pytest.raises(AssertionError):
        KV.needs_legal_token_check(None, [LO - 1])
    with pytest.raises(AssertionError):
        KV.needs_legal_token_check(None, [TOK_MAX + 1])
