#!/usr/bin/env python3
"""Reading the sampler must never consume a sample (issue #26 item 5).

M2's resume log used to call sampler.next() and then decrement position. It happens to be a no-op
today, but a logging line may never depend on next()'s side effect, and the property must hold at
permutation boundaries too. peek() is the read-only accessor the log uses now.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "training"))
import r15_resume as RS  # noqa: E402

FRAMES = [("push_T/ep_%03d.hdf5" % i, 5 + i) for i in range(7)] + \
         [("stack_bowls/ep_%03d.hdf5" % i, 9 + i) for i in range(7)]


def test_peek_does_not_advance_and_matches_the_next_sample():
    s = RS.FrameSampler(FRAMES, seed=0)
    for _ in range(5):
        pos = s.position
        look = s.peek()
        assert s.position == pos, "peek must not advance the sampler"
        assert look == s.next()
    assert s.position == 5


def test_peek_matches_across_a_save_restore_round_trip_at_perm_boundaries():
    n = len(FRAMES)
    for pos in (0, 1, n - 2, n - 1, n, n + 1, 3 * n - 1):
        s = RS.FrameSampler(FRAMES, seed=7, position=pos)
        st = s.state()
        restored = RS.FrameSampler.from_state(FRAMES, st)
        assert restored.peek() == s.peek(), "pos=%d" % pos
        # and the CONTINUOUS sampler and the RESTORED sampler then take identical samples
        cont = RS.FrameSampler(FRAMES, seed=7, position=pos)
        a = [cont.next() for _ in range(5)]
        b = [restored.next() for _ in range(5)]
        assert a == b
        assert restored.position == pos + 5


def test_peek_follows_the_permutation_not_the_frame_order():
    s = RS.FrameSampler(FRAMES, seed=3)
    assert s.perm != list(range(len(FRAMES)))          # a real shuffle, not identity
    assert s.peek() == FRAMES[s.perm[0]]
    s.position = 2
    assert s.peek() == FRAMES[s.perm[2]]
    assert s.peek(1) == FRAMES[s.perm[3]]


def test_restore_after_a_partial_run_resumes_the_same_frames():
    """The M2 scenario: run 3 steps, save, restore, then continue - the sequence must not shift."""
    s = RS.FrameSampler(FRAMES, seed=11)
    first = [s.next() for _ in range(3)]
    st = s.state()
    r = RS.FrameSampler.from_state(FRAMES, st)
    # the log line peeks; it must not shift what step 4 consumes
    _ = r.peek()
    assert r.next() == s.next(), "the resumed sampler consumed a different frame"
    assert first[2] == FRAMES[s.perm[2]]
