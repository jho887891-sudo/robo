#!/usr/bin/env python3
"""R24 regression: the fair comparator must treat all three policies identically.

Two real defects are pinned here.

1. AUDIT PR#30's finding: the R17 comparator clipped the MODEL but not the baselines, and applied the
   deployment gripper fallback to nothing. A ranking produced that way is not a ranking of policies.
2. MY OWN defect while fixing it: r42_adapter.decode_command returns (cmd, r_hat, info), and I unpacked
   it as (raw, safe, info), so "SAFE" was actually the residual vector. Every policy then appeared to
   have an L1 of ~0.38 (a residual compared against an action).

Also pinned: the frame manifest must not contain frames the training contract cannot consume (t >= 2),
and it must be deterministic and frozen before any model runs.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _s in ("training", "code/xpolicylab/policy/OpenVLA_Pure",
           "code/xpolicylab/policy/OpenVLA_R42", "tools"):
    sys.path.insert(0, os.path.join(REPO, _s))

import r42_adapter as R              # noqa: E402
import m2_task_normalizer as TN      # noqa: E402
import r24_fair_audit as FA          # noqa: E402
import r24_frame_manifest as FM      # noqa: E402

NORM = os.path.join(REPO, "data", "m2_task_normalizer_2task.json")
MANIFEST = os.path.join(REPO, "mission", "r24", "frame_manifest.json")


@pytest.fixture(scope="module")
def ctx():
    payload = json.load(open(NORM))
    return TN.decode_context(payload, "push_T"), payload


def test_decode_policy_returns_a_command_not_a_residual(ctx):
    """decode_command yields (cmd, r_hat, info); the helper must expose cmd as the command."""
    c, _p = ctx
    limits = {"amin": list(map(float, c["pmin"])), "amax": list(map(float, c["pmax"]))}
    # a non-trivial state so the command and the residual are separated in magnitude: with s = v = 0 the
    # command IS the (clipped) residual and the two are not distinguishable
    s = np.full(14, 0.01, np.float32)
    v = np.full(14, 0.001, np.float32)
    bins = np.full(14, 128, np.int64)
    raw, safe, rhat, info = FA.decode_policy(c, limits, (6, 13), bins, s, v)
    assert raw.shape == safe.shape == rhat.shape == (14,)
    # assert against the CANONICAL codec rather than re-deriving its convention: the point of this test is
    # that the helper surfaces the codec's own residual and not some other vector
    expect = R.residual_from_indices(bins, c["q01"], c["q99"])
    assert np.allclose(rhat, expect, atol=1e-9), "r_hat must be the codec inverse, not a command"
    # a command must be on the scale of a joint value (state + velocity + residual), not of a residual
    assert np.abs(safe).max() > 5.0 * np.abs(rhat).max(), \
        "the SAFE command must not be mistaken for the (much smaller) residual"
    # and it must be the state plus velocity plus that residual, before clipping
    assert np.allclose(raw[[6, 13]], (s + v)[[6, 13]], atol=1e-6), "gripper fallback on the raw too"


def test_gripper_fallback_is_applied_to_every_policy(ctx):
    """cmd[6] and cmd[13] must be S+V regardless of residual: the deployment contract."""
    c, _p = ctx
    limits = {"amin": list(map(float, c["pmin"])), "amax": list(map(float, c["pmax"]))}
    s = np.zeros(14, np.float32)
    v = np.zeros(14, np.float32)
    for bins in (np.zeros(14, np.int64), np.full(14, 255, np.int64), np.full(14, 128, np.int64)):
        raw, safe, _r, _i = FA.decode_policy(c, limits, (6, 13), bins, s, v)
        assert safe[6] == pytest.approx(float(s[6] + v[6]), abs=1e-6)
        assert safe[13] == pytest.approx(float(s[13] + v[13]), abs=1e-6)


def test_baselines_and_model_share_the_same_bounds_and_codec(ctx):
    """A CV residual of 0 must decode to A = S + V through the SAME path the model uses."""
    c, _p = ctx
    limits = {"amin": list(map(float, c["pmin"])), "amax": list(map(float, c["pmax"]))}
    s = np.full(14, 0.01, np.float32)
    v = np.full(14, 0.002, np.float32)
    lo = np.asarray(limits["amin"], np.float32)
    hi = np.asarray(limits["amax"], np.float32)
    bins = FA.bins_for_residual(c, np.zeros(14, np.float32))
    _raw, safe, _r, _i = FA.decode_policy(c, limits, (6, 13), bins, s, v)
    # The authority for "what does CV command" is decode_command itself, called with a zero residual and
    # going through the codec. Compare against THAT instead of a hand-written formula.
    q01 = np.asarray(c["q01"], np.float64)
    q99 = np.asarray(c["q99"], np.float64)
    zero_r = R.residual_from_indices(bins, q01, q99)
    ref_cmd, _rh, _i = R.decode_command(bins, q01, q99, s, v, limits=limits, prev_cmd=None, dt=None,
                                       gripper_dims=(6, 13), on_unsafe="raise")
    assert np.allclose(safe, ref_cmd, atol=1e-9), "the helper must reproduce decode_command exactly"
    # and the interesting physical statement: a ZERO residual is not exactly representable, so the CV
    # command differs from the ideal S+V by the codec's own quantisation.
    assert np.abs(zero_r).max() > 0.0, "a zero residual must not be exactly representable"
    # one FULL codec bin: the observed worst case is 1.16e-4 against a bin width of 1.73e-4, i.e. the
    # codec rounds rather than truncates and a zero residual can sit most of a bin away from 0
    one_bin = float(((q99 - q01) / 256.0).max())
    ideal = np.clip(s + v, lo, hi)
    for d in (6, 13):
        ideal[d] = s[d] + v[d]
    arm = [d for d in range(14) if d not in (6, 13)]
    assert (np.abs(safe[arm] - ideal[arm]) <= one_bin + 1e-6).all(), \
        "the CV command must be clip(S + V) to within one codec bin"
    assert np.allclose(safe[[6, 13]], (s + v)[[6, 13]], atol=1e-6)


@pytest.mark.skipif(not os.path.isfile(MANIFEST), reason="manifest not generated")
def test_manifest_is_frozen_and_consumable():
    m = json.load(open(MANIFEST))
    assert m["frozen_before_model_run"] is True
    assert m.get("manifest_sha256")
    import r42_train_contract as TC
    for task, block in m["frames"].items():
        for split in ("train", "dev"):
            frames = block[split]
            assert frames, "%s/%s is empty" % (task, split)
            assert all(f["t"] >= TC.MIN_HISTORY_FRAMES for f in frames), \
                "the manifest must not contain frames the contract cannot build a row for"
            assert all(f["stratum"] for f in frames)


@pytest.mark.skipif(not os.path.isfile(MANIFEST), reason="manifest not generated")
def test_manifest_hash_covers_its_own_content():
    import hashlib
    m = json.load(open(MANIFEST))
    body = json.dumps({k: v for k, v in m.items() if k != "manifest_sha256"}, sort_keys=True)
    assert hashlib.sha256(body.encode()).hexdigest() == m["manifest_sha256"], \
        "the manifest was edited after freezing"


def test_manifest_selection_is_deterministic_and_model_free():
    """Same strata definition, same hash rule: picking twice must give the same frames."""
    recs = [{"episode": "e%d.hdf5" % i, "t": 3 + i, "speed": 0.001 * (i % 5),
             "resid_mag": 0.0001 * (i % 3)} for i in range(60)]
    stats = FM.stratify(recs)
    assert stats["speed_median"] >= 0
    a = FM.pick(recs, 3, seed=0)
    b = FM.pick(recs, 3, seed=0)
    assert [(r["episode"], r["t"]) for r in a] == [(r["episode"], r["t"]) for r in b]
    assert all("stratum" in r for r in a)
