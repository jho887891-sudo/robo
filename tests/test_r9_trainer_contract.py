#!/usr/bin/env python3
"""Regression for training/r9_trainer_contract.py (R9 Gate B 42D/R wiring).

Fast, CPU-only, no trainer import and no 7B model: it pins the contract semantics and
every fail-closed guard that stops 14D and 42D paths from being mixed silently.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("training", "diagnostics", "code/xpolicylab/policy/OpenVLA_Pure"):
    sys.path.insert(0, os.path.join(REPO, sub))

import action_codec as AC  # noqa: E402
import action_target as AT  # noqa: E402
import r9_trainer_contract as RC  # noqa: E402

RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
CONTRACT = os.path.join(REPO, "evidence", "r9_gate_20261020", "frozen_contract_64train.json")
LEGACY = os.path.join(REPO, "evidence", "r8_vla_42d_smoke_20261020", "frozen_r8_contract.json")
SPLIT = os.path.join(REPO, "data", "split80.json")
BOUNDS = os.environ.get("PUSHT_BOUNDS", RT + "/push_T_bounds.npz")
RESULTS = []


def ok(name, cond, detail=None):
    RESULTS.append({"check": name, "pass": bool(cond), "detail": detail or {}})
    print("[%s] %-46s %s" % ("PASS" if cond else "FAIL", name,
                             json.dumps(detail or {}, default=str)[:220]))
    return bool(cond)


def raises(fn, exc=ValueError):
    try:
        fn()
        return False
    except exc:
        return True


def main():
    contract, meta = RC.load_frozen_contract(CONTRACT)
    q01 = np.asarray(meta["q01"], np.float32)
    q99 = np.asarray(meta["q99"], np.float32)
    nb = np.load(BOUNDS)
    splits = RC.select_splits(SPLIT)

    ok("split_64_16_20_disjoint",
       len(splits["train"]) == 64 and len(splits["dev"]) == 16 and len(splits["holdout"]) == 20
       and not (set(splits["train"]) & set(splits["dev"])),
       {k: len(v) for k, v in splits.items()})
    ok("contract_mode_signed_motion", contract.mode == "signed_motion", {"mode": contract.mode})
    ok("rejects_legacy_80_episode_contract", raises(lambda: RC.load_frozen_contract(LEGACY)))

    bad = json.loads(json.dumps(meta))
    bad["normalization_mode"] = "legacy_position"
    tmp = os.path.join(REPO, "evidence", "r9_gate_20261020", "_tmp_bad_contract.json")
    with open(tmp, "w") as fh:
        json.dump(bad, fh)
    try:
        ok("rejects_legacy_normalisation", raises(lambda: RC.load_frozen_contract(tmp)))
    finally:
        os.remove(tmp)

    ok("frame_pool_starts_at_2",
       int(RC.frame_pool(10)[0]) == 2 and int(RC.frame_pool(10)[-1]) == 8,
       {"pool": RC.frame_pool(10).tolist()})
    ok("rejects_frame_t_below_2", raises(lambda: RC.check_frame_index(1))
       and raises(lambda: RC.check_frame_index(0)))
    ok("accepts_frame_t_2", RC.check_frame_index(2) == 2)

    rng = np.random.RandomState(0)
    lo = nb["amin"].astype(np.float32)
    hi = nb["amax"].astype(np.float32)
    # stay strictly inside the joint limits so clipping cannot mask a decode error
    state = (lo + 0.1 * (hi - lo) + rng.rand(50, 14) * 0.8 * (hi - lo)).astype(np.float32)
    t = 10
    f = RC.proprio42(state[t], state[t - 1], state[t - 2], contract)
    v = state[t] - state[t - 1]
    acc = (state[t] - state[t - 1]) - (state[t - 1] - state[t - 2])
    manual = np.concatenate([
        np.clip(2.0 * (state[t] - nb["pmin"]) / (nb["pmax"] - nb["pmin"] + 1e-8) - 1.0, -1, 1),
        np.clip(v / meta["v_scale"], -1, 1), np.clip(acc / meta["a_scale"], -1, 1)]).astype(np.float32)
    ok("proprio42_matches_manual", f.shape == (42,) and float(np.abs(f - manual).max()) <= 1e-6,
       {"shape": list(f.shape), "max_delta": float(np.abs(f - manual).max())})

    # small motion so S+V+R stays inside the joint limits (a decode error, not clipping)
    v = (rng.randn(14) * 1e-4).astype(np.float32)
    action = (state[t] + v + np.asarray(meta["q01"], np.float32)).astype(np.float32)
    r = RC.target_residual(state[t], action, v)
    ref = AT.training_target(state[t], action, AT.RESIDUAL) - v
    ok("target_R_identity", float(np.abs(r - ref).max()) <= 1e-6,
       {"max_delta": float(np.abs(r - ref).max())})

    idx, ids = RC.encode_tokens(r, q01, q99)
    ok("tokens_are_legal_action_ids",
       idx.shape == (14,) and int(ids.min()) >= 31744 and int(ids.max()) <= 31999,
       {"token_range": [int(ids.min()), int(ids.max())]})

    cmd, decoded, excluded = RC.decode_command(ids, q01, q99, state[t], v, BOUNDS)
    step = (q99 - q01) / 255.0
    model_dims = list(RC.A.DEV_MODEL_DIMS)
    inrange = (r >= q01) & (r <= q99)
    err = float(np.abs(cmd - action)[model_dims][inrange[model_dims]].max()) \
        if inrange[model_dims].any() else 0.0
    cv = state[t] + v
    ok("decode_roundtrip_and_gripper_policy",
       err <= float(step.max()) + 1e-5 and excluded == [6, 13]
       and bool(np.allclose(cmd[[6, 13]], np.clip(cv[[6, 13]], nb["amin"][[6, 13]],
                                                          nb["amax"][[6, 13]]), atol=1e-5)),
       {"max_abs_error_in_range_model_dims": err, "one_bin_step_max": float(step.max()),
        "excluded": excluded})

    ok("rejects_14d_projector_with_42d_features", raises(lambda: RC.assert_projector_dim(14)))
    ok("rejects_absolute_target_with_42d", raises(lambda: RC.assert_target_mode(AT.ABSOLUTE)))
    ok("accepts_42d_projector", RC.assert_projector_dim(42) is True)

    ic = RC.integration_contract(contract, meta, CONTRACT, BOUNDS, SPLIT)
    ok("integration_contract_complete",
       ic["proprio_dim"] == 42 and ic["target_representation"] == "velocity_residual"
       and ic["normalization_mode"] == "signed_motion" and len(ic["frozen_contract_sha256"]) == 64
       and ic["gripper_policy"]["gripper_dims"] == [6, 13],
       {"keys": sorted(ic.keys())})

    gaps = RC.standard_entrypoint_gaps()
    ok("entrypoint_gaps_documented", len(gaps) >= 6 and any(g["gap"] == "target" for g in gaps),
       {"n_gaps": len(gaps)})

    n = sum(1 for x in RESULTS if x["pass"])
    print("\n%d/%d checks passed" % (n, len(RESULTS)))
    out = os.path.join(REPO, "evidence", "r9_trainer_trace_20261020", "test_r9_trainer_contract.json")
    if os.path.isdir(os.path.dirname(out)):
        with open(out, "w") as fh:
            json.dump({"checks": RESULTS, "n_pass": n, "n_total": len(RESULTS)}, fh, indent=1)
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
