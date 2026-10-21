#!/usr/bin/env python3
"""R10.2 D3 regression: OpenVLA_R42 adapter (CPU only, no GPU, no 7B)."""
import json
import os
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_R42"))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))

import r42_adapter as R  # noqa: E402
from r8_normalization import NormalizationContract  # noqa: E402

CONTRACT = os.path.join(REPO, "evidence", "r9_gate_20261020", "frozen_contract_64train.json")
RESULTS = []


def ok(name, cond, detail=None):
    RESULTS.append({"check": name, "pass": bool(cond), "detail": detail or {}})
    print("[%s] %-46s %s" % ("PASS" if cond else "FAIL", name,
                             json.dumps(detail or {}, default=str)[:160]))
    return bool(cond)


def raises(fn, exc=Exception):
    try:
        fn()
        return False
    except exc:
        return True


def main():
    meta = json.load(open(CONTRACT))
    contract = NormalizationContract.from_metadata(meta)
    rs = np.random.RandomState(0)
    s0 = (rs.randn(14) * 0.02).astype(np.float32)
    s1 = (s0 + rs.randn(14) * 0.002).astype(np.float32)
    s2 = (s1 + rs.randn(14) * 0.002).astype(np.float32)

    h = R.EpisodeHistory(contract)
    f0 = h.observe(0, s0, timestamp=0.0)
    ok("t0_motion_slots_are_zero", bool(np.allclose(f0[14:], 0.0)), {"max": float(np.abs(f0[14:]).max())})
    f1 = h.observe(0, s1, timestamp=0.04)
    v = s1 - s0
    want_v = np.clip(v / np.asarray(meta["v_scale"], np.float32), -1, 1)
    ok("t1_velocity_matches_manual_and_acc_zero",
       bool(np.allclose(f1[14:28], want_v, atol=1e-6)) and bool(np.allclose(f1[28:], 0.0)),
       {"v_max_delta": float(np.abs(f1[14:28] - want_v).max())})
    f2 = h.observe(0, s2, timestamp=0.08)
    acc = (s2 - s1) - (s1 - s0)
    want_acc = np.clip(acc / np.asarray(meta["a_scale"], np.float32), -1, 1)
    ok("t2_acceleration_matches_manual", bool(np.allclose(f2[28:], want_acc, atol=1e-6)),
       {"acc_max_delta": float(np.abs(f2[28:] - want_acc).max())})
    ok("feature_is_42d_finite", f2.shape == (42,) and bool(np.isfinite(f2).all()))

    # per-env isolation
    h2 = R.EpisodeHistory(contract)
    h2.observe(0, s0)
    h2.observe(1, s2)
    a = h2.features(0)
    b = h2.features(1)
    ok("envs_are_isolated", not np.allclose(a, b))
    h2.reset(0)
    ok("reset_clears_env", raises(lambda: h2.features(0), R.R42Error)
       and bool(np.isfinite(h2.features(1)).all()))
    h2.observe(1, s1, timestamp=1.00)
    ok("timestamp_backwards_rejected",
       raises(lambda: h2.observe(1, s2, timestamp=0.50), R.R42Error))

    ok("camera_order_enforced",
       R.validate_images({"cam_head": np.zeros((4, 4, 3), np.uint8),
                          "cam_left_wrist": np.zeros((4, 4, 3), np.uint8),
                          "cam_right_wrist": np.zeros((4, 4, 3), np.uint8)}) is not None,
       {"ok": True})
    ok("camera_order_wrong_rejected",
       raises(lambda: R.validate_images({"cam_left_wrist": np.zeros((4, 4, 3), np.uint8),
                                        "cam_head": np.zeros((4, 4, 3), np.uint8),
                                        "cam_right_wrist": np.zeros((4, 4, 3), np.uint8)}),
              R.R42Error))
    ok("wrong_dtype_rejected",
       raises(lambda: R.validate_images({"cam_head": np.zeros((4, 4, 3), np.float32),
                                        "cam_left_wrist": np.zeros((4, 4, 3), np.uint8),
                                        "cam_right_wrist": np.zeros((4, 4, 3), np.uint8)}),
              R.R42Error))

    q01 = np.asarray(meta["q01"], np.float32)
    q99 = np.asarray(meta["q99"], np.float32)
    r_true = (rs.randn(14) * 0.005).astype(np.float32)
    idx, ids = R.token_ids_for_residual(r_true, q01, q99)
    ok("token_ids_in_window", idx.shape == (14,)
       and int(ids.min()) >= R.CODEC_WINDOW[0] and int(ids.max()) <= R.CODEC_WINDOW[1],
       {"window": list(R.CODEC_WINDOW), "min": int(ids.min()), "max": int(ids.max())})
    lab = R.build_labels([1, 2, 3, 4], r_true, q01, q99)
    n_sup = int(sum(1 for x in lab["labels"] if x != R.IGNORE_INDEX))
    # R11 P0-1: the prompt now carries the legacy trailing <space> (29871) token, so a 4-token
    # prompt becomes a 5-token prefix; the supervised block stays 14 action tokens + STOP.
    ok("labels_mask_prompt", n_sup == 15 and lab["n_prompt"] == 5
       and lab["input_ids"][4] == 29871
       and lab["labels"][-1] == R.STOP_ID, {"n_supervised": n_sup})

    amin = np.full(14, -1.3, np.float32)
    amax = np.full(14, 2.5, np.float32)
    cmd, r_hat, info = R.decode_command(idx, q01, q99, s1, v)
    want = s1 + v + r_hat
    ok("decode_is_S_plus_V_plus_Rhat",
       bool(np.allclose(cmd[[d for d in range(14) if d not in R.GRIPPER_DIMS]],
                        want[[d for d in range(14) if d not in R.GRIPPER_DIMS]], atol=1e-5)),
       {"max_arm_delta": float(np.abs(cmd[[d for d in range(14) if d not in R.GRIPPER_DIMS]]
                                      - want[[d for d in range(14) if d not in R.GRIPPER_DIMS]]).max())})
    ok("gripper_uses_constant_velocity_fallback",
       bool(np.allclose(cmd[list(R.GRIPPER_DIMS)],
                        (s1 + v)[list(R.GRIPPER_DIMS)], atol=1e-6)),
       {"gripper": cmd[list(R.GRIPPER_DIMS)].tolist()})
    cmd_lim, _r2, info_lim = R.decode_command(idx, q01, q99, s1, v,
                                              limits={"amin": amin, "amax": amax,
                                                      "max_rate": np.full(14, 0.05, np.float32)},
                                              prev_cmd=s1, dt=0.04)
    ok("rate_limit_actually_applied",
       info_lim["rate_limited"] > 0
       and bool(np.all(np.abs(cmd_lim - s1) <= 0.05 * 0.04 + 1e-6)),
       {"rate_limited": info_lim["rate_limited"]})

    sha = "a" * 64
    # R11 P0-3: an R42 checkpoint must carry the frozen normalizer payload and its SHA.
    norm = R.normalizer_payload(
        normalization_mode=meta["normalization_mode"], pmin=meta["pmin"], pmax=meta["pmax"],
        q01=meta["q01"], q99=meta["q99"], v_scale=meta["v_scale"], a_scale=meta["a_scale"],
        motion_quantile=meta["motion_quantile"],
        n_train_episodes=meta["frozen_from"]["n_train_episodes_for_quantiles"],
        n_dev_episodes_excluded=meta["frozen_from"]["n_dev_episodes_excluded"])
    m = R.build_metadata("push_T", sha, sha, sha, sha, sha, train_steps=0, seed=0,
                         gripper_policy={"gripper_dims": [6, 13]},
                         safety_limits={"amin": amin.tolist(), "amax": amax.tolist()},
                         normalizer=norm)
    ok("metadata_valid", R.validate_metadata(m, expect_task="push_T") is True,
       {"proprio_dim": m["proprio_dim"], "window": m["action_token_window"]})
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "meta.json")
        h1 = R.save_metadata(p, m)
        ok("metadata_roundtrip_sha", len(h1) == 64 and R.load_metadata(p)["task_id"] == "push_T")

    bad = dict(m); bad["proprio_dim"] = 28
    ok("rejects_28d_metadata", raises(lambda: R.validate_metadata(bad), R.R42Error))
    bad = dict(m); bad["proprio_dim"] = 7
    ok("rejects_prismatic_7d_metadata", raises(lambda: R.validate_metadata(bad), R.R42Error))
    bad = dict(m); bad["action_token_window"] = [31745, 32000]
    ok("rejects_wrong_token_window", raises(lambda: R.validate_metadata(bad), R.R42Error))
    bad = dict(m); bad.pop("split_sha256")
    ok("rejects_missing_field", raises(lambda: R.validate_metadata(bad), R.R42Error))
    ok("rejects_task_mismatch",
       raises(lambda: R.validate_metadata(m, expect_task="stack_bowls"), R.R42Error))

    n = sum(1 for x in RESULTS if x["pass"])
    print("\n%d/%d checks passed" % (n, len(RESULTS)))
    out = os.path.join(REPO, "evidence", "r10_2", "test_r10_2_r42.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"checks": RESULTS, "n_pass": n, "n_total": len(RESULTS)}, open(out, "w"), indent=1)
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
