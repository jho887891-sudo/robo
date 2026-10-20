#!/usr/bin/env python3
"""R10.2 D5: offline one-step action metrics for 42D/R policies (pure numpy).

All metrics use the SAME conventions as R7-R10: physical 14D L1 (frame mean, then episode
macro), correction R^2 on R = (A-S)-V, sign agreement, cosine, per-dim/per-arm, motion strata.
"""
import json
import sys

import numpy as np

GRIPPER = (6, 13)


def l1_frame(pred, gt):
    p = np.asarray(pred, np.float64)
    g = np.asarray(gt, np.float64)
    if p.shape != g.shape or p.ndim != 2:
        raise ValueError("pred/gt must share shape (T,14)")
    return float(np.abs(p - g).mean(axis=1).mean())


def episode_macro(per_episode_l1):
    xs = [float(x) for x in per_episode_l1]
    if not xs:
        raise ValueError("no episodes")
    return float(np.mean(xs))


def correction_r2(r_hat, r_gt):
    p = np.asarray(r_hat, np.float64).reshape(-1)
    g = np.asarray(r_gt, np.float64).reshape(-1)
    var = float(np.var(g))
    if var <= 0:
        raise ValueError("degenerate target variance")
    return float(1.0 - np.var(p - g) / var)


def sign_agreement(pred, gt, delta_floor=1e-3, r_floor=1e-4):
    p = np.asarray(pred, np.float64)
    g = np.asarray(gt, np.float64)
    mask = (np.abs(g) >= delta_floor) & (np.abs(p) >= r_floor)
    if not mask.any():
        return {"n": 0, "agreement": None}
    agree = (np.sign(p) == np.sign(g))[mask]
    return {"n": int(mask.sum()), "agreement": float(agree.mean())}


def cosine(pred, gt):
    p = np.asarray(pred, np.float64).reshape(-1)
    g = np.asarray(gt, np.float64).reshape(-1)
    denom = float(np.linalg.norm(p) * np.linalg.norm(g))
    if denom == 0.0:
        return None
    return float(np.dot(p, g) / denom)


def per_dim_l1(pred, gt):
    p = np.asarray(pred, np.float64)
    g = np.asarray(gt, np.float64)
    return [float(x) for x in np.abs(p - g).mean(axis=0)]


def per_arm_l1(pred, gt):
    d = per_dim_l1(pred, gt)
    return {"left": float(np.mean(d[:7])), "right": float(np.mean(d[7:])),
            "left_gripper": d[6], "right_gripper": d[13]}


def motion_strata(pred, gt, edges=(0.0, 0.002, 0.01, 1.0)):
    p = np.asarray(pred, np.float64)
    g = np.asarray(gt, np.float64)
    mag = np.abs(g).max(axis=1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mag >= lo) & (mag < hi)
        if m.any():
            out.append({"range": [lo, hi], "n": int(m.sum()),
                        "l1": float(np.abs(p[m] - g[m]).mean())})
    return out


def clip_rate(pred, amin, amax):
    p = np.asarray(pred, np.float64)
    lo = np.asarray(amin, np.float64)
    hi = np.asarray(amax, np.float64)
    return float(((p < lo) | (p > hi)).mean())


def evaluate(pred, gt, amin=None, amax=None, r_hat=None, r_gt=None):
    res = {"n_frames": int(np.asarray(pred).shape[0]),
           "physical_l1_frame_mean": l1_frame(pred, gt),
           "per_dim_l1": per_dim_l1(pred, gt),
           "per_arm_l1": per_arm_l1(pred, gt),
           "cosine_frame_mean": cosine(pred, gt),
           "sign_agreement": sign_agreement(pred, gt),
           "motion_strata": motion_strata(pred, gt)}
    if r_hat is not None and r_gt is not None:
        res["correction_r2"] = correction_r2(r_hat, r_gt)
    if amin is not None and amax is not None:
        res["clip_rate"] = clip_rate(pred, amin, amax)
    return res


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else ""
    rs = np.random.RandomState(0)
    gt = rs.randn(64, 14) * 0.01
    pred = gt + rs.randn(64, 14) * 0.001
    res = {"toolkit": "r10_2_offline_eval",
           "synthetic_self_check": evaluate(pred, gt, r_hat=pred, r_gt=gt),
           "identity_baseline": evaluate(np.zeros_like(gt), gt),
           "status": "IMPLEMENTED_CPU_SYNTHETIC_ONLY"}
    print(json.dumps({k: v for k, v in res.items() if k != "identity_baseline"}, indent=1)[:800])
    if out:
        json.dump(res, open(out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
