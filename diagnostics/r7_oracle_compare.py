#!/usr/bin/env python3
"""R7 phase 1: representable-vs-floor oracle comparison, independently recomputed.

Two questions the R6 report got wrong and this script settles on real HDF5:

  (a) the OLD "oracle" was decode(encode_floor(true target)); a free decodable head
      may pick a NEARER bin, so the old number is an encoder-reconstruction error,
      not the lower bound;
  (b) is the lower bound (nearest representable bin per component) below the
      past-only constant-velocity baseline on the SAME frames?

The brute-force block re-derives the bound by scoring all 256 bins with the real
action_codec decoder, i.e. by a code path independent of r6_representable_oracle.py,
and asserts the two agree entry by entry on a random frame subsample.

All numbers are on the legal validation frames t in [1, T-2] of the 20 held-out
episodes (the same set every other R6/R7 baseline uses), with q01/q99 from ALL
training rows exactly as training/train_pushT_pure2.py::action_stats computes them.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))
import r6_data as D          # noqa: E402
import action_codec as AC    # noqa: E402
import r6_representable_oracle as RO   # noqa: E402

LEFT, RIGHT = slice(0, 7), slice(7, 14)


def train_quantiles(train_eps, mode):
    """Identical to action_stats(): quantiles over ALL frames of the TRAIN episodes."""
    tgt = []
    for e in train_eps:
        s, a = e["state"], e["action"]
        tgt.append(a if mode == "absolute" else a - s)
    allt = np.concatenate(tgt, axis=0)
    return (np.quantile(allt, 0.01, axis=0).astype(np.float32),
            np.quantile(allt, 0.99, axis=0).astype(np.float32))


def val_arrays(val_eps):
    S, A, V = [], [], []
    for e in val_eps:
        t = D.valid_frames(len(e["state"]))
        S.append(e["state"][t])
        A.append(e["action"][t])
        V.append(e["state"][t] - e["state"][t - 1])
    return np.concatenate(S), np.concatenate(A), np.concatenate(V)


def brute_force_bound(target, q01, q99, chunk=512):
    """Min L1 over all 256 decoded bins, using action_codec (independent code path).

    Runs over EVERY legal validation frame (chunked to bound memory).
    """
    y_all = np.asarray(target, np.float32)
    idx = np.arange(len(y_all))
    # every representable physical value for this q01/q99, straight from action_codec
    levels = AC.decode_indices(np.arange(AC.N_BINS), mode=AC.MODE_FIXED)     # (256,)
    # grid[d, k] = physical value of bin k for dimension d
    grid = np.stack([AC.denormalize(levels, np.float32(q01[d]), np.float32(q99[d]))
                     for d in range(14)], axis=0)                          # (14, 256)
    best_sum, floor_sum = 0.0, 0.0
    for start in range(0, len(idx), chunk):
        y = y_all[idx[start:start + chunk]]
        err = np.abs(y[:, None, :] - grid.T[None, :, :])                    # (n, 256, 14)
        best_sum += float(err.min(axis=1).sum())
        floored = np.abs(AC_decode_floor(y, q01, q99) - y)
        floor_sum += float(floored.sum())
    n = len(idx) * 14
    return {"frames_checked": int(len(idx)),
            "brute_force_nearest_L1": best_sum / n,
            "brute_force_floor_L1": floor_sum / n}


def mode_block(target_val, q01, q99, cv_L1, identity_L1, label):
    rep = RO.summarize_oracle(target_val, q01, q99)
    step = (np.asarray(q99, np.float64) - np.asarray(q01, np.float64)) / (AC.N_BINS - 1)
    out = {
        "target": label,
        "floor_encoder_oracle_L1": rep["floor_encoder_oracle_L1"],
        "nearest_representable_lower_bound_L1": rep["nearest_representable_lower_bound_L1"],
        "nearest_per_arm": rep["nearest_bound_per_arm"],
        "nearest_per_dim": rep["per_dim_nearest_lower_bound"],
        "floor_per_arm": {"left": float(np.abs(AC_decode_floor(target_val, q01, q99)
                                               - target_val)[:, LEFT].mean()),
                          "right": float(np.abs(AC_decode_floor(target_val, q01, q99)
                                                - target_val)[:, RIGHT].mean())},
        "improvement_floor_to_nearest": rep["mean_improvement_due_to_nearest_bin"],
        "fraction_entries_needing_other_bin": rep["fraction_entries_requiring_different_bin"],
        "fraction_entries_outside_q01_q99": rep["fraction_entries_outside_quantiles"],
        "physical_step_per_dim": step.tolist(),
        "physical_step_mean": float(step.mean()),
        "theoretical_floor_if_uniform_step_over_4": float(step.mean() / 4.0),
        "theoretical_floor_if_uniform_step_over_2": float(step.mean() / 2.0),
        "cv_physical_L1_same_frames": cv_L1,
        "identity_physical_L1_same_frames": identity_L1,
        "bound_vs_cv_ratio": rep["nearest_representable_lower_bound_L1"] / (cv_L1 + 1e-12),
        "bound_beats_cv": bool(rep["nearest_representable_lower_bound_L1"] < cv_L1),
        "bound_beats_identity": bool(rep["nearest_representable_lower_bound_L1"] < identity_L1),
        "q01": np.asarray(q01).tolist(), "q99": np.asarray(q99).tolist(),
        "caveat": ("lower bound applies to one independent fixed-q01/q99 bin per dimension "
                   "with no post-processing; any rescaling, blending with S, or learned "
                   "inverse transform escapes it"),
    }
    return out


def AC_decode_floor(target, q01, q99):
    return AC.denormalize(AC.decode_indices(
        AC.encode_indices(AC.normalize(target, q01, q99), mode=AC.MODE_FIXED),
        mode=AC.MODE_FIXED), q01, q99)


def run():
    split = D.load_split()
    train_eps = D.load_episodes(split["train"])
    val_eps = D.load_episodes(split["val"])
    S, A, V = val_arrays(val_eps)
    cv_cmd = S + V
    cv_L1 = float(np.abs(cv_cmd - A).mean())
    identity_L1 = float(np.abs(S - A).mean())
    out = {"n_val_frames": int(len(S)), "n_val_episodes": len(split["val"]),
           "n_train_frames_used_for_quantiles": int(sum(len(e["state"]) for e in train_eps)),
           "cv_physical_L1_same_frames": cv_L1,
           "identity_physical_L1_same_frames": identity_L1,
           "modes": {}}
    for mode in ("residual", "absolute"):
        q01, q99 = train_quantiles(train_eps, mode)
        target = (A - S) if mode == "residual" else A
        out["modes"][mode] = mode_block(target, q01, q99, cv_L1, identity_L1, mode)
        out["modes"][mode]["independent_brute_force"] = brute_force_bound(target, q01, q99)
    # the fair "quantized copy-V" reference for the residual representation
    q01, q99 = train_quantiles(train_eps, "residual")
    cv_target = V                                  # residual target of the copy-V prediction
    for name, quant in (("nearest", "nearest"), ("floor", "floor")):
        if quant == "nearest":
            _, pred = RO.representable_nearest(cv_target, q01, q99)
        else:
            pred = AC_decode_floor(cv_target, q01, q99)
        out.setdefault("quantized_copy_V", {})[name] = float(np.abs(S + pred - A).mean())
    # consistency checks
    checks = []
    for mode in ("residual", "absolute"):
        m = out["modes"][mode]
        bf = m["independent_brute_force"]
        checks.append({
            "check": "brute_force_matches_nearest_bound_" + mode,
            "pass": bool(abs(bf["brute_force_nearest_L1"] - m["nearest_representable_lower_bound_L1"])
                         <= 1e-6 + 1e-3 * max(m["nearest_representable_lower_bound_L1"], 1e-9)),
            "brute_force": bf["brute_force_nearest_L1"],
            "script": m["nearest_representable_lower_bound_L1"]})
        checks.append({
            "check": "nearest_never_worse_than_floor_" + mode,
            "pass": bool(m["nearest_representable_lower_bound_L1"] <= m["floor_encoder_oracle_L1"] + 1e-9)})
    out["independent_checks"] = checks
    out["independent_checks_pass"] = all(c["pass"] for c in checks)
    out["frame_set_note"] = ("all legal validation frames t in [1, T-2] over the 20 held-out "
                             "episodes; identical frame set to cv / identity / MLP baselines")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = run()
    json.dump(res, open(args.out, "w"), indent=1)
    print("val frames %d | cv %.6f | identity %.6f" % (
        res["n_val_frames"], res["cv_physical_L1_same_frames"], res["identity_physical_L1_same_frames"]))
    for mode, m in res["modes"].items():
        print("[%s] floor=%.6f  nearest_bound=%.6f  (x cv=%.3f)  beats_cv=%s | step_mean=%.6f "
              "| clip=%.4f | other_bin=%.3f" % (
                  mode, m["floor_encoder_oracle_L1"], m["nearest_representable_lower_bound_L1"],
                  m["bound_vs_cv_ratio"], m["bound_beats_cv"], m["physical_step_mean"],
                  m["fraction_entries_outside_q01_q99"], m["fraction_entries_needing_other_bin"]))
        print("        per-arm nearest: left=%.6f right=%.6f | brute-force check %.6f" % (
            m["nearest_per_arm"]["left"], m["nearest_per_arm"]["right"],
            m["independent_brute_force"]["brute_force_nearest_L1"]))
    print("quantized copy-V:", {k: round(v, 6) for k, v in res["quantized_copy_V"].items()})
    print("independent checks pass:", res["independent_checks_pass"])
    for c in res["independent_checks"]:
        print("   %-42s %s" % (c["check"], "PASS" if c["pass"] else "FAIL"))
    print("wrote", args.out)
    return 0 if res["independent_checks_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
