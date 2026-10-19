#!/usr/bin/env python3
"""R6: offline token-level calibration analysis (no GPU, no model).

Rebuilds every pilot prediction from the *generated token rows* stored in
metrics.jsonl, re-derives the expert delta from the HDF5, and separates the two
things a physical L1 mixes:

  * DIRECTION  — does the predicted displacement point the right way?  (cosine, sign)
  * MAGNITUDE  — is the predicted displacement the right size?         (|d_pred|/|d_gt|)

It then asks the question the pilot could not answer: **is the whole L1 gap between
the two arms reproducible with one scalar on the control arm?** If a single scale
factor fitted on TRAIN frames brings the no-history arm to the history arm's L1, the
measured A/B gain is magnitude calibration, not learned motion direction.

The reconstruction is exact: the val L1 recomputed here is asserted against the
value the training run logged.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for sub in ("diagnostics", "training"):
    sys.path.insert(0, os.path.join(REPO, sub))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))
import r6_data as D          # noqa: E402
import action_codec as AC    # noqa: E402

ALPHAS = np.round(np.arange(0.0, 2.0001, 0.02), 4)
SIGN_THRESHOLD = 1e-3


def episode_cache(pairs, hdf5_dir):
    need = sorted({e for e, _ in pairs})
    eps = {e: D.load_episode(e, hdf5_dir) for e in need}
    return eps


def deltas_from_tokens(tokens, q01, q99):
    idx = np.asarray(tokens, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 14:
        raise ValueError("token rows must be (N,14)")
    return AC.denormalize(AC.decode_indices(idx, mode=AC.MODE_FIXED), q01, q99)


def expert_deltas(pairs, eps):
    s = np.stack([eps[e]["state"][t] for e, t in pairs])
    a = np.stack([eps[e]["action"][t] for e, t in pairs])
    return s, a, (a - s).astype(np.float64)


def metrics_for(pred_delta, gt_delta):
    """Direction and magnitude descriptors at scale 1."""
    pd = np.asarray(pred_delta, np.float64)
    gd = np.asarray(gt_delta, np.float64)
    num = (pd * gd).sum(axis=1)
    den = np.linalg.norm(pd, axis=1) * np.linalg.norm(gd, axis=1)
    cos = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    m = np.abs(gd) >= SIGN_THRESHOLD
    sign = float(np.mean(np.sign(pd[m]) == np.sign(gd[m]))) if m.any() else None
    corr = None
    if pd.std() > 0 and gd.std() > 0:
        corr = float(np.corrcoef(pd.reshape(-1), gd.reshape(-1))[0, 1])
    return {
        "L1": float(np.abs(pd - gd).mean()),
        "per_dim_L1": np.abs(pd - gd).mean(axis=0).tolist(),
        "magnitude_ratio": float(np.abs(pd).mean() / (np.abs(gd).mean() + 1e-12)),
        "cosine_mean": float(np.nanmean(cos)) if np.isfinite(cos).any() else None,
        "sign_agreement_at_1e-3": sign,
        "sign_n": int(m.sum()),
        "corr_pooled": corr,
    }


def alpha_curve(pred_delta, gt_delta, alphas=ALPHAS):
    return [(float(a), float(np.abs(a * pred_delta - gt_delta).mean())) for a in alphas]


def best_alpha(curve):
    a, l1 = min(curve, key=lambda kv: kv[1])
    return a, l1


def magnitude_oracle(pred_delta, gt_delta):
    """Keep each sample's predicted direction, replace its norm by the expert's norm."""
    pd = np.asarray(pred_delta, np.float64)
    gd = np.asarray(gt_delta, np.float64)
    pn = np.linalg.norm(pd, axis=1, keepdims=True)
    gn = np.linalg.norm(gd, axis=1, keepdims=True)
    scaled = np.where(pn > 0, pd / np.maximum(pn, 1e-12) * gn, pd)
    return float(np.abs(scaled - gd).mean())


def load_run(run_dir, hdf5_dir):
    cfg = json.load(open(os.path.join(run_dir, "_snapshot", "config.json")))
    q01 = np.asarray(cfg["q01"], np.float32)
    q99 = np.asarray(cfg["q99"], np.float32)
    val_pairs = [tuple(x) for x in cfg["val_samples"]]
    evals = {}
    for line in open(os.path.join(run_dir, "metrics.jsonl")):
        rec = json.loads(line)
        if rec.get("stage") == "eval":
            evals.setdefault(rec["split"], []).append(rec)
    if "val" not in evals:
        raise SystemExit("no val evaluation in " + run_dir)
    last_val = evals["val"][-1]
    last_train = evals["train"][-1] if "train" in evals else None
    # the train evaluation may cover any number of the sampled frames: follow the token rows
    n_train = len(last_train["gen_tokens_per_sample"]) if last_train is not None else 0
    train_pairs = [tuple(x) for x in cfg["samples"][:n_train]] if n_train else []
    if n_train and n_train != len(train_pairs):
        raise SystemExit("train token rows exceed the sampled frame list in " + run_dir)
    eps = episode_cache(list(val_pairs) + list(train_pairs), hdf5_dir)
    S, A, gd = expert_deltas(val_pairs, eps)
    pd = deltas_from_tokens(last_val["gen_tokens_per_sample"], q01, q99).astype(np.float64)
    out = {"run": os.path.abspath(run_dir),
           "arm": last_val.get("arm", cfg.get("history_mode", "legacy14d")),
           "step": last_val["step"],
           "n_val": len(val_pairs), "proprio_history_mode": cfg.get("history_mode", "legacy14d"),
           "steps_budget": cfg.get("steps"),
           "reconstructed_val_L1": float(np.abs(pd - gd).mean()),
           "logged_val_L1": float(last_val["gen_L1"]),
           "reconstruction_matches_logged": bool(
               abs(float(np.abs(pd - gd).mean()) - float(last_val["gen_L1"])) < 1e-6),
           "at_scale_1": metrics_for(pd, gd),
           "alpha_curve_val": alpha_curve(pd, gd),
           "magnitude_oracle_L1": magnitude_oracle(pd, gd)}
    # TRAIN-fitted scalar (honest protocol: never fit on validation frames)
    if last_train is not None:
        _, _, gdt = expert_deltas(train_pairs, eps)
        pdt = deltas_from_tokens(last_train["gen_tokens_per_sample"], q01, q99).astype(np.float64)
        a_tr, _ = best_alpha(alpha_curve(pdt, gdt))
        out["alpha_fitted_on_train"] = a_tr
        out["val_L1_at_train_alpha"] = float(np.abs(a_tr * pd - gd).mean())
        out["train_L1_at_scale1"] = float(np.abs(pdt - gdt).mean())
        out["train_best_alpha"] = a_tr
    a_val, l1_val = best_alpha(out["alpha_curve_val"])
    out["alpha_best_on_val_descriptive"] = a_val
    out["val_L1_at_best_alpha"] = l1_val
    out["per_dim_L1_at_best_alpha"] = np.abs(a_val * pd - gd).mean(axis=0).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=path")
    ap.add_argument("--hdf5", default=os.environ.get("PUSHT_HDF5", D.HDF5_DIR))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        runs[label] = load_run(path, args.hdf5)

    by_arm = {}
    for label, r in runs.items():
        by_arm.setdefault(r["arm"], []).append(r)
    summary = {"runs": runs, "by_arm": {}}
    for arm, rs in by_arm.items():
        summary["by_arm"][arm] = {
            "n": len(rs),
            "mean_val_L1_scale1": float(np.mean([x["at_scale_1"]["L1"] for x in rs])),
            "mean_val_L1_at_train_alpha": float(np.mean([x["val_L1_at_train_alpha"] for x in rs])),
            "mean_alpha_train": float(np.mean([x["alpha_fitted_on_train"] for x in rs])),
            "mean_magnitude_ratio": float(np.mean([x["at_scale_1"]["magnitude_ratio"] for x in rs])),
            "mean_cosine": float(np.mean([x["at_scale_1"]["cosine_mean"] for x in rs])),
            "mean_corr": float(np.mean([x["at_scale_1"]["corr_pooled"] for x in rs])),
            "mean_sign": float(np.mean([x["at_scale_1"]["sign_agreement_at_1e-3"] for x in rs])),
            "mean_magnitude_oracle_L1": float(np.mean([x["magnitude_oracle_L1"] for x in rs])),
        }
    # the decisive comparison: does ONE train-fitted scalar close the arm gap?
    if "duplicate" in summary["by_arm"] and "true" in summary["by_arm"]:
        dup, tru = summary["by_arm"]["duplicate"], summary["by_arm"]["true"]
        gap = dup["mean_val_L1_scale1"] - tru["mean_val_L1_scale1"]
        closed = dup["mean_val_L1_at_train_alpha"] - tru["mean_val_L1_scale1"]
        summary["arm_gap_analysis"] = {
            "gap_at_scale1": float(gap),
            "gap_after_one_train_fitted_scalar_on_control": float(closed),
            "fraction_of_gap_explained_by_scalar": float(1.0 - closed / gap) if gap > 0 else None,
            "verdict": ("scalar magnitude calibration alone reproduces the A/B gap"
                        if closed <= 0.1 * max(gap, 1e-12) else
                        "direction information also contributes to the A/B gap"),
        }
    json.dump(summary, open(args.out, "w"), indent=1)

    print("%-10s %-10s %8s %8s %8s %7s %7s %8s" % (
        "label", "arm", "L1@1", "alpha*", "L1@a_tr", "magR", "cos", "sign"))
    for label, r in runs.items():
        print("%-10s %-10s %8.5f %8.2f %8.5f %7.2f %7.3f %8.3f" % (
            label, r["arm"], r["at_scale_1"]["L1"], r["alpha_fitted_on_train"],
            r["val_L1_at_train_alpha"], r["at_scale_1"]["magnitude_ratio"],
            r["at_scale_1"]["cosine_mean"] or float("nan"),
            r["at_scale_1"]["sign_agreement_at_1e-3"] or float("nan")))
    for arm, b in summary["by_arm"].items():
        print("[%s] n=%d mean L1@1=%.5f  L1@train_alpha=%.5f (alpha=%.2f)  mag_ratio=%.2f "
              "cos=%.3f corr=%.3f sign=%.3f  magnitude_oracle_L1=%.5f" % (
                  arm, b["n"], b["mean_val_L1_scale1"], b["mean_val_L1_at_train_alpha"],
                  b["mean_alpha_train"], b["mean_magnitude_ratio"], b["mean_cosine"],
                  b["mean_corr"], b["mean_sign"], b["mean_magnitude_oracle_L1"]))
    if "arm_gap_analysis" in summary:
        g = summary["arm_gap_analysis"]
        print("gap=%.5f  after_scalar_on_control=%.5f  explained=%.0f%%  -> %s" % (
            g["gap_at_scale1"], g["gap_after_one_train_fitted_scalar_on_control"],
            100.0 * (g["fraction_of_gap_explained_by_scalar"] or 0.0), g["verdict"]))
    bad = [l for l, r in runs.items() if not r["reconstruction_matches_logged"]]
    if bad:
        print("WARNING: reconstruction mismatch for", bad)
    print("wrote", args.out)
    return 0 if not bad else 3


if __name__ == "__main__":
    sys.exit(main())
