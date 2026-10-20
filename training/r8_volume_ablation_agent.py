#!/usr/bin/env python3
"""Data-volume ablation for the R7/R8 state-dynamics question.

Question: is the 200-frame pilot really too little data to learn the velocity
correction R, or was that conclusion an artifact of the old 200-frame code?

Fixed conditions (the same in every arm)
  episodes   64 train / 16 dev / 20 holdout, rebuilt from split80.json by the
             same episode carving the audited pipeline uses
  features   [S, V, Acc] restricted to t >= 2 (42 dims, past-only)
  target     R = (A - S) - V
  model      the audited MLP (128, 128) with Adam lr 1e-3, batch 512,
             standardisation from the training rows only
  metric     physical L1 on the 20 holdout episodes, plus per-episode macro L1,
             velocity-correction R^2 / correlation / sign agreement and the count
             of holdout episodes that beat constant velocity

Datasets are sampled from the SAME 64 training episodes by stratified
round-robin over (episode, motion-phase) cells, so different sizes are not
different episode sets by accident.

Three comparisons are reported separately, because data volume and optimisation
budget cannot be varied at once:
  equal_updates  same number of optimiser steps
  equal_epochs   same number of passes over each dataset
  dev_best       each size gets its own epoch count chosen on DEV
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "diagnostics"))
import r6_data as D            # noqa: E402
import r7_metrics as M         # noqa: E402
import r7_state_mlp as T       # noqa: E402

MOTION_EDGES = (1e-3, 5e-3, 2e-2)


def build_blobs(episodes, need_accel=True):
    frames = [T.episode_frames(D.load_episode(name), need_accel=need_accel) for name in episodes]
    return T.stack_frames(frames)


def stratified_subsample(blob, n, seed):
    """Deterministic (episode, motion-phase) stratified round-robin sample."""
    if n <= 0 or n >= len(blob["S"]):
        return np.arange(len(blob["S"]))
    velocity = np.abs(blob["V"]).max(axis=1)
    phase = np.digitize(velocity, MOTION_EDGES)
    cells = {}
    for ep in np.unique(blob["episode"]):
        rows = np.flatnonzero(blob["episode"] == ep)
        for p in np.unique(phase[rows]):
            key = (str(ep), int(p))
            cells[key] = rows[phase[rows] == p]
    keys = sorted(cells)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    picked = []
    cursor = {k: 0 for k in keys}
    while len(picked) < n:
        progressed = False
        for oi in order:
            key = keys[oi]
            rows = cells[key]
            if cursor[key] >= len(rows):
                continue
            picked.append(int(rows[cursor[key]]))
            cursor[key] += 1
            progressed = True
            if len(picked) >= n:
                break
        if not progressed:
            break
    return np.array(sorted(picked), dtype=np.int64)


def holdout_report(cmd, gt, state, V, R_hat, R_gt, episodes):
    ev = M.evaluate(cmd, gt, state, V, episodes)
    ev["correction"] = M.evaluate_correction(R_hat, R_gt)
    cv = state + V
    per_ep, beating = {}, 0
    for ep in sorted(set(episodes.tolist())):
        m = episodes == ep
        l1 = float(np.abs(cmd[m] - gt[m]).mean())
        cv_l1 = float(np.abs(cv[m] - gt[m]).mean())
        per_ep[ep] = {"model": l1, "cv": cv_l1, "beats_cv": bool(l1 < cv_l1)}
        beating += int(l1 < cv_l1)
    ev["per_episode"] = per_ep
    ev["episodes_beating_cv"] = beating
    ev["episode_count"] = len(per_ep)
    ev["macro_L1"] = float(np.mean([v["model"] for v in per_ep.values()])) if per_ep else None
    # did the model actually predict a correction, or is it emitting ~0?
    ev["correction_scale"] = {
        "mean_abs_R_hat": float(np.abs(R_hat).mean()),
        "mean_abs_R_gt": float(np.abs(R_gt).mean()),
        "ratio": float(np.abs(R_hat).mean() / (np.abs(R_gt).mean() + 1e-12)),
        "frac_rows_near_zero_1e-4": float(np.mean(np.abs(R_hat).max(axis=1) < 1e-4)),
    }
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 2000, 0],
                    help="training frames per arm; 0 means all legal training frames")
    ap.add_argument("--modes", nargs="+", default=["equal_updates", "equal_epochs", "dev_best"],
                    choices=["equal_updates", "equal_epochs", "dev_best"])
    ap.add_argument("--updates", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--features", default="SVAcc", choices=["SVAcc", "SV"],
                    help="SVAcc = 42-dim [S,V,Acc]; SV = 28-dim [S,V] as used by the R7 pilot")
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--sampling", default="stratified", choices=["stratified", "uniform"],
                    help="how a size-N subset is drawn from the same 64 training episodes")
    ap.add_argument("--pilot-config", default="evidence/history_proprio_pilot200.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-dir", default="")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    split = json.load(open("/home/dgut/pusht_runtime/split80.json"))
    train_eps, dev_eps = T.split_episodes(split["train"])
    holdout_eps = split["val"]
    overlap = (set(train_eps) & set(dev_eps)) | (set(train_eps) & set(holdout_eps)) | (
        set(dev_eps) & set(holdout_eps))
    if overlap:
        raise AssertionError("episode leakage between train/dev/holdout: %s" % sorted(overlap))

    print("[data] train=%d dev=%d holdout=%d episodes" % (len(train_eps), len(dev_eps), len(holdout_eps)),
          flush=True)
    train_blob = build_blobs(train_eps)
    dev_blob = build_blobs(dev_eps)
    val_blob = build_blobs(holdout_eps)
    print("[data] frames train=%d dev=%d holdout=%d | feature dims=%d"
          % (len(train_blob["S"]), len(dev_blob["S"]), len(val_blob["S"]),
             2 * 14 + 14), flush=True)

    Xdev, Rdev = T.build_xy(dev_blob, args.features)
    Xva, _ = T.build_xy(val_blob, args.features)
    print("[data] features=%s dims=%d" % (args.features, Xdev.shape[1]), flush=True)

    pred_dir = args.pred_dir or os.path.dirname(os.path.abspath(args.out))
    os.makedirs(pred_dir, exist_ok=True)

    result = {
        "config": {k: v for k, v in vars(args).items()},
        "splits": {"train_episodes": train_eps, "dev_episodes": dev_eps, "holdout_episodes": holdout_eps},
        "n_frames": {"train": int(len(train_blob["S"])), "dev": int(len(dev_blob["S"])),
                     "holdout": int(len(val_blob["S"]))},
        "arms": {},
    }

    for size in args.sizes:
        label = "all" if size <= 0 else str(size)
        if size <= 0:
            base_index = np.arange(len(train_blob["S"]))
        for mode in args.modes:
            for seed in args.seeds:
                t0 = time.time()
                if size <= 0:
                    index = np.arange(len(train_blob["S"]))
                elif args.sampling == "uniform":
                    rng = np.random.default_rng(seed)
                    index = np.sort(rng.choice(len(train_blob["S"]), size=size, replace=False))
                else:
                    index = stratified_subsample(train_blob, size, seed)
                sub = {k: v[index] for k, v in train_blob.items()}
                Xtr, Rtr = T.build_xy(sub, args.features)
                steps_per_epoch = max(1, math.ceil(len(Xtr) / args.batch))
                if mode == "equal_updates":
                    epochs = max(1, math.ceil(args.updates / steps_per_epoch))
                    select_best, patience = False, 10 ** 9
                elif mode == "equal_epochs":
                    epochs = args.epochs
                    select_best, patience = False, 10 ** 9
                else:
                    epochs = args.max_epochs
                    select_best, patience = True, args.patience

                net, stats = T.train_mlp(
                    Xtr, Rtr, Xdev, Rdev, seed, tuple(args.hidden), epochs, args.lr,
                    min(args.batch, len(Xtr)), patience, torch=torch, select_best=select_best,
                )
                out_hat = T.predict(net, Xva, stats, torch).astype(np.float32)
                cmd, R_hat, R_gt = T.reconstruct_prediction(out_hat, val_blob, "R")
                ev = holdout_report(cmd, val_blob["A"], val_blob["S"], val_blob["V"],
                                    R_hat, R_gt, val_blob["episode"])
                ev.update({
                    "size_label": label,
                    "n_train_frames": int(len(Xtr)),
                    "episodes_covered": int(len(set(sub["episode"].tolist()))),
                    "mode": mode,
                    "seed": int(seed),
                    "epochs_requested": int(epochs),
                    "epochs_executed": int(len(stats["history"])),
                    "optimiser_steps_approx": int(len(stats["history"]) * steps_per_epoch),
                    "steps_per_epoch": int(steps_per_epoch),
                    "checkpoint_selection": stats["checkpoint_selection"],
                    "best_epoch": int(stats["best_epoch"]),
                    "dev_best_L1": float(stats["dev_best_L1"]),
                    "dev_curve": [{"epoch": h["epoch"], "train_L1": h["train_L1"],
                                   "dev_L1": h["dev_L1"]} for h in stats["history"]][::max(1, len(stats["history"]) // 20)],
                    "seconds": round(time.time() - t0, 2),
                })
                npz_path = os.path.join(pred_dir, "pred_abl_%s_%s_seed%d.npz" % (label, mode, seed))
                np.savez_compressed(
                    npz_path, R_hat=R_hat, R_gt=R_gt, cmd=cmd, gt=val_blob["A"],
                    state=val_blob["S"], V=val_blob["V"], episode=val_blob["episode"],
                    frame=val_blob["frame"],
                )
                ev["prediction_file"] = npz_path
                result["arms"].setdefault(label, {}).setdefault(mode, []).append(ev)
                print("[%s/%s seed=%d] frames=%d eps=%d epochs=%d steps~%d | L1=%.6f cv=%.6f "
                      "macro=%.6f R2=%s sign=%s beat_cv=%d/%d Rscale=%.3f"
                      % (label, mode, seed, len(Xtr), len(set(sub["episode"].tolist())),
                         len(stats["history"]), len(stats["history"]) * steps_per_epoch,
                         ev["L1"], ev["constant_velocity_L1"], ev["macro_L1"],
                         None if ev["correction"]["R2_variance_explained"] is None
                         else round(ev["correction"]["R2_variance_explained"], 4),
                         None if ev["direction"]["sign_agreement_at_1e-3"] is None
                         else round(ev["direction"]["sign_agreement_at_1e-3"], 4),
                         ev["episodes_beating_cv"], ev["episode_count"],
                         ev["correction_scale"]["ratio"]), flush=True)
                with open(args.out, "w") as fh:
                    json.dump(result, fh, indent=1)

    # summary across seeds
    summary = {}
    for label, modes in result["arms"].items():
        for mode, runs in modes.items():
            summary.setdefault(label, {})[mode] = {
                "n_seeds": len(runs),
                "mean_L1": float(np.mean([r["L1"] for r in runs])),
                "std_L1": float(np.std([r["L1"] for r in runs])),
                "mean_macro_L1": float(np.mean([r["macro_L1"] for r in runs])),
                "mean_R2": float(np.nanmean([
                    r["correction"]["R2_variance_explained"]
                    if r["correction"]["R2_variance_explained"] is not None else np.nan
                    for r in runs])),
                "mean_sign": float(np.nanmean([
                    r["direction"]["sign_agreement_at_1e-3"]
                    if r["direction"]["sign_agreement_at_1e-3"] is not None else np.nan
                    for r in runs])),
                "mean_episodes_beating_cv": float(np.mean([r["episodes_beating_cv"] for r in runs])),
                "mean_correction_ratio": float(np.mean([r["correction_scale"]["ratio"] for r in runs])),
                "mean_train_frames": float(np.mean([r["n_train_frames"] for r in runs])),
                "mean_episodes_covered": float(np.mean([r["episodes_covered"] for r in runs])),
                "mean_steps": float(np.mean([r["optimiser_steps_approx"] for r in runs])),
            }
    result["summary"] = summary
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)

    print("\n=== summary (mean over seeds) ===")
    for label in sorted(summary, key=lambda x: (x == "all", x)):
        for mode in ("equal_updates", "equal_epochs", "dev_best"):
            if mode not in summary[label]:
                continue
            s = summary[label][mode]
            print("  %-5s %-14s frames=%6.0f eps=%4.1f steps=%6.0f | L1=%.6f macro=%.6f "
                  "R2=%+.3f sign=%.3f beat_cv=%.1f Rscale=%.3f"
                  % (label, mode, s["mean_train_frames"], s["mean_episodes_covered"],
                     s["mean_steps"], s["mean_L1"], s["mean_macro_L1"], s["mean_R2"],
                     s["mean_sign"], s["mean_episodes_beating_cv"], s["mean_correction_ratio"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
