#!/usr/bin/env python3
"""R7 phase 3: what the lightweight state model actually learned (CPU, no model weights).

Answers, from data:

  Q1 can a small MLP beat copy-V, and by how much (paired per-episode)?
  Q2 is the velocity CORRECTION predictable, and does past acceleration help?
  Q3 which motion magnitudes are predictable?
  Q4 left vs right arm
  Q6 how much of the correction is NOT determined by proprio -> is vision the next lever?

The kNN block estimates the irreducible L1 given (S,V) non-parametrically: it predicts
each held-out frame's R by the mean R of its k nearest TRAIN frames (different episodes
by construction). If the MLP matches that estimate, the proprio input is exhausted for
this target -- which is an empirical bound for these inputs, NOT a mathematical one.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import r6_data as D  # noqa: E402

LEFT, RIGHT = slice(0, 7), slice(7, 14)
MOTION_BINS = (0.0, 1e-3, 5e-3, 2e-2, np.inf)
MOTION_LABELS = ("tiny_lt_1e-3", "small_1e-3_5e-3", "medium_5e-3_2e-2", "large_ge_2e-2")


def _corr(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def load_frames(names, need_accel=True):
    out = []
    for n in names:
        ep = D.load_episode(n)
        t = D.valid_frames(len(ep["state"]))
        if need_accel:
            t = t[t >= 2]
        out.append({"S": ep["state"][t], "A": ep["action"][t],
                    "V": ep["state"][t] - ep["state"][t - 1],
                    "Acc": (ep["state"][t] - ep["state"][t - 1])
                           - (ep["state"][t - 1] - ep["state"][t - 2]),
                    "episode": ep["episode"]})
    return {k: (np.concatenate([f[k] for f in out]) if k != "episode"
                else np.concatenate([[f["episode"]] * len(f["S"]) for f in out]))
            for k in out[0]}


def paired_stats(per_episode, base_key):
    """Per-episode paired L1 differences, sign test and a bootstrap CI over episodes."""
    diffs = np.array([v["model"] - v[base_key] for v in per_episode.values()])
    eps = list(per_episode.keys())
    rng = np.random.default_rng(0)
    boots = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(4000)]
    return {
        "n_episodes": len(diffs),
        "mean_diff": float(diffs.mean()),
        "median_diff": float(np.median(diffs)),
        "episodes_improved": int((diffs < 0).sum()),
        "episodes_worse": int((diffs > 0).sum()),
        "bootstrap_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "bootstrap_prob_improvement": float(np.mean(np.array(boots) < 0)),
        "per_episode_diff": {e: float(d) for e, d in zip(eps, diffs)},
    }


def knn_aleatoric(Xtr, Rtr, Xva, Rva, k=32, chunk=64):
    """Mean-R-of-nearest-TRAIN-frames predictor: an empirical bound given the inputs."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr = ((Xtr - mu) / sd).astype(np.float32)
    Zva = ((Xva - mu) / sd).astype(np.float32)
    pred = np.empty_like(Rva)
    nn1 = np.empty_like(Rva)
    for i in range(0, len(Zva), chunk):
        block = Zva[i:i + chunk]
        d2 = ((block[:, None, :] - Ztr[None, :, :]) ** 2).sum(axis=2)
        idx = np.argpartition(d2, k, axis=1)[:, :k]
        pred[i:i + chunk] = Rtr[idx].mean(axis=1)
        nearest = idx[np.arange(len(block)), np.argmin(np.take_along_axis(
            d2, idx, axis=1), axis=1)]
        nn1[i:i + chunk] = Rtr[nearest]
    return {"knn_mean_R_L1": float(np.abs(pred - Rva).mean()),
            "nearest_neighbour_R_L1": float(np.abs(nn1 - Rva).mean()),
            "k": int(k)}


def run(pred_dir, mlp_json, out_path):
    split = D.load_split()
    tr = load_frames(sorted(split["train"]))
    va = load_frames(split["val"])
    R_tr = (tr["A"] - tr["S"]) - tr["V"]
    R_va = (va["A"] - va["S"]) - va["V"]
    res = {"n_train_frames": int(len(R_tr)), "n_holdout_frames": int(len(R_va)),
           "note": "acceleration-defined frames (t>=2) so every comparison uses one frame set"}

    # ---- Q2: structure of the velocity correction R_t = V_(t+1) - V_t
    res["correction_structure"] = {
        "corr_R_with_past_acceleration": _corr(R_va, va["Acc"]),
        "corr_R_with_negative_velocity": _corr(R_va, -va["V"]),
        "corr_R_with_velocity": _corr(R_va, va["V"]),
        "corr_past_acceleration_with_current_acceleration": _corr(va["Acc"], R_va),
        "mean_abs_R": float(np.abs(R_va).mean()),
        "mean_abs_V": float(np.abs(va["V"]).mean()),
        "ratio_R_over_V": float(np.abs(R_va).mean() / (np.abs(va["V"]).mean() + 1e-12)),
        "corr_R_per_arm": {"left": _corr(R_va[:, LEFT], va["Acc"][:, LEFT]),
                           "right": _corr(R_va[:, RIGHT], va["Acc"][:, RIGHT])},
        "velocity_autocorrelation": {
            "corr_V_t_V_t_minus_1": _corr(va["V"], tr["V"][:len(va["V"])]) if False else None,
            "per_arm": None},
    }
    # velocity / acceleration autocorrelation measured inside the holdout episodes
    vac, aac = [], []
    for name in split["val"]:
        ep = D.load_episode(name)
        S = ep["state"]
        V = S[1:-1] - S[:-2]
        if len(V) < 4:
            continue
        vac.append(_corr(V[1:], V[:-1]))
        A = V[1:] - V[:-1]
        aac.append(_corr(A[1:], A[:-1]))
    res["correction_structure"]["velocity_autocorrelation"] = {
        "per_episode_mean": float(np.nanmean(vac)),
        "min": float(np.nanmin(vac)), "max": float(np.nanmax(vac))}
    res["correction_structure"]["acceleration_autocorrelation"] = {
        "per_episode_mean": float(np.nanmean(aac)),
        "min": float(np.nanmin(aac)), "max": float(np.nanmax(aac)),
        "interpretation": "how much of tomorrow's acceleration is already visible in today's"}

    # ---- Q1 paired comparison from the training JSON (holdout, per episode)
    mlp = json.load(open(mlp_json))
    res["paired_vs_constant_velocity"] = {}
    for key in ("SVAcc_R", "SV_R", "V_R", "SV_delta"):
        if key not in mlp["runs"]:
            continue
        seed0 = mlp["runs"][key]["per_seed"][0]
        res["paired_vs_constant_velocity"][key] = {
            "L1_mean_over_seeds": mlp["runs"][key]["aggregate"]["L1"]["mean"],
            "vs_cv": paired_stats(seed0["per_episode"], "constant_velocity"),
            "vs_identity": paired_stats(seed0["per_episode"], "identity"),
        }

    # ---- Q3/Q4 from the stored predictions (seed 0 of the SV/R model)
    files = [f for f in sorted(os.listdir(pred_dir)) if f.endswith(".npz") and f.startswith("pred_")]
    preferred = [f for f in files if f.startswith("pred_SVAcc_R_")] or                 [f for f in files if f.startswith("pred_SV_R_")]
    if preferred:
        name = preferred[0]
        res["prediction_file_used"] = name
        z = np.load(os.path.join(pred_dir, name))
        R_hat, cmd, gt, S = z["R_hat"], z["cmd"], z["gt"], z["state"]
        cv = z["state"] + z["V"]
        err, err_cv = np.abs(cmd - gt), np.abs(cv - gt)
        motion = np.abs(gt - S).mean(axis=1)
        strata = {}
        for lo, hi, lab in zip(MOTION_BINS[:-1], MOTION_BINS[1:], MOTION_LABELS):
            m = (motion >= lo) & (motion < hi)
            if not m.any():
                strata[lab] = {"n": 0}
                continue
            strata[lab] = {
                "n": int(m.sum()), "share": float(m.mean()),
                "model_L1": float(err[m].mean()), "cv_L1": float(err_cv[m].mean()),
                "improvement": float(err_cv[m].mean() - err[m].mean()),
                "relative_improvement": float(1 - err[m].mean() / (err_cv[m].mean() + 1e-12)),
                "R_corr": _corr((R_hat - (gt - S) + (gt - S))[m] * 0 + R_hat[m], R_va[m])}
        res["motion_strata"] = strata
        res["per_arm"] = {}
        for arm, sl in (("left", LEFT), ("right", RIGHT)):
            res["per_arm"][arm] = {
                "model_L1": float(err[:, sl].mean()), "cv_L1": float(err_cv[:, sl].mean()),
                "identity_L1": float(np.abs(S - gt)[:, sl].mean()),
                "relative_improvement_vs_cv": float(1 - err[:, sl].mean() / (err_cv[:, sl].mean() + 1e-12)),
                "R_corr": _corr(R_hat[:, sl], R_va[:, sl]),
                "R_sign_at_1e-4": float(np.mean(np.sign(R_hat[:, sl])[np.abs(R_va[:, sl]) >= 1e-4]
                                                == np.sign(R_va[:, sl])[np.abs(R_va[:, sl]) >= 1e-4]))}
        # ---- Q6 residual structure: how much is NOT determined by (S,V)
        R_gt_az = (gt - S) - z["V"]
        eps_ids = z["episode"]
        resid = R_gt_az - R_hat
        tot_var = float(resid.var())
        per_ep_mean = np.stack([resid[eps_ids == e].mean(axis=0) for e in sorted(set(eps_ids.tolist()))])
        ep_of_row = np.searchsorted(sorted(set(eps_ids.tolist())),
                                    eps_ids)
        centred = resid - per_ep_mean[ep_of_row]
        res["residual_structure"] = {
            "residual_L1": float(np.abs(resid).mean()),
            "residual_variance": tot_var,
            "episode_offset_variance_explained": float(1.0 - centred.var() / (tot_var + 1e-30)),
            "episode_offset_is_an_oracle_upper_bound": True,
            "interpretation": ("a large episode-offset share means the remaining error is "
                               "configuration-specific (object pose / contact), i.e. exactly "
                               "what an image-conditioned model could resolve -- an upper "
                               "bound, because the offsets are fitted on the holdout itself"),
        }
        # ---- Q6 non-parametric bound given (S,V)
        Xtr = np.concatenate([tr["S"], tr["V"]], axis=1)
        res["proprio_knn_bound"] = knn_aleatoric(Xtr, R_tr, np.concatenate([S, z["V"]], axis=1), R_gt_az)
        res["proprio_knn_bound"]["mlp_R_L1_for_reference"] = float(np.abs(R_hat - R_gt_az).mean())
    json.dump(res, open(out_path, "w"), indent=1)

    print("correction structure: corr(R, accel)=%.3f  corr(R, -V)=%.3f  |R|/|V|=%.3f" % (
        res["correction_structure"]["corr_R_with_past_acceleration"],
        res["correction_structure"]["corr_R_with_negative_velocity"],
        res["correction_structure"]["ratio_R_over_V"]))
    print("acceleration autocorrelation (holdout, per-episode mean)=%.3f" % (
        res["correction_structure"]["acceleration_autocorrelation"]["per_episode_mean"]))
    for k, v in res["paired_vs_constant_velocity"].items():
        p = v["vs_cv"]
        print("[%s] L1=%.6f  vs cv: mean diff=%+.6f, improved %d/%d episodes, "
              "bootstrap P(improve)=%.3f CI[%+.6f,%+.6f]" % (
                  k, v["L1_mean_over_seeds"], p["mean_diff"], p["episodes_improved"],
                  p["n_episodes"], p["bootstrap_prob_improvement"], p["bootstrap_ci95"][0],
                  p["bootstrap_ci95"][1]))
    for lab, s in res.get("motion_strata", {}).items():
        if s.get("n"):
            print("  stratum %-18s n=%5d model=%.6f cv=%.6f rel_improvement=%+.1f%%" % (
                lab, s["n"], s["model_L1"], s["cv_L1"], 100 * s["relative_improvement"]))
    if "proprio_knn_bound" in res:
        b = res["proprio_knn_bound"]
        print("proprio kNN bound: knn_mean_R L1=%.6f  nearest-neighbour L1=%.6f  (MLP R L1=%.6f)" % (
            b["knn_mean_R_L1"], b["nearest_neighbour_R_L1"], b["mlp_R_L1_for_reference"]))
        print("residual: episode-offset variance explained (oracle upper bound)=%.3f" % (
            res["residual_structure"]["episode_offset_variance_explained"]))
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--mlp-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.pred_dir, args.mlp_json, args.out)


if __name__ == "__main__":
    main()
