"""R6 Phase 1 / Task B: is past motion V_t = S_t - S_(t-1) informative about Delta_t = A_t - S_t?

Pure numpy, CPU only. Reports correlation, sign agreement at several motion
thresholds (with the number of valid samples per threshold), magnitude ratio,
per-arm / per-dimension / per-episode breakdowns, speed autocorrelation, and the
best TRAIN-fitted linear map from past motion to expert delta (a history-only
ceiling). Dims with zero variance are excluded from correlation, never treated as 0.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import r6_data as D  # noqa: E402

THRESHOLDS = (0.0, 1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)


def _corr(a, b):
    """Pearson r, or None when either side has zero variance (undefined, not 0)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 3 or a.std() == 0.0 or b.std() == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def episode_series(state, action):
    """Past motion V, expert delta D and the true future motion (alignment proof only)."""
    S, A = D.validate_arrays(state, action)
    t = D.valid_frames(len(S))
    return {"t": t, "V": S[t] - S[t - 1], "D": A[t] - S[t],
            "V_next": S[t + 1] - S[t], "state": S[t],
            "prev_state": S[t - 1], "action": A[t]}


def per_dim_corr(V, D):
    return [_corr(V[:, d], D[:, d]) for d in range(D.shape[1])]


def sign_block(V, D, thresholds=THRESHOLDS):
    """Sign agreement of (V, D) per motion threshold, with valid sample counts."""
    out = []
    for thr in thresholds:
        for basis, series in (("expert_delta", np.abs(D)), ("past_motion", np.abs(V))):
            m = series >= thr
            n = int(m.sum())
            if n == 0:
                out.append({"threshold": thr, "basis": basis, "n": 0})
                continue
            agree = float(np.mean(np.sign(V[m]) == np.sign(D[m])))
            out.append({"threshold": thr, "basis": basis, "n": n,
                        "fraction_of_entries": float(n / series.size),
                        "sign_agreement": agree})
    return out


def fit_linear(V, D):
    """Per-dimension OLS D ~ a*V + b (TRAIN only). Zero-variance dims fall back to the mean."""
    a = np.zeros(D.shape[1]); b = np.zeros(D.shape[1])
    for d in range(D.shape[1]):
        x = np.asarray(V[:, d], np.float64)
        var = x.var()
        if var <= 0 or not np.isfinite(var):
            a[d] = 0.0
            b[d] = float(D[:, d].mean())
            continue
        a[d] = float(np.cov(x, np.asarray(D[:, d], np.float64), bias=True)[0, 1] / var)
        b[d] = float(D[:, d].mean() - a[d] * x.mean())
    return {"slope": a, "intercept": b}


def _arm_corr(V, D, sl, lens=7):
    """Mean of the per-dimension correlations inside one 7-dim arm block."""
    cs = [c for c in per_dim_corr(V[:, sl], D[:, sl]) if c is not None]
    return float(np.mean(cs)) if cs else None


def run():
    split = D.load_split()
    res = {"dim_names": D.DIM_NAMES, "thresholds": list(THRESHOLDS), "splits": {}, "episodes": {}}
    pooled = {}
    for name in ("train", "val"):
        V_all, D_all, Vn_all, S_all, ep_ids = [], [], [], [], []
        per_ep = {}
        for ep in D.load_episodes(split[name]):
            s = episode_series(ep["state"], ep["action"])
            V_all.append(s["V"]); D_all.append(s["D"]); Vn_all.append(s["V_next"])
            S_all.append(s["state"]); ep_ids += [ep["episode"]] * len(s["V"])
            per_ep[ep["episode"]] = {
                "corr_left": _arm_corr(s["V"], s["D"], D.LEFT),
                "corr_right": _arm_corr(s["V"], s["D"], D.RIGHT),
                "mean_abs_delta_left": float(np.abs(s["D"][:, D.LEFT]).mean()),
                "mean_abs_delta_right": float(np.abs(s["D"][:, D.RIGHT]).mean()),
                "mean_abs_V_left": float(np.abs(s["V"][:, D.LEFT]).mean()),
                "mean_abs_V_right": float(np.abs(s["V"][:, D.RIGHT]).mean()),
                "n_frames": int(len(s["V"])),
            }
        V = np.concatenate(V_all); Dm = np.concatenate(D_all); Vn = np.concatenate(Vn_all)
        S = np.concatenate(S_all)
        pooled[name] = {"V": V, "D": Dm, "V_next": Vn, "S": S, "ep": ep_ids}
        res["episodes"][name] = per_ep
    # TRAIN-fitted linear history model, evaluated on validation frames.
    fit = fit_linear(pooled["train"]["V"], pooled["train"]["D"])
    for name in ("train", "val"):
        V, Dm, Vn = pooled[name]["V"], pooled[name]["D"], pooled[name]["V_next"]
        pc = per_dim_corr(V, Dm)
        lin = fit["slope"][None, :] * V + fit["intercept"][None, :]
        ep_corrs = res["episodes"][name]
        res["splits"][name] = {
            "n_frames": int(len(V)),
            "n_episodes": len(split[name]),
            "corr_per_dim": pc,
            "corr_defined_dims": int(sum(c is not None for c in pc)),
            "corr_zero_variance_dims": [D.DIM_NAMES[d] for d, c in enumerate(pc) if c is None],
            "corr_mean_over_dims": float(np.mean([c for c in pc if c is not None])),
            "corr_arm": {"left": _arm_corr(V, Dm, D.LEFT), "right": _arm_corr(V, Dm, D.RIGHT)},
            "corr_pooled_flattened": _corr(V, Dm),
            "corr_pooled_flattened_left": _corr(V[:, D.LEFT], Dm[:, D.LEFT]),
            "corr_pooled_flattened_right": _corr(V[:, D.RIGHT], Dm[:, D.RIGHT]),
            "r2_per_dim": [None if c is None else c * c for c in pc],
            "sign": sign_block(V, Dm),
            "magnitude": {
                "mean_abs_delta": float(np.abs(Dm).mean()),
                "mean_abs_past_motion": float(np.abs(V).mean()),
                "ratio_past_motion_to_delta": float(np.abs(V).mean() / (np.abs(Dm).mean() + 1e-12)),
                "per_arm": {
                    "left": {"mean_abs_delta": float(np.abs(Dm[:, D.LEFT]).mean()),
                             "mean_abs_past_motion": float(np.abs(V[:, D.LEFT]).mean())},
                    "right": {"mean_abs_delta": float(np.abs(Dm[:, D.RIGHT]).mean()),
                              "mean_abs_past_motion": float(np.abs(V[:, D.RIGHT]).mean())}},
                "mean_signed_delta_per_dim": Dm.mean(axis=0).tolist(),
                "mean_signed_past_motion_per_dim": V.mean(axis=0).tolist(),
            },
            "speed_autocorrelation": {
                "corr_V_t_Vs_next": _corr(V, Vn),
                "corr_per_dim": per_dim_corr(V, Vn),
                "corr_arm": {"left": _arm_corr(V, Vn, D.LEFT), "right": _arm_corr(V, Vn, D.RIGHT)},
            },
            "alignment_proof_corr_future_motion_vs_delta": {
                "pooled": _corr(Vn, Dm), "per_dim": per_dim_corr(Vn, Dm)},
            "history_only_predictors": {
                "hold_state_L1": float(np.abs(Dm).mean()),
                "constant_velocity_L1": float(np.abs(V - Dm).mean()),
                "train_linear_history_L1": float(np.abs(lin - Dm).mean()),
                "train_linear_history_per_arm": {
                    "left": float(np.abs(lin[:, D.LEFT] - Dm[:, D.LEFT]).mean()),
                    "right": float(np.abs(lin[:, D.RIGHT] - Dm[:, D.RIGHT]).mean())},
                "train_linear_history_vs_constant_velocity_ratio":
                    float(np.abs(lin - Dm).mean() / (np.abs(V - Dm).mean() + 1e-12)),
            },
            "per_episode_corr_mean": {
                "left": float(np.mean([e["corr_left"] for e in ep_corrs.values() if e["corr_left"] is not None])),
                "right": float(np.mean([e["corr_right"] for e in ep_corrs.values() if e["corr_right"] is not None])),
            },
            "per_episode_corr_min": {
                "left": float(min(e["corr_left"] for e in ep_corrs.values() if e["corr_left"] is not None)),
                "right": float(min(e["corr_right"] for e in ep_corrs.values() if e["corr_right"] is not None)),
            },
            "episodes_with_negative_arm_corr": {
                "left": int(sum(1 for e in ep_corrs.values() if (e["corr_left"] or 0) < 0)),
                "right": int(sum(1 for e in ep_corrs.values() if (e["corr_right"] or 0) < 0)),
            },
        }
    res["train_fitted_linear_model"] = {"slope": fit["slope"].tolist(),
                                        "intercept": fit["intercept"].tolist()}
    res["history_only_predictors_note"] = (
        "constant_velocity uses ONLY past state; hold_state predicts zero delta. "
        "train_linear_history is the best per-dimension affine map of past motion fitted on TRAIN.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = run()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    for name in ("train", "val"):
        b = res["splits"][name]
        print("[%s] frames=%d  corr(V,D) pooled=%.4f  left=%.4f right=%.4f  mean_over_dims=%.4f" % (
            name, b["n_frames"], b["corr_pooled_flattened"], b["corr_arm"]["left"],
            b["corr_arm"]["right"], b["corr_mean_over_dims"]))
        print("   speed autocorr corr(V_t,V_t+1)=%.4f | predict L1: hold=%.6f cv=%.6f linear=%.6f" % (
            b["speed_autocorrelation"]["corr_V_t_Vs_next"], b["history_only_predictors"]["hold_state_L1"],
            b["history_only_predictors"]["constant_velocity_L1"],
            b["history_only_predictors"]["train_linear_history_L1"]))
        for row in b["sign"]:
            if row["basis"] == "expert_delta":
                print("   sign thr=%.0e n=%7d (%.3f) agree=%.3f" % (
                    row["threshold"], row["n"], row.get("fraction_of_entries", 0.0),
                    row.get("sign_agreement", float("nan"))))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
