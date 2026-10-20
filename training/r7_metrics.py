"""R7 metrics: physical L1, direction, and the velocity-correction target R."""
import numpy as np

LEFT, RIGHT = slice(0, 7), slice(7, 14)
SIGN_THRESHOLD = 1e-3
SIGN_THRESHOLD_STRONG = 5e-3
MOTION_BINS = (0.0, 1e-3, 5e-3, 2e-2, np.inf)
MOTION_LABELS = ("tiny_lt_1e-3", "small_1e-3_5e-3", "medium_5e-3_2e-2", "large_ge_2e-2")


def _a14(x, name):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 14:
        raise ValueError("%s must be (N,14)" % name)
    if not np.isfinite(a).all():
        raise ValueError("%s has non-finite values" % name)
    return a


def _corr(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _cos_rows(a, b):
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def direction(pred_cmd, gt, state, thresh=SIGN_THRESHOLD):
    pd, gd = pred_cmd - state, gt - state
    cos = _cos_rows(pd, gd)
    m = np.abs(gd) >= thresh
    return {
        "corr_delta_pooled": _corr(pd, gd),
        "corr_delta_per_dim": [_corr(pd[:, d], gd[:, d]) for d in range(14)],
        "cosine_mean": float(np.nanmean(cos)) if np.isfinite(cos).any() else None,
        "cosine_median": float(np.nanmedian(cos)) if np.isfinite(cos).any() else None,
        "sign_agreement_at_1e-3": float(np.mean(np.sign(pd[m]) == np.sign(gd[m]))) if m.any() else None,
        "sign_n_at_1e-3": int(m.sum()),
        "magnitude_ratio": float(np.abs(pd).mean() / (np.abs(gd).mean() + 1e-12)),
        "pred_std": float(pd.std()), "gt_std": float(gd.std()),
    }


def evaluate_correction(R_hat, R_gt):
    """How much of the velocity CORRECTION (the part copy-V cannot give) is learned."""
    Rh = _a14(R_hat, "R_hat")
    Rg = _a14(R_gt, "R_gt")
    resid = Rh - Rg
    var_gt = float(Rg.var())
    m = np.abs(Rg) >= 1e-4
    return {
        "L1": float(np.abs(resid).mean()),
        "zero_correction_L1": float(np.abs(Rg).mean()),
        "relative_to_zero": float(np.abs(resid).mean() / (np.abs(Rg).mean() + 1e-12)),
        "R2_variance_explained": (1.0 - float(resid.var()) / var_gt) if var_gt > 0 else None,
        "corr": _corr(Rh, Rg),
        "cosine_mean": (float(np.nanmean(_cos_rows(Rh, Rg)))
                        if np.isfinite(_cos_rows(Rh, Rg)).any() else None),
        "sign_agreement": float(np.mean(np.sign(Rh[m]) == np.sign(Rg[m]))) if m.any() else None,
        "sign_n": int(m.sum()),
        "per_dim_L1": np.abs(resid).mean(axis=0).tolist(),
        "per_arm_L1": {"left": float(np.abs(resid)[:, LEFT].mean()),
                       "right": float(np.abs(resid)[:, RIGHT].mean())},
    }


def evaluate(cmd, gt, state, V, episodes):
    """Full physical evaluation of an absolute command against the next observed state."""
    c, g, s = _a14(cmd, "cmd"), _a14(gt, "gt"), _a14(state, "state")
    V = _a14(V, "V")
    if c.shape != g.shape or c.shape != s.shape:
        raise ValueError("cmd/gt/state shape mismatch")
    if len(episodes) != len(c):
        raise ValueError("episode labels must align")
    cv = s + V
    err, err_id, err_cv = np.abs(c - g), np.abs(s - g), np.abs(cv - g)
    eps = sorted(set(episodes))
    per_ep = {e: {"model": float(err[episodes == e].mean()),
                  "identity": float(err_id[episodes == e].mean()),
                  "constant_velocity": float(err_cv[episodes == e].mean())} for e in eps}
    strata = {}
    motion = np.abs(g - s).mean(axis=1)
    for lo, hi, label in zip(MOTION_BINS[:-1], MOTION_BINS[1:], MOTION_LABELS):
        m = (motion >= lo) & (motion < hi)
        if not m.any():
            strata[label] = {"n": 0}
            continue
        cos = _cos_rows(c[m] - s[m], g[m] - s[m])
        strata[label] = {"n": int(m.sum()), "share": float(m.mean()),
                         "L1": float(err[m].mean()), "identity": float(err_id[m].mean()),
                         "constant_velocity": float(err_cv[m].mean()),
                         "cosine": float(np.nanmean(cos)) if np.isfinite(cos).any() else None}
    return {
        "L1": float(err.mean()),
        "identity_L1": float(err_id.mean()),
        "constant_velocity_L1": float(err_cv.mean()),
        "ratio_vs_identity": float(err.mean() / (err_id.mean() + 1e-12)),
        "ratio_vs_constant_velocity": float(err.mean() / (err_cv.mean() + 1e-12)),
        "per_arm_L1": {"left": float(err[:, LEFT].mean()), "right": float(err[:, RIGHT].mean())},
        "per_arm_identity": {"left": float(err_id[:, LEFT].mean()), "right": float(err_id[:, RIGHT].mean())},
        "per_arm_cv": {"left": float(err_cv[:, LEFT].mean()), "right": float(err_cv[:, RIGHT].mean())},
        "per_dim_L1": err.mean(axis=0).tolist(),
        "per_dim_cv": err_cv.mean(axis=0).tolist(),
        "per_episode_macro": {
            "model": float(np.mean([v["model"] for v in per_ep.values()])),
            "identity": float(np.mean([v["identity"] for v in per_ep.values()])),
            "constant_velocity": float(np.mean([v["constant_velocity"] for v in per_ep.values()]))},
        "episodes_beating_cv": int(sum(1 for v in per_ep.values() if v["model"] < v["constant_velocity"])),
        "episodes_beating_identity": int(sum(1 for v in per_ep.values() if v["model"] < v["identity"])),
        "episode_count": len(eps),
        "per_episode": {e: {k: float(v[k]) for k in ("model", "identity", "constant_velocity")}
                        for e, v in per_ep.items()},
        "direction": direction(c, g, s),
        "motion_strata": strata,
        "n": int(len(c)),
    }


def aggregate(per_seed):
    keys = ("L1", "ratio_vs_identity", "ratio_vs_constant_velocity")
    out = {k: {"mean": float(np.mean([r[k] for r in per_seed])),
               "min": float(np.min([r[k] for r in per_seed])),
               "max": float(np.max([r[k] for r in per_seed]))} for k in keys}
    out["correction_L1"] = {"mean": float(np.mean([r["correction"]["L1"] for r in per_seed])),
                            "min": float(np.min([r["correction"]["L1"] for r in per_seed])),
                            "max": float(np.max([r["correction"]["L1"] for r in per_seed]))}
    out["correction_R2"] = {"mean": float(np.nanmean([r["correction"]["R2_variance_explained"]
                                                     for r in per_seed]))}
    out["cosine_mean"] = float(np.nanmean([r["direction"]["cosine_mean"] for r in per_seed]))
    out["sign_at_1e-3"] = float(np.nanmean([r["direction"]["sign_agreement_at_1e-3"] for r in per_seed]))
    out["n_seeds"] = len(per_seed)
    return out
