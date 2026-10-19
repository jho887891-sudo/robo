"""R6 direction metrics: never interpret absolute-position correlation as motion accuracy.

All inputs are PHYSICAL 14D absolute joint commands. Direction statistics are
computed on deltas (command - current state) at several motion thresholds, and
every threshold reports the number of entries it actually kept.
"""
import numpy as np

THRESHOLDS = (0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3)
LEFT = slice(0, 7)
RIGHT = slice(7, 14)


def _as14(x, name):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 14 or not np.isfinite(a).all():
        raise ValueError("%s must be shape (N,14) and finite" % name)
    return a


def _corr(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    if a.size < 3 or a.std() == 0.0 or b.std() == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize_direction(pred, gt, state, prev_state, episode_ids, thresholds=THRESHOLDS):
    """Delta-level direction / magnitude / per-episode diagnostics."""
    p, g = _as14(pred, "pred"), _as14(gt, "gt")
    s, old = _as14(state, "state"), _as14(prev_state, "previous_state")
    if p.shape != g.shape or p.shape != s.shape or p.shape != old.shape:
        raise ValueError("pred/gt/state/previous_state shape mismatch")
    if len(episode_ids) != len(p):
        raise ValueError("episode labels must align to rows")
    pred_delta, gt_delta, past_motion = p - s, g - s, s - old
    per_dim = [_corr(pred_delta[:, d], gt_delta[:, d]) for d in range(14)]
    per_dim_past = [_corr(past_motion[:, d], gt_delta[:, d]) for d in range(14)]
    sign = []
    for thr in thresholds:
        for basis, series in (("expert_delta", np.abs(gt_delta)), ("predicted_delta", np.abs(pred_delta))):
            m = series >= thr
            n = int(m.sum())
            if n == 0:
                sign.append({"threshold": thr, "basis": basis, "n": 0})
                continue
            sign.append({
                "threshold": thr, "basis": basis, "n": n,
                "fraction_of_entries": float(n / series.size),
                "sign_agreement": float(np.mean(np.sign(pred_delta[m]) == np.sign(gt_delta[m]))),
                "past_motion_sign_agreement": float(np.mean(np.sign(past_motion[m]) == np.sign(gt_delta[m]))),
            })
    per_episode = {}
    for e in sorted(set(episode_ids)):
        idx = np.array([i for i, label in enumerate(episode_ids) if label == e])
        model_err = float(np.abs(p[idx] - g[idx]).mean())
        cv_err = float(np.abs((s[idx] + (s[idx] - old[idx])) - g[idx]).mean())
        hold_err = float(np.abs(s[idx] - g[idx]).mean())
        per_episode[e] = {
            "n": int(len(idx)), "model_L1": model_err, "constant_velocity_L1": cv_err,
            "hold_state_L1": hold_err,
            "beats_constant_velocity": bool(model_err < cv_err),
            "beats_hold_state": bool(model_err < hold_err),
            "pred_delta_corr": _corr(pred_delta[idx], gt_delta[idx]),
        }
    return {
        "delta_correlation": {
            "pooled": _corr(pred_delta, gt_delta),
            "mean_over_dims": float(np.mean([c for c in per_dim if c is not None])) if any(
                c is not None for c in per_dim) else None,
            "per_dim": per_dim,
            "left": _corr(pred_delta[:, LEFT], gt_delta[:, LEFT]),
            "right": _corr(pred_delta[:, RIGHT], gt_delta[:, RIGHT]),
            "undefined_dims": [d for d, c in enumerate(per_dim) if c is None],
        },
        "past_motion_delta_correlation": {
            "pooled": _corr(past_motion, gt_delta),
            "per_dim": per_dim_past,
            "note": "upper reference for a policy that simply copies the past motion",
        },
        "sign_agreement": sign,
        "magnitude": {
            "mean_abs_predicted_delta": float(np.abs(pred_delta).mean()),
            "mean_abs_expert_delta": float(np.abs(gt_delta).mean()),
            "ratio": float(np.abs(pred_delta).mean() / (np.abs(gt_delta).mean() + 1e-12)),
            "per_arm": {
                "left": {"predicted": float(np.abs(pred_delta[:, LEFT]).mean()),
                         "expert": float(np.abs(gt_delta[:, LEFT]).mean())},
                "right": {"predicted": float(np.abs(pred_delta[:, RIGHT]).mean()),
                          "expert": float(np.abs(gt_delta[:, RIGHT]).mean())},
            },
            "predicted_std": float(pred_delta.std()),
            "expert_std": float(gt_delta.std()),
        },
        "per_episode": per_episode,
        "episodes_beating_constant_velocity": int(
            sum(1 for v in per_episode.values() if v["beats_constant_velocity"])),
        "episodes_beating_hold_state": int(
            sum(1 for v in per_episode.values() if v["beats_hold_state"])),
        "episode_count": len(per_episode),
        "units_warning": "physical 14D L1 mixes joint and gripper units; read per-dimension values",
    }
