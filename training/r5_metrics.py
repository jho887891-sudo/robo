"""R5 metric extensions: no trivial-state shortcut disguised as control skill.

All predictions and baselines must be decoded to physical 14D absolute actions
BEFORE computing any metric. Gripper units may differ from joint radians, so
report per-dimension and arm metrics; do not treat all 14D as radians.
"""
import numpy as np


def _as14(x, name):
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 14 or not np.isfinite(a).all():
        raise ValueError("%s must have shape (N,14) and finite values" % name)
    return a


def summarize_prediction(pred, gt, state, prior_state, episode_ids, gen_token_rows,
                         median_delta=None):
    p = _as14(pred, "pred")
    a = _as14(gt, "gt")
    s = _as14(state, "state")
    old = _as14(prior_state, "prior_state")
    if p.shape != a.shape or p.shape != s.shape or p.shape != old.shape:
        raise ValueError("prediction/gt/state/prior_state shape mismatch")
    if len(episode_ids) != len(p):
        raise ValueError("episode labels must align to rows")
    toks = np.asarray(gen_token_rows, dtype=np.int64)
    if toks.shape != p.shape or np.any(toks < 0) or np.any(toks > 255):
        raise ValueError("need legal 14D generated action tokens for every sample")
    if median_delta is None:
        d = np.zeros((14,), np.float32)
    else:
        d = np.asarray(median_delta, np.float32)
        if d.shape != (14,) or not np.isfinite(d).all():
            raise ValueError("training-only constant-median velocity must be finite 14D")
    baselines = {
        "model": p,
        "hold_state": s,
        "constant_velocity": s + (s - old),
        "train_median_delta": s + d,
    }
    errors = {name: np.abs(out - a) for name, out in baselines.items()}
    def avg(x):
        return float(np.mean(x))
    per_arm = {
        name: {method: avg(err[:, start:end])
               for method, err in errors.items()}
        for name, start, end in (("left", 0, 7), ("right", 7, 14))
    }
    per_dim = {method: err.mean(axis=0).astype(float).tolist()
               for method, err in errors.items()}
    unique_episode_ids = sorted(set(episode_ids))
    per_episode = {}
    for e in unique_episode_ids:
        inds = [i for i, label in enumerate(episode_ids) if label == e]
        per_episode[e] = {"n": len(inds),
                          **{name: avg(err[inds]) for name, err in errors.items()}}
    macro_l1 = {name: float(np.mean([per_episode[e][name] for e in unique_episode_ids]))
                for name in errors}
    token_diversity = []
    for d in range(14):
        v, counts = np.unique(toks[:, d], return_counts=True)
        token_diversity.append({
            "dimension": d, "unique_tokens": len(v),
            "mode_fraction": float(np.max(counts) / len(toks)),
            "mode_token": int(v[np.argmax(counts)]),
        })
    return {
        "per_arm_physical_L1": per_arm,
        "per_dim_physical_L1": per_dim,
        "per_episode_physical_L1": per_episode,
        "episode_macro_L1": macro_l1,
        "evaluation_episode_count": len(unique_episode_ids),
        "per_dim_generated_token_diversity": token_diversity,
        "unique_generated_token_rows": len({tuple(row) for row in toks.tolist()}),
        "predicted_delta_to_expert_delta_ratio":
            avg(np.abs(p - s)) / (avg(np.abs(a - s)) + 1e-12),
        "units_warning": "14D physical L1 mixes joint and gripper dimensions; inspect per-dimension metrics",
    }
