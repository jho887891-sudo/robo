#!/usr/bin/env python3
"""R6 audit: minimum representable 256-bin L1 vs existing floor-bin encoder.

The old 'encode true / decode' error is NOT the global lower bound for any
256-token predictor: the model can select any bin, including the nearest one.

Here nearest-grid is the coordinatewise minimal L1 prediction representable
by the *fixed* 256-bin action decoder, for specified q01/q99. This tests
rather than assumes whether an absolute action output is incapable of
beating the past-only constant-velocity baseline on the same heldout frames.

This is a CPU-only HDF5 diagnostic. Synthetic tests can import helpers
without requiring h5py, PyTorch or any robot runtime.
"""
import argparse
import json
import numpy as np


N_BINS = 256


def _validate(target, q01, q99):
    y = np.asarray(target, dtype=np.float32)
    lo = np.asarray(q01, dtype=np.float32)
    hi = np.asarray(q99, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != 14 or lo.shape != (14,) or hi.shape != (14,):
        raise ValueError("expected (N,14) targets and per-dim (14,) q01/q99")
    if not np.isfinite(y).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("nonfinite targets or statistics")
    if (hi <= lo).any():
        raise ValueError("q99 must strictly exceed q01 for all 14 dimensions")
    return y, lo, hi


def decode_grid(index, q01, q99):
    """Exact physical-value equivalent of action_codec.decode_indices/denormalize."""
    ids = np.asarray(index, dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= N_BINS):
        raise ValueError("index outside 0..255")
    lo = np.asarray(q01, np.float32)
    hi = np.asarray(q99, np.float32)
    levels = np.linspace(-1.0, 1.0, N_BINS).astype(np.float32)[ids]
    return ((levels + 1.0) / 2.0 * (hi - lo + 1e-8) + lo).astype(np.float32)


def representable_nearest(target, q01, q99):
    """Minimum absolute error among every representable bin, per target entry."""
    y, lo, hi = _validate(target, q01, q99)
    # Exact decoder grid endpoints are lo and hi + ~5e-9; uniformly spaced.
    grid_lo = decode_grid(np.zeros((1, 14), np.int64), lo, hi)[0]
    grid_hi = decode_grid(np.full((1, 14), N_BINS - 1, np.int64), lo, hi)[0]
    step = (grid_hi.astype(np.float64) - grid_lo.astype(np.float64)) / (N_BINS - 1)
    fractional = (y.astype(np.float64) - grid_lo) / step
    k_floor = np.clip(np.floor(fractional), 0, N_BINS - 1).astype(np.int64)
    k_ceil = np.clip(k_floor + 1, 0, N_BINS - 1).astype(np.int64)
    lower = decode_grid(k_floor, lo, hi)
    upper = decode_grid(k_ceil, lo, hi)
    choose_upper = np.abs(upper.astype(np.float64) - y) < np.abs(lower.astype(np.float64) - y)
    chosen = np.where(choose_upper, k_ceil, k_floor)
    nearest = decode_grid(chosen, lo, hi)
    return chosen, nearest


def existing_floor_oracle(target, q01, q99):
    """Reproduce R2/R5 mode=fixed index selection; not the best-bin bound."""
    y, lo, hi = _validate(target, q01, q99)
    # Matches action_codec.normalize / encode_indices: digitize-1, clip.
    normalized = np.clip(2.0 * (y - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0)
    levels = np.linspace(-1.0, 1.0, N_BINS).astype(np.float32)
    idx = np.clip(np.digitize(normalized, levels) - 1, 0, N_BINS - 1)
    return idx, decode_grid(idx, lo, hi)


def summarize_oracle(target, q01, q99):
    y, lo, hi = _validate(target, q01, q99)
    floored_ids, floor_pred = existing_floor_oracle(y, lo, hi)
    nearest_ids, best_pred = representable_nearest(y, lo, hi)
    err_floor = np.abs(floor_pred - y)
    err_best = np.abs(best_pred - y)
    tol = 1e-6
    if (err_best > err_floor + tol).any():
        raise AssertionError("nearest representable bin must never be worse than floor bin")
    return {
        "n_frames": len(y),
        "floor_encoder_oracle_L1": float(err_floor.mean()),
        "nearest_representable_lower_bound_L1": float(err_best.mean()),
        "nearest_bound_per_arm": {
            "left": float(err_best[:, :7].mean()),
            "right": float(err_best[:, 7:].mean())
        },
        "per_dim_nearest_lower_bound": err_best.mean(0).tolist(),
        "mean_improvement_due_to_nearest_bin": float((err_floor - err_best).mean()),
        "fraction_entries_requiring_different_bin": float((nearest_ids != floored_ids).mean()),
        "fraction_entries_outside_quantiles": float(((y < lo) | (y > hi)).mean()),
        "notes": "lower bound holds only for a direct, fixed q01/q99, independent-bin decoder with no post-processing",
    }


def run():
    import os
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "diagnostics"))
    from r6_data import load_split, load_episodes, valid_frames
    split = load_split()
    train_episodes = load_episodes(split["train"])
    val_episodes = load_episodes(split["val"])
    def arrays(episodes):
        all_s, all_a, all_v = [], [], []
        for e in episodes:
            t = valid_frames(len(e["state"]))
            all_s.append(e["state"][t])
            all_a.append(e["action"][t])
            all_v.append(e["state"][t] - e["state"][t-1])
        return np.concatenate(all_s), np.concatenate(all_a), np.concatenate(all_v)
    # q01/q99 MUST match the actual R5/R6 training action_stats contract:
    # train quantiles use ALL frames, not the time-valid 1:-1 diagnostic subset.
    S_tr_all = np.concatenate([e["state"] for e in train_episodes])
    A_tr_all = np.concatenate([e["action"] for e in train_episodes])
    S, A, V = arrays(val_episodes)
    cv_L1 = float(np.abs((S + V) - A).mean())
    modes = {}
    for mode in ("absolute", "residual"):
        T_train = A_tr_all if mode == "absolute" else A_tr_all - S_tr_all
        q01 = np.quantile(T_train, 0.01, axis=0).astype(np.float32)
        q99 = np.quantile(T_train, 0.99, axis=0).astype(np.float32)
        T_val = A if mode == "absolute" else A - S
        report = summarize_oracle(T_val, q01, q99)
        report["best_bin_beats_cv"] = report["nearest_representable_lower_bound_L1"] < cv_L1
        report["q01"] = q01.tolist()
        report["q99"] = q99.tolist()
        modes[mode] = report
    return {"n_val_frames": len(S), "cv_full_val_L1": cv_L1, "modes": modes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run()
    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "n_val_frames": result["n_val_frames"],
        "cv_full_val_L1": result["cv_full_val_L1"],
        "floor_vs_lower_bounds": {
            k: {j: v[j] for j in
                ("floor_encoder_oracle_L1", "nearest_representable_lower_bound_L1",
                 "best_bin_beats_cv")}
            for k, v in result["modes"].items()}},
        indent=2))


if __name__ == "__main__":
    main()
