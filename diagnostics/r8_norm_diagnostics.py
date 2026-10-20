#!/usr/bin/env python3
"""R8-A2: does state-bounds normalization destroy signed V/Acc information? (CPU, read-only)

Runs on the real HDF5 with the TRAIN / DEV / HOLDOUT episode split (64/16/20) and, for
every dimension, reports:

  * the positional bounds and whether zero lies inside them;
  * the normalized value of ZERO velocity / acceleration (should be 0 for a signed
    transform; it is not for a positional one);
  * floor / ceiling clipping shares (frames pinned to -1 or +1);
  * how many DISTINCT normalized values remain for negative motion (1 = full annihilation);
  * the share of frames whose sign information is lost because they collide with the
    zero-motion bucket;
  * per-arm and gripper aggregates;
  * the same quantities under a candidate signed, train-only, zero-centred transform.

Read-only: nothing is written except the requested JSON.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for sub in ("diagnostics", "training", "code/xpolicylab/policy/OpenVLA_Pure"):
    sys.path.insert(0, os.path.join(REPO, sub))
import r6_data as D          # noqa: E402
import r8_contract as C      # noqa: E402

LEFT, RIGHT = slice(0, 7), slice(7, 14)
GRIPPERS = (6, 13)
DIM_NAMES = D.DIM_NAMES


def history_arrays(episodes):
    """(V, Acc) on t >= 2, the same frame definition as the R8 contract and audit."""
    vs, accs = [], []
    for ep in episodes:
        s = np.asarray(ep["state"], np.float32)
        if len(s) < 4:
            continue
        t = np.arange(2, len(s) - 1)
        v = s[t] - s[t - 1]
        a = v - (s[t - 1] - s[t - 2])
        vs.append(v)
        accs.append(a)
    return np.concatenate(vs), np.concatenate(accs)


def signed_scale(train_x, q=0.99):
    x = np.asarray(train_x, np.float32)
    if x.ndim != 2 or x.shape[1] != 14 or len(x) < 100:
        raise ValueError("need 100+ finite 14D TRAIN rows")
    return np.maximum(np.quantile(np.abs(x), q, axis=0).astype(np.float32), 1e-6)


def signed_transform(x, scale):
    return np.clip(np.asarray(x, np.float32) / np.asarray(scale, np.float32), -1.0, 1.0).astype(np.float32)


def zero_mapping(pmin_d, pmax_d):
    """Where does a ZERO motion land under the positional transform, per dimension?"""
    lo = float(pmin_d); hi = float(pmax_d)
    return float(np.clip(2.0 * (0.0 - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0))


def neg_branch_stats(x, z):
    neg = z[x < -1e-8]
    pos = z[x > 1e-8]
    finite_motion = np.abs(x) > 1e-8
    st = {
        "neg_n": int(neg.size), "pos_n": int(pos.size),
        "neg_distinct": int(len(np.unique(neg))) if neg.size else 0,
        "pos_distinct": int(len(np.unique(pos))) if pos.size else 0,
        "neg_all_floor": bool(neg.size > 0 and np.all(neg <= -1.0 + 1e-7)),
        "sign_separable": bool((neg.size == 0 or pos.size == 0) or (neg.max() < pos.min())),
        "floor_share": float(np.mean(z <= -1.0 + 1e-7)),
        "ceiling_share": float(np.mean(z >= 1.0 - 1e-7)),
        "distinct_values": int(len(np.unique(z))),
        "finite_motion_n": int(finite_motion.sum()),
    }
    return st


def dim_report(raw, old, new, pmin_d, pmax_d):
    x = np.asarray(raw, np.float64)
    o = np.asarray(old, np.float64)
    n = np.asarray(new, np.float64)
    z_old = zero_mapping(pmin_d, pmax_d)
    st_old = neg_branch_stats(x, o)
    st_new = neg_branch_stats(x, n)
    finite = np.abs(x) > 1e-8
    out = {
        "pmin": float(pmin_d), "pmax": float(pmax_d),
        "zero_inside_bounds": bool(pmin_d <= 0.0 <= pmax_d),
        "bound_width": float(pmax_d - pmin_d),
        "zero_normalized_old": z_old,
        "zero_normalized_new": 0.0,
        "raw_mean_abs": float(np.abs(x).mean()),
        "raw_std": float(float(x.std())),
        "old": st_old, "new": st_new,
        "old_collides_with_zero_share": float(np.mean(np.abs(o[finite] - z_old) <= 1e-7)) if finite.any() else None,
        "new_collides_with_zero_share": float(np.mean(np.abs(n[finite]) <= 1e-7)) if finite.any() else None,
    }
    neg = x < -1e-8
    if neg.sum() > 10:
        out["old_neg_magnitude_monotone_share"] = float(np.mean(np.diff(np.sort(o[neg])) >= 0))
        out["new_neg_magnitude_monotone_share"] = float(np.mean(np.diff(np.sort(n[neg])) >= 0))
        out["old_neg_distinct_ratio"] = st_old["neg_distinct"] / max(1, st_old["neg_n"])
        out["new_neg_distinct_ratio"] = st_new["neg_distinct"] / max(1, st_new["neg_n"])
    return out


def summarize(raw, old, new, pmin, pmax):
    per_dim = [dim_report(raw[:, d], old[:, d], new[:, d], float(pmin[d]), float(pmax[d]))
               for d in range(14)]
    def agg(idx):
        return {
            "old_floor_share_mean": float(np.mean([per_dim[i]["old"]["floor_share"] for i in idx])),
            "old_floor_share_max": float(np.max([per_dim[i]["old"]["floor_share"] for i in idx])),
            "new_floor_share_mean": float(np.mean([per_dim[i]["new"]["floor_share"] for i in idx])),
            "dims_with_old_neg_annihilation": [DIM_NAMES[i] for i in idx
                                               if per_dim[i]["old"]["neg_all_floor"]],
        }
    return {
        "rows": int(len(raw)),
        "per_dim": per_dim,
        "per_arm": {"left": agg(range(0, 7)), "right": agg(range(7, 14))},
        "grippers": agg(GRIPPERS),
        "arms_only": agg([i for i in range(14) if i not in GRIPPERS]),
        "n_dims_old_neg_annihilated": int(sum(1 for d in per_dim if d["old"]["neg_all_floor"])),
        "n_dims_old_zero_not_zero": int(sum(1 for d in per_dim
                                            if abs(d["zero_normalized_old"]) > 1e-6)),
        "mean_old_floor_share": float(np.mean([d["old"]["floor_share"] for d in per_dim])),
        "mean_new_floor_share": float(np.mean([d["new"]["floor_share"] for d in per_dim])),
        "mean_old_collide_with_zero": float(np.mean(
            [d["old_collides_with_zero_share"] or 0.0 for d in per_dim])),
        "mean_new_collide_with_zero": float(np.mean(
            [d["new_collides_with_zero_share"] or 0.0 for d in per_dim])),
    }


def run(bounds_path, out_path):
    b = np.load(bounds_path)
    pmin = np.asarray(b["pmin"], np.float32)
    pmax = np.asarray(b["pmax"], np.float32)
    if pmin.shape != (14,) or pmax.shape != (14,) or not (pmax > pmin).all():
        raise ValueError("invalid positional bounds")
    split = D.load_split()
    from r7_state_mlp import split_episodes
    tr_eps, dev_eps = split_episodes(split["train"], n_dev=16)
    groups = {"train": tr_eps, "dev": dev_eps, "holdout": split["val"]}
    data = {}
    for name, eps in groups.items():
        v, a = history_arrays([D.load_episode(e) for e in eps])
        data[name] = {"V": v, "Acc": a}
    v_scale = signed_scale(data["train"]["V"])
    a_scale = signed_scale(data["train"]["Acc"])
    out = {
        "bounds_file": os.path.abspath(bounds_path),
        "bounds": {"pmin": pmin.tolist(), "pmax": pmax.tolist()},
        "zero_in_bounds": bool(((pmin <= 0) & (pmax >= 0)).all()),
        "zero_in_bounds_per_dim": ((pmin <= 0) & (pmax >= 0)).tolist(),
        "dims_without_zero_in_bounds": [DIM_NAMES[i] for i in range(14)
                                        if not (pmin[i] <= 0 <= pmax[i])],
        "candidate_train_only_signed_scale": {"V_q99": v_scale.tolist(), "Acc_q99": a_scale.tolist()},
        "splits": {},
        "gripper_dims": [DIM_NAMES[i] for i in GRIPPERS],
    }
    for name, payload in data.items():
        out["splits"][name] = {}
        for label, scale in (("V", v_scale), ("Acc", a_scale)):
            raw = payload[label]
            old = C.normalize_block(raw, pmin, pmax)
            new = signed_transform(raw, scale)
            out["splits"][name][label] = summarize(raw, old, new, pmin, pmax)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)

    print("zero in bounds (all dims):", out["zero_in_bounds"],
          "| dims without zero:", out["dims_without_zero_in_bounds"])
    for name in ("train", "dev", "holdout"):
        for label in ("V", "Acc"):
            s = out["splits"][name][label]
            print("[%s/%s] rows=%d dims_neg_annihilated=%d mean_floor old=%.3f new=%.4f"
                  % (name, label, s["rows"], s["n_dims_old_neg_annihilated"],
                     s["mean_old_floor_share"], s["mean_new_floor_share"]))
    print("dims with old negative annihilation (train/V):",
          [DIM_NAMES[i] for i, d in enumerate(out["splits"]["train"]["V"]["per_dim"])
           if d["old"]["neg_all_floor"]])
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bounds", default="/home/dgut/pusht_runtime/push_T_bounds.npz")
    ap.add_argument("--out", required=True)
    run(ap.parse_args().bounds, ap.parse_args().out)
