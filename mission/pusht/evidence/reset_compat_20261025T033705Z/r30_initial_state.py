#!/usr/bin/env python3
"""R30 CPU-only, read-only initial-state statistics for push_T (TRAIN vs DEV).

Reads the frozen splits + the real HDF5 through the SAME loader the trainer uses
(training/m2_multitask_data.py), and reports, per dimension:
  S[0]            initial state of each episode
  S[1] - S[0]     first state transition
  A[0] - S[0]     first target action minus initial state
plus the first `start-window` transitions and the frame indices of that window.
No model, no CUDA, no writes outside the given output paths.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

DIM_NAMES = ["left_arm%d" % i for i in range(6)] + ["left_ee"] + \
            ["right_arm%d" % i for i in range(6)] + ["right_ee"]
DIM_UNITS = ["rad"] * 14
DIM_UNITS[6] = "normalized[0,1]"
DIM_UNITS[13] = "normalized[0,1]"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dist(x):
    """Per-dimension distribution of a [n, 14] array."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 14:
        raise ValueError("expected [n,14], got %s" % (x.shape,))
    qs = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90, 0.99], axis=0)
    return {
        "n": int(x.shape[0]),
        "min": x.min(0).tolist(), "max": x.max(0).tolist(),
        "mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
        "q10": qs[0].tolist(), "q25": qs[1].tolist(), "q50": qs[2].tolist(),
        "q75": qs[3].tolist(), "q90": qs[4].tolist(), "q99": qs[5].tolist(),
    }


def positional_normalize(x, pmin, pmax):
    v = np.asarray(x, np.float32)
    lo = np.asarray(pmin, np.float32)
    hi = np.asarray(pmax, np.float32)
    return np.clip(2.0 * (v - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0)


def quantile_normalize(x, q01, q99):
    v = np.asarray(x, np.float32)
    lo = np.asarray(q01, np.float32)
    hi = np.asarray(q99, np.float32)
    return np.clip(2.0 * (v - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0)


def scale_normalize(x, scale):
    return np.clip(np.asarray(x, np.float32) / np.asarray(scale, np.float32), -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--hdf5-root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--normalizer", required=True)
    ap.add_argument("--start-window", type=int, default=25)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(a.repo, "training"))
    import m2_multitask_data as MD

    split_doc = json.load(open(a.splits))
    split_map = split_doc.get("splits", split_doc)
    task = "push_T"
    train_eps = sorted(os.path.basename(str(e)) for e in split_map[task]["train"])
    dev_eps = sorted(os.path.basename(str(e)) for e in split_map[task]["dev"])
    if set(train_eps) & set(dev_eps):
        raise SystemExit("TRAIN/DEV overlap in splits")

    norm_doc = json.load(open(a.normalizer))
    nb = norm_doc["tasks"][task]
    pmin, pmax = nb["pmin"], nb["pmax"]
    q01, q99 = nb["action_q01"], nb["action_q99"]
    v_scale = nb["v_scale"]

    index = MD.load_task_index(task, a.hdf5_root, verify_targets=False)
    reader = getattr(MD, "_episode_state_action", None)
    if reader is None:
        raise SystemExit("m2_multitask_data._episode_state_action not found; refusing to guess HDF5 keys")

    out = {"schema": "r30-pusht-initial-state-v1",
           "task": task,
           "loader": "training/m2_multitask_data.py::_episode_state_action",
           "dim_order": "[left_arm(6), left_ee(1), right_arm(6), right_ee(1)]",
           "dim_names": DIM_NAMES, "dim_units": DIM_UNITS,
           "start_window": int(a.start_window),
           "splits_sha256": sha256_file(a.splits),
           "normalizer_sha256": sha256_file(a.normalizer),
           "normalizer_mode": nb.get("normalization_mode"),
           "hdf5_dir": index["hdf5_dir"],
           "split_counts": {"train": len(train_eps), "dev": len(dev_eps)},
           "splits": {}}

    rows = []
    for split, eps in (("TRAIN", train_eps), ("DEV", dev_eps)):
        s0, d1, a0 = [], [], []
        first_motion = []
        align_max = []
        per_ep = []
        for ep in eps:
            s, act = reader(index["hdf5_dir"], ep)
            s = np.asarray(s, np.float64)
            act = np.asarray(act, np.float64)
            if s.shape[1] != 14 or act.shape[1] != 14:
                raise SystemExit("%s: unexpected width %s/%s" % (ep, s.shape, act.shape))
            T = s.shape[0]
            align = float(np.max(np.abs(act[0:T - 1] - s[1:T])))
            align_max.append(align)
            s0.append(s[0])
            d1.append(s[1] - s[0])
            a0.append(act[0] - s[0])
            w = min(int(a.start_window), T - 1)
            first_motion.append(s[1:w + 1] - s[0:w])
            per_ep.append({"episode": ep, "T": int(T),
                           "alignment_max_abs": align,
                           "S0": s[0].tolist(),
                           "S1_minus_S0": (s[1] - s[0]).tolist(),
                           "A0_minus_S0": (act[0] - s[0]).tolist()})
        S0 = np.asarray(s0)
        D1 = np.asarray(d1)
        A0 = np.asarray(a0)
        FM = np.concatenate(first_motion, axis=0)
        block = {
            "n_episodes": len(eps),
            "n_first_window_transitions": int(FM.shape[0]),
            "camera_frame_indices_in_window": list(range(int(a.start_window))),
            "S0_raw": dist(S0),
            "S1_minus_S0_raw": dist(D1),
            "A0_minus_S0_raw": dist(A0),
            "first_window_motion_raw": dist(FM),
            "S0_positional_normalized": dist(positional_normalize(S0, pmin, pmax)),
            "A0_quantile_normalized": dist(quantile_normalize(A0 + S0, q01, q99)),
            "S1_minus_S0_motion_normalized": dist(scale_normalize(D1, v_scale)),
            "A0_minus_S0_normalized_space": dist(
                quantile_normalize(A0 + S0, q01, q99) - positional_normalize(S0, pmin, pmax)),
            "alignment_max_abs_per_episode": {
                "min": float(np.min(align_max)), "max": float(np.max(align_max)),
                "n_over_1e-6": int(np.sum(np.asarray(align_max) > 1e-6))},
            "episodes": per_ep,
        }
        out["splits"][split] = block
        for qty, arr, unit in (("S0_raw", S0, "raw"),
                               ("S1_minus_S0_raw", D1, "raw"),
                               ("A0_minus_S0_raw", A0, "raw"),
                               ("first_window_motion_raw", FM, "raw")):
            d = dist(arr)
            for i in range(14):
                rows.append([split, qty, i, DIM_NAMES[i], DIM_UNITS[i], d["n"],
                             d["min"][i], d["q10"][i], d["q25"][i], d["q50"][i],
                             d["q75"][i], d["q90"][i], d["q99"][i], d["max"][i],
                             d["mean"][i], d["std"][i]])

    with open(a.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "quantity", "dim", "dim_name", "unit", "n",
                    "min", "q10", "q25", "q50", "q75", "q90", "q99", "max",
                    "mean", "std"])
        w.writerows(rows)

    out["csv_note"] = ("units: arm joints listed as rad per the HDF5 contract docstring; "
                       "gripper dims 6 and 13 are the normalized [0,1] channel. "
                       "n is the number of episodes for A0/S1-S0 rows and the number of "
                       "pooled transitions for first_window_motion_raw rows.")
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({"ok": True, "train": out["splits"]["TRAIN"]["n_episodes"],
                      "dev": out["splits"]["DEV"]["n_episodes"],
                      "csv_rows": len(rows)}, indent=1))


if __name__ == "__main__":
    main()
