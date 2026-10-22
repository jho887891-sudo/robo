#!/usr/bin/env python3
"""Independent M2 matched-bounds SAFE comparison v2.

DO NOT overwrite historical mission/m2_physical_audit.json (v1).
This is a CPU-only audit of already frozen 769-prefix-corrected DEV predictions.

Important: deployment discards predicted gripper residuals, and CV/no-op must be
subjected to the same physical bounds. Rate limiting needs real prev_cmd/dt and
is NOT inferred from offline frame pairs; the output explicitly flags that.
"""
import argparse
import json
import os
import sys
import numpy as np
import h5py

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("training", "code/xpolicylab/policy/OpenVLA_Pure",
            "code/xpolicylab/policy/OpenVLA_R42"):
    sys.path.insert(0, os.path.join(REPO, sub))
import r42_adapter as R
import m2_task_normalizer as TN
import m2_multitask_data as MD
import r42_train_contract as TC


def matched_commands(state, velocity, bins, norm):
    """Identical codec, gripper fallback and bounds in model/CV paths.

    CV is encoded to the nearest legal zero-residual bin, as done by the real
    R42_POLICY=cv branch. Holding position is also bounded.
    """
    lo, hi = (np.asarray(norm[k], np.float32) for k in ("pmin", "pmax"))
    limits = {"amin": lo, "amax": hi}
    zeros = np.zeros(14, np.float32)
    cv_bins, _ = R.token_ids_for_residual(zeros, norm["q01"], norm["q99"])
    model, model_residual, model_info = R.decode_command(
        bins, norm["q01"], norm["q99"], state, velocity,
        limits=limits, gripper_dims=R.GRIPPER_DIMS)
    cv, cv_residual, cv_info = R.decode_command(
        cv_bins, norm["q01"], norm["q99"], state, velocity,
        limits=limits, gripper_dims=R.GRIPPER_DIMS)
    noop = np.clip(state, lo, hi)
    return model, cv, noop, model_info, cv_info


def mean_l1(a, b):
    return float(np.abs(a - b).mean())


def bootstrap_ci(x, seed=0):
    x = np.asarray(x, np.float64)
    if len(x) < 2:
        return [None, None]
    rng = np.random.RandomState(seed)
    boot = np.mean(rng.choice(x, (4000, len(x)), replace=True), axis=1)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", default=os.path.join(REPO, "evidence", "dev_eval_v2_aligned.npz"))
    p.add_argument("--normalizer", default=os.path.join(REPO, "data", "m2_task_normalizer_2task.json"))
    p.add_argument("--hdf5-root", default=None)
    p.add_argument("--tasks", default="push_T,stack_bowls")
    p.add_argument("--out", default=os.path.join(REPO, "mission", "m2_physical_audit_v2.json"))
    a = p.parse_args()
    z = np.load(a.archive, allow_pickle=True)
    ver = str(z["action_window_alignment"]) if "action_window_alignment" in z.files else None
    if ver != "vision-prefix-aware-v2":
        raise SystemExit("refuse LEGACY or unverified action-window alignment: %r" % ver)
    norm_all = json.load(open(a.normalizer))
    res = {
        "schema": "m2-physical-matched-bounds-v2",
        "archive_alignment": ver,
        "archive": os.path.abspath(a.archive),
        "normalizer": os.path.abspath(a.normalizer),
        "checkpoint_global_step": int(z["checkpoint_global_step"]) if "checkpoint_global_step" in z.files else None,
        "lora_sha256": str(z["lora_sha256"]) if "lora_sha256" in z.files else None,
        "assumptions": {
            "bounds": "per-task TRAIN pmin,pmax; verify against actual serving checkpoint safety_limits",
            "gripper": "drop learned residual at dims 6,13 just as r42_adapter.decode_command",
            "cv": "zero residual encoded through 256-bin action codec, then same decoder/bounds",
            "rate_limiter": "NOT applied: offline DEV archive lacks actual prev_cmd and dt",
            "confidence_interval": "frame bootstrap over only 8 DEV frames/task, not independent episode CI"},
        "tasks": {}}
    for task in [t.strip() for t in a.tasks.split(",") if t.strip()]:
        norm = TN.decode_context(norm_all, task)
        idx = MD.load_task_index(task, a.hdf5_root)
        ep = [str(e) for e in z[task + "/episodes"]]
        ts = [int(t) for t in z[task + "/t"]]
        bins = np.asarray(z[task + "/pred_bins"], np.int64)
        rows = []
        for i, (episode, tick) in enumerate(zip(ep, ts)):
            with h5py.File(os.path.join(idx["hdf5_dir"], episode), "r") as h:
                states = np.asarray(TC.read14(h, "state"), np.float32)
                targets = np.asarray(TC.read14(h, "action"), np.float32)
            state, target = states[tick], targets[tick]
            vel = states[tick] - states[tick - 1]
            model, cv, noop, mi, ci = matched_commands(state, vel, bins[i], norm)
            rhat = R.residual_from_indices(bins[i], norm["q01"], norm["q99"])
            raw_legacy = state + vel + rhat
            old_clipped = np.clip(raw_legacy, norm["pmin"], norm["pmax"])
            rows.append({
                "episode": episode, "t": tick,
                "l1_model_safe_matched": mean_l1(model, target),
                "l1_cv_safe_matched": mean_l1(cv, target),
                "l1_noop_safe_matched": mean_l1(noop, target),
                "l1_model_safe_legacy": mean_l1(old_clipped, target),
                "l1_model_raw_legacy": mean_l1(raw_legacy, target),
                "l1_cv_unclipped_legacy": mean_l1(state + vel, target),
                "l1_noop_unclipped_legacy": mean_l1(state, target),
                "model_clipped_dims": int(mi["clipped_joint"]),
                "cv_clipped_dims": int(ci["clipped_joint"]),
                "l1_model_arm": mean_l1(model[[j for j in range(14) if j not in R.GRIPPER_DIMS]],
                                         target[[j for j in range(14) if j not in R.GRIPPER_DIMS]]),
                "l1_cv_arm": mean_l1(cv[[j for j in range(14) if j not in R.GRIPPER_DIMS]],
                                      target[[j for j in range(14) if j not in R.GRIPPER_DIMS]])})
        def avg(k):
            return float(np.mean([r[k] for r in rows]))
        dif = [r["l1_model_safe_matched"] - r["l1_cv_safe_matched"] for r in rows]
        res["tasks"][task] = {
            "summary": {
                "n_frames": len(rows),
                "model_safe_l1": avg("l1_model_safe_matched"),
                "cv_safe_l1": avg("l1_cv_safe_matched"),
                "noop_safe_l1": avg("l1_noop_safe_matched"),
                "model_safe_legacy": avg("l1_model_safe_legacy"),
                "cv_unclipped_legacy": avg("l1_cv_unclipped_legacy"),
                "mean_model_minus_cv": float(np.mean(dif)),
                "frame_bootstrap_ci95_model_minus_cv": bootstrap_ci(dif),
                "model_clipped_dims_mean": avg("model_clipped_dims"),
                "cv_clipped_dims_mean": avg("cv_clipped_dims"),
                "model_arm_l1": avg("l1_model_arm"),
                "cv_arm_l1": avg("l1_cv_arm")},
            "frames": rows}
    parent = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(parent, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps({t: d["summary"] for t, d in res["tasks"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
