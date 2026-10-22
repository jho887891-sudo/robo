#!/usr/bin/env python3
"""Physical SAFE/Raw 14D audit of the M2 shared model against trivial baselines (workline B2).

Everything here is CPU: the model's per-frame predicted bins already exist in the corrected DEV archive
(evidence/dev_eval_v2_aligned.npz, alignment vision-prefix-aware-v2), so no GPU and no re-inference is
needed and the numbers are exactly the ones the P6 gate used.

For every DEV frame it reconstructs what the robot would actually be COMMANDED:

    V       = S[t] - S[t-1]
    R_true  = (A[t] - S[t]) - V            (the residual the model is supposed to predict)
    R_hat   = q01 + (bin + 0.5)/256 * (q99 - q01)     (codec inverse, per task)
    RAW     = S[t] + V + R_hat
    SAFE    = clip(RAW, pmin, pmax)        (the deploy bounds-normalisation clip)

and compares it with the true action and with three baselines: no-op (A_hat = S[t]), constant velocity
(A_hat = S[t] + V) and the legacy zero-position predictor. It reports per-task, per-group (arm vs
gripper), per-motion-stratum, clip rate, residual magnitude and the wrong-prompt counterfactual, with
paired bootstrap CIs over frames.

The question it exists to answer is not 'is the CE lower' but 'does the commanded 14D motion actually
resemble the demonstration'.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _s in ("training", "code/xpolicylab/policy/OpenVLA_Pure",
           "code/xpolicylab/policy/OpenVLA_R42", "tools"):
    sys.path.insert(0, os.path.join(REPO, _s))

import m2_multitask_data as MD      # noqa: E402
import m2_task_normalizer as TN     # noqa: E402
import m2_train_shared as MT        # noqa: E402
import r42_adapter as R             # noqa: E402

GRIP = tuple(R.GRIPPER_DIMS)        # (6, 13)


def codec_residual(ctx, bins):
    q01 = np.asarray(ctx["q01"], np.float32)
    q99 = np.asarray(ctx["q99"], np.float32)
    return q01 + ((np.asarray(bins, np.float32) + 0.5) / 256.0) * (q99 - q01)


def bootstrap_ci(x, n=4000, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    if x.size < 2:
        return [float("nan"), float("nan")]
    rs = np.random.RandomState(seed)
    means = [rs.choice(x, size=x.size, replace=True).mean() for _ in range(n)]
    return [float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))]


def groups_l1(pred, true):
    dims = np.arange(14)
    arm = np.array([d for d in dims if d not in GRIP])
    return {"arm": float(np.abs(pred[arm] - true[arm]).mean()),
            "gripper": float(np.abs(pred[list(GRIP)] - true[list(GRIP)]).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.path.join(REPO, "evidence", "dev_eval_v2_aligned.npz"))
    ap.add_argument("--normalizer", default=os.path.join(REPO, "data", "m2_task_normalizer_2task.json"))
    ap.add_argument("--tasks", default="push_T,stack_bowls")
    ap.add_argument("--hdf5-root", default=None)
    ap.add_argument("--out", default=os.path.join(REPO, "mission", "m2_physical_audit.json"))
    a = ap.parse_args()

    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    payload = json.load(open(a.normalizer))
    z = np.load(a.archive, allow_pickle=True)
    alignment = str(z["action_window_alignment"]) if "action_window_alignment" in z.files else None
    if alignment != "vision-prefix-aware-v2":
        raise SystemExit("archive alignment %r is not vision-prefix-aware-v2; refusing to audit" % alignment)
    import h5py
    import r42_train_contract as TC

    out = {"schema": "m2-physical-audit-v1", "archive": os.path.abspath(a.archive),
           "archive_alignment": alignment,
           "checkpoint_global_step": int(z["checkpoint_global_step"]) if "checkpoint_global_step" in z.files else None,
           "lora_sha256": str(z["lora_sha256"]) if "lora_sha256" in z.files else None,
           "normalizer_sha256": str(z["normalizer_sha256"]) if "normalizer_sha256" in z.files else None,
           "gripper_dims": list(GRIP), "tasks": {}}

    for task in tasks:
        ctx = TN.decode_context(payload, task)
        pmin = np.asarray(ctx["pmin"], np.float32)
        pmax = np.asarray(ctx["pmax"], np.float32)
        idx = MD.load_task_index(task, a.hdf5_root)
        eps = [str(x) for x in z[task + "/episodes"]]
        ts = np.asarray(z[task + "/t"], np.int64)
        pred = np.asarray(z[task + "/pred_bins"], np.int64)
        true_bins = np.asarray(z[task + "/bins_true"], np.int64)
        cf_key = "x|%s|%s" % (task, [t for t in tasks if t != task][0]) if len(tasks) > 1 else None
        cf = None
        if cf_key and cf_key in z.files:
            raw_cf = np.asarray(z[cf_key])
            # the counterfactual block stores the full 256-bin DISTRIBUTIONS, not the bins
            if raw_cf.size == len(eps) * 14 * 256:
                cf = raw_cf.reshape(len(eps), 14, 256).argmax(axis=-1).astype(np.int64)
            elif raw_cf.size == len(eps) * 14:
                cf = raw_cf.reshape(len(eps), 14).astype(np.int64)
            else:
                raise SystemExit("unexpected counterfactual block shape %s" % (raw_cf.shape,))

        rows = []
        for i, (ep, t) in enumerate(zip(eps, ts)):
            with h5py.File(os.path.join(idx["hdf5_dir"], ep), "r") as h:
                S = TC.read14(h, "state"); A = TC.read14(h, "action")
            S = np.asarray(S, np.float32); A = np.asarray(A, np.float32)
            t = int(t)
            v = S[t] - S[t - 1]
            r_true = (A[t] - S[t]) - v
            r_hat = codec_residual(ctx, pred[i])
            raw = S[t] + v + r_hat
            safe = np.clip(raw, pmin, pmax)
            noop = S[t]
            cv = S[t] + v
            zero = np.zeros_like(A[t])
            rec = {"episode": ep, "t": t,
                   "l1_safe_model": float(np.abs(safe - A[t]).mean()),
                   "l1_raw_model": float(np.abs(raw - A[t]).mean()),
                   "l1_noop": float(np.abs(noop - A[t]).mean()),
                   "l1_cv": float(np.abs(cv - A[t]).mean()),
                   "l1_zero_position": float(np.abs(zero - A[t]).mean()),
                   "resid_mag_model": float(np.abs(r_hat).mean()),
                   "resid_mag_true": float(np.abs(r_true).mean()),
                   "resid_r2": float(1.0 - ((r_hat - r_true) ** 2).sum() / max(((r_true - r_true.mean()) ** 2).sum(), 1e-12)),
                   "clip_rate": float((np.abs(raw - safe) > 1e-7).mean()),
                   "speed_mean_abs": float(np.abs(v).mean()),
                   "groups_model": groups_l1(safe, A[t]),
                   "groups_noop": groups_l1(noop, A[t]),
                   "groups_cv": groups_l1(cv, A[t]),
                   "bin_hit_rate": float((pred[i] == true_bins[i]).mean())}
            if cf is not None:
                cf_raw = S[t] + v + codec_residual(ctx, cf[i])
                cf_safe = np.clip(cf_raw, pmin, pmax)
                rec["l1_safe_wrong_prompt"] = float(np.abs(cf_safe - A[t]).mean())
                rec["counterfactual_changes_bins"] = bool((cf[i] != pred[i]).any())
            rows.append(rec)

        def agg(key):
            vals = np.array([r[key] for r in rows], float)
            return {"mean": float(vals.mean()), "median": float(np.median(vals)),
                    "ci95": bootstrap_ci(vals)}
        # motion strata by the frame's mean |V|
        speed = np.array([r["speed_mean_abs"] for r in rows])
        order = np.argsort(speed)
        half = max(1, len(rows) // 2)
        slow, fast = order[:half], order[half:]

        def strat(key, sel):
            return float(np.mean([rows[i][key] for i in sel])) if len(sel) else None

        summary = {
            "n_frames": len(rows),
            "model_safe_l1": agg("l1_safe_model"), "model_raw_l1": agg("l1_raw_model"),
            "noop_l1": agg("l1_noop"), "cv_l1": agg("l1_cv"),
            "zero_position_l1_legacy": agg("l1_zero_position"),
            "paired_model_minus_noop": {"mean": float(np.mean([r["l1_safe_model"] - r["l1_noop"] for r in rows])),
                                        "ci95": bootstrap_ci([r["l1_safe_model"] - r["l1_noop"] for r in rows])},
            "paired_model_minus_cv": {"mean": float(np.mean([r["l1_safe_model"] - r["l1_cv"] for r in rows])),
                                      "ci95": bootstrap_ci([r["l1_safe_model"] - r["l1_cv"] for r in rows])},
            "resid_mag_model": agg("resid_mag_model"), "resid_mag_true": agg("resid_mag_true"),
            "resid_r2": agg("resid_r2"), "clip_rate": agg("clip_rate"), "bin_hit_rate": agg("bin_hit_rate"),
            "groups": {g: {"model": float(np.mean([r["groups_model"][g] for r in rows])),
                           "noop": float(np.mean([r["groups_noop"][g] for r in rows])),
                           "cv": float(np.mean([r["groups_cv"][g] for r in rows]))}
                       for g in ("arm", "gripper")},
            "motion_strata": {"slow_frames": {"model": strat("l1_safe_model", slow),
                                              "noop": strat("l1_noop", slow),
                                              "cv": strat("l1_cv", slow)},
                              "fast_frames": {"model": strat("l1_safe_model", fast),
                                              "noop": strat("l1_noop", fast),
                                              "cv": strat("l1_cv", fast)}}}
        if cf is not None:
            summary["wrong_prompt"] = agg("l1_safe_wrong_prompt")
            summary["counterfactual_changed_bins_fraction"] = float(
                np.mean([1.0 if r["counterfactual_changes_bins"] else 0.0 for r in rows]))
        out["tasks"][task] = {"summary": summary, "frames": rows}
        print("[audit] %-12s safe=%.4f raw=%.4f | noop=%.4f cv=%.4f zero=%.4f | resid model=%.4f true=%.4f "
              "| r2=%.3f clip=%.2f hit=%.2f"
              % (task, summary["model_safe_l1"]["mean"], summary["model_raw_l1"]["mean"],
                 summary["noop_l1"]["mean"], summary["cv_l1"]["mean"],
                 summary["zero_position_l1_legacy"]["mean"],
                 summary["resid_mag_model"]["mean"], summary["resid_mag_true"]["mean"],
                 summary["resid_r2"]["mean"], summary["clip_rate"]["mean"],
                 summary["bin_hit_rate"]["mean"]), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
