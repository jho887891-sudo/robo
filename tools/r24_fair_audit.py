#!/usr/bin/env python3
"""R24 Stage B: a FAIR physical comparison of model / no-op / CV (supersedes the v1 audit).

The R17 audit (PR#30) found a real flaw in tools/m2_physical_audit.py: the MODEL's action was clipped to
[amin, amax] while the no-op and constant-velocity baselines were scored UNCLIPPED, and the model was run
through the 256-bin codec while the baselines were exact reals, and the deployment gripper fallback
(cmd[6] = cmd[13] = S[d] + V[d], i.e. the residual is IGNORED on the gripper dims) was not applied to
anything. Three different postprocessings were therefore compared as if they were one.

This tool has no re-implementation to drift: every policy's command is produced by calling the SAME
deployment function, r42_adapter.decode_command(), with the SAME metadata, codec and bounds. The only
thing that differs is the residual each policy predicts:

    model : R = codec_inverse(predicted 14 bins)
    CV    : R = 0                      (A_hat = S + V, the constant-velocity extrapolator)
    no-op : R = -V                     (A_hat = S, hold the current pose)

Consequences that are REPORTED rather than hidden:
  * because of the gripper fallback, the "no-op" policy is not exactly A = S on the gripper dims: it
    commands S + V there, exactly as the model and CV do. That is the deployment contract, not a bug,
    and it is why the old A = S comparison was not the baseline the robot would actually execute.
  * rate limiting is NOT evaluated: it needs prev_cmd and dt, which an offline archive does not have.
    The report says so instead of silently assuming it does nothing.
  * every policy is reported in BOTH forms: RAW (before the clip) and SAFE (after it), so a ranking that
    only survives under clipping is visible as such.
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

import m2_task_normalizer as TN     # noqa: E402
import m2_multitask_data as MD      # noqa: E402
import r42_adapter as R             # noqa: E402

GRIP = tuple(R.GRIPPER_DIMS)


def bins_for_residual(ctx, residual):
    _idx, ids = R.token_ids_for_residual(np.asarray(residual, np.float32),
                                         np.asarray(ctx["q01"], np.float32),
                                         np.asarray(ctx["q99"], np.float32))
    return np.asarray(ids, np.int64) - R.TOKEN_BEGIN_IDX - 1


def decode_policy(ctx, limits, gripper_dims, bins, s, v):
    """The ONE deployment path: r42_adapter.decode_command.

    It returns (cmd, r_hat, info), NOT (raw, safe, info). An earlier version of this tool unpacked the
    residual as if it were the SAFE command, which made every policy look like it had L1 ~0.38 (it was
    comparing a residual vector to an action). RAW is therefore reconstructed from the returned r_hat by
    the same documented formula the function itself uses, and SAFE is the function's own clipped cmd -
    both carry the gripper fallback, so the three policies still share one postprocessing.
    """
    cmd, r_hat, info = R.decode_command(np.asarray(bins, np.int64), ctx["q01"], ctx["q99"], s, v,
                                        limits=limits, prev_cmd=None, dt=None,
                                        gripper_dims=gripper_dims, on_unsafe="raise")
    raw = (np.asarray(s, np.float32) + np.asarray(v, np.float32) + r_hat).astype(np.float32)
    for d in gripper_dims:                     # the deployment gripper fallback, on the raw too
        raw[d] = s[d] + v[d]
    return np.asarray(raw, np.float32), np.asarray(cmd, np.float32), r_hat, info


def boot(xs, n=4000, seed=0):
    xs = np.asarray(xs, float)
    if xs.size < 2:
        return [float("nan"), float("nan")]
    rs = np.random.RandomState(seed)
    m = [rs.choice(xs, xs.size, replace=True).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True,
                    help="npz with <task>|<split>/{pred_bins,pred_probs,bins_true,episodes,t}")
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--normalizer", default=os.path.join(REPO, "data", "m2_task_normalizer_2task.json"))
    ap.add_argument("--metadata", default=None,
                    help="r42_metadata.json from the SEALED checkpoint; the bounds used here must be the "
                         "ones the robot actually gets. Defaults to the normalizer payload's own bounds.")
    ap.add_argument("--tasks", default="push_T,stack_bowls")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="sealed checkpoint dir; if it has no r42_metadata.json the bridge is tried and "
                         "its refusal (if any) is recorded in the report")
    ap.add_argument("--out", default=os.path.join(REPO, "mission", "r24", "fair_physical_audit.json"))
    a = ap.parse_args()

    import h5py
    import r42_train_contract as TC

    payload = json.load(open(a.normalizer))
    meta = json.load(open(a.metadata)) if a.metadata else None
    bridge_status = None
    if meta is None and a.checkpoint_dir:
        # Record WHY there is no sealed metadata for this checkpoint. The deploy model falls back to
        # r42_metadata_bridge when a checkpoint carries no r42_metadata.json, and that bridge is the
        # R15 SINGLE-task contract, so it refuses a multi-task payload. That is a real deployment gap
        # and it must be visible in the audit, not silently papered over.
        try:
            import r42_metadata_bridge as B
            _m, _r = B.build_stage_metadata(a.checkpoint_dir,
                                            task_id=str(payload.get("tasks") and
                                                        sorted(payload["tasks"])[0] or "push_T"))
            meta = _m
            bridge_status = {"bridge": "ok", "report": _r}
        except Exception as exc:                        # noqa: BLE001 - recorded verbatim
            bridge_status = {"bridge": "REFUSED", "error": repr(exc)}
    z = np.load(a.archive, allow_pickle=True)
    alignment = str(z["action_window_alignment"]) if "action_window_alignment" in z.files else None
    out = {"schema": "r24-fair-physical-audit-v2", "archive": os.path.abspath(a.archive),
           "archive_alignment": alignment,
           "checkpoint_global_step": int(z["checkpoint_global_step"])
           if "checkpoint_global_step" in z.files else None,
           "metadata_source": ("sealed checkpoint r42_metadata.json" if (meta and a.metadata) else
                                ("r42_metadata_bridge (built)" if (meta and bridge_status
                                 and bridge_status.get("bridge") == "ok") else
                                 "per-task normalizer payload pmin/pmax")),
           "metadata_bridge": bridge_status,
           "compares": "model vs no-op vs CV, ALL through r42_adapter.decode_command",
           "gripper_dims": list(GRIP),
           "not_evaluated": ["rate limiting (needs prev_cmd + dt, unavailable offline)"],
           "tasks": {}}

    for task in [t.strip() for t in a.tasks.split(",") if t.strip()]:
        ctx = TN.decode_context(payload, task)
        if meta is not None:
            nm = R.normalizer_for_task(meta["normalizer"], task) if meta["normalizer"].get("tasks") \
                else meta["normalizer"]
            if not np.allclose(nm["q01"], ctx["q01"]) or not np.allclose(nm["q99"], ctx["q99"]):
                raise SystemExit("%s: the eval quantiles disagree with the sealed checkpoint metadata; "
                                 "refusing to compare" % task)
            limits = meta["safety_limits"]
            gripper_dims = tuple(meta["gripper_policy"]["gripper_dims"])
        else:
            limits = {"amin": list(map(float, ctx["pmin"])), "amax": list(map(float, ctx["pmax"]))}
            gripper_dims = GRIP
        lo = np.asarray(limits["amin"], np.float32)
        hi = np.asarray(limits["amax"], np.float32)
        idx = MD.load_task_index(task)
        out["tasks"][task] = {"bounds_used": {"amin": lo.tolist(), "amax": hi.tolist()},
                              "gripper_dims": list(gripper_dims), "splits": {}}
        for split in a.splits:
            # two archive layouts exist: the fit archive uses "<task>|<split>", the R16 regenerated
            # DEV archive uses a bare "<task>" for its dev block
            candidates = ["%s|%s" % (task, split)]
            if split == "dev":
                candidates.append(task)
            key = next((k for k in candidates if k + "/pred_bins" in z.files), None)
            if key is None:
                out["tasks"][task]["splits"][split] = {"MISSING": True, "looked_for": candidates}
                continue
            eps = [str(x) for x in z[key + "/episodes"]]
            ticks = np.asarray(z[key + "/t"], np.int64)
            pred = np.asarray(z[key + "/pred_bins"], np.int64)
            rows = []
            cache = {}
            for i, (ep, tk) in enumerate(zip(eps, ticks)):
                if ep not in cache:
                    with h5py.File(os.path.join(idx["hdf5_dir"], ep), "r") as h:
                        cache[ep] = (np.asarray(TC.read14(h, "state"), np.float32),
                                     np.asarray(TC.read14(h, "action"), np.float32))
                S, A = cache[ep]
                tk = int(tk)
                s, v, truth = S[tk], S[tk] - S[tk - 1], A[tk]
                rec = {"episode": ep, "t": tk}
                for name, bins in (("model", pred[i]),
                                   ("cv", bins_for_residual(ctx, np.zeros(14, np.float32))),
                                   ("noop", bins_for_residual(ctx, -v))):
                    raw, safe, _rhat, info = decode_policy(ctx, limits, gripper_dims, bins, s, v)
                    rec[name] = {
                        "raw_l1": float(np.abs(raw - truth).mean()),
                        "safe_l1": float(np.abs(safe - truth).mean()),
                        "raw_l1_arm": float(np.abs(raw[[d for d in range(14) if d not in gripper_dims]]
                                                   - truth[[d for d in range(14)
                                                            if d not in gripper_dims]]).mean()),
                        "safe_l1_grip": float(np.abs(safe[list(gripper_dims)]
                                                     - truth[list(gripper_dims)]).mean()),
                        "clipped_dims": int(info["clipped_joint"])}
                rows.append(rec)

            def agg(name, field):
                v = np.array([r[name][field] for r in rows], float)
                return {"mean": float(v.mean()), "median": float(np.median(v)), "ci95": boot(v)}

            summary = {"n_frames": len(rows)}
            for name in ("model", "cv", "noop"):
                summary[name] = {"raw": agg(name, "raw_l1"), "safe": agg(name, "safe_l1"),
                                 "arm_raw": agg(name, "raw_l1_arm"),
                                 "grip_safe": agg(name, "safe_l1_grip"),
                                 "clip_dims_mean": float(np.mean([r[name]["clipped_dims"]
                                                                  for r in rows]))}
            # paired deltas, computed on SAFE (the command the robot receives)
            for ref in ("noop", "cv"):
                d = [r["model"]["safe_l1"] - r[ref]["safe_l1"] for r in rows]
                summary["paired_model_minus_%s_safe" % ref] = {"mean": float(np.mean(d)),
                                                               "ci95": boot(d)}
                dr = [r["model"]["raw_l1"] - r[ref]["raw_l1"] for r in rows]
                summary["paired_model_minus_%s_raw" % ref] = {"mean": float(np.mean(dr)),
                                                              "ci95": boot(dr)}
            out["tasks"][task]["splits"][split] = summary
            print("[fair] %-12s %-5s n=%-3d SAFE model %.4f | no-op %.4f | CV %.4f || RAW model "
                  "%.4f | no-op %.4f | CV %.4f"
                  % (task, split, len(rows),
                     summary["model"]["safe"]["mean"], summary["noop"]["safe"]["mean"],
                     summary["cv"]["safe"]["mean"],
                     summary["model"]["raw"]["mean"], summary["noop"]["raw"]["mean"],
                     summary["cv"]["raw"]["mean"]), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, a.out)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
