#!/usr/bin/env python3
"""R24: per-stratum physical audit joined with the frozen manifest.

The closed loop starts from REST. If a policy cannot produce motion there, it can never start a push, no
matter how good its steady-state tracking is. This tool therefore reports the fair SAFE-14D comparison
separately per manifest stratum (still / slow_motion / fast_motion / +large_residual), which is the
offline quantity that actually predicts the closed-loop freeze.

CPU only; reads the archive a model produced plus the frozen manifest.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _s in ("training", "code/xpolicylab/policy/OpenVLA_Pure",
           "code/xpolicylab/policy/OpenVLA_R42"):
    sys.path.insert(0, os.path.join(REPO, _s))

import m2_train_shared as M2        # noqa: E402
import m2_multitask_data as MD      # noqa: E402
import m2_task_normalizer as TN     # noqa: E402
import r42_adapter as R             # noqa: E402

GRIP = tuple(R.GRIPPER_DIMS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.path.join(REPO, "mission", "r24", "fit_step2000.npz"))
    ap.add_argument("--manifest", default=os.path.join(REPO, "mission", "r24", "frame_manifest.json"))
    ap.add_argument("--normalizer", default=os.path.join(REPO, "data", "m2_task_normalizer_2task.json"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default=os.path.join(REPO, "mission", "r24", "stratum_audit.json"))
    a = ap.parse_args()

    import h5py
    import r42_train_contract as TC

    man = json.load(open(a.manifest))
    payload = json.load(open(a.normalizer))
    z = np.load(a.archive, allow_pickle=True)
    out = {"schema": "r24-stratum-audit-v1", "frame_manifest_sha256": man["manifest_sha256"],
           "archive": os.path.abspath(a.archive), "split": a.split, "tasks": {}}

    for task in man["tasks"]:
        ctx = TN.decode_context(payload, task)
        q01 = np.asarray(ctx["q01"], np.float64)
        q99 = np.asarray(ctx["q99"], np.float64)
        limits = {"amin": list(map(float, ctx["pmin"])), "amax": list(map(float, ctx["pmax"]))}
        key = "%s|%s" % (task, a.split)
        if key + "/pred_bins" not in z.files:
            out["tasks"][task] = {"MISSING": key}
            continue
        frames = man["frames"][task][a.split]
        pred = np.asarray(z[key + "/pred_bins"], np.int64)
        eps = [str(x) for x in z[key + "/episodes"]]
        ticks = [int(x) for x in z[key + "/t"]]
        assert [(f["episode"], int(f["t"])) for f in frames] == list(zip(eps, ticks)), \
            "the archive frame order does not match the frozen manifest; refusing to join"
        idx = MD.load_task_index(task)
        cache = {}
        rows = []
        for i, f in enumerate(frames):
            ep, t = f["episode"], int(f["t"])
            if ep not in cache:
                with h5py.File(os.path.join(idx["hdf5_dir"], ep), "r") as h:
                    cache[ep] = (np.asarray(TC.read14(h, "state"), np.float32),
                                 np.asarray(TC.read14(h, "action"), np.float32))
            S, A = cache[ep]
            s, v, truth = S[t], S[t] - S[t - 1], A[t]
            rec = {"stratum": f["stratum"], "speed": f["speed"], "resid_mag": f["resid_mag"]}
            for name, res in (("model", None),
                              ("cv", np.zeros(14, np.float32)),
                              ("noop", -v)):
                if name == "model":
                    _i, ids = R.token_ids_for_residual(
                        np.zeros(14, np.float32), q01, q99)   # placeholder, replaced below
                    bins = pred[i]
                else:
                    _i, ids = R.token_ids_for_residual(np.asarray(res, np.float32), q01, q99)
                    bins = np.asarray(ids, np.int64) - R.TOKEN_BEGIN_IDX - 1
                cmd, _rh, _info = R.decode_command(bins, q01, q99, s, v, limits=limits,
                                                   prev_cmd=None, dt=None, gripper_dims=GRIP,
                                                   on_unsafe="raise")
                rec[name] = float(np.abs(cmd - truth).mean())
                # how much motion does the policy COMMAND relative to holding still?
                rec[name + "_cmd_delta_from_state"] = float(np.abs(cmd - s).mean())
            rows.append(rec)

        strata = sorted({r["stratum"] for r in rows})
        summ = {}
        for st in strata:
            sel = [r for r in rows if r["stratum"] == st]
            summ[st] = {k: float(np.mean([r[k] for r in sel]))
                        for k in ("model", "cv", "noop",
                                  "model_cmd_delta_from_state", "cv_cmd_delta_from_state",
                                  "noop_cmd_delta_from_state")}
            summ[st]["n"] = len(sel)
        out["tasks"][task] = {"strata": summ, "n_frames": len(rows), "frames": rows}
        print("[stratum] " + task)
        for st in strata:
            s = summ[st]
            print("   %-26s n=%-3d SAFE model %.5f | no-op %.5f | CV %.5f || commanded motion from "
                  "state: model %.5f CV %.5f (truth %.5f)"
                  % (st, s["n"], s["model"], s["noop"], s["cv"],
                     s["model_cmd_delta_from_state"], s["cv_cmd_delta_from_state"],
                     float(np.mean([r["resid_mag"] for r in rows if r["stratum"] == st]))))

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
