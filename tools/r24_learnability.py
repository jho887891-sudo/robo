#!/usr/bin/env python3
"""R24: is there anything to learn in the residual? (the H4 discriminator)

The action the R42 policy must produce is a residual around constant velocity:

    R_t = (A_t - S_t) - V_t

If R_t has a near-zero conditional mean given everything the policy can see, then the Bayes-optimal
L2/L1 predictor IS approximately zero, i.e. exactly the constant-velocity baseline. In that case no
number of extra optimisation steps or extra episodes can make the model beat CV, and the "under-training"
hypothesis (H1) is not the binding constraint.

This tool measures, on the same frozen manifest the model is evaluated on:
  1. R_true statistics: per-dim mean, std, and the magnitude a zero predictor still pays (|R|).
  2. A TRAIN-only linear map 42D features -> R, evaluated on DEV (R^2 and L1). If R^2 <= 0 the LINEAR
     signal in proprio is nil; combined with the model's own R^2 ~ 0 this bounds what 1-step imitation
     can achieve here.
  3. The "predict the TRAIN mean residual" predictor, scored in PHYSICAL L1 through the same codec and
     SAFE clip as everything else - this is the ceiling a policy that only learned the mean can reach.
CPU only; no GPU, no model, no training.
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
    ap.add_argument("--manifest", default=os.path.join(REPO, "mission", "r24", "frame_manifest.json"))
    ap.add_argument("--normalizer", default=os.path.join(REPO, "data", "m2_task_normalizer_2task.json"))
    ap.add_argument("--splits", default="data/m2_splits.json")
    ap.add_argument("--out", default=os.path.join(REPO, "mission", "r24", "residual_learnability.json"))
    a = ap.parse_args()

    import h5py
    import r42_train_contract as TC

    man = json.load(open(a.manifest))
    payload = json.load(open(a.normalizer))
    out = {"schema": "r24-residual-learnability-v1",
           "frame_manifest_sha256": man["manifest_sha256"],
           "question": ("does the residual have a conditional mean the policy could learn, or is a "
                        "zero (constant-velocity) prediction already the Bayes optimum?"),
           "tasks": {}}

    for task in man["tasks"]:
        idx = MD.load_task_index(task)
        ctx = TN.decode_context(payload, task)
        q01 = np.asarray(ctx["q01"], np.float64)
        q99 = np.asarray(ctx["q99"], np.float64)
        data = {}
        for split in ("train", "dev"):
            X, Y = [], []
            for rec in man["frames"][task][split]:
                ep, t = rec["episode"], int(rec["t"])
                with h5py.File(os.path.join(idx["hdf5_dir"], ep), "r") as h:
                    S = np.asarray(TC.read14(h, "state"), np.float32)
                    A = np.asarray(TC.read14(h, "action"), np.float32)
                v = S[t] - S[t - 1]
                r = (A[t] - S[t]) - v
                feat = np.concatenate([S[t], v, S[t] - S[t - 1]])
                feats = np.concatenate([S[t], v])          # 28D of state+velocity
                X.append(feats.astype(np.float64))
                Y.append(r.astype(np.float64))
            data[split] = (np.stack(X), np.stack(Y))

        Xtr, Ytr = data["train"]
        Xdv, Ydv = data["dev"]
        mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-9
        mu_y = Ytr.mean(0)
        # ridge-regularised linear map, TRAIN only
        Ztr = np.hstack([(Xtr - mu_x) / sd_x, np.ones((len(Xtr), 1))])
        Zdv = np.hstack([(Xdv - mu_x) / sd_x, np.ones((len(Xdv), 1))])
        lam = 1e-2
        W = np.linalg.solve(Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1]), Ztr.T @ Ytr)
        Pdv = Zdv @ W
        ss_res = float(((Ydv - Pdv) ** 2).sum())
        ss_tot = float(((Ydv - Ydv.mean(0)) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        mean_pred = np.repeat(mu_y[None, :], len(Ydv), axis=0)
        ss_res_mean = float(((Ydv - mean_pred) ** 2).sum())

        def phys_l1(res_pred):
            """Physical SAFE L1 of a residual predictor, through the same codec and clip."""
            tot = []
            for i, rec in enumerate(man["frames"][task]["dev"]):
                ep, t = rec["episode"], int(rec["t"])
                with h5py.File(os.path.join(idx["hdf5_dir"], ep), "r") as h:
                    S = np.asarray(TC.read14(h, "state"), np.float32)
                    A = np.asarray(TC.read14(h, "action"), np.float32)
                v = S[t] - S[t - 1]
                _idx, ids = R.token_ids_for_residual(np.asarray(res_pred[i], np.float32), q01, q99)
                bins = np.asarray(ids, np.int64) - R.TOKEN_BEGIN_IDX - 1
                cmd, _rh, _info = R.decode_command(
                    bins, q01, q99, S[t], v,
                    limits={"amin": list(map(float, ctx["pmin"])), "amax": list(map(float, ctx["pmax"]))},
                    prev_cmd=None, dt=None, gripper_dims=GRIP, on_unsafe="raise")
                tot.append(float(np.abs(cmd - A[t]).mean()))
            return float(np.mean(tot))

        out["tasks"][task] = {
            "n_train": len(Xtr), "n_dev": len(Xdv),
            "residual_stats_dev": {"mean_abs": float(np.abs(Ydv).mean()),
                                   "per_dim_mean": [round(float(x), 8) for x in Ydv.mean(0)],
                                   "per_dim_std": [round(float(x), 8) for x in Ydv.std(0)],
                                   "mean_abs_of_mean": float(np.abs(Ydv.mean(0)).mean())},
            "train_mean_residual": [round(float(x), 8) for x in mu_y],
            "linear_probe_dev_r2": float(r2),
            "mean_predictor_dev_ss_res_vs_linear": float(ss_res_mean / max(ss_res, 1e-30)),
            "physical_safe_l1": {"zero_residual_CV_equivalent": phys_l1(
                                     np.zeros((len(Xdv), 14), np.float32)),
                                 "train_mean_residual": phys_l1(mean_pred.astype(np.float32)),
                                 "linear_probe": phys_l1(Pdv.astype(np.float32))},
            "codec_effective_levels": float(
                np.abs(Ydv).mean() / float(((q99 - q01) / 256.0).mean()))}
        print("[learn] %-12s linear R2 %+0.4f | SAFE L1 zero %.5f  train-mean %.5f  linear %.5f"
              % (task, r2, out["tasks"][task]["physical_safe_l1"]["zero_residual_CV_equivalent"],
                 out["tasks"][task]["physical_safe_l1"]["train_mean_residual"],
                 out["tasks"][task]["physical_safe_l1"]["linear_probe"]), flush=True)

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
