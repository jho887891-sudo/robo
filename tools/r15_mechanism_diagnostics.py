#!/usr/bin/env python3
"""Why can this model not beat constant velocity on Push-T? (mechanism diagnostics, CPU only)

The gate compares a model against the constant-velocity predictor CV: A_hat = S_t + V_t, V_t = S_t - S_{t-1}.
That baseline has an exact algebraic meaning here, because the labels satisfy A[t] == S[t+1]:

    |CV - A_t| = |S_t + V_t - A_t| = |R_t|      with R_t = (A_t - S_t) - V_t

so the CV L1 IS the mean residual magnitude. Beating CV therefore requires predicting R from the
observation better than the trivial prediction R_hat = 0, and that is a falsifiable question the DEV
split can answer without any 7B model:

  * CV floor          : mean|R_t| per dimension (and per arm/gripper)
  * TRAIN-only fit    : an affine/MLP map features -> R fitted on TRAIN, evaluated on DEV (R^2 vs zero)
  * constant-R fit    : predicting the TRAIN mean of R as a one-number model
  * noise estimate    : the fraction of |R| that is not linearly predictable from the features

If the DEV R^2 of a TRAIN-only fit is negative, no amount of 7B capacity with these features can be
expected to beat CV, and G5 failing at 1000 steps is a DATA/RESIDUAL finding, not a training bug.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("training", "diagnostics", "code/xpolicylab/policy/OpenVLA_Pure"):
    _p = os.path.join(REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def blocks(states, actions, vels):
    """Per-group error decomposition of the residual R = (A - S) - V."""
    R = (actions - states) - vels
    out = {"all": float(np.abs(R).mean())}
    for name, dims in (("left_arm", range(0, 6)), ("left_gripper", [6]),
                       ("right_arm", range(7, 13)), ("right_gripper", [13])):
        out[name] = float(np.abs(R[:, list(dims)]).mean())
    return R, out


def r2(pred, true):
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_ridge(X, Y, ridge=1e-6):
    Xa = np.concatenate([X, np.ones((len(X), 1), np.float32)], axis=1)
    G = Xa.T @ Xa + ridge * np.eye(Xa.shape[1], dtype=np.float32)
    return np.linalg.solve(G, Xa.T @ Y), int(Xa.shape[1])


def apply_ridge(W, X):
    return np.concatenate([X, np.ones((len(X), 1), np.float32)], axis=1) @ W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-episode", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import r15_dev_eval as E
    contract, meta, splits = E.load_splits()
    nb = np.load(E.BOUNDS)
    bounds = (nb["amin"].astype(np.float32), nb["amax"].astype(np.float32))

    dev_frames = E.frozen_frames(sorted(splits["dev"]), a.per_episode, a.seed)
    feats, st, vel, act = E.build_samples(dev_frames, contract)
    train_frames = E.frozen_frames(sorted(splits["train"])[:32], 8, seed=a.seed + 1)
    ftr, str_, vtr, atr = E.build_samples(train_frames, contract)
    if set(e for e, _t in dev_frames) & set(e for e, _t in train_frames):
        raise SystemExit("TRAIN/DEV frame overlap")

    R_dev, dev_blocks = blocks(st, act, vel)
    R_tr, tr_blocks = blocks(str_, atr, vtr)

    rep = {"n_dev_frames": len(dev_frames), "n_train_frames": len(train_frames),
           "eval_version": E.EVAL_VERSION, "metric_version": E.METRIC_VERSION,
           "cv_identity": "CV L1 == mean|R_t| exactly, because A[t] == S[t+1] and CV = S_t + V_t",
           "cv_floor_dev": dev_blocks, "cv_floor_train": tr_blocks,
           "dev_frames_sha256": E.frames_sha256(dev_frames)}

    # 1) constant predictor: predict the TRAIN mean residual, every frame
    mean_R = R_tr.mean(axis=0)
    const_pred = np.tile(mean_R, (len(R_dev), 1))
    l1_const = float(np.abs((st + vel + const_pred) - act).mean())
    rep["train_mean_residual_model"] = {"l1_dev": l1_const, "r2_dev": r2(const_pred, R_dev),
                                        "cv_l1_dev": dev_blocks["all"],
                                        "beats_cv": bool(l1_const < dev_blocks["all"])}

    # 2) linear map features -> R, fitted on TRAIN only
    W, npar = fit_ridge(ftr, R_tr)
    pred_R = apply_ridge(W, feats)
    l1_lin = float(np.abs((st + vel + pred_R) - act).mean())
    rep["train_only_linear_residual"] = {
        "n_features": int(ftr.shape[1]), "n_params": npar,
        "l1_dev": l1_lin, "r2_dev": r2(pred_R, R_dev),
        "cv_l1_dev": dev_blocks["all"], "beats_cv": bool(l1_lin < dev_blocks["all"])}

    # 3) per-dimension: is the residual predictable at all, and is it inside the joint box?
    per_dim = []
    for d in range(14):
        per_dim.append({"dim": d,
                        "mean_abs_R_dev": float(np.abs(R_dev[:, d]).mean()),
                        "std_R_train": float(R_tr[:, d].std()),
                        "r2_linear": r2(pred_R[:, d], R_dev[:, d])})
    rep["per_dim"] = per_dim
    rep["labelled_action_oob"] = {
        "coord_rate": float(((act < bounds[0] - 1e-6) | (act > bounds[1] + 1e-6)).mean()),
        "frames": int(((act < bounds[0] - 1e-6) | (act > bounds[1] + 1e-6)).any(axis=1).sum()),
        "note": "even the human expert labels are (rarely) outside the frozen joint box; this is why a "
                "perfect prediction scores 4e-6 under the SAFE clamp instead of exactly 0"}

    if a.out:
        d = os.path.dirname(os.path.abspath(a.out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(rep, fh, indent=1)
    print("CV floor (== mean|R|), DEV    %.6f  (train %.6f)" % (dev_blocks["all"], tr_blocks["all"]))
    print("  left_arm %.6f  left_grip %.6f  right_arm %.6f  right_grip %.6f"
          % (dev_blocks["left_arm"], dev_blocks["left_gripper"],
             dev_blocks["right_arm"], dev_blocks["right_gripper"]))
    print("TRAIN-mean residual model    L1 %.6f  R2 %+.4f  beats CV: %s"
          % (l1_const, rep["train_mean_residual_model"]["r2_dev"],
             rep["train_mean_residual_model"]["beats_cv"]))
    print("TRAIN-only linear R model    L1 %.6f  R2 %+.4f  beats CV: %s"
          % (l1_lin, rep["train_only_linear_residual"]["r2_dev"],
             rep["train_only_linear_residual"]["beats_cv"]))
    print("expert labels outside box    coord-rate %.5f over %d frames"
          % (rep["labelled_action_oob"]["coord_rate"], rep["labelled_action_oob"]["frames"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
