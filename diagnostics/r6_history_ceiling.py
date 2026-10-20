#!/usr/bin/env python3
"""R6: evaluate selected history-only predictors, NOT an information ceiling (CPU).

Phase 1 showed a per-dimension affine map of V_t cannot beat plain V_t. This script
pushes harder: k-nearest-neighbour regression and a quadratic-feature ridge on the 14D
past motion, fitted on TRAIN frames and evaluated on VAL frames, against the trivial
baselines. If tested models fail to beat copy-V, this establishes a strong
observed benchmark, NOT a mathematical or information-theoretic ceiling.

kNN, joint-linear, and quadratic ridge are evaluated on a deterministic 2,000-frame
subsample of the 6,711 validation frames. The per-dimension linear is also scored
on all held-out frames separately, never directly compared to the 2,000-frame CV.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import r6_data as D  # noqa: E402

KNN_KS = (1, 5, 10, 25)
KNN_VAL_FRAMES = 2000
CHUNK = 32


def collect(episodes):
    V, Dlt, eps = [], [], []
    for ep in D.load_episodes(episodes):
        t = D.valid_frames(len(ep["state"]))
        V.append(ep["state"][t] - ep["state"][t - 1])
        Dlt.append(ep["action"][t] - ep["state"][t])
        eps += [ep["episode"]] * len(t)
    return np.concatenate(V), np.concatenate(Dlt), np.array(eps)


def knn_predict(V_tr, D_tr, V_va, k=10):
    out = np.empty_like(V_va)
    for i in range(0, len(V_va), CHUNK):
        block = V_va[i:i + CHUNK]
        d2 = ((block[:, None, :] - V_tr[None, :, :]) ** 2).sum(axis=2)
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        out[i:i + CHUNK] = D_tr[idx].mean(axis=1)
    return out


def quadratic(V):
    feats = [V]
    for sl in (slice(0, 7), slice(7, 14)):
        block = V[:, sl]
        feats.append(np.einsum("ni,nj->nij", block, block).reshape(len(V), -1))
    return np.concatenate(feats, axis=1)


def ridge_fit_predict(Xtr, Ytr, Xva, lam=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = np.concatenate([np.ones((len(Xtr), 1)), (Xtr - mu) / sd], axis=1)
    B = np.concatenate([np.ones((len(Xva), 1)), (Xva - mu) / sd], axis=1)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Ytr)
    return B @ W


def run():
    split = D.load_split()
    V_tr, D_tr, _ = collect(split["train"])
    V_va, D_va, _ = collect(split["val"])
    keep = np.linspace(0, len(V_va) - 1, min(KNN_VAL_FRAMES, len(V_va))).astype(np.int64)
    V_knn, D_knn = V_va[keep], D_va[keep]
    print("train frames %d | val frames %d (kNN on %d)" % (len(V_tr), len(V_va), len(V_knn)), flush=True)

    res = {"n_train_frames": int(len(V_tr)), "n_val_frames": int(len(V_va)),
           "n_knn_eval_frames": int(len(V_knn)),
           "baselines": {"hold_state_zero_delta": float(np.abs(D_knn).mean()),
                         "constant_velocity_copy_V": float(np.abs(V_knn - D_knn).mean()),
                         "hold_state_zero_delta_all_val": float(np.abs(D_va).mean()),
                         "constant_velocity_copy_V_all_val": float(np.abs(V_va - D_va).mean())},
           "models": {}}
    lin = np.zeros_like(D_va)
    for d in range(14):
        x = V_tr[:, d]
        if x.var() <= 0:
            lin[:, d] = D_tr[:, d].mean()
            continue
        a = float(np.cov(x, D_tr[:, d], bias=True)[0, 1] / x.var())
        b = float(D_tr[:, d].mean() - a * x.mean())
        lin[:, d] = a * V_va[:, d] + b
    res["models"]["per_dim_linear_all_val"] = float(np.abs(lin - D_va).mean())
    lin_knn = lin[keep]
    res["models"]["per_dim_linear_knn_set"] = float(np.abs(lin_knn - D_knn).mean())
    print("  per-dim linear done", flush=True)
    for k in KNN_KS:
        pred = knn_predict(V_tr, D_tr, V_knn, k=k)
        res["models"]["knn_%d" % k] = float(np.abs(pred - D_knn).mean())
        print("  knn_%d done -> %.6f" % (k, res["models"]["knn_%d" % k]), flush=True)
    quad = ridge_fit_predict(quadratic(V_tr), D_tr, quadratic(V_knn))
    res["models"]["quadratic_ridge"] = float(np.abs(quad - D_knn).mean())
    print("  quadratic ridge done", flush=True)
    joint = ridge_fit_predict(V_tr, D_tr, V_knn)
    res["models"]["joint_linear_14x14"] = float(np.abs(joint - D_knn).mean())

    cv = res["baselines"]["constant_velocity_copy_V"]
    # Compare models on identical 2,000 held-out frames, not mixed validation sets.
    same_sample_models = {k: v for k, v in res["models"].items()
                          if k != "per_dim_linear_all_val"}
    best = min(same_sample_models, key=same_sample_models.get)
    res["best_model"] = best
    res["best_model_L1"] = same_sample_models[best]
    res["best_model_eval_n"] = int(len(V_knn))
    res["constant_velocity_L1_knn_set"] = cv
    res["best_vs_copy_velocity_ratio"] = same_sample_models[best] / (cv + 1e-12)
    res["information_ceiling_proven"] = False
    res["interpretation"] = (
        "none of these tested models beats copy-V on the matched validation subset"
        if res["best_vs_copy_velocity_ratio"] >= 1.0 else
        "at least one tested model beats copy-V on the matched validation subset")

    m = np.abs(D_va).mean(axis=1) >= 5e-3
    mv, md = V_va[m][:400], D_va[m][:400]
    res["motion_rich_subset"] = {
        "n": int(m.sum()), "share": float(m.mean()),
        "hold": float(np.abs(D_va[m]).mean()),
        "cv": float(np.abs(V_va[m] - D_va[m]).mean()),
        "knn_10_on_first_400": float(np.abs(knn_predict(V_tr, D_tr, mv, k=10) - md).mean())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = run()
    json.dump(res, open(args.out, "w"), indent=1)
    b = res["baselines"]
    print("baselines on kNN set: hold=%.6f  copy_V=%.6f" % (b["hold_state_zero_delta"],
                                                           b["constant_velocity_copy_V"]))
    for k, v in sorted(res["models"].items(), key=lambda kv: kv[1]):
        if k == "per_dim_linear_all_val":
            print("  %-26s %.6f  (full heldout set; NOT same subset as copy-V below)" % (k, v))
        else:
            print("  %-26s %.6f  (x copy_V = %.3f)" % (k, v, v / (b["constant_velocity_copy_V"] + 1e-12)))
    print("best=%s %.6f -> %s" % (res["best_model"], res["best_model_L1"], res["interpretation"]))
    m = res["motion_rich_subset"]
    print("motion-rich subset n=%d (%.1f%%): hold=%.6f cv=%.6f knn10=%.6f" % (
        m["n"], 100 * m["share"], m["hold"], m["cv"], m["knn_10_on_first_400"]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
