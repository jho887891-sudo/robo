#!/usr/bin/env python3
"""R8-A3: controlled comparison of motion-normalization schemes (CPU only).

Same split (64 train / 16 dev / 20 holdout), same 42D layout [S,V,Acc], same velocity-
residual target R, same MLP, optimizer, seeds and DEV-selection rule as the R8 coverage
ablation. Only the transform of the two motion slots changes:

  A        legacy_position  : V, Acc mapped with the POSITION bounds pmin/pmax and clipped
                              to [-1,1] (this is what r8_contract.build_proprio does today)
  A_noclip offset_only      : the same affine map WITHOUT clipping (diagnostic control:
                              isolates how much damage comes from clipping alone, since the
                              MLP re-standardises the inputs anyway)
  B99      signed_motion    : V -> v/max(q99(|v|),eps), Acc -> a/max(q99(|a|),eps), TRAIN only
  B999     signed_motion    : same with the 0.999 quantile

Selection is made on DEV only (mean over seeds). The 20 holdout episodes are reported as a
RETROSPECTIVE diagnostic and are never used to choose the scheme.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for sub in ("training", "diagnostics", "code/xpolicylab/policy/OpenVLA_Pure"):
    sys.path.insert(0, os.path.join(REPO, sub))
import r6_data as D                      # noqa: E402
import r7_metrics as M                   # noqa: E402
from r7_state_mlp import episode_frames, stack_frames, split_episodes   # noqa: E402
from r8_coverage_ablation import train_fixed                            # noqa: E402
from r8_normalization import (NormalizationContract, SIGNED_MOTION,     # noqa: E402
                              LEGACY_POSITION)

SCHEMES = ("A_legacy_position", "A_noclip_offset", "B99_signed", "B999_signed")


def motion_transform(scheme, state, velocity, acceleration, contract, pmin, pmax):
    s_n = contract.transform(state, np.zeros_like(velocity), np.zeros_like(acceleration))[:, :14]
    if scheme == "A_legacy_position":
        v = contract.__class__(LEGACY_POSITION, pmin, pmax).transform(
            state, velocity, acceleration)[:, 14:]
        return np.concatenate([s_n, v], axis=1)
    if scheme == "A_noclip_offset":
        raw_v = 2.0 * (velocity - pmin) / (pmax - pmin + 1e-8) - 1.0
        raw_a = 2.0 * (acceleration - pmin) / (pmax - pmin + 1e-8) - 1.0
        return np.concatenate([s_n, raw_v.astype(np.float32), raw_a.astype(np.float32)], axis=1)
    return contract.transform(state, velocity, acceleration)


def slot_floor_shares(x, slots=(slice(14, 28), slice(28, 42))):
    out = {}
    for name, sl in (("velocity", slots[0]), ("acceleration", slots[1])):
        block = x[:, sl]
        out[name] = {"floor_share": float(np.mean(block <= -1.0 + 1e-7)),
                     "ceiling_share": float(np.mean(block >= 1.0 - 1e-7)),
                     "distinct_values_mean": float(np.mean([len(np.unique(block[:, d]))
                                                            for d in range(14)]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.manual_seed(0)
    split = D.load_split()
    tr_eps, dev_eps = split_episodes(split["train"], n_dev=16)
    nb = np.load("/home/dgut/pusht_runtime/push_T_bounds.npz")
    pmin, pmax = nb["pmin"].astype(np.float32), nb["pmax"].astype(np.float32)

    blobs = {}
    for name, eps in (("train", tr_eps), ("dev", dev_eps), ("holdout", split["val"])):
        blobs[name] = stack_frames([episode_frames(D.load_episode(e), need_accel=True) for e in eps])
    R = {k: ((b["A"] - b["S"]) - b["V"]).astype(np.float32) for k, b in blobs.items()}
    contracts = {
        "A_legacy_position": NormalizationContract(LEGACY_POSITION, pmin, pmax),
        "A_noclip_offset": NormalizationContract(LEGACY_POSITION, pmin, pmax),
        "B99_signed": NormalizationContract.fit(blobs["train"]["S"], blobs["train"]["V"],
                                                blobs["train"]["Acc"], pmin, pmax,
                                                SIGNED_MOTION, 0.99),
        "B999_signed": NormalizationContract.fit(blobs["train"]["S"], blobs["train"]["V"],
                                                 blobs["train"]["Acc"], pmin, pmax,
                                                 SIGNED_MOTION, 0.999),
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out)),
                           "norm_contracts.json"), "w") as fh:
        json.dump({k: c.to_metadata() for k, c in contracts.items()}, fh, indent=1)

    cv_holdout = float(np.abs(blobs["holdout"]["S"] + blobs["holdout"]["V"]
                              - blobs["holdout"]["A"]).mean())
    result = {"schemes": {}, "cv_holdout_L1": cv_holdout,
              "n_frames": {k: int(len(b["S"])) for k, b in blobs.items()},
              "selection_rule": "lowest mean DEV physical L1 across seeds; holdout retrospective only"}
    print("[data] train=%d dev=%d holdout=%d | cv holdout=%.6f"
          % (len(blobs["train"]["S"]), len(blobs["dev"]["S"]),
             len(blobs["holdout"]["S"]), cv_holdout), flush=True)

    for scheme in SCHEMES:
        c = contracts[scheme]
        X = {k: motion_transform(scheme, blobs[k]["S"], blobs[k]["V"], blobs[k]["Acc"],
                                 c, pmin, pmax).astype(np.float32)
             for k in blobs}
        shares = {k: slot_floor_shares(X[k]) for k in X}
        dev_l1, hold = [], []
        per_seed = []
        for seed in args.seeds:
            net, st = train_fixed(X["train"], R["train"], X["dev"], R["dev"], seed,
                                  max_steps=None, epochs=args.epochs, select="dev_best",
                                  patience=40, torch=torch)
            out = {}
            for name in ("dev", "holdout"):
                R_hat = __import__("r7_state_mlp").predict(net, X[name], st, torch).astype(np.float32)
                cmd = blobs[name]["S"] + blobs[name]["V"] + R_hat
                ev = M.evaluate(cmd, blobs[name]["A"], blobs[name]["S"], blobs[name]["V"],
                                blobs[name]["episode"])
                ev["correction"] = M.evaluate_correction(R_hat, R[name])
                out[name] = ev
                if name == "holdout":
                    side = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                        "pred_norm_%s_seed%d.npz" % (scheme, seed))
                    np.savez_compressed(side, R_hat=R_hat, R_gt=R[name], cmd=cmd,
                                        gt=blobs[name]["A"], state=blobs[name]["S"],
                                        V=blobs[name]["V"], episode=blobs[name]["episode"])
            dev_l1.append(out["dev"]["L1"])
            hold.append(out["holdout"]["L1"])
            per_seed.append({"seed": int(seed), "dev_L1": out["dev"]["L1"],
                             "holdout_L1": out["holdout"]["L1"],
                             "holdout_R2": out["holdout"]["correction"]["R2_variance_explained"],
                             "holdout_cos": out["holdout"]["direction"]["cosine_mean"],
                             "holdout_sign_delta": out["holdout"]["direction"]["sign_agreement_at_1e-3"],
                             "holdout_wins": out["holdout"]["episodes_beating_cv"],
                             "steps": st["steps"], "epochs": st["epochs"]})
            print("[%s seed=%d] DEV L1=%.6f | holdout L1=%.6f R2=%.3f cos=%.4f sign=%.4f wins=%d/20"
                  % (scheme, seed, out["dev"]["L1"], out["holdout"]["L1"],
                     out["holdout"]["correction"]["R2_variance_explained"] or float("nan"),
                     out["holdout"]["direction"]["cosine_mean"] or float("nan"),
                     out["holdout"]["direction"]["sign_agreement_at_1e-3"] or float("nan"),
                     out["holdout"]["episodes_beating_cv"]), flush=True)
        result["schemes"][scheme] = {
            "dev_L1_mean": float(np.mean(dev_l1)), "dev_L1_min": float(np.min(dev_l1)),
            "dev_L1_max": float(np.max(dev_l1)),
            "holdout_L1_mean": float(np.mean(hold)), "holdout_L1_min": float(np.min(hold)),
            "holdout_L1_max": float(np.max(hold)),
            "holdout_ratio_vs_cv": float(np.mean(hold) / cv_holdout),
            "slot_floor_shares": shares,
            "contract": c.to_metadata(), "per_seed": per_seed}
        json.dump(result, open(args.out, "w"), indent=1)

    best = min(result["schemes"], key=lambda k: result["schemes"][k]["dev_L1_mean"])
    result["selected_by_dev"] = best
    result["selected_contract"] = result["schemes"][best]["contract"]
    json.dump(result, open(args.out, "w"), indent=1)
    print("\n=== DEV selection ===")
    for k in SCHEMES:
        s = result["schemes"][k]
        print("  %-20s DEV=%.6f [%.6f,%.6f] | holdout=%.6f (x cv %.3f) | V floor share=%.4f"
              % (k, s["dev_L1_mean"], s["dev_L1_min"], s["dev_L1_max"],
                 s["holdout_L1_mean"], s["holdout_ratio_vs_cv"],
                 s["slot_floor_shares"]["train"]["velocity"]["floor_share"]))
    print("selected by DEV:", best)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
