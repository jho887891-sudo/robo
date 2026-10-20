#!/usr/bin/env python3
"""Freeze the R8 normalization contract into checkpoint metadata (CPU, no model).

Input : workstream-A `norm_contracts.json` (scheme -> NormalizationContract metadata)
        + the TRAIN episodes (for the R-target q01/q99)
Output: a single JSON blob carrying everything the training run and the deployment
        path must agree on (proprio mode/dim/layout, normalization mode + scales,
        target representation, quantiles, gripper policy).

The scheme is selected by DEV in workstream A; this tool only serializes the winner.
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
import r6_data as D          # noqa: E402
import r8_contract as C      # noqa: E402
import r8_vla_adapter as A   # noqa: E402
from r7_state_mlp import split_episodes  # noqa: E402
from r8_normalization import NormalizationContract  # noqa: E402


def select_quantile_train_episodes(split, limit=0):
    """Use the SAME 64/16/20 partition as R7/R8, never silently include DEV."""
    train, dev = split_episodes(split["train"], n_dev=16)
    if set(train) & set(dev) or set(train) & set(split["val"]) or set(dev) & set(split["val"]):
        raise ValueError("TRAIN/DEV/holdout overlap in frozen-contract split")
    if limit < 0 or limit > len(train):
        raise ValueError("episode limit must be 0..%d (0 means ALL 64 TRAIN)" % len(train))
    return (train[:limit] if limit else train), dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", required=True, help="workstream-A norm_contracts.json")
    ap.add_argument("--scheme", default="B99_signed")
    ap.add_argument("--episodes", type=int, default=0,
                    help="limit 64 TRAIN-only episodes after 64/16/20 split (0 = all 64)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    blob = json.load(open(args.contracts))
    if args.scheme not in blob:
        raise SystemExit("scheme %r not in %s (have %s)" % (args.scheme, args.contracts, sorted(blob)))
    contract = NormalizationContract.from_metadata(blob[args.scheme])
    split = D.load_split()
    eps, dev = select_quantile_train_episodes(split, args.episodes)
    S, Aa, V = [], [], []
    for name in eps:
        ep = D.load_episode(name)
        t = D.valid_frames(len(ep["state"]))
        t = t[t >= 2]
        S.append(ep["state"][t]); Aa.append(ep["action"][t])
        V.append(ep["state"][t] - ep["state"][t - 1])
    q01, q99 = C.velocity_residual_quantiles(np.concatenate(S), np.concatenate(Aa),
                                             np.concatenate(V))
    meta = A.metadata(contract, q01, q99)
    meta["frozen_from"] = {"contracts_file": os.path.abspath(args.contracts),
                          "scheme": args.scheme,
                          "n_train_episodes_for_quantiles": len(eps),
                           "n_dev_episodes_excluded": len(dev),
                           "train_episode_names": list(eps)}
    with open(args.out, "w") as fh:
        json.dump(meta, fh, indent=1)
    print("frozen contract ->", args.out)
    print("  normalization_mode:", meta["normalization_mode"], "quantile:", meta.get("motion_quantile"))
    print("  proprio_dim:", meta["proprio_dim"], "target:", meta["target_representation"])
    print("  q01[:4]:", np.round(q01[:4], 6).tolist(), " q99[:4]:", np.round(q99[:4], 6).tolist())


if __name__ == "__main__":
    main()
