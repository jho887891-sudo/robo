#!/usr/bin/env python3
"""R8 phase 2: matched data-coverage ablation on the corrected R7 protocol (CPU only).

Everything except the amount of training data is held fixed:

  episodes   64 train / 16 dev / 20 holdout   (identical to the R7 split, re-derived)
  input      [S_t, V_t, Acc_t]  (42D, past-only, t >= 2)
  target     R_t = (A_t - S_t) - V_t,  command = S + V + R_hat
  model      MLP 42-128-128-14, GELU, standardised inputs, physical L1 loss
  optimiser  Adam lr 1e-3, batch 512, seeds 0/1/2
  eval       the untouched 20 holdout episodes, 6,691 legal t>=2 frames

Varied: the number of DISTINCT training frames (200 / 2,000 / all 19,127), sampled from
the same 64 training episodes by episode coverage + motion-phase strata, and NESTED
(A subset of B subset of C) so the ladder has no sampling luck.

Three protocols, reported separately because they answer different questions:

  equal_updates : exactly U optimiser steps for every dataset size
  equal_epochs  : exactly E passes over the dataset (steps = E * ceil(n / batch))
  dev_selected  : up to 400 epochs, early stop on DEV (patience 40), best DEV checkpoint

This script is additive: it imports helpers from the frozen R7 modules and never edits
them.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "diagnostics"))
import r6_data as D                       # noqa: E402
import r7_metrics as M                    # noqa: E402
from r7_state_mlp import (episode_frames, stack_frames, build_xy,   # noqa: E402
                          split_episodes, make_mlp, predict)

N_DIMS = 14
MOTION_BINS = 4


# ------------------------------------------------------------------ sampling pool
def pool_from_episodes(names):
    """All legal t>=2 frames of the given episodes, with a motion score per frame."""
    rows = []
    for name in names:
        ep = D.load_episode(name)
        t = D.valid_frames(len(ep["state"]))
        t = t[t >= 2]
        S, A = ep["state"][t], ep["action"][t]
        V = ep["state"][t] - ep["state"][t - 1]
        Acc = V - (ep["state"][t - 1] - ep["state"][t - 2])
        motion = np.abs(A - S).mean(axis=1)
        rows.append({"episode": name, "frame": t, "S": S, "A": A, "V": V, "Acc": Acc,
                     "motion": motion})
    return rows


def stratified_order(rows, seed=0):
    """Coverage-first, motion-stratified, nesting-friendly frame order.

    Every outer pass gives each episode exactly one frame, and the frames inside an
    episode are ordered by a per-episode rotation over the motion strata, so a prefix
    of any size spreads over BOTH episodes and motion phases. Prefixes nest by
    construction (A subset of B subset of C).
    """
    rng = np.random.default_rng(seed)
    all_motion = np.concatenate([r["motion"] for r in rows])
    edges = np.quantile(all_motion, np.linspace(0, 1, MOTION_BINS + 1)[1:-1])
    episodes = sorted({r["episode"] for r in rows})
    by_ep = {r["episode"]: r for r in rows}
    rng.shuffle(episodes)
    per_episode = {}
    for e in episodes:
        r = by_ep[e]
        strata = np.digitize(r["motion"], edges)
        buckets = {s: [] for s in range(MOTION_BINS)}
        for i, sd in enumerate(strata):
            buckets[int(sd)].append(i)
        for k in buckets:
            rng.shuffle(buckets[k])
        rotation = list(range(MOTION_BINS))
        rng.shuffle(rotation)
        order_e = []
        ptr = {k: 0 for k in range(MOTION_BINS)}
        while len(order_e) < len(r["motion"]):
            moved = False
            for sd in rotation:
                if ptr[sd] < len(buckets[sd]):
                    order_e.append(buckets[sd][ptr[sd]])
                    ptr[sd] += 1
                    moved = True
                    if len(order_e) >= len(r["motion"]):
                        break
            if not moved:
                break
            rotation = rotation[1:] + rotation[:1]
        per_episode[e] = order_e
    total = sum(len(v) for v in per_episode.values())
    order, ptr = [], {e: 0 for e in episodes}
    while len(order) < total:
        for e in episodes:                       # one frame per episode per pass
            if ptr[e] < len(per_episode[e]):
                order.append((e, per_episode[e][ptr[e]]))
                ptr[e] += 1
    return order, edges


def take_prefix(rows, order, n):
    chosen = order[:n] if n is not None else order
    by_ep = {r["episode"]: r for r in rows}
    S = np.stack([by_ep[e]["S"][i] for e, i in chosen])
    A = np.stack([by_ep[e]["A"][i] for e, i in chosen])
    V = np.stack([by_ep[e]["V"][i] for e, i in chosen])
    Acc = np.stack([by_ep[e]["Acc"][i] for e, i in chosen])
    motion = np.stack([by_ep[e]["motion"][i] for e, i in chosen])
    episode = np.array([e for e, _ in chosen])
    frame = np.array([int(by_ep[e]["frame"][i]) for e, i in chosen])
    return {"S": S, "A": A, "V": V, "Acc": Acc, "motion": motion,
            "episode": episode, "frame": frame}


# ----------------------------------------------------------------------- training
def train_fixed(Xtr, Rtr, Xdev, Rdev, seed, *, max_steps=None, epochs=400, lr=1e-3,
                batch=512, patience=40, select="dev_best", torch=None):
    """Adam on physical L1(R_hat, R) with an explicit step/epoch budget."""
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr = torch.from_numpy(((Xtr - mu) / sd).astype(np.float32))
    Zdev = torch.from_numpy(((Xdev - mu) / sd).astype(np.float32))
    tR, dR = torch.from_numpy(Rtr), torch.from_numpy(Rdev)
    net = make_mlp(Xtr.shape[1], (128, 128), seed, torch)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 1)
    n = len(Ztr)
    steps_per_epoch = max(1, int(np.ceil(n / batch)))
    best = {"epoch": 0, "dev": float("inf"), "state": None}
    hist, steps, ep = [], 0, 0
    t0 = time.time()
    while ep < epochs:
        ep += 1
        net.train()
        perm = torch.randperm(n, generator=g)
        for start in range(0, n, batch):
            if max_steps is not None and steps >= max_steps:
                break
            idx = perm[start:start + batch]
            opt.zero_grad(set_to_none=True)
            loss = (net(Ztr[idx]) - tR[idx]).abs().mean()
            loss.backward()
            opt.step()
            steps += 1
        net.eval()
        with torch.no_grad():
            dev_l1 = float((net(Zdev) - dR).abs().mean())
        hist.append({"epoch": ep, "dev_L1": dev_l1, "steps": steps})
        if dev_l1 < best["dev"] - 1e-7:
            best = {"epoch": ep, "dev": dev_l1,
                    "state": {k: v.clone() for k, v in net.state_dict().items()}}
        elif select == "dev_best" and ep - best["epoch"] >= patience:
            break
        if max_steps is not None and steps >= max_steps:
            break
    if select == "dev_best":
        net.load_state_dict(best["state"])
    net.eval()
    return net, {"x_mean": mu, "x_std": sd, "steps": steps, "epochs": ep,
                 "steps_per_epoch": steps_per_epoch, "dev_best_L1": best["dev"],
                 "best_epoch": best["epoch"], "select": select,
                 "seconds": round(time.time() - t0, 1),
                 "history_tail": hist[-5:]}


def holdout_eval(net, stats, val_blob, torch):
    R_hat = predict(net, np.concatenate([val_blob["S"], val_blob["V"], val_blob["Acc"]], 1),
                    stats, torch).astype(np.float32)
    R_gt = (val_blob["A"] - val_blob["S"]) - val_blob["V"]
    cmd = val_blob["S"] + val_blob["V"] + R_hat
    ev = M.evaluate(cmd, val_blob["A"], val_blob["S"], val_blob["V"], val_blob["episode"])
    ev["correction"] = M.evaluate_correction(R_hat, R_gt)
    return ev, {"R_hat": R_hat, "R_gt": R_gt, "cmd": cmd, "gt": val_blob["A"],
                "state": val_blob["S"], "V": val_blob["V"],
                "episode": val_blob["episode"], "frame": val_blob["frame"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 2000, 0],
                    help="0 means all legal training frames")
    ap.add_argument("--equal-updates", type=int, default=8000)
    ap.add_argument("--equal-epochs", type=int, default=200)
    ap.add_argument("--dev-max-epochs", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.manual_seed(0)
    split = D.load_split()
    tr_eps, dev_eps = split_episodes(split["train"], n_dev=16)
    if set(tr_eps) & set(dev_eps) or set(tr_eps) & set(split["val"]) or set(dev_eps) & set(split["val"]):
        raise SystemExit("episode leakage between train/dev/holdout")

    rows = pool_from_episodes(tr_eps)
    order, edges = stratified_order(rows, seed=0)
    full_n = len(order)
    sizes = [full_n if s == 0 else s for s in args.sizes]
    for s in sizes:
        if s > full_n:
            raise SystemExit("requested %d frames but only %d are available" % (s, full_n))
    print("[pool] %d legal t>=2 frames from %d train episodes; motion-bin edges %s"
          % (full_n, len(tr_eps), np.round(edges, 5).tolist()), flush=True)

    dev_blob = stack_frames([episode_frames(D.load_episode(n), need_accel=True) for n in dev_eps])
    val_blob = stack_frames([episode_frames(D.load_episode(n), need_accel=True) for n in split["val"]])
    Xdev, Rdev = build_xy(dev_blob, "SVAcc", "R")
    cv_L1 = float(np.abs(val_blob["S"] + val_blob["V"] - val_blob["A"]).mean())
    print("[data] dev frames=%d holdout frames=%d | copy-V holdout L1=%.6f"
          % (len(Xdev), len(val_blob["S"]), cv_L1), flush=True)

    result = {"config": vars(args), "pool": {"train_episodes": tr_eps, "dev_episodes": dev_eps,
                                             "holdout_episodes": split["val"],
                                             "n_pool_frames": full_n,
                                             "motion_bin_edges": np.asarray(edges).tolist()},
              "holdout_frames": int(len(val_blob["S"])),
              "copy_V_holdout_L1": cv_L1,
              "identity_holdout_L1": float(np.abs(val_blob["S"] - val_blob["A"]).mean()),
              "runs": {}, "manifests": {}}
    json.dump(result, open(args.out, "w"), indent=1)

    for size in sizes:
        blob = take_prefix(rows, order, size)
        # nesting check: identical sampling rule, prefix of the same order
        man = {"n_frames": int(len(blob["S"])),
               "n_episodes": int(len(set(blob["episode"].tolist()))),
               "frames_per_episode_min": int(min(np.bincount(
                   [sorted(set(blob["episode"].tolist())).index(e) for e in blob["episode"]]))),
               "motion_mean": float(blob["motion"].mean()),
               "motion_p10": float(np.quantile(blob["motion"], 0.10)),
               "motion_p90": float(np.quantile(blob["motion"], 0.90)),
               "episodes": sorted(set(blob["episode"].tolist()))}
        result["manifests"][str(size)] = man
        print("[size %5d] episodes=%d frames/ep_min=%d motion mean=%.5f p10=%.5f p90=%.5f"
              % (size, man["n_episodes"], man["frames_per_episode_min"],
                 man["motion_mean"], man["motion_p10"], man["motion_p90"]), flush=True)
        Xtr, Rtr = build_xy(blob, "SVAcc", "R")
        for protocol in ("equal_updates", "equal_epochs", "dev_selected"):
            key = "%s_%s" % (size, protocol)
            per_seed = []
            for seed in args.seeds:
                if protocol == "equal_updates":
                    net, st = train_fixed(Xtr, Rtr, Xdev, Rdev, seed, max_steps=args.equal_updates,
                                          epochs=10 ** 6, select="last", torch=torch)
                elif protocol == "equal_epochs":
                    net, st = train_fixed(Xtr, Rtr, Xdev, Rdev, seed, max_steps=None,
                                          epochs=args.equal_epochs, select="last", torch=torch)
                else:
                    net, st = train_fixed(Xtr, Rtr, Xdev, Rdev, seed, max_steps=None,
                                          epochs=args.dev_max_epochs, select="dev_best",
                                          patience=40, torch=torch)
                ev, side = holdout_eval(net, st, val_blob, torch)
                ev["seed"] = int(seed)
                ev["protocol"] = protocol
                ev["n_train_frames"] = int(len(Xtr))
                ev["steps"] = st["steps"]
                ev["epochs"] = st["epochs"]
                ev["dev_best_R_L1"] = st["dev_best_L1"]
                ev["checkpoint_selection"] = st["select"]
                ev["seconds"] = st["seconds"]
                ev["sample_passes"] = int(st["steps"] * min(512, len(Xtr)))
                per_seed.append(ev)
                np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                                 "pred_cov_%s_seed%d.npz" % (key, seed)), **side)
                print("[%s seed=%d] holdout L1=%.6f (cv %.6f) | R^2=%.3f corr=%+.3f "
                      "sign=%.3f | steps=%d epochs=%d dev_R=%.6f | %.0fs"
                      % (key, seed, ev["L1"], cv_L1,
                         ev["correction"]["R2_variance_explained"] or float("nan"),
                         ev["correction"]["corr"] or float("nan"),
                         ev["correction"]["sign_agreement"] or float("nan"),
                         st["steps"], st["epochs"], st["dev_best_L1"], st["seconds"]), flush=True)
            result["runs"][key] = {"per_seed": per_seed, "aggregate": aggregate(per_seed),
                                   "protocol": protocol, "n_train_frames": int(len(Xtr))}
            json.dump(result, open(args.out, "w"), indent=1)
    print("wrote", args.out)


def aggregate(per_seed):
    def stat(key):
        v = [float(s[key]) for s in per_seed]
        return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
    out = {"L1": stat("L1"), "ratio_vs_cv": stat("ratio_vs_constant_velocity"),
           "R2": {"mean": float(np.nanmean([s["correction"]["R2_variance_explained"] for s in per_seed]))},
           "R_corr": {"mean": float(np.nanmean([s["correction"]["corr"] for s in per_seed]))},
           "R_sign": {"mean": float(np.nanmean([s["correction"]["sign_agreement"] for s in per_seed]))},
           "cosine": {"mean": float(np.nanmean([s["direction"]["cosine_mean"] for s in per_seed]))},
           "sign_delta_at_1e-3": {"mean": float(np.nanmean(
               [s["direction"]["sign_agreement_at_1e-3"] for s in per_seed]))},
           "episodes_beating_cv": [int(s["episodes_beating_cv"]) for s in per_seed],
           "steps": [int(s["steps"]) for s in per_seed],
           "epochs": [int(s["epochs"]) for s in per_seed],
           "n_seeds": len(per_seed)}
    return out


if __name__ == "__main__":
    main()
