#!/usr/bin/env python3
"""R7 phase 2: lightweight state-dynamics baseline (no OpenVLA, CPU by default).

Question: from the FULL set of trainable frames, can a small MLP learn the velocity
correction R_t = (A_t - S_t) - V_t better than predicting R = 0 (i.e. constant velocity)?

    X_t   = [S_t, V_t],  V_t = S_t - S_{t-1}          (past-only, strict causality)
    R_t   = (A_t - S_t) - V_t                          (supervision)
    A_hat = S_t + V_t + R_hat                          (physical command)

Baselines on the same frames: identity (S), constant velocity (S+V), a per-dimension
affine map of [S,V] fitted on TRAIN episodes, and the R=0 model itself.
Evaluation: the 20 held-out episodes only (6,711 legal frames), never used for
selection. Model selection uses 16 dev episodes carved out of the 80 train episodes
so no episode appears in both training and selection.

Also trains the same architecture on the R5/R6 200-frame pilot subset, which measures
what the 200-frame regime was missing.
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
import r6_data as D          # noqa: E402
import r7_metrics as M       # noqa: E402

LEFT, RIGHT = slice(0, 7), slice(7, 14)
N_DIMS = 14


# ----------------------------------------------------------------- data assembly
def episode_frames(ep, need_accel=False):
    """Legal frames t with t-1 (and t-2 when needed) available; strict past-only inputs.

    need_accel=True restricts to t >= 2 so that the past acceleration
    Acc_t = V_t - V_(t-1) is defined for every retained frame.
    """
    t = D.valid_frames(len(ep["state"]))
    if need_accel:
        t = t[t >= 2]
    S, A = ep["state"][t], ep["action"][t]
    V = ep["state"][t] - ep["state"][t - 1]
    acc = (ep["state"][t] - ep["state"][t - 1]) - (ep["state"][t - 1] - ep["state"][t - 2]) if need_accel else None
    return {"S": S, "A": A, "V": V, "Acc": acc, "episode": ep["episode"], "frame": t}


def stack_frames(frames):
    out = {"S": np.concatenate([f["S"] for f in frames]),
           "A": np.concatenate([f["A"] for f in frames]),
           "V": np.concatenate([f["V"] for f in frames]),
           "episode": np.concatenate([[f["episode"]] * len(f["S"]) for f in frames])}
    if frames[0].get("Acc") is not None:
        out["Acc"] = np.concatenate([f["Acc"] for f in frames])
    if "frame" in frames[0]:
        out["frame"] = np.concatenate([f["frame"] for f in frames])
    return out


def build_xy(blob, feature, target="R"):
    """feature: 'SV' | 'S' | 'V'; target: 'R' (velocity correction) or 'delta'."""
    S, V = blob["S"], blob["V"]
    delta = blob["A"] - S
    R = (delta - V) if target == "R" else delta
    if feature == "SV":
        X = np.concatenate([S, V], axis=1)
    elif feature == "SVAcc":
        X = np.concatenate([S, V, blob["Acc"]], axis=1)
    elif feature == "S":
        X = S.copy()
    elif feature == "V":
        X = V.copy()
    else:
        raise ValueError("unknown feature set: " + feature)
    return X.astype(np.float32), R.astype(np.float32)


def reconstruct_prediction(raw_prediction, blob, target="R"):
    """Return A_hat, predicted R, and ground-truth R for either target.

    For target=delta, raw_prediction estimates A-S, not R=A-S-V.
    Both target modes must use the SAME true correction for R² and sign tests.
    """
    S, V, A = blob["S"], blob["V"], blob["A"]
    if target == "R":
        R_hat = raw_prediction
    elif target == "delta":
        R_hat = raw_prediction - V
    else:
        raise ValueError("unknown prediction target: " + str(target))
    return S + V + R_hat, R_hat, (A - S) - V


def choose_hybrid_dims_on_dev(dev_command, dev_blob):
    """Select model-corrected channels from DEV only, not the holdout set."""
    cv = dev_blob["S"] + dev_blob["V"]
    model_error = np.abs(dev_command - dev_blob["A"]).mean(axis=0)
    cv_error = np.abs(cv - dev_blob["A"]).mean(axis=0)
    return [int(i) for i in np.flatnonzero(model_error < cv_error)]


# ------------------------------------------------------------------------ model
def make_mlp(n_in, hidden, seed, torch):
    g = torch.Generator().manual_seed(seed)
    layers, prev = [], n_in
    for h in hidden:
        lin = torch.nn.Linear(prev, h)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(h, prev, generator=g) * (1.0 / prev ** 0.5))
            lin.bias.zero_()
        layers += [lin, torch.nn.GELU()]
        prev = h
    out = torch.nn.Linear(prev, N_DIMS)
    with torch.no_grad():
        out.weight.copy_(torch.randn(N_DIMS, prev, generator=g) * (1.0 / prev ** 0.5))
        out.bias.zero_()
    layers.append(out)
    return torch.nn.Sequential(*layers)


def train_mlp(Xtr, Rtr, Xdev, Rdev, seed, hidden=(128, 128), epochs=400, lr=1e-3,
              batch=512, patience=40, torch=None, select_best=True):
    """Full-batch-ish Adam on physical L1(R_hat, R); early stop on dev L1."""
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr = (Xtr - mu_x) / sd_x
    Zdev = (Xdev - mu_x) / sd_x
    net = make_mlp(Xtr.shape[1], hidden, seed, torch)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    tZ = torch.from_numpy(Ztr)
    tR = torch.from_numpy(Rtr)
    dZ = torch.from_numpy(Zdev)
    dR = torch.from_numpy(Rdev)
    n = len(tZ)
    g = torch.Generator().manual_seed(seed + 1)
    best = {"epoch": -1, "dev_L1": float("inf"), "state": None}
    hist = []
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            opt.zero_grad(set_to_none=True)
            pred = net(tZ[idx])
            loss = (pred - tR[idx]).abs().mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        net.eval()
        with torch.no_grad():
            dev_l1 = float((net(dZ) - dR).abs().mean())
        hist.append({"epoch": ep, "train_L1": tot / n, "dev_L1": dev_l1})
        if dev_l1 < best["dev_L1"] - 1e-7:
            best = {"epoch": ep, "dev_L1": dev_l1,
                    "state": {k: v.clone() for k, v in net.state_dict().items()}}
        elif ep - best["epoch"] >= patience:
            break
    if select_best:
        net.load_state_dict(best["state"])
    # The 200-frame ablation is a fixed-budget experiment, not training
    # checkpoint selection against the same 200 training frames.
    net.eval()
    return net, {"best_epoch": best["epoch"] if select_best else len(hist),
                 "checkpoint_selection": "dev_best" if select_best else "last",
                 "dev_best_L1": best["dev_L1"],
                 "x_mean": mu_x, "x_std": sd_x, "history": hist}


def predict(net, X, stats, torch):
    Z = (X - stats["x_mean"]) / stats["x_std"]
    with torch.no_grad():
        return net(torch.from_numpy(Z.astype(np.float32))).numpy()


# --------------------------------------------------------------------- baselines
def fit_affine(blob, feature):
    """Per-dimension affine map [S,V] -> Delta (not R), fitted on TRAIN episodes."""
    X, _ = build_xy(blob, feature)
    delta = blob["A"] - blob["S"]
    A = np.concatenate([X, np.ones((len(X), 1), np.float32)], axis=1).astype(np.float64)
    W = np.linalg.lstsq(A, delta.astype(np.float64), rcond=None)[0]
    return W


def apply_affine(W, blob, feature):
    X, _ = build_xy(blob, feature)
    A = np.concatenate([X, np.ones((len(X), 1), np.float32)], axis=1).astype(np.float64)
    return (A @ W).astype(np.float32)


# ------------------------------------------------------------------------ splits
def split_episodes(train_episodes, n_dev=16):
    """Episode-level dev split: no episode contributes frames to both sides."""
    eps = sorted(train_episodes)
    dev = eps[::max(1, len(eps) // n_dev)][:n_dev]
    dev = sorted(set(dev))
    tr = [e for e in eps if e not in set(dev)]
    if set(tr) & set(dev):
        raise AssertionError("train/dev episode overlap")
    return tr, dev


def load_pilot_200_frames(path):
    cfg = json.load(open(path))
    pairs = [tuple(x) for x in cfg["samples"]]
    return sorted({e for e, _ in pairs}), pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--features", nargs="+", default=["SV", "S", "V"])
    ap.add_argument("--targets", nargs="+", default=["R"], choices=["R", "delta"])
    ap.add_argument("--restrict-t2", action="store_true",
                    help="restrict every feature to t>=2 so SV and SVAcc share one frame set")
    ap.add_argument("--pilot-config", default="evidence/history_proprio_pilot_20261019T161033Z/config_duplicate.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.manual_seed(0)
    split = D.load_split()
    tr_eps, dev_eps = split_episodes(split["train"])
    if set(tr_eps) & set(dev_eps) or set(tr_eps) & set(split["val"]) or set(dev_eps) & set(split["val"]):
        raise AssertionError("episode leakage between train/dev/holdout")

    def load(names):
        return [episode_frames(D.load_episode(n), need_accel=args.restrict_t2) for n in names]

    train_frames = load(tr_eps)
    dev_frames = load(dev_eps)
    val_frames = load(split["val"])
    train_blob = stack_frames(train_frames)
    dev_blob = stack_frames(dev_frames)
    val_blob = stack_frames(val_frames)
    print("[data] train episodes=%d frames=%d | dev episodes=%d frames=%d | holdout episodes=%d frames=%d"
          % (len(tr_eps), len(train_blob["S"]), len(dev_eps), len(dev_blob["S"]),
             len(split["val"]), len(val_blob["S"])), flush=True)

    result = {"config": vars(args), "splits": {"train_episodes": tr_eps, "dev_episodes": dev_eps,
                                               "holdout_episodes": split["val"]},
              "n_frames": {"train": int(len(train_blob["S"])), "dev": int(len(dev_blob["S"])),
                           "holdout": int(len(val_blob["S"]))},
              "baselines": {}, "runs": {}, "pilot200": {}}

    # ---- deterministic baselines on the holdout episodes
    identity_cmd = val_blob["S"]
    cv_cmd = val_blob["S"] + val_blob["V"]
    result["baselines"]["identity"] = M.evaluate(identity_cmd, val_blob["A"], val_blob["S"], val_blob["V"],
                                                 val_blob["episode"])
    result["baselines"]["constant_velocity"] = M.evaluate(cv_cmd, val_blob["A"], val_blob["S"], val_blob["V"],
                                                          val_blob["episode"])
    for feat in ("SV", "S", "V"):
        W = fit_affine(train_blob, feat)
        pred = apply_affine(W, val_blob, feat)
        result["baselines"]["train_affine_" + feat] = M.evaluate(
            val_blob["S"] + pred, val_blob["A"], val_blob["S"], val_blob["V"], val_blob["episode"])
    json.dump(result, open(args.out, "w"), indent=1)

    # ---- MLP runs
    for feat in args.features:
      for tgt in args.targets:
        Xtr, Rtr = build_xy(train_blob, feat, tgt)
        Xdev, Rdev = build_xy(dev_blob, feat, tgt)
        Xva, Rva = build_xy(val_blob, feat, tgt)
        per_seed = []
        for seed in args.seeds:
            t0 = time.time()
            net, stats = train_mlp(Xtr, Rtr, Xdev, Rdev, seed, tuple(args.hidden),
                                   args.epochs, args.lr, args.batch, args.patience, torch=torch)
            out_hat = predict(net, Xva, stats, torch).astype(np.float32)
            cmd, R_hat, R_gt = reconstruct_prediction(out_hat, val_blob, tgt)
            ev = M.evaluate(cmd, val_blob["A"], val_blob["S"], val_blob["V"], val_blob["episode"])
            ev["correction"] = M.evaluate_correction(R_hat, R_gt)
            ev["seed"] = int(seed)
            ev["target"] = tgt
            ev["best_epoch"] = int(stats["best_epoch"])
            ev["dev_best_R_L1"] = float(stats["dev_best_L1"])
            ev["seconds"] = round(time.time() - t0, 1)
            dev_command, _, _ = reconstruct_prediction(
                predict(net, Xdev, stats, torch), dev_blob, tgt)
            ev["dev"] = M.evaluate(
                dev_command, dev_blob["A"], dev_blob["S"], dev_blob["V"], dev_blob["episode"])
            if feat == "SVAcc" and tgt == "R":
                # Holdout-selected gripper gating is exploratory. Choose which
                # channels to correct on DEV; then evaluate once on HOLDOUT.
                selected = choose_hybrid_dims_on_dev(dev_command, dev_blob)
                cv_holdout = val_blob["S"] + val_blob["V"]
                hybrid = cv_holdout.copy()
                hybrid[:, selected] = cmd[:, selected]
                ev["hybrid_dev_selected_model_dims"] = selected
                ev["hybrid_holdout_L1"] = float(np.abs(
                    hybrid - val_blob["A"]).mean())
            if feat in ("SV", "SVAcc") and tgt == "R":
                side = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                    "pred_%s_%s_%d.npz" % (feat, tgt, seed))
                np.savez_compressed(side, R_hat=R_hat, R_gt=Rva, cmd=cmd,
                                    gt=val_blob["A"], state=val_blob["S"], V=val_blob["V"],
                                    episode=val_blob["episode"], frame=val_blob["frame"])
            per_seed.append(ev)
            print("[%s/%s seed=%d] holdout L1=%.6f (cv %.6f, identity %.6f) | corr_delta=%+.3f "
                  "sign@1e-3=%.3f | R_corr=%+.3f R_sign=%.3f | best_epoch=%d %.0fs" % (
                      feat, tgt, seed, ev["L1"], result["baselines"]["constant_velocity"]["L1"],
                      result["baselines"]["identity"]["L1"],
                      ev["direction"]["corr_delta_pooled"] or float("nan"),
                      ev["direction"]["sign_agreement_at_1e-3"] or float("nan"),
                      ev["correction"]["corr"] or float("nan"),
                      ev["correction"]["sign_agreement"] or float("nan"),
                      stats["best_epoch"], ev["seconds"]), flush=True)
        result["runs"]["%s_%s" % (feat, tgt)] = {"per_seed": per_seed,
                                                 "aggregate": M.aggregate(per_seed),
                                                 "feature": feat, "target": tgt}
        json.dump(result, open(args.out, "w"), indent=1)

    # ---- what did the 200-frame pilot regime miss?
    if os.path.isfile(args.pilot_config):
        pilot_eps, _pairs = load_pilot_200_frames(args.pilot_config)
        pilot_blob = stack_frames([episode_frames(D.load_episode(n)) for n in pilot_eps])
        keep = None
        cfg = json.load(open(args.pilot_config))
        pairs = [tuple(x) for x in cfg["samples"]]
        index = {(e, int(t)): i for i, (e, t) in enumerate(zip(pilot_blob["episode"], pilot_blob["frame"]))}
        keep = np.array([index[(e, int(t))] for e, t in pairs])
        pilot_blob = {k: v[keep] for k, v in pilot_blob.items()}
        Xp, Rp = build_xy(pilot_blob, "SV")
        Xva, Rva = build_xy(val_blob, "SV")
        # No early stopping: the 200-frame ablation has no honest dev set, so the epoch
        # count is FIXED to the median best epoch of the full-frame SV runs (selection
        # therefore comes from the full-frame dev episodes, not from these 200 frames).
        sv_epochs = [r["best_epoch"] for r in
                     result["runs"].get("SV_R", result["runs"].get("SV", {})).get("per_seed", [])]
        fixed_epochs = int(np.median(sv_epochs)) if sv_epochs else args.epochs
        result["pilot200"]["fixed_epochs_from_full_run"] = fixed_epochs
        print("[pilot200] fixed epoch budget = %d (median best epoch of full-frame SV)"
              % fixed_epochs, flush=True)
        for seed in args.seeds:
            net, stats = train_mlp(Xp, Rp, Xp, Rp, seed, tuple(args.hidden),
                                   fixed_epochs, args.lr, min(args.batch, len(Xp)),
                                   10 ** 9, torch=torch, select_best=False)
            out_hat = predict(net, Xva, stats, torch).astype(np.float32)
            # The pilot is trained on R even if the last grid target is
            # "delta": never use the stale outer loop variable `tgt` here.
            cmd, R_hat, R_gt = reconstruct_prediction(out_hat, val_blob, "R")
            ev = M.evaluate(cmd, val_blob["A"], val_blob["S"], val_blob["V"], val_blob["episode"])
            ev["correction"] = M.evaluate_correction(R_hat, R_gt)
            ev["seed"] = int(seed)
            ev["n_train_frames"] = int(len(Xp))
            ev["episodes_covered"] = int(len(set(pilot_blob["episode"].tolist())))
            ev["checkpoint_selection"] = stats["checkpoint_selection"]
            ev["fixed_epochs_executed"] = int(len(stats["history"]))
            side = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                "pred_pilot200_R_%d.npz" % seed)
            np.savez_compressed(side, R_hat=R_hat, R_gt=R_gt, cmd=cmd,
                                gt=val_blob["A"], state=val_blob["S"],
                                V=val_blob["V"], episode=val_blob["episode"],
                                frame=val_blob["frame"])
            result["pilot200"].setdefault("per_seed", []).append(ev)
            print("[pilot200 seed=%d] holdout L1=%.6f | R_corr=%+.3f (trained on %d frames / %d episodes)"
                  % (seed, ev["L1"], ev["correction"]["corr"] or float("nan"),
                     len(Xp), ev["episodes_covered"]), flush=True)
        result["pilot200"]["aggregate"] = M.aggregate(result["pilot200"]["per_seed"])

    json.dump(result, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
