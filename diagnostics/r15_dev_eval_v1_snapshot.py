#!/usr/bin/env python3
"""R15 Lane E: frozen 16-DEV evaluator with same-sample baselines (P1-METRIC-3).

Design rules enforced here:
  * the DEV cohort is exactly the 16 episodes produced by `r9_trainer_contract.select_splits`
    from the frozen `data/split80.json`; the old 20 holdout and any new FINAL are never read;
  * one FROZEN frame list (deterministic, hashed) is used by every model and every baseline;
  * baselines are fitted on TRAIN frames only and are marked NOT_AVAILABLE when their weights
    do not exist instead of being invented;
  * every model sees the same pre-processing contract (q01/q99, gripper policy, bounds);
  * episode-macro metrics are reported alongside the pooled numbers, because a macro average over
    episodes is what the R15 gates are written against.

Metrics: physical joint-14D L1, per-arm/per-gripper L1, correction R^2, action magnitude ratio,
sign agreement, joint-clip rate, motion strata (by `|V|` quartiles computed on DEV), and the
still-action rate.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("training", "code/xpolicylab/policy/OpenVLA_Pure",
             "code/xpolicylab/policy/OpenVLA_R42"):
    _p = os.path.join(REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONTRACT = os.environ.get("PUSHT_CONTRACT",
                          os.path.join(REPO, "evidence", "r9_gate_20261020",
                                       "frozen_contract_64train.json"))
SPLIT = os.environ.get("PUSHT_SPLIT", os.path.join(REPO, "data", "split80.json"))
BOUNDS = os.environ.get("PUSHT_BOUNDS", os.path.join(REPO, "data", "push_T_bounds.npz"))
HDF5_DIR = os.environ.get("PUSHT_HDF5",
                          "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
GRIPPER = (6, 13)
ARM = tuple(d for d in range(14) if d not in GRIPPER)
MIN_FRAME = 2
EVAL_VERSION = "r15-dev-eval-v1"


# --------------------------------------------------------------------------- data
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read14(h, group):
    v = np.concatenate([h[group + "/left_arm_joint_states"][:],
                        h[group + "/left_ee_joint_states"][:],
                        h[group + "/right_arm_joint_states"][:],
                        h[group + "/right_ee_joint_states"][:]], axis=1).astype(np.float32)
    if v.shape[1] != 14:
        raise ValueError("%s is not 14D" % group)
    return v


def load_splits():
    import r9_trainer_contract as RC
    contract, meta = RC.load_frozen_contract(CONTRACT)
    splits = RC.select_splits(SPLIT)
    if set(splits["train"]) & set(splits["dev"]):
        raise ValueError("TRAIN/DEV overlap")
    return contract, meta, splits


def frozen_frames(episodes, per_episode, seed=0):
    """Deterministic frame list: `per_episode` history-safe frames from each episode."""
    import h5py
    rs = np.random.RandomState(seed)
    out = []
    for name in episodes:
        with h5py.File(os.path.join(HDF5_DIR, name), "r") as h:
            T = len(h["action/left_arm_joint_states"])
        pool = np.arange(MIN_FRAME, max(MIN_FRAME + 1, T - 1))
        k = min(per_episode, len(pool))
        picks = np.sort(rs.choice(pool, size=k, replace=False))
        out.extend([(name, int(t)) for t in picks])
    return out


def frames_sha256(frames):
    return hashlib.sha256(json.dumps([[e, int(t)] for e, t in frames], sort_keys=True)
                          .encode()).hexdigest()


def build_samples(frames, contract):
    """(features42, S, V, A) for every frozen frame; history-only, no future rows."""
    import h5py
    import r8_vla_adapter as A
    feats, states, vels, acts = [], [], [], []
    for name, t in frames:
        with h5py.File(os.path.join(HDF5_DIR, name), "r") as h:
            st = read14(h, "state")
            ac = read14(h, "action")
        if t < MIN_FRAME or t >= len(st) - 1:
            raise ValueError("frame %d of %s is not history-safe" % (t, name))
        v = (st[t] - st[t - 1]).astype(np.float32)
        feats.append(A.build_proprio(st[t], st[t - 1], st[t - 2], contract))
        states.append(st[t].copy())
        vels.append(v)
        acts.append(ac[t].copy())
    return (np.stack(feats).astype(np.float32), np.stack(states).astype(np.float32),
            np.stack(vels).astype(np.float32), np.stack(acts).astype(np.float32))


# ------------------------------------------------------------------------ metrics
def _macro(values, episodes):
    """Episode-macro mean: mean over episodes of the per-episode mean."""
    out = {}
    eps = sorted(set(episodes))
    per = [float(np.mean([v for v, e in zip(values, episodes) if e == ep])) for ep in eps]
    out["macro"] = float(np.mean(per))
    out["per_episode"] = dict(zip(eps, per))
    out["worst_episode"] = float(np.max(per))
    return out


def metrics(pred, target, state, velocity, episodes, bounds=None, still_eps=1e-3):
    """All Gate quantities for one model prediction set (all [N,14])."""
    pred = np.asarray(pred, np.float32)
    target = np.asarray(target, np.float32)
    err = np.abs(pred - target)
    pooled_l1 = float(err.mean(axis=1).mean())
    arm_l1 = float(err[:, ARM].mean())
    grip_l1 = float(err[:, GRIPPER].mean())
    macro = _macro(err.mean(axis=1), episodes)

    corr_t = (target - state) - velocity
    corr_p = (pred - state) - velocity
    ss_res = float(((corr_p - corr_t) ** 2).sum())
    ss_tot = float(((corr_t - corr_t.mean(axis=0)) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    mag_ratio = float(np.mean(np.abs(pred).mean(axis=0) /
                              np.maximum(np.abs(target).mean(axis=0), 1e-9)))
    sign_agree = float(np.mean(np.sign(pred) == np.sign(target)))
    still = float(np.mean(np.abs(pred - state).max(axis=1) < still_eps))
    clip = float("nan")
    if bounds is not None:
        lo, hi = bounds
        outside = ((pred < lo - 1e-6) | (pred > hi + 1e-6))
        clip = float(outside.mean())

    speed = np.abs(velocity).mean(axis=1)
    q = np.quantile(speed, [0.25, 0.5, 0.75]) if len(speed) > 3 else [0, 0, 0]
    strata = {}
    edges = [-np.inf] + list(q) + [np.inf]
    for i in range(4):
        m = (speed > edges[i]) & (speed <= edges[i + 1])
        strata["q%d" % (i + 1)] = {"n": int(m.sum()),
                                   "l1": float(err[m].mean()) if m.any() else None}
    return {"physical_l1_pooled": pooled_l1, "physical_l1_macro": macro["macro"],
            "physical_l1_per_episode": macro["per_episode"],
            "physical_l1_worst_episode": macro["worst_episode"],
            "arm_l1": arm_l1, "gripper_l1": grip_l1,
            "correction_r2": r2, "magnitude_ratio": mag_ratio,
            "sign_agreement": sign_agree, "still_action_rate": still,
            "joint_clip_rate": clip, "motion_strata": strata, "n_samples": int(len(pred))}


# ----------------------------------------------------------------------- baselines
def baseline_identity(state, velocity):
    return state.copy()


def baseline_constant_velocity(state, velocity):
    return (state + velocity).astype(np.float32)


def fit_affine(feats_train, state_train, vel_train, act_train, ridge=1e-6):
    """TRAIN-only affine map from the 42D feature to the correction, A_hat = S + V + C_hat."""
    X = np.concatenate([feats_train, np.ones((len(feats_train), 1), np.float32)], axis=1)
    Y = ((act_train - state_train) - vel_train).astype(np.float32)
    G = X.T @ X + ridge * np.eye(X.shape[1], dtype=np.float32)
    W = np.linalg.solve(G, X.T @ Y)
    return W


def apply_affine(W, feats, state, velocity):
    X = np.concatenate([feats, np.ones((len(feats), 1), np.float32)], axis=1)
    return (state + velocity + X @ W).astype(np.float32)


def fit_mlp(feats_train, state_train, vel_train, act_train, seed=0, epochs=300, hidden=64,
            lr=1e-3):
    """TRAIN-only small MLP on the same features and target as the affine baseline."""
    import torch
    torch.manual_seed(seed)
    X = torch.from_numpy(feats_train)
    Y = torch.from_numpy(((act_train - state_train) - vel_train).astype(np.float32))
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    net = torch.nn.Sequential(torch.nn.Linear(42, hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, 14))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net((X - mu) / sd) - Y) ** 2).mean()
        loss.backward()
        opt.step()
    return {"net": net, "mu": mu, "sd": sd}


def apply_mlp(m, feats, state, velocity):
    import torch
    with torch.no_grad():
        c = m["net"]((torch.from_numpy(feats) - m["mu"]) / m["sd"]).numpy()
    return (state + velocity + c).astype(np.float32)


# ----------------------------------------------------------------------------- main
def run(pred_npz=None, per_episode=8, seed=0, out_dir=None, train_frames=256,
        skip_mlp=False):
    contract, meta, splits = load_splits()
    q01 = np.asarray(meta["q01"], np.float32)
    q99 = np.asarray(meta["q99"], np.float32)
    nb = np.load(BOUNDS)
    bounds = (nb["amin"].astype(np.float32), nb["amax"].astype(np.float32))

    dev_frames = frozen_frames(sorted(splits["dev"]), per_episode, seed=seed)
    dev_eps = [e for e, _t in dev_frames]
    feats, states, vels, acts = build_samples(dev_frames, contract)
    train_pick = frozen_frames(sorted(splits["train"])[:16], max(1, train_frames // 16),
                               seed=seed + 1)
    feats_t, states_t, vels_t, acts_t = build_samples(train_pick, contract)

    if set(e for e, _t in dev_frames) & set(e for e, _t in train_pick):
        raise ValueError("frozen TRAIN fit frames overlap DEV")

    ev = {"eval_version": EVAL_VERSION, "delta_contract_sha256": sha256_file(CONTRACT),
          "split_sha256": sha256_file(SPLIT), "bounds_sha256": sha256_file(BOUNDS),
          "hdf5_dir": HDF5_DIR, "n_dev_frames": len(dev_frames),
          "dev_episodes": sorted(set(dev_eps)), "dev_frames_sha256": frames_sha256(dev_frames),
          "train_fit_frames_sha256": frames_sha256(train_pick),
          "train_fit_episodes": sorted(set(e for e, _t in train_pick)),
          "old_holdout_used_for_selection": False, "new_final_used": False,
          "q01": q01.tolist(), "q99": q99.tolist(),
          "gripper_dims": list(GRIPPER), "models": {}}

    def record(name, pred, avail=True, note=""):
        if not avail:
            ev["models"][name] = {"status": "NOT_AVAILABLE", "note": note}
            return
        ev["models"][name] = {"status": "OK", "metrics": metrics(pred, acts, states, vels,
                                                                 dev_eps, bounds)}
        ev["models"][name]["note"] = note

    record("identity_A=S", baseline_identity(states, vels), note="A_hat = S_t")
    record("constant_velocity_A=S+V", baseline_constant_velocity(states, vels),
           note="A_hat = S_t + V_t")
    W = fit_affine(feats_t, states_t, vels_t, acts_t)
    record("train_only_affine", apply_affine(W, feats, states, vels),
           note="TRAIN-only affine on the 42D feature -> correction")
    if skip_mlp:
        record("train_only_mlp", None, avail=False, note="skipped by flag")
    else:
        try:
            m = fit_mlp(feats_t, states_t, vels_t, acts_t, seed=seed)
            record("train_only_mlp", apply_mlp(m, feats, states, vels),
                   note="TRAIN-only 42->64->14 MLP")
        except Exception as exc:
            record("train_only_mlp", None, avail=False, note="fit failed: %r" % (exc,))

    if pred_npz:
        d = np.load(pred_npz, allow_pickle=True)
        p = np.asarray(d["action"], np.float32)
        if p.shape != acts.shape:
            raise SystemExit("prediction shape %s != DEV sample shape %s"
                             % (p.shape, acts.shape))
        if "frames_sha256" in d and str(d["frames_sha256"]) != ev["dev_frames_sha256"]:
            raise SystemExit("prediction archive was produced on a DIFFERENT frame list")
        record("vla_r42", p, note="model predictions from %s" % os.path.basename(pred_npz))
        ev["models"]["vla_r42"]["model_sha256"] = str(d["model_sha256"]) \
            if "model_sha256" in d else None

    cv = ev["models"]["constant_velocity_A=S+V"]["metrics"]["physical_l1_macro"]
    vla = ev["models"].get("vla_r42", {}).get("metrics")
    ev["gate"] = {"cv_macro_l1": cv,
                  "vla_macro_l1": None if vla is None else vla["physical_l1_macro"],
                  "improvement_vs_cv": None if vla is None else
                  float((cv - vla["physical_l1_macro"]) / max(cv, 1e-9))}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "dev_eval.json"), "w") as fh:
            json.dump(ev, fh, indent=1)
        np.savez_compressed(os.path.join(out_dir, "dev_samples.npz"),
                            frames=np.array([[e, t] for e, t in dev_frames]),
                            features=feats, state=states, velocity=vels, action=acts,
                            frames_sha256=ev["dev_frames_sha256"])
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-npz", default=None)
    ap.add_argument("--per-episode", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frames", type=int, default=256)
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ev = run(args.pred_npz, args.per_episode, args.seed, args.out, args.train_frames,
             args.skip_mlp)
    for name, m in ev["models"].items():
        if m["status"] != "OK":
            print("%-28s %s (%s)" % (name, m["status"], m["note"]))
            continue
        q = m["metrics"]
        print("%-28s macroL1=%.6f pooled=%.6f arm=%.6f grip=%.6f R2=%+.4f mag=%.3f "
              "sign=%.3f still=%.3f clip=%.4f"
              % (name, q["physical_l1_macro"], q["physical_l1_pooled"], q["arm_l1"],
                 q["gripper_l1"], q["correction_r2"], q["magnitude_ratio"],
                 q["sign_agreement"], q["still_action_rate"], q["joint_clip_rate"]))
    print("gate:", json.dumps(ev["gate"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
