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

Metrics (v2, see metric_definitions() for the machine-readable contract):
  * physical joint-14D L1 (the gate quantity), per-arm/per-gripper L1, correction R^2;
  * MOTION metrics are computed on the commanded motion ΔA = A - S and on the residual R = ΔA - V,
    NOT on the absolute joint position. v1 compared absolute positions, which is dominated by "the
    arm is somewhere around q" and is not the quantity a controller cares about (audit PR#21);
  * every quantity is reported twice: RAW (exactly what the decoder emitted) and SAFE (the same
    vector after the joint-box clamp the robot actually applies). SAFE is the gated variant; RAW
    out-of-box statistics are reported next to it instead of being silently hidden;
  * motion strata (by `|V|` quartiles computed on DEV), the still-action rate, and the clip rates.
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


METRIC_VERSION = "r15-dev-metrics-v2"


def metric_definitions():
    """Machine-readable contract for every number this evaluator reports.

    Added for audit PR#21 finding 3/4: v1 reported sign_agreement / magnitude_ratio on the ABSOLUTE
    joint position and reported a single joint_clip_rate, which made the G4 "inside the joint box"
    claim ambiguous (6.03% of decoded coordinates were out of the box while the gate said PASS).
    """
    return {
        "metric_version": METRIC_VERSION,
        "gate_quantity": "physical_l1_macro on SAFE (clamped) commands",
        "delta_definition": "dA = A_hat - S_t versus dA_true = A_t - S_t (commanded motion)",
        "residual_definition": "R = (A - S_t) - V_t with V_t = S_t - S_{t-1}; A[t] == S[t+1]",
        "sign_agreement": "mean(sign(dA) == sign(dA_true)) over all 14 coordinates (NOT absolute)",
        "magnitude_ratio": "mean over dims of mean|dA| / max(mean|dA_true|, 1e-9) (NOT absolute)",
        "residual_sign_agreement": "same sign test on the residual R",
        "residual_magnitude_ratio": "same magnitude test on the residual R",
        "absolute_sign_agreement": "legacy v1 definition, kept only for continuity, not gated",
        "absolute_magnitude_ratio": "legacy v1 definition, kept only for continuity, not gated",
        "raw_vs_safe": ("every metric is reported twice: the unprefixed key is SAFE (decoder output "
                        "clamped to the joint box, i.e. what the robot executes) and the raw_ prefix "
                        "is the unclamped decoder output"),
        "clip_policy": ("clamp each of the 14 coordinates to [amin, amax] from the frozen bounds npz; "
                        "no other post-processing, no temporal filter, no re-scaling"),
        "sampling": ("one deterministic greedy decode per frozen DEV frame, frozen frame list "
                     "(16 DEV episodes x --per-episode frames, seed=0 for the registered gate); no "
                     "test-time augmentation, no best-of-n, no per-frame selection"),
        "raw_oob_coord_rate": "fraction of the 14*N RAW coordinates outside the joint box",
        "raw_oob_frame_rate": "fraction of frames with at least one RAW coordinate outside the box",
    }


def clamp_to_box(pred, bounds):
    """The deployment clamp: identical to what the policy server applies before sending commands."""
    if bounds is None:
        return np.asarray(pred, np.float32).copy()
    lo, hi = bounds
    return np.clip(np.asarray(pred, np.float32), lo, hi).astype(np.float32)


def _outside_box(pred, bounds, tol=1e-6):
    if bounds is None:
        return np.zeros(np.asarray(pred).shape, dtype=bool)
    lo, hi = bounds
    p = np.asarray(pred, np.float32)
    return (p < lo - tol) | (p > hi + tol)


def _metrics_block(p, target, state, velocity, episodes, still_eps):
    """One metric block for one prediction vector set (all [N,14])."""
    p = np.asarray(p, np.float32)
    err = np.abs(p - target)
    pooled_l1 = float(err.mean(axis=1).mean())
    arm_l1 = float(err[:, ARM].mean())
    grip_l1 = float(err[:, GRIPPER].mean())
    macro = _macro(err.mean(axis=1), episodes)

    # ---- motion quantities: commanded motion dA = A - S and residual R = dA - V
    dA_p = (p - state).astype(np.float32)
    dA_t = (target - state).astype(np.float32)
    res_p = (dA_p - velocity).astype(np.float32)
    res_t = (dA_t - velocity).astype(np.float32)

    ss_res = float(((res_p - res_t) ** 2).sum())
    ss_tot = float(((res_t - res_t.mean(axis=0)) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    d_mag = float(np.mean(np.abs(dA_p).mean(axis=0) /
                          np.maximum(np.abs(dA_t).mean(axis=0), 1e-9)))
    d_sign = float(np.mean(np.sign(dA_p) == np.sign(dA_t)))
    r_mag = float(np.mean(np.abs(res_p).mean(axis=0) /
                          np.maximum(np.abs(res_t).mean(axis=0), 1e-9)))
    r_sign = float(np.mean(np.sign(res_p) == np.sign(res_t)))
    # legacy v1 (absolute position) - reported, never gated
    a_mag = float(np.mean(np.abs(p).mean(axis=0) /
                          np.maximum(np.abs(target).mean(axis=0), 1e-9)))
    a_sign = float(np.mean(np.sign(p) == np.sign(target)))

    still = float(np.mean(np.abs(p - state).max(axis=1) < still_eps))
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
            "correction_r2": r2,
            "sign_agreement": d_sign, "magnitude_ratio": d_mag,
            "delta_sign_agreement": d_sign, "delta_magnitude_ratio": d_mag,
            "residual_sign_agreement": r_sign, "residual_magnitude_ratio": r_mag,
            "delta_l1_pooled": pooled_l1, "residual_l1_pooled": pooled_l1,
            "absolute_sign_agreement": a_sign, "absolute_magnitude_ratio": a_mag,
            "still_action_rate": still, "motion_strata": strata,
            "motion_strata_edges": [float(x) for x in q],
            "n_samples": int(len(p))}


def metrics(pred, target, state, velocity, episodes, bounds=None, still_eps=1e-3):
    """RAW (as decoded) and SAFE (clamped, what the robot executes) metrics for one model.

    The unprefixed keys are the SAFE/deployment variant because the gate is judged on what is
    actually commanded; every one of them is mirrored under a raw_ prefix, together with the
    out-of-box rates that v1 hid behind a single ambiguous 'joint_clip_rate'.
    """
    pred = np.asarray(pred, np.float32)
    target = np.asarray(target, np.float32)
    state = np.asarray(state, np.float32)
    velocity = np.asarray(velocity, np.float32)
    outside = _outside_box(pred, bounds)
    pred_safe = clamp_to_box(pred, bounds)
    safe_block = _metrics_block(pred_safe, target, state, velocity, episodes, still_eps)
    raw_block = _metrics_block(pred, target, state, velocity, episodes, still_eps)
    out = dict(safe_block)
    for k, v in raw_block.items():
        out["raw_" + k] = v
    changed = outside.any(axis=1)
    delta = np.abs(pred_safe - pred)
    out.update({
        "metric_version": METRIC_VERSION,
        "gated_on": "SAFE",
        "raw_oob_coord_rate": float(outside.mean()) if bounds is not None else float("nan"),
        "raw_oob_frame_rate": float(changed.mean()) if bounds is not None else float("nan"),
        "raw_oob_coord_count": int(outside.sum()),
        "raw_oob_frame_count": int(changed.sum()),
        "raw_oob_max_excursion": float(np.maximum(
            0.0, np.maximum(np.asarray(pred) - np.asarray(bounds[1]) if bounds else 0.0,
                            np.asarray(bounds[0]) - np.asarray(pred) if bounds else 0.0)).max())
        if bounds is not None else float("nan"),
        "safe_clip_frame_rate": float(changed.mean()) if bounds is not None else 0.0,
        "safe_clip_coord_rate": float((delta > 0).mean()),
        "safe_clip_mean_abs": float(delta.mean()),
        "safe_clip_max_abs": float(delta.max()),
        # legacy alias: v1 called the RAW out-of-box coordinate rate "joint_clip_rate"
        "joint_clip_rate": float(outside.mean()) if bounds is not None else float("nan"),
    })
    return out


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

    ev = {"eval_version": EVAL_VERSION, "metric_definitions": metric_definitions(),
          "delta_contract_sha256": sha256_file(CONTRACT),
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

    cv_m = ev["models"]["constant_velocity_A=S+V"]["metrics"]
    cv = cv_m["physical_l1_macro"]
    vla = ev["models"].get("vla_r42", {}).get("metrics")
    ev["gate"] = {"cv_macro_l1": cv,
                  "cv_macro_l1_raw": cv_m["raw_physical_l1_macro"],
                  "cv_raw_oob_coord_rate": cv_m["raw_oob_coord_rate"],
                  "cv_delta_sign_agreement": cv_m["delta_sign_agreement"],
                  "cv_delta_magnitude_ratio": cv_m["delta_magnitude_ratio"],
                  "vla_macro_l1": None if vla is None else vla["physical_l1_macro"],
                  "vla_macro_l1_raw": None if vla is None else vla["raw_physical_l1_macro"],
                  "vla_raw_oob_coord_rate": None if vla is None else vla["raw_oob_coord_rate"],
                  "vla_safe_clip_frame_rate": None if vla is None else vla["safe_clip_frame_rate"],
                  "vla_delta_sign_agreement": None if vla is None
                  else vla["delta_sign_agreement"],
                  "vla_delta_magnitude_ratio": None if vla is None
                  else vla["delta_magnitude_ratio"],
                  "gated_on": "SAFE",
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
        print("%-28s SAFE macroL1=%.6f | RAW macroL1=%.6f | arm=%.6f grip=%.6f R2=%+.4f"
              % (name, q["physical_l1_macro"], q["raw_physical_l1_macro"], q["arm_l1"],
                 q["gripper_l1"], q["correction_r2"]))
        print("%-28s   motion: dA sign=%.4f mag=%.4f | residual sign=%.4f mag=%.4f | "
              "still=%.3f | RAW-OOB coord=%.4f frame=%.4f | SAFE clip frames=%.4f"
              % ("", q["delta_sign_agreement"], q["delta_magnitude_ratio"],
                 q["residual_sign_agreement"], q["residual_magnitude_ratio"],
                 q["still_action_rate"], q["raw_oob_coord_rate"], q["raw_oob_frame_rate"],
                 q["safe_clip_frame_rate"]))
    print("gate:", json.dumps(ev["gate"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
