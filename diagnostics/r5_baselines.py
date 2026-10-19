#!/usr/bin/env python3
"""R5 附加：独立计算四个"简单策略"基线（纯 CPU，不训练、不动 GPU）。

目的：R5 的 r5_metrics 用 prior_state = 上一帧 **action** 计算"恒定速度"，
而本数据集 A[t-1] == S[t]，导致该基线退化成恒等策略。这里额外用真实相邻
**state** 计算真正的恒定速度基线，并给出训练集残差中位数策略。

基线（全部先还原为绝对关节命令，再算 14 维物理 L1）：
  hold_state          : Â = S[t]
  cv_prev_action      : Â = S[t] + (S[t] - A[t-1])      ← R5 脚本内部实现
  cv_prev_state       : Â = S[t] + (S[t] - S[t-1])      ← 真正的相邻状态外推
  train_median_delta  : Â = S[t] + median_train(A - S)

用法：python diagnostics/r5_baselines.py --out /tmp/r5_baselines.json
"""
import argparse
import json
import os
import sys

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "training"))
from r5_sampling import spread_validation_frames  # noqa: E402

RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
HDF5 = os.environ.get("PUSHT_HDF5", "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
SPLIT = os.environ.get("PUSHT_SPLIT", RT + "/split80.json")
DIMS = (["left_arm_j%d" % i for i in range(6)] + ["left_gripper"] +
        ["right_arm_j%d" % i for i in range(6)] + ["right_gripper"])


def c14(h, g):
    return np.concatenate([h[g + "/left_arm_joint_states"][:], h[g + "/left_ee_joint_states"][:],
                           h[g + "/right_arm_joint_states"][:], h[g + "/right_ee_joint_states"][:]],
                          axis=1).astype(np.float32)


def stats(err, episodes):
    per_ep = {}
    for e in sorted(set(episodes)):
        idx = [i for i, x in enumerate(episodes) if x == e]
        per_ep[e] = float(np.mean(err[idx]))
    return {"overall": float(np.mean(err)),
            "left": float(np.mean(err[:, :7])),
            "right": float(np.mean(err[:, 7:])),
            "macro_over_episodes": float(np.mean(list(per_ep.values()))),
            "per_dim": np.mean(err, axis=0).tolist(),
            "per_episode": per_ep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args, _u = ap.parse_known_args()

    split = json.load(open(SPLIT))
    train, val = list(split["train"]), list(split["val"])
    val_lengths = {}
    for e in sorted(val):
        with h5py.File(os.path.join(HDF5, e), "r") as h:
            val_lengths[e] = int(h["action/left_arm_joint_states"].shape[0])
    val_pairs = spread_validation_frames(split, val_lengths, args.val_n, seed=args.seed)

    S, G, PA, PS, EP, frame_ids = [], [], [], [], [], []
    for e, t in val_pairs:
        with h5py.File(os.path.join(HDF5, e), "r") as h:
            st, ac = c14(h, "state"), c14(h, "action")
        S.append(st[t]); G.append(ac[t])
        PA.append(ac[max(t - 1, 0)]); PS.append(st[max(t - 1, 0)])
        EP.append(e); frame_ids.append([e, int(t)])
    S, G, PA, PS = map(np.stack, (S, G, PA, PS))

    deltas = []
    for e in sorted(train):
        with h5py.File(os.path.join(HDF5, e), "r") as h:
            st, ac = c14(h, "state"), c14(h, "action")
        deltas.append((ac - st)[2:-1])
    median_delta = np.median(np.concatenate(deltas, 0), axis=0).astype(np.float32)

    cand = {
        "hold_state": S,
        "cv_prev_action": S + (S - PA),
        "cv_prev_state": S + (S - PS),
        "train_median_delta": S + median_delta,
    }
    res = {"git_commit": os.popen("git -C %s rev-parse HEAD" % REPO).read().strip(),
           "n_val_frames": len(S), "n_val_episodes": len({e for e, _ in val_pairs}),
           "median_train_delta": median_delta.tolist(),
           "mean_abs_prior_action_minus_state": float(np.abs(PA - S).mean()),
           "mean_abs_prior_state_minus_state": float(np.abs(PS - S).mean()),
           "expert_delta_mean_abs": float(np.abs(G - S).mean()),
           "baselines": {}}
    for name, cmd in cand.items():
        res["baselines"][name] = stats(np.abs(cmd - G), EP)
    res["degeneracy_check"] = {
        "cv_prev_action_equals_hold_state": bool(np.allclose(cand["cv_prev_action"], cand["hold_state"], atol=1e-6)),
        "max_abs_diff_cv_prev_action_vs_hold": float(np.abs(cand["cv_prev_action"] - cand["hold_state"]).max()),
    }

    print("commit:", res["git_commit"])
    print("val frames=%d episodes=%d | mean|A[t-1]-S[t]|=%.6f | mean|S[t-1]-S[t]|=%.6f | expert |Δ|=%.6f" % (
        res["n_val_frames"], res["n_val_episodes"], res["mean_abs_prior_action_minus_state"],
        res["mean_abs_prior_state_minus_state"], res["expert_delta_mean_abs"]))
    print("%-22s %10s %10s %10s %12s" % ("baseline", "overall", "left", "right", "macro/ep"))
    for name, s in res["baselines"].items():
        print("%-22s %10.5f %10.5f %10.5f %12.5f" % (name, s["overall"], s["left"], s["right"],
                                                       s["macro_over_episodes"]))
    print("degeneracy: cv_prev_action == hold_state ? %s (maxdiff %.2e)" % (
        res["degeneracy_check"]["cv_prev_action_equals_hold_state"],
        res["degeneracy_check"]["max_abs_diff_cv_prev_action_vs_hold"]))
    print("per-dim hold_state :", [round(x, 5) for x in res["baselines"]["hold_state"]["per_dim"]])
    print("per-dim cv_prev_st :", [round(x, 5) for x in res["baselines"]["cv_prev_state"]["per_dim"]])
    print("per-dim median_dlt :", [round(x, 5) for x in res["baselines"]["train_median_delta"]["per_dim"]])
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
        print("wrote", args.out)
    print("R5_BASELINES_DONE")


if __name__ == "__main__":
    main()
