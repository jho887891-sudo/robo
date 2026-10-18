#!/usr/bin/env python3
"""五阶段（可选）：100 个 push_T 专家回合的动作统计 —— 绝对动作 vs 残差动作。

纯 CPU，不训练、不改数据。回答三个问题：

  1. 专家每步的真实位移有多小（|A[t] - S[t]| 的逐维分布）？
  2. 若把目标改成残差 Δ = A[t] - S[t]，256 个 bin 的分辨率与动态范围是否更合适？
  3. 三个参照误差（恒等 S、上一步动作 A[t-1]、预测均值）分别是多少 —— 用作下一轮训练的基线。

用法：python diagnostics/action_delta_analysis.py --out /tmp/action_delta.json
"""
import argparse
import json
import os
import sys

import numpy as np

HDF5 = os.environ.get("PUSHT_HDF5", "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
SPLIT = os.environ.get("PUSHT_SPLIT", "/home/dgut/pusht_runtime/split80.json")
BOUNDS = os.environ.get("PUSHT_BOUNDS", "/home/dgut/pusht_runtime/push_T_bounds.npz")
DIMS = (["left_arm_j%d" % i for i in range(6)] + ["left_gripper"] +
        ["right_arm_j%d" % i for i in range(6)] + ["right_gripper"])


def c14(h, g):
    return np.concatenate([
        h[g + "/left_arm_joint_states"][:], h[g + "/left_ee_joint_states"][:],
        h[g + "/right_arm_joint_states"][:], h[g + "/right_ee_joint_states"][:],
    ], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="回合数上限")
    ap.add_argument("--out", default="")
    args, _u = ap.parse_known_args()

    import h5py
    split = json.load(open(SPLIT))
    eps = sorted(split["train"]) + sorted(split["val"])
    eps = eps[:args.n]
    A, S, APREV, dA = [], [], [], []
    for e in eps:
        p = os.path.join(HDF5, e)
        if not os.path.exists(p):
            continue
        with h5py.File(p, "r") as h:
            a = c14(h, "action")
            s = c14(h, "state")
        A.append(a[1:]); S.append(s[:-1])          # A[t] 与 S[t-1] 对齐
        APREV.append(a[:-1]); dA.append(a[1:] - a[:-1])     # 上一步动作 / 相邻步变化
    A = np.concatenate(A, 0); S = np.concatenate(S, 0)
    APREV = np.concatenate(APREV, 0); dA = np.concatenate(dA, 0)
    D = A - S                                       # 残差目标
    n = A.shape[0]

    q = lambda x, p: np.quantile(x, p, axis=0).astype(np.float64)
    abs_q01, abs_q99 = q(A, 0.01), q(A, 0.99)
    abs_min, abs_max = A.min(0).astype(np.float64), A.max(0).astype(np.float64)
    res_q01, res_q99 = q(D, 0.01), q(D, 0.99)
    res_absmax = np.abs(D).max(0).astype(np.float64)

    bin_abs = (abs_q99 - abs_q01) / 255.0
    bin_res = (res_q99 - res_q01) / 255.0
    bin_abs_full = (abs_max - abs_min) / 255.0
    step_abs = np.abs(D).mean(0)                    # 每维平均每步位移
    step_bins_abs = step_abs / np.maximum(bin_abs, 1e-12)
    step_bins_res = step_abs / np.maximum(bin_res, 1e-12)

    rows = []
    for i, name in enumerate(DIMS):
        rows.append({
            "dim": name,
            "action_range_q01_q99": [float(abs_q01[i]), float(abs_q99[i])],
            "bin_width_abs_q": float(bin_abs[i]),
            "bin_width_abs_full": float(bin_abs_full[i]),
            "residual_q01_q99": [float(res_q01[i]), float(res_q99[i])],
            "bin_width_res": float(bin_res[i]),
            "mean_abs_step": float(step_abs[i]),
            "step_in_bins_abs": float(step_bins_abs[i]),
            "step_in_bins_res": float(step_bins_res[i]),
            "frac_moves_lt_1bin_abs": float((np.abs(D[:, i]) < bin_abs[i]).mean()),
        })

    nb = np.load(BOUNDS)
    amin, amax = nb["amin"], nb["amax"]
    nrm = lambda x: np.clip(2.0 * (x - amin) / (amax - amin + 1e-8) - 1.0, -1, 1)
    res = {
        "n_episodes": len(A) and len(eps),
        "n_transitions": int(n),
        "reference_errors": {
            "identity_S_vs_A_bounds": float(np.abs(nrm(S) - nrm(A)).mean()),
            "identity_S_vs_A_physical": float(np.abs(S - A).mean()),
            "previous_action_vs_A_bounds": float(np.abs(nrm(APREV) - nrm(A)).mean()),
            "previous_action_vs_A_physical": float(np.abs(APREV - A).mean()),
        },
        "absolute_action": {
            "mean_abs_value": float(np.abs(A).mean()),
            "mean_abs_step": float(np.abs(D).mean()),
            "per_dim_mean_abs_step": np.abs(D).mean(0).tolist(),
            "corr_with_state": float(np.corrcoef(A.reshape(-1), S.reshape(-1))[0, 1]),
            "r2_predict_A_from_S": float(1.0 - np.var(A - S) / np.var(A)),
        },
        "residual_action": {
            "mean_abs_value": float(np.abs(D).mean()),
            "std": float(D.std()),
            "per_dim_mean_abs": np.abs(D).mean(0).tolist(),
            "per_dim_q99_abs": np.abs(q(D, 0.99)).tolist(),
            "fraction_of_range_used_by_abs_norm": float(((A - abs_q01) / np.maximum(abs_q99 - abs_q01, 1e-9)).std()),
        },
        "binning": {
            "note": "bin_width = (q99-q01)/255；step_in_bins = mean|Δ| / bin_width",
            "abs_bin_width_mean": float(bin_abs.mean()),
            "res_bin_width_mean": float(bin_res.mean()),
            "mean_step_in_bins_abs": float(np.mean(step_bins_abs)),
            "mean_step_in_bins_res": float(np.mean(step_bins_res)),
            "per_dim": rows,
        },
        "temporal": {
            "action_change_between_steps_mean_abs": float(np.abs(dA).mean()),
            "action_change_q99_abs": float(np.abs(q(dA, 0.99)).max()),
        },
    }

    print("episodes=%d transitions=%d" % (res["n_episodes"], res["n_transitions"]))
    print("\n参照误差（同一批转移，BOUNDS 归一化 / 物理单位）:")
    for k, v in res["reference_errors"].items():
        print("  %-34s %.6f" % (k, v))
    print("\n绝对动作: mean|A|=%.4f  mean|Δ|=%.4f  corr(A,S)=%.4f  R2(A|S)=%.4f" % (
        res["absolute_action"]["mean_abs_value"], res["absolute_action"]["mean_abs_step"],
        res["absolute_action"]["corr_with_state"], res["absolute_action"]["r2_predict_A_from_S"]))
    print("残差动作: mean|Δ|=%.4f  std=%.4f" % (
        res["residual_action"]["mean_abs_value"], res["residual_action"]["std"]))
    print("分箱尺度: 绝对 bin 宽均值 %.5f（平均每步 %.2f bin）; 残差 bin 宽均值 %.5f（平均每步 %.2f bin）" % (
        res["binning"]["abs_bin_width_mean"], res["binning"]["mean_step_in_bins_abs"],
        res["binning"]["res_bin_width_mean"], res["binning"]["mean_step_in_bins_res"]))
    print("\n%-14s %10s %10s %12s %12s %10s" % ("dim", "mean|Δ|", "bin_abs", "step_bin_abs", "bin_res", "step_bin_res"))
    for r in rows:
        print("%-14s %10.3e %10.3e %12.2f %12.3e %10.2f" % (
            r["dim"], r["mean_abs_step"], r["bin_width_abs_q"], r["step_in_bins_abs"],
            r["bin_width_res"], r["step_in_bins_res"]))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("\nwrote " + args.out)
    print("ACTION_DELTA_DONE")


if __name__ == "__main__":
    main()
