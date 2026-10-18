#!/usr/bin/env python3
"""统一离线评测：现有 6000 步纯 OpenVLA 检查点 vs 恒等策略（固定 20 个留出样本）。

要点（对应第一轮审查问题 2）
----------------------------
* 输入：三路相机（cam_head / cam_left_wrist / cam_right_wrist）+ 14 维本体状态；
* 权重：直接实例化部署类 XPolicyLab.policy.OpenVLA_Pure.model.Model，
  因此 adapter / proprio_projector / q01-q99 / BOUNDS 与线上服务同一套加载路径；
* 样本：PUSHT_SAMPLES（fixed_samples.json）前 N 个，全部来自 20 个留出回合；
* 归一化：统一用 BOUNDS（push_T_bounds.npz 的 amin/amax），与 OFT 评测可比；
  同时报告物理单位误差；
* 额外报告：逐维误差、预测动作变化量 mean|A_hat - S| 与真实变化量 mean|A - S|、
  动作 token 合法性（PAD 数必须为 0）、解码唯一性。

不做的事：不训练、不修改检查点、不接仿真。

用法
----
  python diagnostics/eval_unified.py --n 20 --out /home/dgut/pusht_runtime/r2_eval.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pure_bootstrap as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--task", type=str, default="push_T")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    m = B.load_model(task=args.task)
    amin, amax = B.bounds()
    samples = B.iter_samples(n=args.n)
    print("[eval] model=%s samples=%d task=%s" % (m.model_cfg["adapter_dir"], len(samples), args.task), flush=True)

    P_raw, P_id, G_raw, S_raw = [], [], [], []
    tok_all, pad_like, top1 = [], 0, []
    for i, (e, t, imgs, state, action) in enumerate(samples):
        m.update_obs(B.make_obs(imgs, state, m.instr))
        pred = np.asarray(m.infer(m.observation_window[0])[0], dtype=np.float32)
        dec = m.last_decode
        tok_all.append(dec["tokens"])
        if any(int(x) == 256 or int(x) < 0 for x in dec["tokens"]):
            pad_like += 1
        top1.extend([p["p"] for p in dec["probs"]])
        P_raw.append(pred); P_id.append(state.copy()); G_raw.append(action); S_raw.append(state.copy())
        print("  [%02d] %s t=%d |pred-S|=%.5f |gt-S|=%.5f" % (
            i, e, t, float(np.abs(pred - state).mean()), float(np.abs(action - state).mean())), flush=True)

    P = np.stack(P_raw); PI = np.stack(P_id); G = np.stack(G_raw); S = np.stack(S_raw)
    nb = lambda a: B.norm_bounds(a, amin, amax)
    Pb, PIb, Gb, Sb = nb(P), nb(PI), nb(G), nb(S)

    def summ(u, v):
        d = np.abs(u - v)
        return {"bounds_L1": float(d.mean()), "phys_L1": float(np.abs(u - v).mean()),
                "bounds_per_dim": d.mean(0).tolist()}

    model_b = float(np.abs(Pb - Gb).mean())
    ident_b = float(np.abs(PIb - Gb).mean())
    model_p = float(np.abs(P - G).mean())
    ident_p = float(np.abs(PI - G).mean())
    d_pred = np.abs(P - S)
    d_gt = np.abs(G - S)
    res = {
        "n_samples": len(samples),
        "task": args.task,
        "adapter_dir": m.model_cfg["adapter_dir"],
        "proprio_projector_path": m.model_cfg["proprio_projector_path"],
        "stats_path": m.model_cfg["stats_path"],
        "normalization": "BOUNDS (push_T_bounds.npz amin/amax)",
        "decode_mode": getattr(m, "last_decode", {}).get("action_codec_mode"),
        "metrics": {
            "model": {
                "bounds_L1": model_b,
                "phys_L1": model_p,
                "bounds_per_dim": np.abs(Pb - Gb).mean(0).tolist(),
                "phys_per_dim": np.abs(P - G).mean(0).tolist(),
                "delta_pred_mean_abs": float(d_pred.mean()),
                "delta_pred_per_dim": d_pred.mean(0).tolist(),
                "action_std_over_samples": P.std(0).tolist(),
                "unique_rows": len({tuple(np.round(r, 3)) for r in P}),
            },
            "identity": {
                "bounds_L1": ident_b,
                "phys_L1": ident_p,
                "bounds_per_dim": np.abs(PIb - Gb).mean(0).tolist(),
                "phys_per_dim": np.abs(PI - G).mean(0).tolist(),
                "delta_pred_mean_abs": float(np.abs(PI - S).mean()),
            },
            "gt": {
                "delta_gt_mean_abs": float(d_gt.mean()),
                "delta_gt_per_dim": d_gt.mean(0).tolist(),
                "action_std_over_samples": G.std(0).tolist(),
            },
            "ratio_model_over_identity": model_b / (ident_b + 1e-12),
            "pred_vs_state_mae": float(np.abs(P - S).mean()),
            "gt_vs_state_mae": float(np.abs(G - S).mean()),
            "corr_pred_state": float(np.corrcoef(P.reshape(-1), S.reshape(-1))[0, 1]),
            "corr_gt_state": float(np.corrcoef(G.reshape(-1), S.reshape(-1))[0, 1]),
            "corr_pred_gt": float(np.corrcoef(P.reshape(-1), G.reshape(-1))[0, 1]),
            "ratio_pred_delta_over_gt_delta": float(d_pred.mean() / (d_gt.mean() + 1e-12)),
        },
        "tokens": {
            "n_positions": int(np.asarray(tok_all).size),
            "min_idx": int(np.asarray(tok_all).min()),
            "max_idx": int(np.asarray(tok_all).max()),
            "pad_or_illegal": pad_like,
            "frac_at_idx0": float((np.asarray(tok_all) == 0).mean()),
            "frac_at_idx255": float((np.asarray(tok_all) == 255).mean()),
            "mean_top1_prob": float(np.mean(top1)),
            "min_top1_prob": float(np.min(top1)),
        },
        "raw": {
            "episodes": [s[0] for s in samples],
            "frames": [int(s[1]) for s in samples],
            "state": S.tolist(),
            "pred": P.tolist(),
            "gt": G.tolist(),
            "identity": PI.tolist(),
        },
        "per_dim_table": [
            {"dim": B.DIM_NAMES[i],
             "model_bounds": float(np.abs(Pb - Gb).mean(0)[i]),
             "identity_bounds": float(np.abs(PIb - Gb).mean(0)[i]),
             "model_phys": float(np.abs(P - G).mean(0)[i]),
             "identity_phys": float(np.abs(PI - G).mean(0)[i]),
             "pred_delta": float(d_pred.mean(0)[i]),
             "gt_delta": float(d_gt.mean(0)[i])}
            for i in range(14)
        ],
    }

    print("\n=== summary (%d samples) ===" % len(samples))
    print("model    : BOUNDS L1 = %.4f   physical L1 = %.5f" % (model_b, model_p))
    print("identity : BOUNDS L1 = %.4f   physical L1 = %.5f" % (ident_b, ident_p))
    print("model/identity = %.2fx (>1 means worse than copying current state)" % res["metrics"]["ratio_model_over_identity"])
    print("delta    : mean|A_hat - S| = %.5f  vs  mean|A - S| = %.5f  (ratio %.3f)" % (
        d_pred.mean(), d_gt.mean(), res["metrics"]["ratio_pred_delta_over_gt_delta"]))
    print("\n=== per-dim (all 14) ===")
    print("%-14s %10s %10s %10s %10s %10s %10s" % ("dim", "model_bnd", "ident_bnd", "model_phys", "ident_phys", "pred_dlt", "gt_dlt"))
    for r in res["per_dim_table"]:
        print("%-14s %10.4f %10.4f %10.5f %10.5f %10.5f %10.5f" % (
            r["dim"], r["model_bounds"], r["identity_bounds"], r["model_phys"], r["identity_phys"], r["pred_delta"], r["gt_delta"]))
    print("\n=== token ===")
    print(json.dumps(res["tokens"], indent=1))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("\nwrote " + args.out)
    print("EVAL_UNIFIED_DONE")


if __name__ == "__main__":
    main()
