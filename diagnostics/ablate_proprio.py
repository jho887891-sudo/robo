#!/usr/bin/env python3
"""本体状态（proprio）消融：固定图像与指令，比较真实状态 / 零状态 / 中点状态。

对应第一轮审查问题 4。相比旧版 diagnostics/ablate.py 的改进：
  * 固定同一帧图像与同一条指令（旧版会同时换图像，混淆了变量）；
  * 14 维动作全部打印（旧版只打印前 8 维）；
  * 额外打印 14 步动作 token 的概率（top1 概率 + top3 候选）；
  * 额外打印 ProprioProjector 的输出特征（范数、统计量、与真实状态的 L2 距离）；
  * 明确区分“原始零状态”与“归一化后的零特征”（state=(pmin+pmax)/2）。

用法：
  python diagnostics/ablate_proprio.py --sample 0 --out /home/dgut/pusht_runtime/r2_ablate.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pure_bootstrap as B


def proj_feats(m, state):
    """把 14 维状态投影成 LLM 维度特征（与模型内部完全相同的调用）。"""
    p = np.clip(2.0 * (np.asarray(state, np.float32) - m.pmin) /
                (m.pmax - m.pmin + 1e-8) - 1.0, -1.0, 1.0).astype(np.float32)
    t = torch.from_numpy(p).unsqueeze(0)
    w = next(m.proprio_projector.parameters())
    out = m.proprio_projector(t.to(w.dtype).to(m.device))
    if isinstance(out, (tuple, list)):
        out = out[0]
    return p, out.detach().float().cpu().numpy().reshape(-1)


def run_cond(m, imgs, state, tag):
    m.update_obs(B.make_obs(imgs, state, m.instr))
    action = np.asarray(m.infer(m.observation_window[0])[0], dtype=np.float32)
    dec = m.last_decode
    pnorm, feat = proj_feats(m, state)
    return {"tag": tag, "state": np.asarray(state, np.float32), "proprio_norm": pnorm,
            "action": action, "delta": action - np.asarray(state, np.float32),
            "tokens": dec["tokens"], "token_ids": dec["token_ids"],
            "probs": dec["probs"], "proj": feat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="fixed_samples 下标（固定图像来源）")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    m = B.load_model(task="push_T")
    samples = B.iter_samples(n=args.n)
    e, t, imgs, state, gt_action = samples[args.sample]
    print("[ablate] image = %s t=%d (all 3 cameras fixed)" % (e, t), flush=True)
    print("[ablate] instruction = %r (fixed)" % m.instr, flush=True)

    zero = np.zeros(14, dtype=np.float32)
    mid = ((m.pmin + m.pmax) / 2.0).astype(np.float32)
    A = run_cond(m, imgs, state, "real_state")
    Z = run_cond(m, imgs, zero, "zero_state")
    M = run_cond(m, imgs, mid, "mid_state")

    print("\n=== proprio input (14 dims, normalized) ===")
    print("%-14s %10s %10s %10s" % ("dim", "real", "zero(raw0)", "mid"))
    for i in range(14):
        print("%-14s %10.4f %10.4f %10.4f" % (B.DIM_NAMES[i], A["proprio_norm"][i],
                                              Z["proprio_norm"][i], M["proprio_norm"][i]))

    print("\n=== action output (all 14 dims) ===")
    print("%-14s %11s %11s %11s %11s %11s" % ("dim", "real", "zero", "mid", "|real-zero|", "gt"))
    for i in range(14):
        print("%-14s %11.5f %11.5f %11.5f %11.5f %11.5f" % (
            B.DIM_NAMES[i], A["action"][i], Z["action"][i], M["action"][i],
            abs(A["action"][i] - Z["action"][i]), gt_action[i]))

    print("\n=== action - current state (14 dims) ===")
    print("%-14s %11s %11s %11s" % ("dim", "real-S", "zero-S", "mid-S"))
    for i in range(14):
        print("%-14s %11.5f %11.5f %11.5f" % (B.DIM_NAMES[i], A["delta"][i], Z["delta"][i], M["delta"][i]))

    print("\n=== action tokens and probabilities (14 steps, 256-bin window) ===")
    print("%-14s %6s %6s %8s %8s %s" % ("step", "idx", "token", "top1_p", "zero_p", "top3(idx:prob)"))
    for i in range(14):
        pa, pz = A["probs"][i], Z["probs"][i]
        top3 = " ".join("%d:%.3f" % (j, q) for j, q in pa["top3"])
        print("%-14s %6d %6d %8.4f %8.4f %s" % (
            B.DIM_NAMES[i], A["tokens"][i], A["token_ids"][i], pa["p"], pz["p"], top3))
    same_tok = sum(1 for i in range(14) if A["tokens"][i] == Z["tokens"][i])
    print("identical token positions: %d/14 ; max |action(real)-action(zero)| = %.6f" % (
        same_tok, float(np.abs(A["action"] - Z["action"]).max())))

    print("\n=== ProprioProjector output features ===")
    for C in (A, Z, M):
        f = C["proj"]
        print("%-11s dim=%d  |f|=%.4f  mean=%.5f  std=%.5f  min=%.4f  max=%.4f  f[:8]=%s" % (
            C["tag"], f.size, float(np.linalg.norm(f)), float(f.mean()), float(f.std()),
            float(f.min()), float(f.max()), np.round(f[:8], 4).tolist()))
    print("L2(real,zero)=%.5f   L2(real,mid)=%.5f   cos(real,zero)=%.5f" % (
        float(np.linalg.norm(A["proj"] - Z["proj"])),
        float(np.linalg.norm(A["proj"] - M["proj"])),
        float(np.dot(A["proj"], Z["proj"]) / (np.linalg.norm(A["proj"]) * np.linalg.norm(Z["proj"]) + 1e-12))))

    res = {
        "sample": {"episode": e, "t": t, "instruction": m.instr},
        "conditions": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in C.items() if k != "proj"} | {"proj_stats": {
                            "norm": float(np.linalg.norm(C["proj"])),
                            "mean": float(C["proj"].mean()),
                            "std": float(C["proj"].std()),
                            "first8": C["proj"][:8].tolist()}}
                       for C in (A, Z, M)],
        "l2_real_zero": float(np.linalg.norm(A["proj"] - Z["proj"])),
        "max_action_diff_real_zero": float(np.abs(A["action"] - Z["action"]).max()),
        "same_token_positions": same_tok,
        "dim_names": B.DIM_NAMES,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("\nwrote " + args.out)
    print("ABLATE_PROPRIO_DONE")


if __name__ == "__main__":
    main()
