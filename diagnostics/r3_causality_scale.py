#!/usr/bin/env python3
"""第三轮：因果性取证 + 多样本机制差异定量（修正上一版探针的比较口径）。

上一版探针的因果性检查混用了两种注意力实现（第一次 output_attentions=True、
第二次 False），logits 会因 eager/sdpa 数值差异而整体改变，不能用来判断因果性。
本脚本把两次调用统一到同一实现，并额外：

  1. 确定性对照：同一输入跑两次，logits 必须完全一致；
  2. 注意力矩阵：逐行统计严格上三角的注意力质量（应为 0）；
  3. LM 隔离测试：直接对 language_model 做“篡改未来 token”，看过去位置是否变化；
  4. Prismatic 多模态路径同样测试（keep 机制下）；
  5. 多样本定量：20 个固定留出样本上 masked vs kept 的全序列预测差异；
  6. 自由运行：5 个样本上 masked vs kept 逐 token 生成的差异。

用法：python diagnostics/r3_causality_scale.py --n 20 --free-n 5 --out /tmp/r3_causal.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pure_bootstrap as B

B.bootstrap()
import r3_mask_probe as P3
from XPolicyLab.policy.OpenVLA_Pure import action_codec as AC

DEV = "cuda"


def attn_stats(attn):
    """attn: (heads, S, S) -> 严格上三角注意力质量统计。"""
    up = torch.triu(attn.float(), diagonal=1)
    row_above = up.abs().sum(-1)                     # (heads, S) 每行泄漏到未来的总权重
    diag = torch.diagonal(attn.float(), dim1=-2, dim2=-1)
    return {
        "shape": list(attn.shape),
        "max_abs_strictly_upper": float(up.abs().max()),
        "max_row_leak": float(row_above.max()),
        "mean_row_leak": float(row_above.mean()),
        "diag_max": float(diag.abs().max()),
        "diag_mean": float(diag.abs().mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--free-n", type=int, default=5)
    ap.add_argument("--out", default="")
    args, _u = ap.parse_known_args()

    m = B.load_model()
    lm = m.vla.language_model
    cfg = getattr(lm, "config", None)
    impl = getattr(cfg, "_attn_implementation", "?")
    print("[env] language_model=%s  attn_implementation=%s" % (type(lm).__name__, impl))
    res = {"attn_implementation": str(impl)}

    samples = B.iter_samples(n=args.n)
    p = P3.Probe(m, samples[0])
    seq = torch.cat([p.pids, p.aid], dim=1)

    # ---------- 1. 确定性对照（同实现、同输入）
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o1 = m.vla(input_ids=seq, attention_mask=torch.ones_like(seq), output_attentions=True)
        o2 = m.vla(input_ids=seq, attention_mask=torch.ones_like(seq), output_attentions=True)
    det = float((o1.logits.float() - o2.logits.float()).abs().max())
    res["determinism_maxdiff"] = det
    print("[1] determinism: same input twice (output_attentions=True) maxdiff = %.3e" % det)

    # ---------- 2. 注意力矩阵
    a0 = o1.attentions[0][0]
    st = attn_stats(a0)
    res["attn_layer0"] = st
    print("[2] attention layer0: %s" % json.dumps(st))
    last = o1.attentions[-1][0]
    res["attn_last_layer"] = attn_stats(last)
    print("    attention last layer: %s" % json.dumps(res["attn_last_layer"]))

    # ---------- 3. LM 隔离测试（同一 output_attentions 设置）
    pos_k = 10
    seq2 = seq.clone()
    seq2[0, p.plen + pos_k] = AC.LEGAL_TOKEN_MIN + 123
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        b0 = m.vla(input_ids=seq, attention_mask=torch.ones_like(seq), output_attentions=True)
        b1 = m.vla(input_ids=seq2, attention_mask=torch.ones_like(seq2), output_attentions=True)
    d = [float((b0.logits[0, i].float() - b1.logits[0, i].float()).abs().max())
         for i in range(seq.shape[1] - 1)]
    pred_pos = [p.plen + k - 1 for k in range(14)]
    res["prismatic_perturb"] = {
        "perturbed_action_index": pos_k,
        "logit_maxdiff_at_prediction_positions": [d[i] for i in pred_pos],
        "earlier_positions_unchanged": all(d[i] == 0.0 for i in pred_pos[:pos_k]),
    }
    print("[3] Prismatic path (same impl): perturb action#%d -> maxdiff at 14 prediction positions = %s" % (
        pos_k + 1, [round(d[i], 5) for i in pred_pos]))
    print("    earlier positions unchanged: %s" % res["prismatic_perturb"]["earlier_positions_unchanged"])

    # 正向对照：篡改“过去”的动作标记，其后的位置必须变化
    seq3 = seq.clone()
    seq3[0, p.plen + 3] = AC.LEGAL_TOKEN_MIN + 200
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        b2 = m.vla(input_ids=seq3, attention_mask=torch.ones_like(seq3), output_attentions=True)
    d3 = [float((b0.logits[0, i].float() - b2.logits[0, i].float()).abs().max()) for i in pred_pos]
    res["prismatic_perturb_past"] = {"perturbed_action_index": 3, "logit_maxdiff": d3,
                                     "later_positions_changed": any(x > 0 for x in d3[4:])}
    print("    control: perturb action#4 -> maxdiff = %s ; later changed=%s" % (
        [round(x, 5) for x in d3], res["prismatic_perturb_past"]["later_positions_changed"]))

    # ---------- 5. 多样本 masked vs kept（全序列，GT 前缀）
    rows = []
    for idx in range(len(samples)):
        pp = p if idx == 0 else P3.Probe(m, samples[idx])
        outT = pp._full_seq(mask_on=True)
        outF = pp._full_seq(mask_on=False)
        _, iT, aT = pp.action_metrics(outT)
        _, iF, aF = pp.action_metrics(outF)
        dec = pp.stage_deployment()
        rows.append({
            "episode": pp.e, "frame": int(pp.t),
            "tokens_equal": int((iT == iF).sum()),
            "action_l1": float(np.abs(aT - aF).mean()),
            "action_max": float(np.abs(aT - aF).max()),
            "deploy_vs_kept_tokens_equal": int(sum(a == b for a, b in zip(dec["tokens"], iF.tolist()))),
            "deploy_vs_masked_tokens_equal": int(sum(a == b for a, b in zip(dec["tokens"], iT.tolist()))),
        })
        print("  [%02d] %s t=%d tokens_equal=%2d/14 |dA|L1=%.5f deploy==kept %2d/14 deploy==masked %2d/14" % (
            idx, pp.e, pp.t, rows[-1]["tokens_equal"], rows[-1]["action_l1"],
            rows[-1]["deploy_vs_kept_tokens_equal"], rows[-1]["deploy_vs_masked_tokens_equal"]), flush=True)
    res["multi_full_seq"] = rows
    res["summary_full_seq"] = {
        "n": len(rows),
        "mean_tokens_equal": float(np.mean([r["tokens_equal"] for r in rows])),
        "mean_action_l1": float(np.mean([r["action_l1"] for r in rows])),
        "mean_deploy_vs_kept": float(np.mean([r["deploy_vs_kept_tokens_equal"] for r in rows])),
        "mean_deploy_vs_masked": float(np.mean([r["deploy_vs_masked_tokens_equal"] for r in rows])),
    }
    print("\n[5] full-seq masked vs kept over %d samples: mean tokens equal %.2f/14, mean |dA| L1 %.5f" % (
        len(rows), res["summary_full_seq"]["mean_tokens_equal"], res["summary_full_seq"]["mean_action_l1"]))
    print("    deployment vs kept-prefix: %.2f/14 ; deployment vs masked-prefix: %.2f/14" % (
        res["summary_full_seq"]["mean_deploy_vs_kept"], res["summary_full_seq"]["mean_deploy_vs_masked"]))

    # ---------- 6. 自由运行 masked vs kept
    free = []
    for idx in range(min(args.free_n, len(samples))):
        pp = p if idx == 0 else P3.Probe(m, samples[idx])
        fr = pp.stage_free_running()
        d = pp.stage_deployment()
        free.append({"episode": pp.e, "frame": int(pp.t),
                     "tokens_equal": fr["tokens_equal"],
                     "action_l1": fr["action_l1"],
                     "deploy_eq_kept": d["tokens"] == fr["idx_kept_mechanism"],
                     "deploy_eq_masked": d["tokens"] == fr["idx_masked_mechanism"]})
        print("  free[%02d] %s t=%d tokens_equal=%2d/14 |dA|L1=%.5f deploy==kept %s deploy==masked %s" % (
            idx, pp.e, pp.t, free[-1]["tokens_equal"], free[-1]["action_l1"],
            free[-1]["deploy_eq_kept"], free[-1]["deploy_eq_masked"]), flush=True)
    res["multi_free_run"] = free
    res["summary_free_run"] = {
        "n": len(free),
        "mean_tokens_equal": float(np.mean([r["tokens_equal"] for r in free])) if free else None,
        "mean_action_l1": float(np.mean([r["action_l1"] for r in free])) if free else None,
        "deploy_eq_kept_all": all(r["deploy_eq_kept"] for r in free),
    }
    print("\n[6] free-run masked vs kept over %d samples: mean tokens equal %.2f/14, mean |dA| L1 %.5f" % (
        len(free), res["summary_free_run"]["mean_tokens_equal"], res["summary_free_run"]["mean_action_l1"]))
    print("    deployment == free-run(kept) for all samples: %s" % res["summary_free_run"]["deploy_eq_kept_all"])

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("\nwrote " + args.out)
    print("R3_CAUSAL_SCALE_DONE")


if __name__ == "__main__":
    main()
