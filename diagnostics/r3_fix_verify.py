#!/usr/bin/env python3
"""第三轮修复后的验证：机制可配置 + 部署一致 + 因果性（修正版）+ 上下文敏感性。

用法：
  python diagnostics/r3_fix_verify.py --baseline /tmp/r3_probe.json --out /tmp/r3_fix.json

检查项（失败会打印 FAIL 并以非零码退出）：
  T1 默认行为未变：_resolve_action_input_masking() 默认 True（zero），且 set_action_input_masking 可用
  T2 OFT 回归：mask_action_inputs=True 的 14 个标记 == 修复前探针(A) 的结果
  T3 修复生效：mask_action_inputs=False 的 14 个标记 == 修复前探针(B, 不置零) 的结果
  T4 嵌入事实：True 时动作位置输入嵌入全零；False 时非零
  T5 部署一致：自由运行(False) 与 model.Model.infer(KV 缓存) 逐 token 一致
  T6 因果性（logits 层面，两次调用都 output_attentions=False）：
       篡改第 11 个动作标记，前 10 个预测位置的 logits 必须完全相同；
       正向对照：篡改第 4 个动作标记，其后位置必须变化
  T7 上下文敏感性：把 14 个前序动作标记整体替换后，预测变化多少（keep 机制下模型是否真的用前序动作）
  INFO 记录一个容易被误读的现象：本机 attn_implementation=sdpa 时，output_attentions=True 会退回
       eager 且不建因果掩码，返回的注意力矩阵上三角非零 —— 这是 HF 的已知陷阱，不是模型缺陷；
       因此因果性判定只看 logits 层面。
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


PREFIX_REFERENCE = {   # 修复前探针日志（2026-10-18 16:59 UTC）记录的 14 个窗口下标
    "masked": [212, 187, 146, 44, 81, 143, 1, 106, 1, 1, 255, 185, 189, 255],
    "kept_same_sample": [212, 187, 146, 44, 81, 143, 1, 106, 1, 1, 255, 185, 189, 255],
}


def fwd(p, m, mask_kwarg, with_labels=True, ids=None, output_attentions=False):
    ids = p.inp if ids is None else ids
    if not with_labels:
        lab = None
    elif ids is p.inp:
        lab = p.lab
    else:
        # 与训练一致：提示段 IGNORE，动作段（含 STOP）用真实 token 作为标签
        lab = torch.full_like(ids, -100)
        lab[:, p.plen:] = ids[:, p.plen:]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = p.vla(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values=p.pv,
            labels=lab,
            proprio=p.proprio,
            proprio_projector=m.proprio_projector,
            mask_action_inputs=mask_kwarg,
            output_attentions=output_attentions,
        )
    if p.n_vis is None:
        p.n_vis = int(out.logits.shape[1]) - int(ids.shape[1])
    return out


def lm_only(m, ids, output_attentions=False):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return m.vla(input_ids=ids, attention_mask=torch.ones_like(ids),
                     output_attentions=output_attentions)


def free_run(p, m, mask_kwarg):
    toks = []
    for _ in range(14):
        ids = p.pids if not toks else torch.cat(
            [p.pids, torch.tensor([[AC.ACTION_TOKEN_BEGIN_IDX + 1 + t for t in toks]],
                                  dtype=torch.long, device=DEV)], dim=1)
        out = fwd(p, m, mask_kwarg, with_labels=True, ids=ids)
        toks.append(P3.window_probs(out.logits[0, -1])["idx"])
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="/tmp/r3_probe.json")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default="")
    args, _unk = ap.parse_known_args()

    base = json.load(open(args.baseline)) if os.path.exists(args.baseline) else None
    if base is None:
        base = {"full_seq": {"per_position": [{"idx_masked": PREFIX_REFERENCE["masked"][i],
                                              "idx_kept": PREFIX_REFERENCE["kept_same_sample"][i]}
                                             for i in range(14)]}}
        print("[T2/T3] using pre-fix reference tokens recorded in the 16:59 UTC probe log")
    m = B.load_model()
    sample = B.iter_samples(n=args.sample + 1)[args.sample]
    p = P3.Probe(m, sample)
    res, checks = {}, {}

    # ---- T1
    default_zero = bool(p.vla._resolve_action_input_masking())
    p.vla.set_action_input_masking("zero")
    attr_zero = p.vla.action_input_masking == "zero"
    p.vla.set_action_input_masking("keep")
    res["T1"] = {"resolve_default_zero": default_zero, "attr": p.vla.action_input_masking}
    checks["T1_default_zero"] = default_zero and attr_zero
    print("[T1] default masking zero = %s ; set_action_input_masking ok = %s" % (default_zero, attr_zero))

    # ---- T2 / T3
    out_true = fwd(p, m, True)
    out_false = fwd(p, m, False)
    _, iT, aT = p.action_metrics(out_true)
    _, iF, aF = p.action_metrics(out_false)
    res["T2_T3"] = {"idx_mask_true": iT.tolist(), "idx_mask_false": iF.tolist()}
    if base is not None:
        pref_masked = [r["idx_masked"] for r in base["full_seq"]["per_position"]]
        pref_kept = [r["idx_kept"] for r in base["full_seq"]["per_position"]]
        checks["T2_oft_regression"] = iT.tolist() == pref_masked
        checks["T3_fix_effective"] = iF.tolist() == pref_kept
        print("[T2] mask=True  == pre-fix masked? %s   %s" % (checks["T2_oft_regression"], iT.tolist()))
        print("[T3] mask=False == pre-fix kept?   %s   %s" % (checks["T3_fix_effective"], iF.tolist()))
    else:
        print("[T2/T3] baseline json missing; skipped strict comparison")

    # ---- T4
    res["T4"] = {}
    for tag, flag in (("zero", True), ("keep", False)):
        h = p.vla.language_model.register_forward_hook(p._hook(tag), with_kwargs=True)
        try:
            fwd(p, m, flag)
        finally:
            h.remove()
        emb = p.captured[tag][0]
        p0 = p.n_vis + p.plen
        mx = float(emb[p0:p0 + 14].abs().max())
        res["T4"][tag] = mx
        print("[T4] mask=%-5s max|emb| at 14 action positions = %.6f" % (flag, mx))
    checks["T4_zero_when_true"] = res["T4"]["zero"] == 0.0
    checks["T4_nonzero_when_false"] = res["T4"]["keep"] > 1e-3

    # ---- T5 部署一致（用当前样本重建 obs，避免 stage_deployment 依赖上一次 update_obs）
    free = free_run(p, m, False)
    m.update_obs(B.make_obs(sample[2], sample[3], m.instr))
    dep = p.stage_deployment()
    checks["T5_deploy_matches_free_run"] = free == dep["tokens"]
    res["T5"] = {"free_run_keep": free, "deployment_kv": dep["tokens"],
                 "deployment_action_masking": dep.get("action_input_masking")}
    print("[T5] free-run(keep) = %s" % free)
    print("     deployment(KV) = %s  equal=%s" % (dep["tokens"], checks["T5_deploy_matches_free_run"]))

    # ---- T6 因果性（logits 层面；两次调用都 output_attentions=False）
    ids_pert = p.inp.clone()
    ids_pert[0, p.plen + 10] = AC.LEGAL_TOKEN_MIN + 123
    ids_ctrl = p.inp.clone()
    ids_ctrl[0, p.plen + 3] = AC.LEGAL_TOKEN_MIN + 200
    o_base = lm_only(m, p.inp)
    o_again = lm_only(m, p.inp)
    o_pert = lm_only(m, ids_pert)
    o_past = lm_only(m, ids_ctrl)
    pred_pos = [p.plen + k - 1 for k in range(14)]
    det = float((o_base.logits.float() - o_again.logits.float()).abs().max())
    d_fut = [float((o_base.logits[0, i].float() - o_pert.logits[0, i].float()).abs().max()) for i in pred_pos]
    d_past = [float((o_base.logits[0, i].float() - o_past.logits[0, i].float()).abs().max()) for i in pred_pos]
    res["T6"] = {"determinism_maxdiff": det, "perturb_future_maxdiff": d_fut,
                 "perturb_past_maxdiff": d_past}
    checks["T6_deterministic"] = det == 0.0
    checks["T6_no_future_leak"] = all(x == 0.0 for x in d_fut[:10])
    checks["T6_past_does_matter"] = any(x > 0.0 for x in d_past[4:])
    print("[T6] determinism maxdiff = %.3e" % det)
    print("     perturb future action#11 -> maxdiff = %s" % [round(x, 5) for x in d_fut])
    print("     perturb past   action#4  -> maxdiff = %s" % [round(x, 5) for x in d_past])
    print("     no-future-leak=%s ; past-affects-future=%s" % (
        checks["T6_no_future_leak"], checks["T6_past_does_matter"]))

    # INFO: sdpa + output_attentions 的假象
    cfg = getattr(m.vla.language_model, "config", None)
    impl = getattr(cfg, "_attn_implementation", "?")
    oa = lm_only(m, p.inp, output_attentions=True)
    upper_max = float(torch.triu(oa.attentions[0][0].float(), diagonal=1).abs().max())
    res["INFO_sdpa_attn_artifact"] = {"attn_implementation": str(impl), "layer0_upper_max": upper_max}
    print("[INFO] attn_implementation=%s ; with output_attentions=True the returned layer0 "
          "upper-triangle max = %.4f (HF falls back to eager without a causal mask; NOT a model bug)" % (
              impl, upper_max))

    # ---- T7 上下文敏感性（keep 机制下，前序动作标记是否真的起作用）
    # 注意：多模态分支要求 labels 非 None（cfg._process_action_masks(None) 会抛异常），
    # 所以这里按训练口径构造 labels；labels 只影响 loss，不改变 logits。
    ids_swap = p.inp.clone()
    ids_swap[0, p.plen:p.plen + 14] = AC.LEGAL_TOKEN_MIN + 255   # 全部替换成极端动作标记
    o_swap = fwd(p, m, False, with_labels=True, ids=ids_swap)
    _, iS, aS = p.action_metrics(o_swap)
    o_keep_lab = fwd(p, m, False, with_labels=True)
    _, iK, aK = p.action_metrics(o_keep_lab)
    res["T7"] = {"tokens_equal_gt_vs_swapped": int((iK == iS).sum()),
                 "action_l1": float(np.abs(aK - aS).mean()),
                 "action_max_abs_diff": float(np.abs(aK - aS).max())}
    print("[T7] prefix actions GT vs all-255 (keep mode): tokens equal %d/14, |dA| L1 = %.5f (max %.5f)" % (
        res["T7"]["tokens_equal_gt_vs_swapped"], res["T7"]["action_l1"], res["T7"]["action_max_abs_diff"]))

    res["checks"] = checks
    res["passed"] = all(checks.values())
    print("\n=== RESULT ===")
    for k, v in checks.items():
        print("  %-30s %s" % (k, "PASS" if v else "FAIL"))
    print("R3_FIX_VERIFY: %s" % ("ALL_PASS" if res["passed"] else "FAILED"))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("wrote " + args.out)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
