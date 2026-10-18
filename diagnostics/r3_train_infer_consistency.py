#!/usr/bin/env python3
"""B 阶段：训练/离线评测/部署 机制一致性测试（GPU）。

对应任务书 二.3 / 二.4 / 二.5 与 四.B。全部为“同一引擎、同一机制”的可判等测试：

  B1  train() 与 eval() 的前向 logits 是否逐位相同（dropout=0 前提下应相同）
  B2  完整序列前向（训练口径：一次前向 + labels）与 逐 token teacher-forced 缓存解码
      在 14 个动作位置的 logits 是否逐位相同  <-- 这条直接回答“训练与部署是否同一机制”
  B3  KV 缓存逐 token 贪心生成 与 每步全量重算的贪心生成 是否逐 token 相同
  B4  多模态路径（含 3 路相机）在 keep 机制下的注意力是否严格因果
      （强制 attn_implementation=eager 后检查注意力的严格上三角；注意 sdpa+output_attentions=True
       会给出非因果假象，见 r3 报告第 5 节）
  B5  train() 模式下“篡改未来动作标记”是否影响过去位置（训练期也不许泄漏）

用法：python diagnostics/r3_train_infer_consistency.py --sample 0 --out /tmp/r3_B.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pure_bootstrap as B

B.bootstrap()
from XPolicyLab.policy.OpenVLA_Pure import action_codec as AC

DEV = "cuda"
W0 = AC.LEGAL_TOKEN_MIN
PASS, FAIL = "PASS", "FAIL"


def window_logits(logits):
    return logits[W0: W0 + AC.N_BINS].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default="")
    args, _u = ap.parse_known_args()

    m = B.load_model()
    sample = B.iter_samples(n=args.sample + 1)[args.sample]
    e, t, imgs, state, gt_action = sample
    m.update_obs(B.make_obs(imgs, state, m.instr))
    obs = m.observation_window[0]
    pv = torch.cat([m.processor.image_processor.apply_transform(Image.fromarray(im))
                    for im in obs["images_raw"]], dim=0).unsqueeze(0).to(torch.bfloat16).to(DEV)
    pn = np.clip(2.0 * (state - m.pmin) / (m.pmax - m.pmin + 1e-8) - 1.0, -1, 1).astype(np.float32)
    proprio = torch.from_numpy(pn).unsqueeze(0).to(DEV)
    pids = m._prompt_ids()
    plen = int(pids.shape[1])

    legacy = AC.encode_indices(AC.normalize(gt_action, m.q01, m.q99), mode=AC.MODE_LEGACY)
    n_pad = int((legacy == 256).sum())
    idx = np.clip(legacy, 1, 256)
    aid = torch.tensor([[AC.ACTION_TOKEN_BEGIN_IDX + 1 + int(x) for x in idx]],
                       dtype=torch.long, device=DEV)
    stop = torch.full((1, 1), AC.STOP_INDEX, dtype=torch.long, device=DEV)
    inp = torch.cat([pids, aid, stop], dim=1)
    lab = torch.cat([torch.full_like(pids, -100), aid, stop], dim=1)
    vla = m.vla
    res, checks = {"sample": {"episode": e, "frame": int(t)},
                   "gt_legacy_pad_for_teacher_forcing": n_pad}, {}

    def full_seq(labels=True, ids=None, keep=True):
        """keep=True -> mask_action_inputs=False（不置零，自回归口径）。"""
        ids = inp if ids is None else ids
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pv,
                    labels=(lab if labels else None), proprio=proprio,
                    proprio_projector=m.proprio_projector,
                    mask_action_inputs=(not keep))
        return o

    # ---------- B1 train() vs eval()
    vla.train()
    o_train = full_seq()
    vla.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o_eval = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                     labels=lab, proprio=proprio, proprio_projector=m.proprio_projector,
                     mask_action_inputs=False)
    d1 = float((o_train.logits.float() - o_eval.logits.float()).abs().max())
    n_vis = int(o_eval.logits.shape[1]) - int(inp.shape[1])
    res["B1_train_vs_eval_maxdiff"] = d1
    checks["B1_train_eval_identical"] = d1 == 0.0
    print("[B1] train() vs eval() logits maxdiff = %.3e  (%d vision tokens)" % (d1, n_vis))

    # ---------- B2 完整序列 vs 逐 token teacher-forced
    #  B2a: KV 缓存逐 token（部署口径）—— 与完整序列比较标记/概率（bf16 累加顺序不同，允许极小数值差）
    #  B2b: 不做缓存、每步重算整段序列 —— 与完整序列必须逐位完全相同（同核同序）
    pred_pos = [n_vis + plen + k - 1 for k in range(14)]
    full_vec = [window_logits(o_eval.logits[0, pos]).cpu() for pos in pred_pos]
    tok_full = [int(torch.argmax(a).item()) for a in full_vec]

    step_vec = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pre = vla(input_ids=pids, attention_mask=torch.ones_like(pids), pixel_values=pv,
                  labels=pids.clone(), proprio=proprio, proprio_projector=m.proprio_projector,
                  mask_action_inputs=False, use_cache=True)
        past = pre.past_key_values
        step_vec.append(window_logits(pre.logits[0, -1]).cpu())
        for k in range(1, 14):
            tok = torch.tensor([[int(aid[0, k - 1])]], dtype=torch.long, device=DEV)
            o = vla(input_ids=tok, past_key_values=past, labels=None, use_cache=True)
            past = o.past_key_values
            step_vec.append(window_logits(o.logits[0, -1]).cpu())
    tok_step = [int(torch.argmax(a).item()) for a in step_vec]
    d2a = float(max((a - b).abs().max().item() for a, b in zip(full_vec, step_vec)))
    pa = [torch.softmax(a, -1) for a in full_vec]
    pb = [torch.softmax(b, -1) for b in step_vec]
    dp = float(max((x - y).abs().max().item() for x, y in zip(pa, pb)))
    res["B2a_kv_stepwise"] = {"logit_maxdiff": d2a, "prob_maxdiff": dp,
                              "tokens_full": tok_full, "tokens_stepwise": tok_step}
    act_full = AC.actions_from_token_ids(AC.token_ids(np.asarray(tok_full, np.int64)), m.q01, m.q99)
    act_step = AC.actions_from_token_ids(AC.token_ids(np.asarray(tok_step, np.int64)), m.q01, m.q99)
    checks["B2a_tokens_identical_kv"] = tok_full == tok_step
    checks["B2a_actions_identical_kv"] = bool(np.array_equal(act_full, act_step))
    # 数值说明（不作判据）：KV 增量计算与整段前向的 bf16 累加顺序不同 → logits 差约 1 个 bf16 ulp，
    # 概率差约 1e-2；B2b 用同一内核重算整段则逐位相同，证明差异只是精度而非机制。
    print("      actions identical = %s ; |logit| max ~ %.1f (bf16 ulp at that scale = 0.125)" % (
        checks["B2a_actions_identical_kv"], float(max(a.abs().max().item() for a in full_vec))))
    print("[B2a] full-seq vs KV teacher-forced: logit maxdiff=%.3e prob maxdiff=%.2e tokens equal=%s" % (
        d2a, dp, tok_full == tok_step))
    print("      full=%s" % tok_full)
    print("      step=%s" % tok_step)

    step_vec2 = []
    for k in range(14):
        cur = pids if k == 0 else torch.cat([pids, aid[:, :k]], dim=1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=cur, attention_mask=torch.ones_like(cur), pixel_values=pv,
                    labels=cur.clone(), proprio=proprio, proprio_projector=m.proprio_projector,
                    mask_action_inputs=False)
        step_vec2.append(window_logits(o.logits[0, -1]).cpu())
    d2b = float(max((a - b).abs().max().item() for a, b in zip(full_vec, step_vec2)))
    res["B2b_nocache_stepwise_maxdiff"] = d2b
    checks["B2b_nocache_bitwise_identical"] = d2b == 0.0
    print("[B2b] full-seq vs no-cache step-by-step (same kernels): logit maxdiff = %.3e" % d2b)

    # ---------- B3 KV 贪心 vs 每步全量重算贪心
    def gen(cache):
        toks = []
        if cache:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = vla(input_ids=pids, attention_mask=torch.ones_like(pids), pixel_values=pv,
                        labels=pids.clone(), proprio=proprio,
                        proprio_projector=m.proprio_projector,
                        mask_action_inputs=False, use_cache=True)
            past, lg = o.past_key_values, o.logits[0, -1]
            for k in range(14):
                toks.append(int(torch.argmax(window_logits(lg)).item()))
                if k == 13:
                    break
                tok = torch.tensor([[W0 + toks[-1]]], dtype=torch.long, device=DEV)
                o = vla(input_ids=tok, past_key_values=past, labels=None, use_cache=True)
                past, lg = o.past_key_values, o.logits[0, -1]
        else:
            cur = pids
            for k in range(14):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    o = vla(input_ids=cur, attention_mask=torch.ones_like(cur), pixel_values=pv,
                            labels=cur.clone(), proprio=proprio,
                            proprio_projector=m.proprio_projector, mask_action_inputs=False)
                toks.append(int(torch.argmax(window_logits(o.logits[0, -1])).item()))
                nxt = torch.tensor([[W0 + toks[-1]]], dtype=torch.long, device=DEV)
                cur = torch.cat([cur, nxt], dim=1)
        return toks

    g_cache = gen(True)
    g_full = gen(False)
    res["B3_cached"] = g_cache
    res["B3_full_recompute"] = g_full
    checks["B3_cached_equals_full"] = g_cache == g_full
    print("[B3] cached greedy  = %s" % g_cache)
    print("     full recompute = %s  equal=%s" % (g_full, checks["B3_cached_equals_full"]))

    # ---------- B4 多模态路径注意力严格因果（强制 eager）
    lm = vla.language_model
    old = lm.config._attn_implementation
    lm.config._attn_implementation = "eager"
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                    labels=lab, proprio=proprio, proprio_projector=m.proprio_projector,
                    mask_action_inputs=False, output_attentions=True)
        up = torch.triu(o.attentions[0][0].float(), diagonal=1)
        mx = float(up.abs().max())
        res["B4_eager_attn_upper_max"] = mx
        res["B4_eager_attn_shape"] = list(o.attentions[0][0].shape)
        checks["B4_attn_strictly_causal"] = mx == 0.0
        print("[B4] eager multimodal attention: shape %s, strict upper-triangle max = %.3e" % (
            res["B4_eager_attn_shape"], mx))
    finally:
        lm.config._attn_implementation = old

    # ---------- B5 train() 模式下的因果性（未来动作不影响过去）
    vla.train()
    ids_fut = inp.clone()
    ids_fut[0, plen + 10] = W0 + 123
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a0 = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                 labels=lab, proprio=proprio, proprio_projector=m.proprio_projector,
                 mask_action_inputs=False)
        a1 = vla(input_ids=ids_fut, attention_mask=torch.ones_like(ids_fut), pixel_values=pv,
                 labels=lab, proprio=proprio, proprio_projector=m.proprio_projector,
                 mask_action_inputs=False)
    dv = [float((a0.logits[0, p].float() - a1.logits[0, p].float()).abs().max()) for p in pred_pos]
    res["B5_train_mode_perturb_future"] = dv
    checks["B5_train_mode_no_future_leak"] = all(x == 0.0 for x in dv[:10])
    print("[B5] train() mode, perturb future action#11 -> maxdiff at 14 positions = %s" % [round(x, 5) for x in dv])
    vla.eval()

    res["checks"] = checks
    res["passed"] = all(checks.values())
    print("\n=== B RESULT ===")
    for k, v in checks.items():
        print("  %-34s %s" % (k, PASS if v else FAIL))
    print("R3_B_STAGE: %s" % ("ALL_PASS" if res["passed"] else "FAILED"))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("wrote " + args.out)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
