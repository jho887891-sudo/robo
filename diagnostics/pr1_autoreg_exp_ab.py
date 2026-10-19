#!/usr/bin/env python3
"""第三轮 PR 验证 · 实验 A + B：在**未打补丁的原始** modeling_prismatic.py 上验证
chatgpt 分支（audit/r3-pure-autoreg-input-fix）的 pure_autoreg 修复。

实验 A（动作输入嵌入）
  1. 原始 OFT 行为下 _process_action_masks(labels) 的 14 个布尔值
  2. enable_pure_autoreg 之后同一调用的 14 个布尔值
  3. 置零前后：进入 language_model 的动作位置嵌入 max|emb| 与范数；并与“原始词嵌入”比较
  4. 真实进入语言模型的动作输入是否保留了前序真实动作信息（逐位置真值比较）
  5. 多模态序列中的实际布局：图像 patch / 本体状态 token / 提示 token / 动作 token 的位置

实验 B（完整序列 vs 逐 token 缓存推理，教师强制同一历史）
  * 方法一：一次整段前向（因果注意力），按正确位移取 14 个动作位置的 logits
  * 方法二：前缀 + KV Cache，逐步喂**同样的真实历史动作 token**，取每步 logits
  * 比较 14 个位置的 argmax、GT token 概率、logits 差异；BF16 容差另计
  * 因果性：篡改第 11 个动作 token，前 10 个预测位置必须不变

用法：
  python diagnostics/pr1_autoreg_exp_ab.py --out /tmp/pr1_exp_ab.json
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
HDF5 = os.environ.get("PUSHT_HDF5", "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
SPLIT = os.environ.get("PUSHT_SPLIT", RT + "/split80.json")
BOUNDS = os.environ.get("PUSHT_BOUNDS", RT + "/push_T_bounds.npz")
META = os.environ.get("PUSHT_META", RT + "/model_cache/pure3c/run_meta.json")
MODEL = os.environ.get("PUSHT_MODEL_PRISTINE", RT + "/model_cache/openvla-7b-pristine")
OFT = os.environ.get("PUSHT_OFT", "/home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_OFT/openvla_oft")
CODE = os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure")

sys.path.insert(0, CODE)
sys.path.insert(0, OFT)
# 只追加 --aloha，不要替换 sys.argv（否则 --out 等命令行参数会被吞掉）
if "--aloha" not in sys.argv:
    sys.argv.append("--aloha")

import action_codec as AC
from transformers import AutoModelForVision2Seq, AutoProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder

DEV = "cuda"
W0 = AC.LEGAL_TOKEN_MIN
INSTR = "Push the T-shaped block to align it precisely with the gray T-shaped pad."
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
PASS, FAIL = "PASS", "FAIL"


def load_helper():
    p = os.path.join(CODE, "pure_autoreg_mode.py")
    spec = importlib.util.spec_from_file_location("pure_autoreg_mode", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def c14(h, g):
    return np.concatenate([
        h[g + "/left_arm_joint_states"][:], h[g + "/left_ee_joint_states"][:],
        h[g + "/right_arm_joint_states"][:], h[g + "/right_ee_joint_states"][:],
    ], axis=1).astype(np.float32)


def dec(b):
    import cv2
    return cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR)[..., ::-1].copy()


def window(logits):
    return logits[W0: W0 + AC.N_BINS].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default="")
    args, _u = ap.parse_known_args()

    helper = load_helper()
    import h5py
    split = json.load(open(SPLIT))
    meta = json.load(open(META))
    q01 = np.asarray(meta["q01"], np.float32); q99 = np.asarray(meta["q99"], np.float32)
    nb = np.load(BOUNDS)
    pmin = np.asarray(nb["pmin"], np.float32); pmax = np.asarray(nb["pmax"], np.float32)

    fixed = sorted(split["val"])[0]
    with h5py.File(os.path.join(HDF5, fixed), "r") as h:
        t = 20
        imgs = [dec(h["vision/" + c + "/colors"][t]) for c in CAMS]
        state = c14(h, "state")[t].copy()
        act = c14(h, "action")[t].copy()

    print("[env] pristine modeling sha must be b5431a07… :", flush=True)
    import hashlib
    mp = os.path.join(MODEL, "modeling_prismatic.py")
    print("      modeling_prismatic.py sha256 =",
          hashlib.sha256(open(mp, "rb").read()).hexdigest(), flush=True)
    print("[env] helper file =", os.path.join(CODE, "pure_autoreg_mode.py"), flush=True)

    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to(DEV).eval()
    vla.vision_backbone.set_num_images_in_input(3)
    print("[env] loaded core:", type(vla).__name__, "| module:",
          type(vla).__module__, flush=True)
    print("[env] has _process_action_masks:", hasattr(vla, "_process_action_masks"),
          "| has _pure_autoreg_input_mode:", getattr(vla, "_pure_autoreg_input_mode", False), flush=True)

    pv = torch.cat([proc.image_processor.apply_transform(Image.fromarray(im)) for im in imgs],
                   dim=0).unsqueeze(0).to(torch.bfloat16).to(DEV)
    pn = np.clip(2.0 * (state - pmin) / (pmax - pmin + 1e-8) - 1.0, -1, 1).astype(np.float32)
    proprio = torch.from_numpy(pn).unsqueeze(0).to(DEV)

    b = PurePromptBuilder("openvla")
    b.add_turn("human", "What action should the robot take to " + INSTR + "?")
    pids = proc.tokenizer(b.get_prompt(), return_tensors="pt").input_ids.to(DEV)
    if int(pids[0, -1].item()) != 29871:
        pids = torch.cat([pids, torch.tensor([[29871]], dtype=pids.dtype, device=DEV)], dim=1)
    n_text = int(pids.shape[1])

    idx = AC.encode_indices(AC.normalize(act, q01, q99), mode=AC.MODE_FIXED)   # R2 修复后的合法编码
    tok_ids = AC.token_ids(idx)
    aid = torch.tensor([[int(x) for x in tok_ids]], dtype=torch.long, device=DEV)
    stop = torch.full((1, 1), AC.STOP_INDEX, dtype=torch.long, device=DEV)
    inp = torch.cat([pids, aid, stop], dim=1)
    lab = torch.cat([torch.full_like(pids, -100), aid, stop], dim=1)
    AC.validate_token_ids(tok_ids)

    res = {"sample": {"episode": fixed, "frame": t},
           "tokens": [int(x) for x in idx],
           "token_ids": [int(x) for x in tok_ids],
           "gt_action": act.tolist(), "checks": {}}

    captured = {}

    def hook_lm(module, argl, kwargs, output):
        ie = kwargs.get("inputs_embeds", None)
        if ie is not None:
            captured["mm"] = ie.detach().float().cpu()

    def hook_emb(module, argl, output):
        if isinstance(output, torch.Tensor):
            captured["pre"] = output.detach().float().cpu()

    # ---- 本体投影：这里不加载 ProprioProjector 权重，只用随机初始化会污染数值，
    #      因此实验 A 用“真实 proprio 输入 + 与训练一致的投影层”需要权重文件；为不引入旧检查点，
    #      本实验对 proprio 只做位置检查（不评估其数值影响）。
    # 与部署一致：LoRA 适配器 + 训练好的 proprio 投影层（只为得到真实量级的 logits；
    # A/B 比较的是“机制一致性”，两条路径用完全相同的权重）
    from prismatic.models.projectors import ProprioProjector
    from peft import PeftModel
    ADAPTER = os.environ.get("PUSHT_ADAPTER", RT + "/model_cache/pure3c/lora_adapter")
    PPF = os.environ.get("PUSHT_PROJ", RT + "/model_cache/pure3c/proprio_projector--pure.pt")
    if os.path.isdir(ADAPTER):
        vla = PeftModel.from_pretrained(vla, ADAPTER, is_trainable=False)
        vla = vla.merge_and_unload()
        print("[env] merged LoRA adapter:", ADAPTER, flush=True)
    pp = ProprioProjector(proprio_dim=14, llm_dim=vla.llm_dim)
    pp.load_state_dict(torch.load(PPF, map_location="cpu"))
    pp = pp.to(torch.bfloat16).to(DEV)
    print("[env] loaded trained proprio projector:", PPF, flush=True)

    def fwd2(fix, ids=None):
        ids = inp if ids is None else ids
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pv,
                      labels=lab, proprio=proprio, proprio_projector=pp)
        return out

    # ---------------- A1/A2: masks
    m_orig = vla._process_action_masks(lab)
    res["masks"] = {"oft_original": [bool(x) for x in m_orig[0, n_text:n_text + 14]],
                    "oft_original_sum": int(m_orig.sum().item()),
                    "prompt_positions_masked": int(m_orig[0, :n_text].sum().item())}
    print("\n[A1] OFT original _process_action_masks on the 14 action positions:")
    print("     %s  (sum=%d, prompt masked=%d)" % (
        res["masks"]["oft_original"], res["masks"]["oft_original_sum"],
        res["masks"]["prompt_positions_masked"]), flush=True)

    h1 = vla.get_input_embeddings().register_forward_hook(hook_emb)
    h2 = vla.language_model.register_forward_hook(hook_lm, with_kwargs=True)
    try:
        o_oft = fwd2(False)
    finally:
        h1.remove(); h2.remove()
    n_vis = int(o_oft.logits.shape[1]) - int(inp.shape[1])
    pre_oft, mm_oft = captured["pre"], captured["mm"]

    # 应用 R3 修复（实例级）
    helper.enable_pure_autoreg(vla)
    m_fix = vla._process_action_masks(lab)
    res["masks"]["r3_fixed"] = [bool(x) for x in m_fix[0, n_text:n_text + 14]]
    res["masks"]["r3_fixed_sum"] = int(m_fix.sum().item())
    res["masks"]["r3_flag"] = bool(getattr(vla, "_pure_autoreg_input_mode", False))
    print("[A2] after enable_pure_autoreg:")
    print("     %s  (sum=%d, flag=%s)" % (
        res["masks"]["r3_fixed"], res["masks"]["r3_fixed_sum"], res["masks"]["r3_flag"]), flush=True)

    h1 = vla.get_input_embeddings().register_forward_hook(hook_emb)
    h2 = vla.language_model.register_forward_hook(hook_lm, with_kwargs=True)
    try:
        o_fix = fwd2(False)
    finally:
        h1.remove(); h2.remove()
    pre_fix, mm_fix = captured["pre"], captured["mm"]

    # ---------------- A3: embedding norms across the mask boundary
    a_pos_oft = n_vis + n_text        # 动作 token 在 mm 序列中的真实位置（预测位置比它早 1）
    rows_oft = mm_oft[0, a_pos_oft:a_pos_oft + 14]
    rows_fix = mm_fix[0, a_pos_oft:a_pos_oft + 14]
    raw_rows = pre_fix[0, n_text:n_text + 14]
    res["embeddings"] = {
        "n_vis_tokens": int(n_vis), "n_text_tokens": int(n_text),
        "oft_max_abs": float(rows_oft.abs().max()),
        "oft_norms": [float(x) for x in rows_oft.norm(dim=-1)],
        "r3_max_abs": float(rows_fix.abs().max()),
        "r3_norms": [float(x) for x in rows_fix.norm(dim=-1)],
        "raw_token_emb_norms": [float(x) for x in raw_rows.norm(dim=-1)],
        "r3_equals_raw_token_emb": bool(torch.allclose(rows_fix, raw_rows, atol=0.0)),
        "image_patch_row_norm_example": float(mm_fix[0, 1].norm()),
        "proprio_row_norm": float(mm_fix[0, 1 + 768].norm()),
        "prompt_row_norm_example": float(mm_fix[0, 1 + 769].norm()),
    }
    print("\n[A3] action-position input embeddings (multimodal rows %d..%d):" % (
        a_pos_oft, a_pos_oft + 13))
    print("     OFT original : max|emb|=%.6f  norms[0:4]=%s" % (
        res["embeddings"]["oft_max_abs"], [round(x, 4) for x in res["embeddings"]["oft_norms"][:4]]))
    print("     R3 fix       : max|emb|=%.6f  norms[0:4]=%s" % (
        res["embeddings"]["r3_max_abs"], [round(x, 4) for x in res["embeddings"]["r3_norms"][:4]]))
    print("     raw token emb: norms[0:4]=%s" % ([round(x, 4) for x in res["embeddings"]["raw_token_emb_norms"][:4]],))
    print("     R3 rows == raw token embeddings: %s" % res["embeddings"]["r3_equals_raw_token_emb"], flush=True)

    # ---------------- A5: 多模态布局
    res["layout"] = {
        "total_multimodal_len": int(1 + n_vis + (n_text - 1) + 15),
        "token0_index": 0,
        "image_patches": [1, 768],
        "proprio_token_index": 1 + 768,
        "prompt_and_actions_block": [1 + 769, 1 + 769 + (n_text - 1) + 15 - 1],
        "action_positions": [n_vis + n_text + k for k in range(14)],
        "pre_action_prediction_positions": [n_vis + n_text - 1 + k for k in range(14)],
        "stop_position": n_vis + n_text + 14,
    }
    print("\n[A5] multimodal layout: total=%d | token0=0 | image patches=[1,768] | proprio=%d | "
          "prompt/action block starts at %d | actions at %s | STOP at %d" % (
              res["layout"]["total_multimodal_len"], res["layout"]["proprio_token_index"],
              1 + 769, res["layout"]["action_positions"], res["layout"]["stop_position"]), flush=True)
    print("     row norms: image=%.1f  proprio=%.3f  first prompt row=%.3f  action=%.3f" % (
        res["embeddings"]["image_patch_row_norm_example"], res["embeddings"]["proprio_row_norm"],
        res["embeddings"]["prompt_row_norm_example"], res["embeddings"]["r3_norms"][0]), flush=True)

    # ---------------- B: full-seq vs KV teacher-forced（R3 修复后）
    pred_pos = [n_vis + n_text - 1 + k for k in range(14)]

    def full_seq_vec(ids=None, labels=None):
        ids = inp if ids is None else ids
        lb = lab if labels is None else labels
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pv,
                    labels=lb, proprio=proprio, proprio_projector=pp)
        return o, [window(o.logits[0, p]).cpu() for p in pred_pos]

    o_full, vec_full = full_seq_vec()

    vec_kv = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pre = vla(input_ids=pids, attention_mask=torch.ones_like(pids), pixel_values=pv,
                  labels=pids.clone(), proprio=proprio, proprio_projector=pp, use_cache=True)
        past = pre.past_key_values
        vec_kv.append(window(pre.logits[0, -1]).cpu())
        for k in range(1, 14):
            tok = torch.tensor([[int(aid[0, k - 1])]], dtype=torch.long, device=DEV)
            o = vla(input_ids=tok, past_key_values=past, labels=None, use_cache=True)
            past = o.past_key_values
            vec_kv.append(window(o.logits[0, -1]).cpu())

    per_pos = []
    for k in range(14):
        pf, pk = vec_full[k], vec_kv[k]
        g = int(idx[k])
        pf_s, pk_s = torch.softmax(pf, -1), torch.softmax(pk, -1)
        per_pos.append({
            "dim": ["left_arm_j%d" % i for i in range(6)][k] if k < 6 else ("left_gripper" if k == 6 else
                    ("right_arm_j%d" % (k - 7) if k < 13 else "right_gripper")),
            "gt_idx": g,
            "top1_full": int(torch.argmax(pf).item()),
            "top1_kv": int(torch.argmax(pk).item()),
            "p_gt_full": float(pf_s[g].item()),
            "p_gt_kv": float(pk_s[g].item()),
            "logit_maxdiff": float((pf - pk).abs().max()),
            "logit_reldiff": float((pf - pk).abs().max() / (pf.abs().max() + 1e-9)),
            "prob_maxdiff": float((pf_s - pk_s).abs().max()),
            "argmax_equal": int(torch.argmax(pf).item()) == int(torch.argmax(pk).item()),
        })
    n_arg = sum(1 for r in per_pos if r["argmax_equal"])
    gt_top1_full = sum(1 for r in per_pos if r["top1_full"] == r["gt_idx"])
    gt_top1_kv = sum(1 for r in per_pos if r["top1_kv"] == r["gt_idx"])
    res["B"] = {"per_position": per_pos, "argmax_equal_positions": n_arg,
                "gt_top1_full": gt_top1_full, "gt_top1_kv": gt_top1_kv,
                "max_logit_diff": max(r["logit_maxdiff"] for r in per_pos),
                "max_prob_diff": max(r["prob_maxdiff"] for r in per_pos)}
    print("\n[B] full-seq vs KV teacher-forced (R3 mode), 14 positions:")
    print("    %-14s %6s %6s %6s %9s %9s %10s" % ("dim", "gt", "top1F", "top1KV", "p_gtF", "p_gtKV", "logitdiff"))
    for r in per_pos:
        print("    %-14s %6d %6d %6d %9.4f %9.4f %10.4f" % (
            r["dim"], r["gt_idx"], r["top1_full"], r["top1_kv"], r["p_gt_full"], r["p_gt_kv"],
            r["logit_maxdiff"]))
    print("    argmax equal %d/14 | GT is top1: full %d/14, kv %d/14 | max logit diff %.4f, max prob diff %.2e" % (
        n_arg, gt_top1_full, gt_top1_kv, res["B"]["max_logit_diff"], res["B"]["max_prob_diff"]), flush=True)

    # 因果性：篡改第 11 个动作 token，前 10 个预测位置必须不变
    ids_p = inp.clone()
    ids_p[0, n_text + 10] = W0 + 123
    _, vec_p = full_seq_vec(ids=ids_p)
    d = [float((vec_full[k] - vec_p[k]).abs().max()) for k in range(14)]
    res["B"]["perturb_future_maxdiff"] = d
    print("    causality: perturb action#11 -> per-position maxdiff %s" % [round(x, 5) for x in d], flush=True)

    ck = res["checks"]
    ck["A1_oft_masks_all_action_positions"] = res["masks"]["oft_original_sum"] == 14 and all(res["masks"]["oft_original"])
    ck["A1_oft_does_not_mask_prompt"] = res["masks"]["prompt_positions_masked"] == 0
    ck["A2_fix_masks_nothing"] = res["masks"]["r3_fixed_sum"] == 0
    ck["A2_fix_flag_set"] = res["masks"]["r3_flag"]
    ck["A3_oft_embeddings_zeroed"] = res["embeddings"]["oft_max_abs"] == 0.0
    ck["A3_fix_embeddings_preserved"] = (res["embeddings"]["r3_max_abs"] > 1e-3
                                         and res["embeddings"]["r3_equals_raw_token_emb"])
    ck["B_argmax_all_equal"] = n_arg == 14
    ck["B_logits_close"] = res["B"]["max_prob_diff"] < 0.02
    ck["B_no_future_leak"] = all(x == 0.0 for x in d[:10])
    res["passed"] = all(ck.values())

    print("\n=== RESULT ===")
    for k, v in ck.items():
        print("  %-40s %s" % (k, PASS if v else FAIL))
    print("PR1_EXP_AB: %s" % ("ALL_PASS" if res["passed"] else "FAILED"))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("wrote " + args.out)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
