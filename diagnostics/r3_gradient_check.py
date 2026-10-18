#!/usr/bin/env python3
"""C 阶段：梯度检查 —— 关键可训练模块是否真正收到有效梯度（GPU）。

对应任务书 四.C。用**全新的 base + LoRA + ProprioProjector**（不加载旧适配器）
在一帧真实样本上做一次前向 + 反向，回答三个问题：

  C1 LoRA 适配器与 proprio 投影层是否都收到非零梯度（无 NaN/Inf）
  C2 keep 机制下，14 个动作 token 的**输入词嵌入行**是否能收到梯度
     —— 这是“模型能否利用前序动作标记”的梯度层面证据（因果注意力 ⇒ 后续位置把梯度传回来）
  C3 zero 机制（OFT 默认）下，同样位置的动作 token 词嵌入行梯度必须**严格为 0**
     —— 这从梯度上证明“置零”切断了自条件通路

用法：
  python diagnostics/r3_gradient_check.py --out /tmp/r3_C.json
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


def build_fresh_model():
    """独立构建 trainable 结构（LoRA r=32 all-linear + ProprioProjector）。

    注意：**不加载部署用的 merged 模型**（那会白占 15 GB，实测直接 OOM）。
    只用基础模型路径与统计量文件。"""
    import yaml
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import LoraConfig, get_peft_model
    from prismatic.models.projectors import ProprioProjector

    cfg = yaml.safe_load(open(os.environ.get(
        "PUSHT_DEPLOY_YML",
        "/home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_Pure/deploy.yml")))
    base = cfg["base_model_path"]
    meta = json.load(open(cfg["stats_path"]))
    nb = np.load(cfg["bounds_path"])
    proc = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to(DEV)
    vla.vision_backbone.set_num_images_in_input(int(cfg.get("num_images_in_input", 3)))
    lcfg = LoraConfig(r=32, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear",
                      init_lora_weights="gaussian")
    vla = get_peft_model(vla, lcfg)
    pp = ProprioProjector(proprio_dim=14, llm_dim=vla.llm_dim).to(DEV)
    m = {"processor": proc,
         "q01": np.asarray(meta["q01"], np.float32), "q99": np.asarray(meta["q99"], np.float32),
         "pmin": np.asarray(nb["pmin"], np.float32), "pmax": np.asarray(nb["pmax"], np.float32),
         "instr": cfg.get("prompt", "")}
    return vla, pp, m


def prep_sample(m, sample):
    e, t, imgs, state, gt = sample
    pv = torch.cat([m["processor"].image_processor.apply_transform(Image.fromarray(im))
                    for im in imgs], dim=0).unsqueeze(0).to(torch.bfloat16).to(DEV)
    pn = np.clip(2.0 * (state - m["pmin"]) / (m["pmax"] - m["pmin"] + 1e-8) - 1.0, -1, 1).astype(np.float32)
    proprio = torch.from_numpy(pn).unsqueeze(0).to(DEV)
    legacy = AC.encode_indices(AC.normalize(gt, m["q01"], m["q99"]), mode=AC.MODE_LEGACY)
    idx = np.clip(legacy, 1, 256)
    return pv, proprio, np.asarray(idx, dtype=np.int64)


def prompt_ids(m):
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    b = PurePromptBuilder("openvla")
    b.add_turn("human", "What action should the robot take to " + m["instr"] + "?")
    ids = m["processor"].tokenizer(b.get_prompt(), return_tensors="pt").input_ids.to(DEV)
    if int(ids[0, -1].item()) != 29871:
        ids = torch.cat([ids, torch.tensor([[29871]], dtype=ids.dtype, device=DEV)], dim=1)
    return ids


def run_case(vla, pp, m, pv, proprio, idx, mask_action_inputs, want_emb_grad=True):
    pids = prompt_ids(m)
    aid = torch.tensor([[AC.ACTION_TOKEN_BEGIN_IDX + 1 + int(x) for x in idx]],
                       dtype=torch.long, device=DEV)
    stop = torch.full((1, 1), AC.STOP_INDEX, dtype=torch.long, device=DEV)
    inp = torch.cat([pids, aid, stop], dim=1)
    lab = torch.cat([torch.full_like(pids, -100), aid, stop], dim=1)

    vla.zero_grad(set_to_none=True)
    pp.zero_grad(set_to_none=True)
    emb_grad = {}

    def emb_hook(module, args, output):
        # 钩住词嵌入模块（**置零之前**的张量）：zero 机制下这里的梯度会被 ~mask 截断为 0
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.retain_grad()
            emb_grad["t"] = output
        return None

    h = vla.get_input_embeddings().register_forward_hook(emb_hook)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                      labels=lab, proprio=proprio, proprio_projector=pp,
                      mask_action_inputs=mask_action_inputs)
            loss = out.loss
        loss.backward()
    finally:
        h.remove()

    lora = [p for n, p in vla.named_parameters() if p.requires_grad]
    lora_g = [p.grad for p in lora if p.grad is not None]
    lora_norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in lora_g)))
    lora_nonzero = float(np.mean([float((g != 0).float().mean()) for g in lora_g])) if lora_g else 0.0
    pp_g = [p.grad for p in pp.parameters() if p.grad is not None]
    pp_norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in pp_g))) if pp_g else 0.0

    # 动作 token 位置（置零之前）的嵌入梯度
    plen = int(pids.shape[1])
    emb = emb_grad.get("t", None)
    if emb is None or emb.grad is None:
        rows = [0.0] * 14
        dbg = "no-grad-captured"
    else:
        g = emb.grad[0]
        rows = [float(g[plen + k].float().norm()) for k in range(14)]
        dbg = "captured"
    n_vis = None
    fin = bool(torch.isfinite(loss).item())
    bad = any(not torch.isfinite(g).all() for g in lora_g)
    return {
        "loss": float(loss.item()),
        "loss_finite": fin,
        "lora_trainable_tensors": len(lora_g),
        "lora_grad_norm": lora_norm,
        "lora_grad_nonzero_frac": lora_nonzero,
        "proprio_grad_norm": pp_norm,
        "action_token_embedding_grad_norms": rows,
        "action_token_embedding_grad_max": float(max(rows)) if rows else 0.0,
        "action_token_embedding_grad_debug": dbg,
        "grad_has_nonfinite": bool(bad),
        "embedding_grad_captured": ("t" in emb_grad),
        "pids_len": plen,
        "any_grad": bool(lora_norm > 0.0 and pp_norm > 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--grad-checkpoint", type=int, default=1)
    ap.add_argument("--out", default="")
    args, _u = ap.parse_known_args()

    # 只构建一份 trainable 模型（不加载部署 merged 模型，避免 15GB×2 的 OOM）
    vla, pp, m = build_fresh_model()
    torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()
    print("[env] free=%.1fGB total=%.1fGB" % (free / 1e9, total / 1e9), flush=True)
    if args.grad_checkpoint:
        vla.gradient_checkpointing_enable()
        try:
            vla.enable_input_require_grads()
        except Exception as ex:
            print("[warn] enable_input_require_grads: %s" % ex)
        print("[env] gradient checkpointing enabled", flush=True)
    vla.train()
    pp.train()

    sample = B.iter_samples(n=args.sample + 1)[args.sample]
    pv, proprio, idx = prep_sample(m, sample)
    res = {"sample": {"episode": sample[0], "frame": int(sample[1])}, "checks": {}}

    case_keep = run_case(vla, pp, m, pv, proprio, idx, mask_action_inputs=False)
    case_zero = run_case(vla, pp, m, pv, proprio, idx, mask_action_inputs=True)
    res["keep"] = case_keep
    res["zero"] = case_zero

    print("\n[keep] loss=%.4f finite=%s | LoRA grads: tensors=%d norm=%.3e nonzero_frac=%.3f | proprio norm=%.3e" % (
        case_keep["loss"], case_keep["loss_finite"], case_keep["lora_trainable_tensors"],
        case_keep["lora_grad_norm"], case_keep["lora_grad_nonzero_frac"], case_keep["proprio_grad_norm"]))
    print("[keep] d(loss)/d(action token embedding row) per dim: %s" % (
        [round(x, 6) for x in case_keep["action_token_embedding_grad_norms"]]))
    print("[zero] loss=%.4f finite=%s | LoRA grads: tensors=%d norm=%.3e nonzero_frac=%.3f | proprio norm=%.3e" % (
        case_zero["loss"], case_zero["loss_finite"], case_zero["lora_trainable_tensors"],
        case_zero["lora_grad_norm"], case_zero["lora_grad_nonzero_frac"], case_zero["proprio_grad_norm"]))
    print("[zero] d(loss)/d(action token embedding row) per dim: %s" % (
        [round(x, 6) for x in case_zero["action_token_embedding_grad_norms"]]))

    ck = res["checks"]
    ck["C1_loss_finite_both"] = case_keep["loss_finite"] and case_zero["loss_finite"]
    ck["C1_lora_receives_grad"] = case_keep["lora_grad_norm"] > 0 and case_zero["lora_grad_norm"] > 0
    ck["C1_proprio_receives_grad"] = case_keep["proprio_grad_norm"] > 0 and case_zero["proprio_grad_norm"] > 0
    ck["C1_no_nonfinite_grad"] = not (case_keep["grad_has_nonfinite"] or case_zero["grad_has_nonfinite"])
    ck["C2_keep_action_emb_grad_nonzero"] = (case_keep["action_token_embedding_grad_max"] > 0
                                             and sum(case_keep["action_token_embedding_grad_norms"]) > 0)
    ck["C3_zero_action_emb_grad_is_zero"] = case_zero["action_token_embedding_grad_max"] == 0.0
    res["passed"] = all(ck.values())

    print("\n=== C RESULT ===")
    for k, v in ck.items():
        print("  %-36s %s" % (k, PASS if v else FAIL))
    print("R3_C_STAGE: %s" % ("ALL_PASS" if res["passed"] else "FAILED"))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("wrote " + args.out)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
