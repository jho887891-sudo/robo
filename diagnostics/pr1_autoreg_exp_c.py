#!/usr/bin/env python3
"""第三轮 PR 验证 · 实验 C：真实前向 + 反向的梯度检查。

与训练脚本一致地构建（基础模型 -> enable_pure_autoreg -> LoRA -> ProprioProjector），
在同一帧真实样本上做一次前向/反向，并逐项核对：

  C1 动作 token 交叉熵损失有效（有限），且只落在 14 个动作位置 + STOP 上
  C2 LoRA 参数收到有限、非零梯度
  C3 ProprioProjector 收到有效梯度
  C4 全程无 NaN / Inf
  C5 监督位置与因果预测位置一一对应（手动复算逐位置 CE 与 out.loss 对齐）
  C6 保留前序动作嵌入后没有未来泄漏（训练模式下篡改未来 token，过去位置 logits 不变），
     并且零机制（OFT 原行为）下动作位置梯度严格为 0 —— 作为对照

用法：
  python diagnostics/pr1_autoreg_exp_c.py --out /tmp/pr1_exp_c.json
"""
import argparse
import hashlib
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
from peft import LoraConfig, get_peft_model
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--grad-checkpoint", default="auto", choices=["auto", "1", "0"])
    ap.add_argument("--out", default="")
    args, _u = ap.parse_known_args()

    helper = load_helper()
    import h5py
    meta = json.load(open(META))
    q01 = np.asarray(meta["q01"], np.float32); q99 = np.asarray(meta["q99"], np.float32)
    nb = np.load(BOUNDS)
    pmin = np.asarray(nb["pmin"], np.float32); pmax = np.asarray(nb["pmax"], np.float32)
    split = json.load(open(SPLIT))
    fixed = sorted(split["val"])[0]
    with h5py.File(os.path.join(HDF5, fixed), "r") as h:
        t = 20
        imgs = [dec(h["vision/" + c + "/colors"][t]) for c in CAMS]
        state = c14(h, "state")[t].copy(); act = c14(h, "action")[t].copy()

    print("[env] modeling sha256 =", hashlib.sha256(open(os.path.join(MODEL, "modeling_prismatic.py"), "rb").read()).hexdigest(), flush=True)

    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    core = AutoModelForVision2Seq.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to(DEV)
    core.vision_backbone.set_num_images_in_input(3)
    orig_method = core._process_action_masks          # OFT 原始实现（bound method），用于零机制对照
    helper.enable_pure_autoreg(core)                  # 与训练脚本一致的顺序：包 LoRA 之前
    print("[env] pure_autoreg flag after enable:", getattr(core, "_pure_autoreg_input_mode", False), flush=True)
    vla = get_peft_model(core, LoraConfig(r=32, lora_alpha=16, lora_dropout=0.0,
                                          target_modules="all-linear", init_lora_weights="gaussian"))
    pp = ProprioProjector(proprio_dim=14, llm_dim=vla.llm_dim).to(DEV)
    free, total = torch.cuda.mem_get_info()
    print("[env] free=%.1fGB total=%.1fGB" % (free / 1e9, total / 1e9), flush=True)
    use_ckpt = (args.grad_checkpoint == "1") or (args.grad_checkpoint == "auto" and free < 20e9)
    if use_ckpt:
        vla.gradient_checkpointing_enable()
        try:
            vla.enable_input_require_grads()
        except Exception as ex:
            print("[warn]", ex, flush=True)
        print("[env] gradient checkpointing ON", flush=True)
    vla.train(); pp.train()

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
    idx = AC.encode_indices(AC.normalize(act, q01, q99), mode=AC.MODE_FIXED)
    tok_ids = AC.token_ids(idx)
    aid = torch.tensor([[int(x) for x in tok_ids]], dtype=torch.long, device=DEV)
    stop = torch.full((1, 1), AC.STOP_INDEX, dtype=torch.long, device=DEV)
    inp = torch.cat([pids, aid, stop], dim=1)
    lab = torch.cat([torch.full_like(pids, -100), aid, stop], dim=1)

    res = {"sample": {"episode": fixed, "frame": t}, "checks": {},
           "sup_positions_original": list(range(n_text, n_text + 15)),
           "action_token_ids": [int(x) for x in tok_ids], "checks_grad_checkpoint": bool(use_ckpt)}

    def run_case(mode):
        """mode: 'fix' | 'zero'"""
        if mode == "zero":
            core._process_action_masks = orig_method
        else:
            helper.enable_pure_autoreg(core)
        masks = core._process_action_masks(lab)
        vla.zero_grad(set_to_none=True); pp.zero_grad(set_to_none=True)
        grab = {}

        def emb_hook(module, argl, output):
            # 冻结词嵌入时 output 默认不 require_grad；这里显式打开以便测量
            # “梯度是否回流到动作 token 的输入嵌入”（不改动任何参数权重）
            if isinstance(output, torch.Tensor):
                if not output.requires_grad:
                    output.requires_grad_(True)
                output.retain_grad()
                grab["pre"] = output

        h = vla.get_input_embeddings().register_forward_hook(emb_hook)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                          labels=lab, proprio=proprio, proprio_projector=pp)
                loss = out.loss
            loss.backward()
        finally:
            h.remove()
        n_vis = int(out.logits.shape[1]) - int(inp.shape[1])
        lora = [p for p in vla.parameters() if p.requires_grad and p.grad is not None]
        lora_norm = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in lora)))
        lora_nz = float(np.mean([float((p.grad != 0).float().mean()) for p in lora])) if lora else 0.0
        ppg = [p.grad for p in pp.parameters() if p.grad is not None]
        pp_norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in ppg))) if ppg else 0.0
        emb = grab.get("pre", None)
        if emb is not None and emb.grad is not None:
            g = emb.grad[0]
            emb_rows = [float(g[n_text + k].float().norm()) for k in range(14)]
        else:
            emb_rows = [0.0] * 14
        # C5: 手动复算逐位置 CE（因果位移：位置 p 的 logits 预测 label[p+1]）
        mm_lab = torch.cat([lab[:, :1], torch.full((1, n_vis), -100, dtype=lab.dtype, device=DEV),
                            lab[:, 1:]], dim=1)
        logits = out.logits.float().detach()
        per_pos = []
        total = 0.0; cnt = 0
        for s in range(1, mm_lab.shape[1]):
            target = int(mm_lab[0, s].item())
            if target == -100:
                continue
            ce = float(torch.nn.functional.cross_entropy(logits[0, s - 1].unsqueeze(0),
                                                         torch.tensor([target], device=DEV)).item())
            per_pos.append({"label_pos_in_mm": s, "target": target, "ce": ce})
            total += ce; cnt += 1
        return {"mode": mode, "masks_sum": int(masks.sum().item()),
                "masks_action_positions": [bool(x) for x in masks[0, n_text:n_text + 14]],
                "loss": float(loss.item()), "loss_finite": bool(torch.isfinite(loss).item()),
                "lora_tensors": len(lora), "lora_grad_norm": lora_norm, "lora_grad_nonzero_frac": lora_nz,
                "proprio_grad_norm": pp_norm,
                "action_input_grad_norms": emb_rows, "action_input_grad_max": max(emb_rows),
                "n_supervised_tokens": cnt, "mean_ce_manual": (total / cnt) if cnt else None,
                "per_supervised_ce": per_pos,
                "grad_nonfinite": any(not torch.isfinite(p.grad).all() for p in lora),
                "n_vis": n_vis}

    case_fix = run_case("fix")
    case_zero = run_case("zero")
    res["fix"] = {k: v for k, v in case_fix.items() if k != "per_supervised_ce"}
    res["zero"] = {k: v for k, v in case_zero.items() if k != "per_supervised_ce"}
    res["fix_per_supervised_ce"] = case_fix["per_supervised_ce"]

    print("\n[C] fixed mode : masks_sum=%d  loss=%.4f finite=%s | LoRA %d tensors norm=%.3e nz=%.3f | proprio=%.3e" % (
        case_fix["masks_sum"], case_fix["loss"], case_fix["loss_finite"], case_fix["lora_tensors"],
        case_fix["lora_grad_norm"], case_fix["lora_grad_nonzero_frac"], case_fix["proprio_grad_norm"]), flush=True)
    print("[C] zero  mode : masks_sum=%d  loss=%.4f finite=%s | LoRA %d tensors norm=%.3e nz=%.3f | proprio=%.3e" % (
        case_zero["masks_sum"], case_zero["loss"], case_zero["loss_finite"], case_zero["lora_tensors"],
        case_zero["lora_grad_norm"], case_zero["lora_grad_nonzero_frac"], case_zero["proprio_grad_norm"]), flush=True)
    print("[C] d(loss)/d(action token input emb) fixed: %s" % (
        [round(x, 4) for x in case_fix["action_input_grad_norms"]],), flush=True)
    print("[C] d(loss)/d(action token input emb) zero : %s" % (
        [round(x, 4) for x in case_zero["action_input_grad_norms"]],), flush=True)
    print("[C] supervised tokens=%d (14 actions + STOP) | manual mean CE=%.4f vs model loss=%.4f" % (
        case_fix["n_supervised_tokens"], case_fix["mean_ce_manual"], case_fix["loss"]), flush=True)
    print("[C] supervised label positions inside multimodal seq: %s" % (
        [p["label_pos_in_mm"] for p in case_fix["per_supervised_ce"]],), flush=True)

    # C6: 训练模式下无未来泄漏
    helper.enable_pure_autoreg(core)
    ids_f = inp.clone(); ids_f[0, n_text + 10] = W0 + 123
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o0 = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv, labels=lab,
                 proprio=proprio, proprio_projector=pp)
        o1 = vla(input_ids=ids_f, attention_mask=torch.ones_like(ids_f), pixel_values=pv, labels=lab,
                 proprio=proprio, proprio_projector=pp)
    pred_pos = [case_fix["n_vis"] + n_text - 1 + k for k in range(14)]
    dv = [float((o0.logits[0, p].float() - o1.logits[0, p].float()).abs().max()) for p in pred_pos]
    res["perturb_future_maxdiff"] = dv
    print("[C6] train() mode, perturb future action#11 -> per-position maxdiff = %s" % (
        [round(x, 5) for x in dv],), flush=True)

    ck = res["checks"]
    ck["C1_loss_finite"] = case_fix["loss_finite"] and case_fix["loss"] > 0
    ck["C1_supervised_15_tokens"] = case_fix["n_supervised_tokens"] == 15
    ck["C2_lora_grad_finite_nonzero"] = (case_fix["lora_grad_norm"] > 0 and not case_fix["grad_nonfinite"]
                                         and case_fix["lora_grad_nonzero_frac"] > 0)
    ck["C3_proprio_grad_nonzero"] = case_fix["proprio_grad_norm"] > 0
    ck["C4_no_nan_inf"] = all(np.isfinite([case_fix["loss"], case_fix["lora_grad_norm"],
                                           case_fix["proprio_grad_norm"]]))
    ck["C5_manual_ce_matches_loss"] = abs(case_fix["mean_ce_manual"] - case_fix["loss"]) < 5e-3
    ck["C6_no_future_leak"] = all(x == 0.0 for x in dv[:10])
    ck["CTRL_zero_mode_action_input_grad_is_zero"] = case_zero["action_input_grad_max"] == 0.0
    ck["CTRL_fix_mode_action_input_grad_nonzero"] = case_fix["action_input_grad_max"] > 0
    res["passed"] = all(ck.values())

    print("\n=== RESULT ===")
    for k, v in ck.items():
        print("  %-42s %s" % (k, PASS if v else FAIL))
    print("PR1_EXP_C: %s" % ("ALL_PASS" if res["passed"] else "FAILED"))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("wrote " + args.out)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
