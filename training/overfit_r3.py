#!/usr/bin/env python3
"""D 阶段：受限小样本过拟合实验（任务书 四.D）。

严格遵守的边界
--------------
  * 32 个**固定**样本（由 split80.json 的 val 回合按固定步长确定性抽取，落盘 samples.json）
  * 最多 300 次参数更新（--steps 上限硬编码 300，不接受更大值）
  * 独立输出目录（默认 /home/dgut/pusht_runtime/r3_overfit/<masking>_<时间戳>）
  * 先保存代码/配置快照，再开始训练
  * 只用 LoRA(r=32, all-linear) + ProprioProjector；OpenVLA-7B 主干冻结
  * 使用第二轮修复后的合法动作编码器（action_codec MODE_FIXED：token ∈ 31744..31999）
  * 出现 NaN/Inf 损失、CUDA OOM、加载错误 → 立刻停止并保存日志
  * 评测只认**真正的逐个生成**（KV 缓存自由运行），不拿教师强制准确率冒充

用法：
  python training/overfit_r3.py --masking keep --steps 300 --out <dir>
  python training/overfit_r3.py --masking zero --steps 300 --out <dir>   # 对照组
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import h5py
import cv2
from PIL import Image

RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
SUBMISSION = "/home/T7/ojh/robot_actual/submission"
HDF5_DIR = os.environ.get("PUSHT_HDF5",
                          "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
SPLIT = os.environ.get("PUSHT_SPLIT", RT + "/split80.json")
BOUNDS = os.environ.get("PUSHT_BOUNDS", RT + "/push_T_bounds.npz")
BOOT = os.path.join(SUBMISSION, "diagnostics", "pure_bootstrap.py")

sys.path.insert(0, os.path.dirname(BOOT))
import pure_bootstrap as B

B.bootstrap()
from XPolicyLab.policy.OpenVLA_Pure import action_codec as AC
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.projectors import ProprioProjector
from peft import LoraConfig, get_peft_model

DEV = "cuda"
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
MAX_STEPS = 300
PASS, FAIL = "PASS", "FAIL"


def log(msg, fh=None):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def c14(h, g):
    return np.concatenate([
        h[g + "/left_arm_joint_states"][:], h[g + "/left_ee_joint_states"][:],
        h[g + "/right_arm_joint_states"][:], h[g + "/right_ee_joint_states"][:],
    ], axis=1).astype(np.float32)


def decode_img(buf):
    return cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR)[..., ::-1].copy()


def pick_samples(n=32):
    """确定性抽取 n 个 (episode, frame)：val 回合按名字排序，帧按固定步长。"""
    split = json.load(open(SPLIT))
    out = []
    for e in sorted(split["val"]):
        with h5py.File(os.path.join(HDF5_DIR, e), "r") as h:
            T = h["action/left_arm_joint_states"].shape[0]
        for t in range(20, max(21, T - 2), 40):
            out.append([e, int(t)])
            if len(out) >= n:
                return out
    return out


def q01_q99(train_eps):
    A = []
    for e in train_eps[:40]:
        with h5py.File(os.path.join(HDF5_DIR, e), "r") as h:
            A.append(c14(h, "action"))
    A = np.concatenate(A, 0)
    return (np.quantile(A, 0.01, axis=0).astype(np.float32),
            np.quantile(A, 0.99, axis=0).astype(np.float32))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def snapshot(out, args, samples):
    d = os.path.join(out, "_snapshot")
    os.makedirs(d, exist_ok=True)
    for src in (os.path.join(SUBMISSION, "code/openvla_dynamic_modules/modeling_prismatic.py"),
                os.path.join(SUBMISSION, "code/xpolicylab/policy/OpenVLA_Pure/action_codec.py"),
                os.path.join(SUBMISSION, "training/train_pushT_pure2.py"),
                os.path.join(SUBMISSION, "training/overfit_r3.py")):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(d, os.path.basename(src)))
    cfg = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "max_steps_cap": MAX_STEPS,
        "n_samples": len(samples),
        "samples": samples,
        "codec": {"mode": AC.MODE_FIXED, "legal_window": [AC.LEGAL_TOKEN_MIN, AC.LEGAL_TOKEN_MAX],
                  "n_bins": AC.N_BINS, "max_quant_error": AC.MAX_QUANT_ERROR},
        "torch": torch.__version__,
        "hashes": {f: sha(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".py")},
    }
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masking", choices=["keep", "zero"], default="keep")
    ap.add_argument("--steps", type=int, default=MAX_STEPS)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--grad-checkpoint", default="auto", choices=["auto", "1", "0"])
    ap.add_argument("--out", default="")
    args, _u = ap.parse_known_args()
    if args.steps > MAX_STEPS:
        raise SystemExit("steps 上限为 %d（任务书硬约束），收到 %d" % (MAX_STEPS, args.steps))

    out = args.out or os.path.join(RT, "r3_overfit",
                                   "%s_%s" % (args.masking, time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "train.log"), "w")
    jsonl = open(os.path.join(out, "metrics.jsonl"), "w")
    status = {"masking": args.masking, "steps": args.steps, "out": out, "result": "RUNNING"}

    def finish(res, code=0):
        status["result"] = res
        with open(os.path.join(out, "status.json"), "w") as fh:
            json.dump(status, fh, indent=1)
        try:
            logf.close(); jsonl.close()
        except Exception:
            pass
        return code

    try:
        samples = pick_samples(args.n)
        split = json.load(open(SPLIT))
        cfg = snapshot(out, args, samples)
        log("out=%s masking=%s steps=%d n=%d lr=%g" % (out, args.masking, args.steps, len(samples), args.lr), logf)
        log("snapshot hashes: %s" % json.dumps(cfg["hashes"]), logf)
        if len(samples) < args.n:
            log("WARN: 只找到 %d 个样本（目标 %d）" % (len(samples), args.n), logf)

        q01, q99 = q01_q99(split["train"])
        nb = np.load(BOUNDS)
        amin, amax = np.asarray(nb["amin"], np.float32), np.asarray(nb["amax"], np.float32)
        pmin, pmax = np.asarray(nb["pmin"], np.float32), np.asarray(nb["pmax"], np.float32)

        model_path = "/home/dgut/pusht_runtime/model_cache/openvla-7b"
        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True).to(DEV)
        vla.vision_backbone.set_num_images_in_input(3)
        vla = get_peft_model(vla, LoraConfig(r=32, lora_alpha=16, lora_dropout=0.0,
                                            target_modules="all-linear",
                                            init_lora_weights="gaussian"))
        pp = ProprioProjector(proprio_dim=14, llm_dim=vla.llm_dim).to(DEV)
        mode = "keep" if args.masking == "keep" else "zero"
        if hasattr(vla, "set_action_input_masking"):
            vla.set_action_input_masking(mode)
        else:
            raise SystemExit("模型没有 set_action_input_masking，修复未生效，拒绝继续")
        mask_kwarg = (mode == "zero")   # True = 置零

        free, total = torch.cuda.mem_get_info()
        log("GPU free=%.1fGB total=%.1fGB ; masking=%s(mask_action_inputs=%s)" % (
            free / 1e9, total / 1e9, mode, mask_kwarg), logf)
        use_ckpt = (args.grad_checkpoint == "1") or (args.grad_checkpoint == "auto" and free < 26e9)
        if use_ckpt:
            vla.gradient_checkpointing_enable()
            try:
                vla.enable_input_require_grads()
            except Exception as ex:
                log("warn enable_input_require_grads: %s" % ex, logf)
            log("gradient checkpointing = ON（显存保护，不改变数学行为）", logf)
        status["grad_checkpoint"] = bool(use_ckpt)

        ps = [p for p in vla.parameters() if p.requires_grad] + list(pp.parameters())
        log("trainable=%.1fM" % (sum(p.numel() for p in ps) / 1e6), logf)
        opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)

        def prep(e, t):
            with h5py.File(os.path.join(HDF5_DIR, e), "r") as h:
                imgs = [decode_img(h["vision/" + c + "/colors"][t]) for c in CAMS]
                state = c14(h, "state")[t].copy()
                act = c14(h, "action")[t].copy()
            pv = torch.cat([proc.image_processor.apply_transform(Image.fromarray(im)) for im in imgs],
                           dim=0).unsqueeze(0).to(torch.bfloat16).to(DEV)
            pn = np.clip(2.0 * (state - pmin) / (pmax - pmin + 1e-8) - 1.0, -1, 1).astype(np.float32)
            pr = torch.from_numpy(pn).unsqueeze(0).to(DEV)
            gt_idx = AC.encode_indices(AC.normalize(act, q01, q99), mode=AC.MODE_FIXED)
            return pv, pr, state, act, np.asarray(gt_idx, np.int64)

        pids = None
        cache = [prep(e, t) for (e, t) in samples]

        def build_ids():
            from prismatic.models.backbones.llm.prompting import PurePromptBuilder
            b = PurePromptBuilder("openvla")
            b.add_turn("human", "What action should the robot take to "
                       + "Push the T-shaped block to align it precisely with the gray T-shaped pad." + "?")
            ids = proc.tokenizer(b.get_prompt(), return_tensors="pt").input_ids.to(DEV)
            if int(ids[0, -1].item()) != 29871:
                ids = torch.cat([ids, torch.tensor([[29871]], dtype=ids.dtype, device=DEV)], dim=1)
            return ids

        pids = build_ids()
        plen = int(pids.shape[1])

        def forward_loss(pv, pr, gt_idx):
            aid = torch.tensor([[AC.ACTION_TOKEN_BEGIN_IDX + 1 + int(x) for x in gt_idx]],
                               dtype=torch.long, device=DEV)
            stop = torch.full((1, 1), AC.STOP_INDEX, dtype=torch.long, device=DEV)
            inp = torch.cat([pids, aid, stop], dim=1)
            lab = torch.cat([torch.full_like(pids, -100), aid, stop], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                          labels=lab, proprio=pr, proprio_projector=pp,
                          mask_action_inputs=mask_kwarg)
            logits = out.logits
            n_vis = int(logits.shape[1]) - int(inp.shape[1])
            preds = []
            for k in range(14):
                lg = logits[0, n_vis + plen + k - 1]
                preds.append(int(torch.argmax(lg[AC.LEGAL_TOKEN_MIN: AC.LEGAL_TOKEN_MIN + AC.N_BINS]).item()))
            return out.loss, np.asarray(preds, np.int64)

        @torch.no_grad()
        def gen_actions(pv, pr):
            """真正的逐个生成：KV 缓存，喂自己的预测，不用教师强制。"""
            vla.eval()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = vla(input_ids=pids, attention_mask=torch.ones_like(pids), pixel_values=pv,
                        labels=pids.clone(), proprio=pr, proprio_projector=pp,
                        mask_action_inputs=mask_kwarg, use_cache=True)
            past, lg = o.past_key_values, o.logits[0, -1]
            toks = []
            for k in range(14):
                toks.append(int(torch.argmax(lg[AC.LEGAL_TOKEN_MIN: AC.LEGAL_TOKEN_MIN + AC.N_BINS]).item()))
                if k == 13:
                    break
                nxt = torch.tensor([[AC.LEGAL_TOKEN_MIN + toks[-1]]], dtype=torch.long, device=DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = vla(input_ids=nxt, past_key_values=past, labels=None, use_cache=True)
                past, lg = o.past_key_values, o.logits[0, -1]
            vla.train()
            return AC.actions_from_token_ids(AC.token_ids(np.asarray(toks, np.int64)),
                                             q01, q99, mode=AC.MODE_FIXED)

        def nb_norm(a):
            return np.clip(2.0 * (np.asarray(a, np.float32) - amin) / (amax - amin + 1e-8) - 1.0, -1, 1)

        @torch.no_grad()
        def evaluate(step):
            vla.eval()
            P, G, S, TF = [], [], [], []
            for (e, t), (pv, pr, state, act, gt_idx) in zip(samples, cache):
                P.append(gen_actions(pv, pr))
                G.append(act); S.append(state)
                _, tp = forward_loss(pv, pr, gt_idx)
                TF.append(tp)
            P, G, S = np.stack(P), np.stack(G), np.stack(S)
            gt_flat = np.concatenate([c[4] for c in cache])          # (n*14,)
            tf_flat = np.concatenate([np.asarray(x) for x in TF])    # (n*14,)
            tf_acc = float(np.mean(tf_flat == gt_flat))
            tf_per_dim = (tf_flat.reshape(len(P), 14) == gt_flat.reshape(len(P), 14)).mean(0).tolist()
            rec = {
                "stage": "eval", "step": int(step), "n": len(P),
                "teacher_forced_tokens_acc": tf_acc,
                "teacher_forced_tokens_acc_per_dim": tf_per_dim,
                "gen_phys_L1": float(np.abs(P - G).mean()),
                "identity_phys_L1": float(np.abs(S - G).mean()),
                "gen_bounds_L1": float(np.abs(nb_norm(P) - nb_norm(G)).mean()),
                "identity_bounds_L1": float(np.abs(nb_norm(S) - nb_norm(G)).mean()),
                "pred_delta_mean_abs": float(np.abs(P - S).mean()),
                "gt_delta_mean_abs": float(np.abs(G - S).mean()),
                "gen_per_dim_phys": np.abs(P - G).mean(0).tolist(),
                "uniq_pred_rows": len({tuple(np.round(r, 3)) for r in P}),
            }
            rec["ratio_vs_identity"] = rec["gen_phys_L1"] / (rec["identity_phys_L1"] + 1e-12)
            log("EVAL step=%d | gen L1(phys)=%.5f identity=%.5f ratio=%.3f | gen bounds L1=%.4f identity=%.4f"
                " | pred_delta=%.5f gt_delta=%.5f | tf_tok_acc=%.3f uniq=%d/%d" % (
                    step, rec["gen_phys_L1"], rec["identity_phys_L1"], rec["ratio_vs_identity"],
                    rec["gen_bounds_L1"], rec["identity_bounds_L1"],
                    rec["pred_delta_mean_abs"], rec["gt_delta_mean_abs"], tf_acc,
                    rec["uniq_pred_rows"], rec["n"]), logf)
            jsonl.write(json.dumps(rec) + "\n"); jsonl.flush()
            vla.train()
            return rec

        vla.train(); pp.train()
        evaluate(0)
        t0 = time.time()
        for step in range(1, args.steps + 1):
            i = (step - 1) % len(samples)
            e, t = samples[i]
            pv, pr, state, act, gt_idx = cache[i]
            opt.zero_grad(set_to_none=True)
            loss, preds = forward_loss(pv, pr, gt_idx)
            if not torch.isfinite(loss):
                log("FATAL: 损失非有限值（step=%d, loss=%r），立即停止" % (step, float(loss)), logf)
                return finish("FAILED_NAN_LOSS", 3)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(ps, 1.0)
            opt.step()
            if step % args.log_every == 0 or step == 1:
                acc = float(np.mean(preds == gt_idx))
                rec = {"stage": "train", "step": step, "loss": float(loss.item()),
                       "token_acc": acc, "gnorm": float(gn),
                       "lr": float(opt.param_groups[0]["lr"]),
                       "mem_gb": float(torch.cuda.max_memory_allocated() / 1e9),
                       "elapsed_s": round(time.time() - t0, 1)}
                log("step=%d loss=%.4f tok_acc=%.3f gnorm=%.2f mem=%.1fGB %.0fs" % (
                    step, rec["loss"], acc, rec["gnorm"], rec["mem_gb"], rec["elapsed_s"]), logf)
                jsonl.write(json.dumps(rec) + "\n"); jsonl.flush()
            if args.eval_every and step % args.eval_every == 0:
                evaluate(step)

        final = evaluate(args.steps)
        d = os.path.join(out, "checkpoints", "final")
        os.makedirs(d, exist_ok=True)
        vla.save_pretrained(os.path.join(d, "lora_adapter"))
        torch.save(pp.state_dict(), os.path.join(d, "proprio_projector--pure.pt"))
        status["final_eval"] = final
        log("saved %s" % d, logf)
        return finish("DONE", 0)
    except RuntimeError as ex:
        msg = str(ex)
        log("RUNTIME_ERROR: %s" % msg, logf)
        code = 4 if ("out of memory" in msg.lower() or "CUDA" in msg) else 5
        return finish("FAILED_RUNTIME", code)


if __name__ == "__main__":
    sys.exit(main())
