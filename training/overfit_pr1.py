#!/usr/bin/env python3
"""第四阶段：受控 32 样本过拟合（直接复用第三轮 PR 的真实训练代码）。

不做任何“另写一套训练”：本脚本 **import training/train_pushT_pure2.py**，
按其原样调用 build() / obs() / make_ids() / tok() / detok() / action_stats()，
只把外层循环换成“固定 32 个样本 + 最多 300 次更新”，并加上：

  * 真正的逐 token 生成评测（KV 缓存、喂自身预测），不用教师强制冒充
  * 与恒等策略 action=state 的对照、pred/gt 位移量、逐关节误差
  * 退化/饱和检查（唯一行数、极值 bin 占比、长期固定输出）
  * 独立输出目录 + 代码/配置快照 + 失败即停（NaN / OOM）

--mechanism keep : 使用 PR 的 enable_pure_autoreg（R3 修复后的因果自回归）
--mechanism zero : 把 PR 的 enable_pure_autoreg 替换为空操作（= R2/OFT 的置零机制，对照组）

用法：
  python training/overfit_pr1.py --mechanism keep --steps 300 --out <dir>
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

RT = os.environ.setdefault("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
os.environ.setdefault("PUSHT_MODEL", RT + "/model_cache/openvla-7b-pristine")
os.environ.setdefault("PUSHT_HDF5", "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
os.environ.setdefault("PUSHT_SPLIT", RT + "/split80.json")
os.environ.setdefault("PUSHT_BOUNDS", RT + "/push_T_bounds.npz")
os.environ.setdefault("PUSHT_OFT", "/home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_OFT/openvla_oft")

import train_pushT_pure2 as T          # noqa: E402  —— 真实训练代码（PR 分支原文件）
from overfit_protocol import select_train_samples, validated_generation_indices
import action_target as AT

MAX_STEPS = 300
PASS, FAIL = "PASS", "FAIL"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(msg, fh=None):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def pick_samples(n):
    """Select overfit examples from TRAIN episodes; never consume held-out VAL episodes."""
    split = json.load(open(T.SPLIT))
    episode_lengths = {}
    for e in sorted(split["train"]):
        with h5py.File(os.path.join(T.HDF5_DIR, e), "r") as h:
            episode_lengths[e] = int(h["action/left_arm_joint_states"].shape[0])
    return select_train_samples(split, episode_lengths, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", choices=["keep", "zero"], default="keep")
    ap.add_argument("--steps", type=int, default=MAX_STEPS)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--val-n", type=int, default=20)
    ap.add_argument("--target-mode", choices=AT.TARGET_MODES, default=AT.ABSOLUTE)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--out", default="")
    args, _u = ap.parse_known_args()
    if args.steps > MAX_STEPS:
        raise SystemExit("steps 上限 %d" % MAX_STEPS)

    torch.manual_seed(0); np.random.seed(0)
    import random as _r; _r.seed(0)

    out = args.out or os.path.join(RT, "r3_pr1_overfit",
                                   "%s_%s_%s" % (args.target_mode, args.mechanism, time.strftime("%Y%m%d_%H%M%S")))
    if os.path.exists(out) and os.listdir(out):
        raise FileExistsError("Output is non-empty. Never overwrite a previous run: " + out)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "train.log"), "w")
    mj = open(os.path.join(out, "metrics.jsonl"), "w")
    status = {"mechanism": args.mechanism, "steps": args.steps, "out": out, "result": "RUNNING"}

    def finish(res, code=0):
        status["result"] = res
        json.dump(status, open(os.path.join(out, "status.json"), "w"), indent=1)
        try:
            logf.close(); mj.close()
        except Exception:
            pass
        return code

    try:
        # ---------- snapshot
        snap = os.path.join(out, "_snapshot"); os.makedirs(snap, exist_ok=True)
        for f in ("train_pushT_pure2.py", "overfit_pr1.py"):
            src = os.path.join(HERE, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(snap, f))
        for f in ("pure_autoreg_mode.py", "action_codec.py", "action_target.py"):
            src = os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure", f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(snap, f))
        samples = pick_samples(args.n)
        split = json.load(open(T.SPLIT))
        train_set, val_set = set(split["train"]), set(split["val"])
        if train_set & val_set:
            raise ValueError("train/val episode overlap")
        all_fixed = [tuple(x) for x in json.load(open(T.SAMPLE_FIX))]
        val_pairs = all_fixed[:args.val_n]
        if args.val_n <= 0 or len(val_pairs) != args.val_n or any(e not in val_set for e, _ in val_pairs):
            raise ValueError("held-out validation list missing or contains non-validation episodes")
        if any(e not in train_set for e, _ in samples):
            raise ValueError("training samples contain held-out episodes")
        log("samples: train=%d / val=%d; episode-disjoint" % (len(samples), len(val_pairs)), logf)

        # ---------- 机制选择：keep = 直接使用 PR 的 build()（内部调用 enable_pure_autoreg）
        if args.mechanism == "zero":
            T.enable_pure_autoreg = lambda m: m      # 对照组：退化成 R2/OFT 的置零机制
            log("mechanism=zero -> enable_pure_autoreg disabled (OFT original masking)", logf)
        else:
            log("mechanism=keep -> PR enable_pure_autoreg active (causal autoreg)", logf)

        train_eps, val_eps = split["train"], split["val"]
        q01, q99 = T.action_stats(train_eps, target_mode=args.target_mode)
        nb = np.load(T.BOUNDS_FILE); pmin, pmax = nb["pmin"], nb["pmax"]
        log("samples=%d  train_eps=%d  q01/q99 computed from train split" % (len(samples), len(train_eps)), logf)

        proc, vla, pp = T.build(3, True)          # ← 真实训练代码（含 PR 的 enable_pure_autoreg）
        import inspect
        model_sha = AT.file_sha256(inspect.getfile(type(vla.base_model.model)))
        core = vla.base_model.model if hasattr(vla, "base_model") else vla
        log("pure_autoreg flag = %s ; has attr = %s" % (
            getattr(core, "_pure_autoreg_input_mode", False), hasattr(core, "_pure_autoreg_input_mode")), logf)
        free, total = torch.cuda.mem_get_info()
        log("GPU free=%.1fGB total=%.1fGB" % (free / 1e9, total / 1e9), logf)
        use_ckpt = free < 20e9
        if use_ckpt:
            vla.gradient_checkpointing_enable()
            try:
                vla.enable_input_require_grads()
            except Exception as ex:
                log("warn: %s" % ex, logf)
            log("gradient checkpointing ON (显存保护)", logf)
        status["grad_checkpoint"] = bool(use_ckpt)

        json.dump({"mechanism": args.mechanism, "action_target_mode": args.target_mode, "steps": args.steps, "lr": args.lr,
                   "samples": samples, "val_samples": val_pairs, "sample_source": "train", "q01": q01.tolist(), "q99": q99.tolist(),
                   "hashes": {f: sha(os.path.join(snap, f)) for f in sorted(os.listdir(snap))},
                   "torch": torch.__version__},
                  open(os.path.join(snap, "config.json"), "w"), indent=1)

        ps = T.trainable_list(vla, pp)
        opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)
        log("trainable=%.1fM" % (sum(p.numel() for p in ps) / 1e6), logf)

        def load_samples(pairs):
            result = []
            for (e, t) in pairs:
                with h5py.File(os.path.join(T.HDF5_DIR, e), "r") as h:
                    pv, pr = T.obs(h, t, proc, 3, True, pmin, pmax)
                    act = T.c14(h, "action")[t].copy()
                    st = T.c14(h, "state")[t].copy()
                    previous_action = T.c14(h, "action")[max(t - 1, 0)].copy() if t > 0 else st.copy()
                target = AT.training_target(st, act, args.target_mode)
                result.append({"pv": pv, "pr": pr, "act": act, "state": st,
                               "previous_action": previous_action, "idx": T.tok(T.norm_a(target, q01, q99))})
            return result
        cache = load_samples(samples)
        heldout = load_samples(val_pairs)
        pids = None
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        b = PurePromptBuilder("openvla")
        b.add_turn("human", "What action should the robot take to " + T.INSTR + "?")
        pids = proc.tokenizer(b.get_prompt(), return_tensors="pt").input_ids.to("cuda")
        if int(pids[0, -1].item()) != 29871:
            pids = torch.cat([pids, torch.tensor([[29871]], dtype=pids.dtype, device="cuda")], dim=1)
        n_text = int(pids.shape[1])
        from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX, STOP_INDEX
        W0 = 31744
        BINS = 256

        def fwd_loss(s):
            aid = torch.tensor([[ACTION_TOKEN_BEGIN_IDX + 1 + int(x) for x in s["idx"]]],
                               dtype=torch.long, device="cuda")
            stop = torch.full((1, 1), STOP_INDEX, dtype=torch.long, device="cuda")
            inp = torch.cat([pids, aid, stop], dim=1)
            lab = torch.cat([torch.full_like(pids, -100), aid, stop], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=s["pv"],
                        labels=lab, proprio=s["pr"], proprio_projector=pp)
            n_vis = int(o.logits.shape[1]) - int(inp.shape[1])
            preds = []
            for k in range(14):
                lg = o.logits[0, n_vis + n_text + k - 1]
                preds.append(int(torch.argmax(lg[W0:W0 + BINS]).item()))
            return o.loss, np.asarray(preds)

        @torch.no_grad()
        def gen(s):
            """真正的逐 token 生成（KV 缓存，喂自身预测）"""
            vla.eval()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = vla(input_ids=pids, attention_mask=torch.ones_like(pids), pixel_values=s["pv"],
                        labels=pids.clone(), proprio=s["pr"], proprio_projector=pp, use_cache=True)
            past, lg = o.past_key_values, o.logits[0, -1]
            toks = []
            for k in range(14):
                toks.append(int(torch.argmax(lg[W0:W0 + BINS]).item()))
                if k == 13:
                    break
                nxt = torch.tensor([[W0 + toks[-1]]], dtype=torch.long, device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = vla(input_ids=nxt, past_key_values=past, labels=None, use_cache=True)
                past, lg = o.past_key_values, o.logits[0, -1]
            vla.train()
            decoded = T.detok(np.asarray(toks), q01, q99)
            absolute_command = AT.joint_command(s["state"], decoded, args.target_mode)
            return absolute_command, validated_generation_indices(toks)

        @torch.no_grad()
        def evaluate(step, dataset, split_name):
            P, G, S, Prev, TF, generated_token_rows = [], [], [], [], [], []
            for s in dataset:
                generated_action, generated_indices = gen(s)
                P.append(generated_action)
                generated_token_rows.append(generated_indices)
                G.append(s["act"]); S.append(s["state"]); Prev.append(s["previous_action"])
                _, tp = fwd_loss(s); TF.append(tp)
            P, G, S, Prev = np.stack(P), np.stack(G), np.stack(S), np.stack(Prev)
            gt = np.stack([s["idx"] for s in dataset])
            tf_flat = np.concatenate([x for x in TF]); gt_flat = gt.reshape(-1)
            rec = {"stage": "eval", "split": split_name, "target_mode": args.target_mode, "step": int(step), "n": len(P),
                   "tf_tok_acc": float(np.mean(tf_flat == gt_flat)),
                   "tf_tok_acc_per_dim": (tf_flat.reshape(len(P), 14) == gt_flat.reshape(len(P), 14)).mean(0).tolist(),
                   "gen_L1": float(np.abs(P - G).mean()),
                   "identity_L1": float(np.abs(S - G).mean()),
                   "repeat_previous_L1": float(np.abs(Prev - G).mean()),
                   "pred_delta": float(np.abs(P - S).mean()),
                   "gt_delta": float(np.abs(G - S).mean()),
                   "per_dim_gen": np.abs(P - G).mean(0).tolist(),
                   "per_dim_identity": np.abs(S - G).mean(0).tolist(),
                   "per_dim_repeat_previous": np.abs(Prev - G).mean(0).tolist(),
                   "uniq_rows": len({tuple(np.round(r, 3)) for r in P}),
                   "gen_tokens_per_sample": generated_token_rows,
                   "gt_tokens_per_sample": gt.tolist()}
            rec["ratio"] = rec["gen_L1"] / (rec["identity_L1"] + 1e-12)
            jsonl_like = rec
            log("EVAL %s target=%s step=%d | gen_L1=%.5f identity=%.5f ratio=%.2f | pred_delta=%.5f gt_delta=%.5f | "
                "tf_acc=%.3f uniq=%d/%d" % (split_name, args.target_mode, step, rec["gen_L1"], rec["identity_L1"], rec["ratio"],
                                             rec["pred_delta"], rec["gt_delta"], rec["tf_tok_acc"],
                                             rec["uniq_rows"], rec["n"]), logf)
            mj.write(json.dumps(jsonl_like) + "\n")
            mj.flush()
            return rec

        vla.train(); pp.train()
        evaluate(0, cache, "train")
        evaluate(0, heldout, "val")
        t0 = time.time()
        for step in range(1, args.steps + 1):
            s = cache[(step - 1) % len(cache)]
            opt.zero_grad(set_to_none=True)
            loss, preds = fwd_loss(s)
            if not torch.isfinite(loss):
                log("FATAL: non-finite loss at step %d" % step, logf)
                return finish("FAILED_NAN_LOSS", 3)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(ps, 1.0)
            opt.step()
            if step % args.log_every == 0 or step == 1:
                rec = {"stage": "train", "step": step, "loss": float(loss.item()),
                       "tok_acc": float(np.mean(preds == s["idx"])), "gnorm": float(gn),
                       "mem_gb": float(torch.cuda.max_memory_allocated() / 1e9),
                       "elapsed_s": round(time.time() - t0, 1)}
                log("step=%d loss=%.4f tok_acc=%.3f gnorm=%.2f mem=%.1fGB %.0fs" % (
                    step, rec["loss"], rec["tok_acc"], rec["gnorm"], rec["mem_gb"], rec["elapsed_s"]), logf)
                mj.write(json.dumps(rec) + "\n"); mj.flush()
            if args.eval_every and step % args.eval_every == 0:
                evaluate(step, cache, "train")
                evaluate(step, heldout, "val")
        final = evaluate(args.steps, cache, "train")
        final_val = evaluate(args.steps, heldout, "val")
        d = os.path.join(out, "checkpoints", "final"); os.makedirs(d, exist_ok=True)
        vla.save_pretrained(os.path.join(d, "lora_adapter"))
        torch.save(pp.state_dict(), os.path.join(d, "proprio_projector--pure.pt"))
        checkpoint_meta = AT.record_checkpoint_hashes(
            {"q01": q01.tolist(), "q99": q99.tolist(),
             "action_target_mode": args.target_mode,
             "action_input_masking": args.mechanism,
             "codec_mode": T.CODEC_MODE,
             "samples_from": "train", "steps": args.steps,
             "modeling_sha256": model_sha, "variant": "pure_openvla"},
            os.path.join(d, "lora_adapter"), os.path.join(d, "proprio_projector--pure.pt"))
        with open(os.path.join(out, "run_meta.json"), "w") as fh:
            json.dump(checkpoint_meta, fh, indent=1)
        status["final_eval"] = final
        status["final_val_eval"] = final_val
        log("saved %s" % d, logf)
        return finish("DONE", 0)
    except RuntimeError as ex:
        log("RUNTIME_ERROR: %s" % ex, logf)
        return finish("FAILED_RUNTIME", 4)


if __name__ == "__main__":
    sys.exit(main())
