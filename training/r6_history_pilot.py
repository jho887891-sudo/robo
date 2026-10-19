#!/usr/bin/env python3
"""R6 pilot: 28D history proprio A/B on real push_T data.

Two arms, IDENTICAL in every respect except the value of proprio slot 1:

  --history-mode duplicate   P_t = [S_t, S_t]        control (no history information)
  --history-mode true        P_t = [S_t, S_(t-1)]    treatment (real past state)

Same base model, LoRA config, ProprioProjector shape (28 -> 4096 -> 4096), residual
action target, train/val samples, seed, optimizer, lr and update budget. The only
difference is what slot 1 contains, so any difference in physical L1 is attributable
to the history information and not to capacity or budget.

This script reuses training/train_pushT_pure2.py for build()/obs()/c14()/tok() and
does NOT modify it, overfit_pr1.py, the codec, or any R2-R5 checkpoint.

Hard limits: at most 300 parameter updates, no 3000/6000-step run, no closed loop.
"""
import argparse
import hashlib
import inspect
import json
import os
import random
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

import train_pushT_pure2 as T          # noqa: E402  (real training code, unmodified)
from overfit_protocol import validated_generation_indices   # noqa: E402
from r5_sampling import (candidate_frames, diversified_manifest,   # noqa: E402
                         summarize_manifest, spread_validation_frames)
from r5_metrics import summarize_prediction                # noqa: E402
from r6_metrics import summarize_direction                 # noqa: E402
import action_target as AT                                  # noqa: E402
import proprio_history as PH                                # noqa: E402

MAX_STEPS = 300          # R6 authorization: pilot only
PASS, FAIL = "PASS", "FAIL"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(msg, fh=None):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def pick_samples(n, strategy):
    split = json.load(open(T.SPLIT))
    rows = []
    for e in sorted(split["train"]):
        with h5py.File(os.path.join(T.HDF5_DIR, e), "r") as h:
            rows.extend(candidate_frames(e, T.c14(h, "state"), T.c14(h, "action"), stride=20))
    chosen = diversified_manifest(split, rows, n=n, seed=0)
    return [[e, t] for e, t, _l, _r in chosen], {"strategy": strategy,
                                                 "summary": summarize_manifest(chosen)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-mode", choices=[PH.DUPLICATE, PH.TRUE], required=True)
    ap.add_argument("--target-mode", choices=AT.TARGET_MODES, default=AT.RESIDUAL)
    ap.add_argument("--steps", type=int, default=MAX_STEPS)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--val-n", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--train-eval-n", type=int, default=60,
                    help="training-split frames used for the curve only (val is always full)")
    ap.add_argument("--out", default="")
    args, _u = ap.parse_known_args()
    if args.steps > MAX_STEPS:
        raise SystemExit("steps is capped at %d for the R6 pilot" % MAX_STEPS)
    if args.steps <= 0:
        raise SystemExit("steps must be positive")

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    out = args.out or os.path.join(RT, "r6_pilot", "%s_%s" % (args.history_mode, time.strftime("%Y%m%d_%H%M%S")))
    if os.path.exists(out) and os.listdir(out):
        raise FileExistsError("Output is non-empty. Never overwrite a previous run: " + out)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "train.log"), "w")
    mj = open(os.path.join(out, "metrics.jsonl"), "w")
    status = {"history_mode": args.history_mode, "target_mode": args.target_mode,
              "steps": args.steps, "out": out, "result": "RUNNING"}

    def finish(res, code=0):
        status["result"] = res
        json.dump(status, open(os.path.join(out, "status.json"), "w"), indent=1)
        try:
            logf.close(); mj.close()
        except Exception:
            pass
        return code

    try:
        snap = os.path.join(out, "_snapshot"); os.makedirs(snap, exist_ok=True)
        for f in ("train_pushT_pure2.py", "r6_history_pilot.py", "r6_metrics.py",
                  "r5_sampling.py", "r5_metrics.py", "overfit_protocol.py"):
            src = os.path.join(HERE, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(snap, f))
        for f in ("proprio_history.py", "pure_autoreg_mode.py", "action_codec.py", "action_target.py"):
            src = os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure", f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(snap, f))

        split = json.load(open(T.SPLIT))
        train_set, val_set = set(split["train"]), set(split["val"])
        samples, manifest_diag = pick_samples(args.n, "diverse")
        val_lengths = {}
        for e in sorted(split["val"]):
            with h5py.File(os.path.join(T.HDF5_DIR, e), "r") as h:
                val_lengths[e] = int(h["action/left_arm_joint_states"].shape[0])
        val_pairs = spread_validation_frames(split, val_lengths, args.val_n, seed=0)
        if len(val_pairs) != args.val_n or any(e not in val_set for e, _ in val_pairs):
            raise ValueError("held-out validation list invalid")
        if any(e not in train_set for e, _ in samples):
            raise ValueError("training samples contain held-out episodes")
        if train_set & val_set:
            raise ValueError("train/val episode overlap")
        log("arm=%s target=%s | train=%d frames/%d episodes val=%d frames/%d episodes (disjoint)" % (
            args.history_mode, args.target_mode, len(samples), len({e for e, _ in samples}),
            len(val_pairs), len({e for e, _ in val_pairs})), logf)

        q01, q99 = T.action_stats(train_set, target_mode=args.target_mode)
        train_deltas = []
        for e in sorted(train_set):
            with h5py.File(os.path.join(T.HDF5_DIR, e), "r") as h:
                train_deltas.append(T.c14(h, "action") - T.c14(h, "state"))
        median_delta = np.median(np.concatenate(train_deltas, axis=0), axis=0).astype(np.float32)
        nb = np.load(T.BOUNDS_FILE); pmin, pmax = nb["pmin"], nb["pmax"]

        proc, vla, _ = T.build(3, False)      # real training build(); own 28D projector below
        model_sha = AT.file_sha256(inspect.getfile(type(vla.base_model.model)))
        dim = PH.proprio_dim(args.history_mode)
        from prismatic.models.projectors import ProprioProjector
        pp = ProprioProjector(proprio_dim=dim, llm_dim=vla.llm_dim).to("cuda")
        n_pp = sum(p.numel() for p in pp.parameters())
        log("proprio interface: mode=%s dim=%d projector_params=%d (28D both arms)" % (
            args.history_mode, dim, n_pp), logf)

        free, total = torch.cuda.mem_get_info()
        log("GPU free=%.1fGB total=%.1fGB" % (free / 1e9, total / 1e9), logf)
        use_ckpt = free < 20e9
        if use_ckpt:
            vla.gradient_checkpointing_enable()
            try:
                vla.enable_input_require_grads()
            except Exception as ex:
                log("warn: %s" % ex, logf)
            log("gradient checkpointing ON", logf)
        status["grad_checkpoint"] = bool(use_ckpt)

        def load_samples(pairs):
            res = []
            for (e, t) in pairs:
                with h5py.File(os.path.join(T.HDF5_DIR, e), "r") as h:
                    pv, _ = T.obs(h, t, proc, 3, False, pmin, pmax)
                    act = T.c14(h, "action")[t].copy()
                    st = T.c14(h, "state")[t].copy()
                    prev_st = T.c14(h, "state")[t - 1].copy() if t > 0 else None
                    prev_act = T.c14(h, "action")[t - 1].copy() if t > 0 else st.copy()
                pr_np = PH.build(st, prev_st, pmin, pmax, args.history_mode)
                target = AT.training_target(st, act, args.target_mode)
                res.append({"pv": pv, "pr": torch.from_numpy(pr_np).unsqueeze(0).to("cuda"),
                            "act": act, "state": st, "prev_state": st if prev_st is None else prev_st,
                            "previous_action": prev_act, "episode": e, "frame": int(t),
                            "proprio": pr_np, "idx": T.tok(T.norm_a(target, q01, q99))})
            return res

        cache = load_samples(samples)
        heldout = load_samples(val_pairs)

        # ---- input-interface guards: the two arms must differ ONLY in slot 1
        slot_gap = np.array([np.abs(s["proprio"][14:] - s["proprio"][:14]).mean() for s in cache])
        if args.history_mode == PH.DUPLICATE:
            if np.abs(slot_gap).max() != 0.0:
                raise SystemExit("control arm must have identical slots; got max gap %.3e" % slot_gap.max())
        else:
            frames_gt0 = [s for s in cache if s["frame"] > 0]
            if slot_gap.max() <= 0.0 or not frames_gt0:
                raise SystemExit("treatment arm has no history signal in slot 1")
        log("slot1-slot0 mean|gap| = %.6f (max %.6f) over %d samples" % (
            float(slot_gap.mean()), float(slot_gap.max()), len(slot_gap)), logf)

        json.dump({"history_mode": args.history_mode, "proprio_dim": dim,
                   "proprio_dims_total": [dim, n_pp], "target_mode": args.target_mode,
                   "steps": args.steps, "lr": args.lr, "seed": 0,
                   "sampling_strategy": "diverse", "validation_strategy": "spread",
                   "samples": samples, "val_samples": val_pairs,
                   "train_manifest_diagnostics": manifest_diag,
                   "slot_gap_mean": float(slot_gap.mean()), "slot_gap_max": float(slot_gap.max()),
                   "median_delta": median_delta.tolist(), "q01": q01.tolist(), "q99": q99.tolist(),
                   "pmin": np.asarray(pmin, np.float32).tolist(), "pmax": np.asarray(pmax, np.float32).tolist(),
                   "first_frame_policy": PH.first_frame_policy(),
                   "checkpoint_fields": PH.checkpoint_fields(args.history_mode),
                   "hashes": {f: sha(os.path.join(snap, f)) for f in sorted(os.listdir(snap))},
                   "torch": torch.__version__},
                  open(os.path.join(snap, "config.json"), "w"), indent=1)

        ps = T.trainable_list(vla, pp)
        opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)
        log("trainable=%.1fM (LoRA+projector)" % (sum(p.numel() for p in ps) / 1e6), logf)

        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        b = PurePromptBuilder("openvla")
        b.add_turn("human", "What action should the robot take to " + T.INSTR + "?")
        pids = proc.tokenizer(b.get_prompt(), return_tensors="pt").input_ids.to("cuda")
        if int(pids[0, -1].item()) != 29871:
            pids = torch.cat([pids, torch.tensor([[29871]], dtype=pids.dtype, device="cuda")], dim=1)
        n_text = int(pids.shape[1])
        from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX, STOP_INDEX
        W0, BINS = 31744, 256

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
            return AT.joint_command(s["state"], decoded, args.target_mode), validated_generation_indices(toks)

        @torch.no_grad()
        def evaluate(step, dataset, split_name, with_tf=True):
            P, G, S, Prev, GEN = [], [], [], [], []
            for s in dataset:
                cmd, idxs = gen(s)
                P.append(cmd); GEN.append(idxs)
                G.append(s["act"]); S.append(s["state"]); Prev.append(s["prev_state"])
            P, G, S, Prev = np.stack(P), np.stack(G), np.stack(S), np.stack(Prev)
            gt = np.stack([s["idx"] for s in dataset])
            rec = {"stage": "eval", "split": split_name, "arm": args.history_mode,
                   "target_mode": args.target_mode, "step": int(step), "n": int(len(P)),
                   "gen_L1": float(np.abs(P - G).mean()),
                   "identity_L1": float(np.abs(S - G).mean()),
                   "gen_tokens_per_sample": GEN, "gt_tokens_per_sample": gt.tolist(),
                   "uniq_rows": len({tuple(np.round(r, 3)) for r in P})}
            rec["ratio_vs_identity"] = rec["gen_L1"] / (rec["identity_L1"] + 1e-12)
            if with_tf:
                TF = np.concatenate([fwd_loss(s)[1] for s in dataset])
                rec["tf_tok_acc"] = float(np.mean(TF == gt.reshape(-1)))
            rec["r5"] = summarize_prediction(P, G, S, Prev, [x["episode"] for x in dataset], GEN, median_delta)
            rec["r6"] = summarize_direction(P, G, S, Prev, [x["episode"] for x in dataset])
            mj.write(json.dumps(rec) + "\n"); mj.flush()
            d = rec["r6"]["delta_correlation"]
            log("EVAL %s arm=%s step=%d | gen_L1=%.5f identity=%.5f ratio=%.3f | corr_delta=%.3f "
                "sign@1e-3=%s | beats_cv=%d/%d uniq_rows=%d/%d" % (
                    split_name, args.history_mode, step, rec["gen_L1"], rec["identity_L1"],
                    rec["ratio_vs_identity"], d["pooled"],
                    next((round(r["sign_agreement"], 3) for r in rec["r6"]["sign_agreement"]
                          if r["basis"] == "expert_delta" and r["threshold"] == 1e-3), None),
                    rec["r6"]["episodes_beating_constant_velocity"], rec["r6"]["episode_count"],
                    rec["uniq_rows"], rec["n"]), logf)
            return rec

        vla.train(); pp.train()
        train_eval_set = cache[:args.train_eval_n]
        evaluate(0, train_eval_set, "train")
        evaluate(0, heldout, "val")
        t0 = time.time()
        grad_log = []
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
                g = pp.fc1.weight.grad
                cur_g = float(g[:, :14].norm()) if g is not None else None
                hist_g = float(g[:, 14:].norm()) if g is not None else None
                rec = {"stage": "train", "step": step, "loss": float(loss.item()),
                       "tok_acc": float(np.mean(preds == s["idx"])), "gnorm": float(gn),
                       "proj_grad_current_slot": cur_g, "proj_grad_history_slot": hist_g,
                       "mem_gb": float(torch.cuda.max_memory_allocated() / 1e9),
                       "elapsed_s": round(time.time() - t0, 1)}
                grad_log.append(rec)
                log("step=%d loss=%.4f tok_acc=%.3f gnorm=%.2f | pp.grad cur=%.4g hist=%.4g | mem=%.1fGB %.0fs" % (
                    step, rec["loss"], rec["tok_acc"], rec["gnorm"], cur_g or 0.0, hist_g or 0.0,
                    rec["mem_gb"], rec["elapsed_s"]), logf)
                mj.write(json.dumps(rec) + "\n"); mj.flush()
            if args.eval_every and step % args.eval_every == 0 and step != args.steps:
                evaluate(step, heldout, "val")
        final_train = evaluate(args.steps, train_eval_set, "train")
        final_val = evaluate(args.steps, heldout, "val")

        ckdir = os.path.join(out, "checkpoints", "final"); os.makedirs(ckdir, exist_ok=True)
        vla.save_pretrained(os.path.join(ckdir, "lora_adapter"))
        torch.save(pp.state_dict(), os.path.join(ckdir, "proprio_projector--pure.pt"))
        meta = {"q01": q01.tolist(), "q99": q99.tolist(), "action_target_mode": args.target_mode,
                "action_input_masking": "keep", "codec_mode": T.CODEC_MODE,
                "samples_from": "train", "sampling_strategy": "diverse",
                "validation_strategy": "spread", "steps": args.steps,
                "modeling_sha256": model_sha, "variant": "pure_openvla",
                "lr": args.lr, "seed": 0, "pmin": np.asarray(pmin, np.float32).tolist(),
                "pmax": np.asarray(pmax, np.float32).tolist()}
        meta.update(PH.checkpoint_fields(args.history_mode))
        checkpoint_meta = AT.record_checkpoint_hashes(
            meta, os.path.join(ckdir, "lora_adapter"), os.path.join(ckdir, "proprio_projector--pure.pt"))
        json.dump(checkpoint_meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1)
        json.dump({"grad_log": grad_log, "history_slot_gradient_seen": bool(
            any((r["proj_grad_history_slot"] or 0.0) > 0.0 for r in grad_log))},
            open(os.path.join(out, "gradient_trace.json"), "w"), indent=1)
        status["final_train_eval"] = final_train
        status["final_val_eval"] = final_val
        log("saved %s" % ckdir, logf)
        return finish("DONE", 0)
    except RuntimeError as ex:
        log("RUNTIME_ERROR: %s" % ex, logf)
        return finish("FAILED_RUNTIME", 4)


if __name__ == "__main__":
    sys.exit(main())
