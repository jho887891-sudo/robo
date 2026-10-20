#!/usr/bin/env python3
"""R9 42D/[S,V,Acc] + velocity-residual entrypoint (NEW file; train_pushT_pure2.py untouched).

Status: IMPLEMENTED, NEVER RUN as training. A single optimizer step is not authorized in R9.
  * `--selfcheck` runs the whole DATA path on CPU (real HDF5 -> 42D -> R -> frozen bins ->
    prompt/label -> decode -> metadata) and needs no GPU; this is what Gate B evidence shows.
  * training refuses to start unless R9_TRAINING_AUTHORIZED=1 is exported by the research lead.
  * the 7B forward/backward path here is UNVERIFIED on GPU and must be smoke-tested (bounded,
    no-update) before any real run.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("training", "diagnostics", "code/xpolicylab/policy/OpenVLA_Pure"):
    sys.path.insert(0, os.path.join(REPO, sub))

import r9_trainer_contract as RC  # noqa: E402


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def selfcheck(contract_path, bounds_path, split_path, hdf5_dir, episodes=2, frames=4):
    """Zero-GPU, zero-step verification of the new entrypoint data path."""
    import h5py
    import train_pushT_pure2 as TP
    import action_codec as AC
    contract, meta = RC.load_frozen_contract(contract_path)
    q01 = np.asarray(meta["q01"], np.float32)
    q99 = np.asarray(meta["q99"], np.float32)
    splits = RC.select_splits(split_path)
    nb = np.load(bounds_path)
    rows = []
    for name in splits["train"][:episodes]:
        with h5py.File(os.path.join(hdf5_dir, name), "r") as h:
            state = TP.c14(h, "state")
            action = TP.c14(h, "action")
        for t in RC.frame_pool(len(state))[:frames]:
            v = state[t] - state[t - 1]
            feat = RC.proprio42(state[t], state[t - 1], state[t - 2], contract)
            r = RC.target_residual(state[t], action[t], v)
            idx, ids = RC.encode_tokens(r, q01, q99)
            cmd, _dec, excl = RC.decode_command(ids, q01, q99, state[t], v, bounds_path)
            step = (q99 - q01) / 255.0
            inrange = (r >= q01) & (r <= q99)
            model_dims = list(RC.A.DEV_MODEL_DIMS)
            err = (float(np.abs(cmd - action[t])[model_dims][inrange[model_dims]].max())
                   if inrange[model_dims].any() else 0.0)
            rows.append({"episode": name, "t": int(t), "feature_shape": list(feat.shape),
                         "token_range": [int(ids.min()), int(ids.max())],
                         "decode_error_in_range": err,
                         "decode_ok": bool(err <= float(step.max()) + 1e-5),
                         "gripper_policy": excl,
                         "in_bounds": bool((cmd >= nb["amin"] - 1e-6).all()
                                           and (cmd <= nb["amax"] + 1e-6).all())})
    ok = all(r["feature_shape"] == [42] and r["decode_ok"] and r["in_bounds"]
             and r["gripper_policy"] == [6, 13] for r in rows)
    ic = RC.integration_contract(contract, meta, contract_path, bounds_path, split_path)
    out = {"mode": "selfcheck", "pass": bool(ok), "n_rows": len(rows), "rows": rows,
           "splits": {k: len(v) for k, v in splits.items()},
           "integration_contract": ic, "gaps_vs_standard_entrypoint": RC.standard_entrypoint_gaps(),
           "cuda_used": False, "optimizer_steps": 0}
    log("selfcheck %s over %d frames (%d TRAIN episodes)"
        % ("PASS" if ok else "FAIL", len(rows), episodes))
    return out


def train(args):  # noqa: C901 - mirrors the standard loop, deliberately explicit
    if os.environ.get("R9_TRAINING_AUTHORIZED") != "1":
        raise SystemExit("refusing to train: R9_TRAINING_AUTHORIZED is not set by the research lead")
    import h5py
    import torch
    import train_pushT_pure2 as TP
    import r8_vla_adapter as A
    import action_codec as AC
    contract, meta = RC.load_frozen_contract(args.contract)
    q01 = np.asarray(meta["q01"], np.float32)
    q99 = np.asarray(meta["q99"], np.float32)
    splits = RC.select_splits(os.environ.get("PUSHT_SPLIT", os.path.join(REPO, "data", "split80.json")))
    nb = np.load(os.environ.get("PUSHT_BOUNDS", "/home/dgut/pusht_runtime/push_T_bounds.npz"))
    out = args.out or os.environ.get("PUSHT_OUT")
    if not out:
        raise SystemExit("set --out or PUSHT_OUT to a NEW empty directory")
    if os.path.exists(out) and os.listdir(out):
        raise FileExistsError("output directory must be empty: %s" % out)
    os.makedirs(out + "/checkpoints", exist_ok=True)
    proc, vla, _pp14 = TP.build(args.cams, False)   # 42D projector is ours, not the 14D one
    proj = A.build_projector(int(vla.llm_dim), device="cuda")
    RC.assert_projector_dim(42)
    params = [p for p in vla.parameters() if p.requires_grad] + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[args.lr_milestone], gamma=0.1)
    meta_out = {"variant": "openvla_pure_r9_svacc", "contract": RC.integration_contract(
        contract, meta, args.contract, os.environ["PUSHT_BOUNDS"],
        os.environ.get("PUSHT_SPLIT", os.path.join(REPO, "data", "split80.json"))),
        "cams": args.cams, "proprio_dim": 42, "steps": args.steps, "seed": args.seed,
        "codec_mode": "fixed", "action_input_masking": "keep"}
    with open(os.path.join(out, "run_meta.json"), "w") as fh:
        json.dump(meta_out, fh, indent=1)
    vla.train(); proj.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        e = random.choice(splits["train"])
        with h5py.File(os.path.join(os.environ["PUSHT_HDF5"], e), "r") as h:
            state = TP.c14(h, "state")
            action = TP.c14(h, "action")
            pool = RC.frame_pool(len(state))
            t = int(pool[random.randrange(len(pool))])
            pv, _ = TP.obs(h, t, proc, args.cams, False, nb["pmin"], nb["pmax"])
            v = state[t] - state[t - 1]
            feat = RC.proprio42(state[t], state[t - 1], state[t - 2], contract)
            r = RC.target_residual(state[t], action[t], v)
            idx, _ids = RC.encode_tokens(r, q01, q99)
        inp, lab = TP.make_ids(proc, idx)
        pr = torch.from_numpy(feat).unsqueeze(0).to("cuda")
        inp = inp.to("cuda"); lab = lab.to("cuda")
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                    labels=lab, proprio=pr, proprio_projector=proj)
            loss = o.loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0 or step == 1:
            log("step=%d loss=%.4f gnorm=%.2f mem=%.1fGB %.1fs" % (
                step, float(loss), float(gn), torch.cuda.max_memory_allocated() / 1e9,
                time.time() - t0))
        if args.save_every and step % args.save_every == 0:
            d = "%s/checkpoints/step%d" % (out, step)
            os.makedirs(d, exist_ok=True)
            vla.save_pretrained(d + "/lora_adapter")
            torch.save(proj.state_dict(), d + "/proprio_projector_42.pt")
            log("saved " + d)
    log("R9_SVACC_TRAIN_STOPPED steps=%d (bounded run, not a long training)" % args.steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_milestone", type=int, default=1000)
    ap.add_argument("--cams", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--selfcheck-out", default="")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    if args.selfcheck:
        res = selfcheck(args.contract,
                        os.environ.get("PUSHT_BOUNDS", "/home/dgut/pusht_runtime/push_T_bounds.npz"),
                        os.environ.get("PUSHT_SPLIT", os.path.join(REPO, "data", "split80.json")),
                        os.environ.get("PUSHT_HDF5",
                                       "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data"))
        if args.selfcheck_out:
            with open(args.selfcheck_out, "w") as fh:
                json.dump(res, fh, indent=1)
            log("selfcheck -> %s" % args.selfcheck_out)
        return 0 if res["pass"] else 1
    if args.steps <= 0:
        raise SystemExit("--steps must be > 0 for training (or pass --selfcheck)")
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
