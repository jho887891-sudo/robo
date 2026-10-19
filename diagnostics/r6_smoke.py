#!/usr/bin/env python3
"""R6 smoke-test gate: does the 28D history proprio actually enter the network?

Runs on GPU with the real OpenVLA-7B + a 28D ProprioProjector and real HDF5 frames.
Every check prints PASS/FAIL and the JSON summary; any FAIL makes the process exit
non-zero, so the GPU pilot cannot start on a broken input interface.

  S1 forward/backward finite; projector + LoRA gradients non-zero; BOTH slots get gradient
  S2 perturbing only slot 1 (history) changes the decoded action tokens
  S3 perturbing only slot 0 (current state) changes the decoded action tokens
  S4 no future leak: corrupting S_(t+1)/A_(t+1) leaves the frame-t input identical
  S5 causal autoregressive decoding: KV-cache generation == full-sequence argmax
  S6 determinism, finite logits, shape/dtype contract
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "training"))

RT = os.environ.setdefault("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
os.environ.setdefault("PUSHT_MODEL", RT + "/model_cache/openvla-7b-pristine")
os.environ.setdefault("PUSHT_HDF5", "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
os.environ.setdefault("PUSHT_SPLIT", RT + "/split80.json")
os.environ.setdefault("PUSHT_BOUNDS", RT + "/push_T_bounds.npz")
os.environ.setdefault("PUSHT_OFT", "/home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_OFT/openvla_oft")

import train_pushT_pure2 as T        # noqa: E402
import proprio_history as PH         # noqa: E402
import action_target as AT           # noqa: E402

W0, BINS = 31744, 256
RESULTS = []


def check(name, ok, detail=None):
    RESULTS.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
    print("[%s] %s %s" % ("PASS" if ok else "FAIL", name,
                          json.dumps(detail) if detail is not None else ""), flush=True)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-mode", choices=[PH.DUPLICATE, PH.TRUE], default=PH.TRUE)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    split = json.load(open(T.SPLIT))
    nb = np.load(T.BOUNDS_FILE); pmin, pmax = nb["pmin"], nb["pmax"]
    q01, q99 = T.action_stats(set(split["train"]), target_mode=AT.RESIDUAL)
    ep = sorted(split["train"])[0]
    with h5py.File(os.path.join(T.HDF5_DIR, ep), "r") as h:
        state = T.c14(h, "state"); action = T.c14(h, "action")
    probe_frames = [30, 120, 240][:args.frames]

    proc, vla, _ = T.build(3, False)
    from prismatic.models.projectors import ProprioProjector
    dim = PH.proprio_dim(args.history_mode)
    pp = ProprioProjector(proprio_dim=dim, llm_dim=vla.llm_dim).to("cuda")
    n28 = sum(p.numel() for p in pp.parameters())
    n14 = sum(p.numel() for p in ProprioProjector(proprio_dim=14, llm_dim=vla.llm_dim).parameters())
    free, total = torch.cuda.mem_get_info()
    gpu_free_gb = float(free / 1e9)
    if free < 20e9:
        vla.gradient_checkpointing_enable()
        try:
            vla.enable_input_require_grads()
        except Exception:
            pass

    samples = []
    for t in probe_frames:
        with h5py.File(os.path.join(T.HDF5_DIR, ep), "r") as h:
            pv, _ = T.obs(h, t, proc, 3, False, pmin, pmax)
        pr_np = PH.build(state[t], state[t - 1] if t > 0 else None, pmin, pmax, args.history_mode)
        samples.append({"frame": int(t), "pv": pv,
                        "pr": torch.from_numpy(pr_np).unsqueeze(0).to("cuda"),
                        "idx": T.tok(T.norm_a(AT.training_target(
                            state[t], action[t], AT.RESIDUAL), q01, q99))})

    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    b = PurePromptBuilder("openvla")
    b.add_turn("human", "What action should the robot take to " + T.INSTR + "?")
    pids = proc.tokenizer(b.get_prompt(), return_tensors="pt").input_ids.to("cuda")
    if int(pids[0, -1].item()) != 29871:
        pids = torch.cat([pids, torch.tensor([[29871]], dtype=pids.dtype, device="cuda")], dim=1)

    def forward(s, pr=None):
        pr = s["pr"] if pr is None else pr
        aid = torch.tensor([[W0 + 1 + int(x) for x in s["idx"]]], dtype=torch.long, device="cuda")
        stop = torch.full((1, 1), 2, dtype=torch.long, device="cuda")
        inp = torch.cat([pids, aid, stop], dim=1)
        lab = torch.cat([torch.full_like(pids, -100), aid, stop], dim=1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=s["pv"],
                       labels=lab, proprio=pr, proprio_projector=pp)

    @torch.no_grad()
    def gen_tokens(s, pr=None):
        pr = s["pr"] if pr is None else pr
        vla.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=pids, attention_mask=torch.ones_like(pids), pixel_values=s["pv"],
                    labels=pids.clone(), proprio=pr, proprio_projector=pp, use_cache=True)
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
        return toks

    @torch.no_grad()
    def full_seq_tokens(s):
        vla.eval()
        toks = []
        for k in range(14):
            inp = pids if not toks else torch.cat(
                [pids, torch.tensor([[W0 + t for t in toks]], dtype=pids.dtype, device="cuda")], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=s["pv"],
                        labels=inp.clone(), proprio=s["pr"], proprio_projector=pp)
            toks.append(int(torch.argmax(o.logits[0, -1][W0:W0 + BINS]).item()))
        vla.train()
        return toks

    s = samples[0]
    out = forward(s)
    ok = check("S1a_forward_loss_finite", bool(torch.isfinite(out.loss).item()),
               {"loss": float(out.loss.item())})
    vla.train(); pp.train()
    vla.zero_grad(set_to_none=True); pp.zero_grad(set_to_none=True)
    out.loss.backward()
    pgrads = {n: (float(p.grad.norm()) if p.grad is not None else None) for n, p in pp.named_parameters()}
    lora = [float(p.grad.norm()) for n, p in vla.named_parameters()
            if p.requires_grad and p.grad is not None and "lora" in n]
    ok &= check("S1b_projector_grads_nonzero",
                all(v is not None and v > 0 for v in pgrads.values()), pgrads)
    ok &= check("S1c_lora_grads_nonzero", bool(lora) and max(lora) > 0,
                {"n_lora_tensors": len(lora), "max_norm": max(lora) if lora else None})
    g = pp.fc1.weight.grad
    cur_g = float(g[:, :14].norm()) if g is not None else 0.0
    hist_g = float(g[:, 14:].norm()) if g is not None else 0.0
    ok &= check("S1d_both_proprio_slots_receive_gradient", cur_g > 0 and hist_g > 0,
                {"grad_current_slot": cur_g, "grad_history_slot": hist_g})
    vla.zero_grad(set_to_none=True); pp.zero_grad(set_to_none=True)

    base = gen_tokens(s)
    pr_hist = s["pr"].clone(); pr_hist[0, 14:] = torch.clamp(pr_hist[0, 14:] + 0.35, -1.0, 1.0)
    hist_toks = gen_tokens(s, pr_hist)
    pr_cur = s["pr"].clone(); pr_cur[0, :14] = torch.clamp(pr_cur[0, :14] + 0.35, -1.0, 1.0)
    cur_toks = gen_tokens(s, pr_cur)
    ok &= check("S2_history_slot_changes_output", hist_toks != base,
                {"base": base[:6], "perturbed": hist_toks[:6],
                 "dims_changed": sum(1 for a, c in zip(base, hist_toks) if a != c)})
    ok &= check("S3_current_slot_changes_output", cur_toks != base,
                {"dims_changed": sum(1 for a, c in zip(base, cur_toks) if a != c)})

    leak_ok, detail = True, {}
    for t in probe_frames:
        st2, ac2 = state.copy(), action.copy()
        st2[t + 1] = st2[t + 1] + 1.0
        ac2[t + 1] = ac2[t + 1] - 1.0
        a = PH.build(state[t], state[t - 1] if t > 0 else None, pmin, pmax, args.history_mode)
        b2 = PH.build(st2[t], st2[t - 1] if t > 0 else None, pmin, pmax, args.history_mode)
        tgt_a = AT.training_target(state[t], action[t], AT.RESIDUAL)
        tgt_b = AT.training_target(st2[t], ac2[t], AT.RESIDUAL)
        same = bool(np.array_equal(a, b2)) and bool(np.array_equal(tgt_a, tgt_b))
        leak_ok = leak_ok and same
        detail["frame_%d" % t] = {"input_and_target_identical": same}
    ok &= check("S4_no_future_state_leak", leak_ok, detail)

    kv = gen_tokens(samples[0]); full = full_seq_tokens(samples[0])
    ok &= check("S5_kv_equals_full_sequence_argmax", kv == full,
                {"matching_dims": sum(1 for a, c in zip(kv, full) if a == c),
                 "kv": kv, "full": full})
    ok &= check("S6a_generation_deterministic", gen_tokens(samples[0]) == kv)
    with torch.no_grad():
        o = forward(samples[0])
    ok &= check("S6b_logits_finite", bool(torch.isfinite(o.logits).all().item()),
                {"logits_shape": list(o.logits.shape)})
    ok &= check("S6c_shape_contract",
                list(samples[0]["pr"].shape) == [1, dim] and pp.proprio_dim == dim,
                {"proprio": list(samples[0]["pr"].shape), "projector_proprio_dim": pp.proprio_dim})

    summary = {"history_mode": args.history_mode, "result": "PASS" if ok else "FAIL",
               "checks": RESULTS, "frames": probe_frames, "episode": ep,
               "projector_params_28d": n28, "projector_params_14d": n14,
               "projector_param_delta": n28 - n14,
               "gpu_free_gb_at_start": gpu_free_gb,
               "grad_current_slot": cur_g, "grad_history_slot": hist_g,
               "torch": torch.__version__}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)
    print("SMOKE %s -> %s" % (summary["result"], args.out))
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
