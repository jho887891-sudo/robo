#!/usr/bin/env python3
"""R6 smoke-test gate: does the 28D history proprio actually reach the network output?

Runs on GPU with the real OpenVLA-7B and real HDF5 frames. Prints PASS/FAIL per
check and writes a JSON summary; any non-informational FAIL exits non-zero, so the
GPU pilot cannot start on a broken input interface.

  S1 forward/backward finite; projector + LoRA gradients non-zero; BOTH slots get gradient
  S2 slot-1 (history) perturbation reaches the action logits (amplitude sweep, token flips reported)
  S3 slot-0 (current state) perturbation reaches the action logits
  S4 no future leak: corrupting S_(t+1)/A_(t+1) leaves the frame-t input and target identical
  S5 causal autoregressive decoding: KV-cache generation == full-sequence argmax
  S6 determinism, finite logits, shape/dtype contract
  S7 informational references: 14D (R2-R5) interface sensitivity, visual sensitivity, top-1 confidence
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
AMPLITUDES = (0.1, 0.35, 1.0)
LOGIT_GATE = 0.05          # max |delta logit| inside the 256-bin action window


def check(name, ok, detail=None, informational=False):
    RESULTS.append({"check": name, "result": "PASS" if ok else "FAIL",
                    "informational": bool(informational), "detail": detail})
    tag = "INFO" if informational else ("PASS" if ok else "FAIL")
    print("[%s] %s %s" % (tag, name, json.dumps(detail) if detail is not None else ""), flush=True)
    return True if informational else bool(ok)


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
    pp14 = ProprioProjector(proprio_dim=14, llm_dim=vla.llm_dim).to("cuda")
    n28 = sum(p.numel() for p in pp.parameters())
    n14 = sum(p.numel() for p in pp14.parameters())
    free, _total = torch.cuda.mem_get_info()
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
                        "pr14": torch.from_numpy(PH.build(state[t], None, pmin, pmax, PH.NONE)).unsqueeze(0).to("cuda"),
                        "idx": T.tok(T.norm_a(AT.training_target(
                            state[t], action[t], AT.RESIDUAL), q01, q99))})

    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    b = PurePromptBuilder("openvla")
    b.add_turn("human", "What action should the robot take to " + T.INSTR + "?")
    pids = proc.tokenizer(b.get_prompt(), return_tensors="pt").input_ids.to("cuda")
    if int(pids[0, -1].item()) != 29871:
        pids = torch.cat([pids, torch.tensor([[29871]], dtype=pids.dtype, device="cuda")], dim=1)

    def forward(s, pr=None, proj=None):
        pr = s["pr"] if pr is None else pr
        proj = pp if proj is None else proj
        aid = torch.tensor([[W0 + 1 + int(x) for x in s["idx"]]], dtype=torch.long, device="cuda")
        stop = torch.full((1, 1), 2, dtype=torch.long, device="cuda")
        inp = torch.cat([pids, aid, stop], dim=1)
        lab = torch.cat([torch.full_like(pids, -100), aid, stop], dim=1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=s["pv"],
                       labels=lab, proprio=pr, proprio_projector=proj)

    @torch.no_grad()
    def first_step_logits(s, pr=None, proj=None, pv=None):
        pr = s["pr"] if pr is None else pr
        proj = pp if proj is None else proj
        pv = s["pv"] if pv is None else pv
        vla.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=pids, attention_mask=torch.ones_like(pids), pixel_values=pv,
                    labels=pids.clone(), proprio=pr, proprio_projector=proj, use_cache=True)
        vla.train()
        return o.logits[0, -1][W0:W0 + BINS].float(), o.past_key_values

    @torch.no_grad()
    def gen_tokens(s, pr=None, proj=None, pv=None):
        pr = s["pr"] if pr is None else pr
        proj = pp if proj is None else proj
        pv = s["pv"] if pv is None else pv
        vla.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=pids, attention_mask=torch.ones_like(pids), pixel_values=pv,
                    labels=pids.clone(), proprio=pr, proprio_projector=proj, use_cache=True)
        past, lg = o.past_key_values, o.logits[0, -1]
        toks, probs = [], []
        for k in range(14):
            cand = lg[W0:W0 + BINS]
            p = torch.softmax(cand.float(), dim=-1)
            probs.append(float(p.max().item()))
            toks.append(int(torch.argmax(cand).item()))
            if k == 13:
                break
            nxt = torch.tensor([[W0 + toks[-1]]], dtype=torch.long, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = vla(input_ids=nxt, past_key_values=past, labels=None, use_cache=True)
            past, lg = o.past_key_values, o.logits[0, -1]
        vla.train()
        return toks, probs

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

    def sweep(s, slot, proj, base_logits, base_toks):
        """Perturb one proprio slot at several amplitudes; report logit and token effects."""
        rows = []
        for amp in AMPLITUDES:
            pr = s["pr"].clone()
            if slot == "history":
                pr[0, 14:] = torch.clamp(pr[0, 14:] + amp, -1.0, 1.0)
            else:
                pr[0, :14] = torch.clamp(pr[0, :14] + amp, -1.0, 1.0)
            lg, _ = first_step_logits(s, pr=pr, proj=proj)
            toks, _ = gen_tokens(s, pr=pr, proj=proj)
            rows.append({"amplitude": amp,
                         "max_abs_delta_logit": float((lg - base_logits).abs().max().item()),
                         "mean_abs_delta_logit": float((lg - base_logits).abs().mean().item()),
                         "tokens_changed": sum(1 for a, c in zip(base_toks, toks) if a != c),
                         "tokens": toks})
        return rows

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
    ok &= check("S1b_projector_grads_nonzero", all(v is not None and v > 0 for v in pgrads.values()), pgrads)
    ok &= check("S1c_lora_grads_nonzero", bool(lora) and max(lora) > 0,
                {"n_lora_tensors": len(lora), "max_norm": max(lora) if lora else None})
    g = pp.fc1.weight.grad
    cur_g = float(g[:, :14].norm()) if g is not None else 0.0
    hist_g = float(g[:, 14:].norm()) if g is not None else 0.0
    ok &= check("S1d_both_proprio_slots_receive_gradient", cur_g > 0 and hist_g > 0,
                {"grad_current_slot": cur_g, "grad_history_slot": hist_g})
    vla.zero_grad(set_to_none=True); pp.zero_grad(set_to_none=True)

    base_logits, _ = first_step_logits(s)
    base_toks, base_probs = gen_tokens(s)
    hist_rows = sweep(s, "history", pp, base_logits, base_toks)
    cur_rows = sweep(s, "current", pp, base_logits, base_toks)
    max_hist = max(r["max_abs_delta_logit"] for r in hist_rows)
    max_cur = max(r["max_abs_delta_logit"] for r in cur_rows)
    ok &= check("S2_history_slot_reaches_action_logits", max_hist > LOGIT_GATE,
                {"sweep": hist_rows, "gate_max_abs_delta_logit": LOGIT_GATE})
    ok &= check("S3_current_slot_reaches_action_logits", max_cur > LOGIT_GATE,
                {"sweep": cur_rows, "gate_max_abs_delta_logit": LOGIT_GATE})

    # informational references: the 14D (R2-R5) interface and a visual perturbation
    flat14 = dict(s); flat14["pr"] = s["pr14"]
    base14_logits, _ = first_step_logits(flat14, pr=s["pr14"], proj=pp14)
    row14 = []
    for amp in AMPLITUDES:
        pr = s["pr14"].clone(); pr[0, :] = torch.clamp(pr[0, :] + amp, -1.0, 1.0)
        lg, _ = first_step_logits(flat14, pr=pr, proj=pp14)
        t14, _ = gen_tokens(flat14, pr=pr, proj=pp14)
        base14_toks, _ = gen_tokens(flat14, pr=s["pr14"], proj=pp14)
        row14.append({"amplitude": amp,
                      "max_abs_delta_logit": float((lg - base14_logits).abs().max().item()),
                      "tokens_changed": sum(1 for a, c in zip(base14_toks, t14) if a != c)})
    pv_noise = torch.clamp(s["pv"] + 0.05, -1.0, 1.0)
    pv_zero = torch.zeros_like(s["pv"])
    lg_noise, _ = first_step_logits(s, pv=pv_noise)
    lg_zero, _ = first_step_logits(s, pv=pv_zero)
    check("S7a_reference_14d_interface_sensitivity", True,
          {"sweep": row14, "note": "same measurement on the R2-R5 14D interface"}, informational=True)
    check("S7b_reference_visual_sensitivity", True,
          {"image_noise_0.05": {"max_abs_delta_logit": float((lg_noise - base_logits).abs().max().item()),
                                "tokens_changed": sum(1 for a, c in zip(
                                    base_toks, gen_tokens(s, pv=pv_noise)[0]) if a != c)},
           "images_zeroed": {"max_abs_delta_logit": float((lg_zero - base_logits).abs().max().item()),
                             "tokens_changed": sum(1 for a, c in zip(
                                 base_toks, gen_tokens(s, pv=pv_zero)[0]) if a != c)}},
          informational=True)
    check("S7c_reference_top1_confidence", True,
          {"per_dim_top1_prob": [round(p, 4) for p in base_probs],
           "min": min(base_probs), "mean": float(np.mean(base_probs))}, informational=True)

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

    kv = gen_tokens(samples[0])[0]; full = full_seq_tokens(samples[0])
    ok &= check("S5_kv_equals_full_sequence_argmax", kv == full,
                {"matching_dims": sum(1 for a, c in zip(kv, full) if a == c), "kv": kv, "full": full})
    ok &= check("S6a_generation_deterministic", gen_tokens(samples[0])[0] == kv)
    with torch.no_grad():
        o = forward(samples[0])
    ok &= check("S6b_logits_finite", bool(torch.isfinite(o.logits).all().item()),
                {"logits_shape": list(o.logits.shape)})
    ok &= check("S6c_shape_contract", list(samples[0]["pr"].shape) == [1, dim] and pp.proprio_dim == dim,
                {"proprio": list(samples[0]["pr"].shape), "projector_proprio_dim": pp.proprio_dim})

    summary = {"history_mode": args.history_mode, "result": "PASS" if ok else "FAIL",
               "checks": RESULTS, "frames": probe_frames, "episode": ep,
               "projector_params_28d": n28, "projector_params_14d": n14,
               "projector_param_delta": n28 - n14, "gpu_free_gb_at_start": gpu_free_gb,
               "grad_current_slot": cur_g, "grad_history_slot": hist_g,
               "max_abs_delta_logit_history_slot": max_hist,
               "max_abs_delta_logit_current_slot": max_cur,
               "torch": torch.__version__}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)
    print("SMOKE %s -> %s" % (summary["result"], args.out))
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
