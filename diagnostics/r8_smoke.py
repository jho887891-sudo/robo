#!/usr/bin/env python3
"""R8 GPU smoke test: real OpenVLA-7B forward/backward with the 42-dim proprio head.

This is a SEPARATE harness from the frozen R6 smoke script (which stays untouched).
It is gated: it refuses to run unless a frozen normalization contract is supplied and
enough VRAM is free, and it never trains (no optimiser steps, no 3000/6000-step run).

Modes
  --plan            print preconditions and the exact checks (CPU, no model)
  --check-contract  validate contract + adapter + codec math on real HDF5 (CPU)
  (default)         run the GPU smoke on the real 7B weights

Checks
  1 real OpenVLA-7B weights load (pristine dynamic module verified)
  2 42D ProprioProjector forward produces finite (1, llm_dim) features
  3 gradients reach the projector (fc1/fc2) and the LoRA adapters
  4 each of the three feature slots (S, V, Acc) independently moves the action logits
  5 perturbation of the 42D input changes decoded tokens (informational)
  6 KV-cache generation == full-sequence argmax (14/14 within tolerance)
  7 token decode -> residual -> physical command reconstruction is exact and bounded,
    including the DEV gripper policy
  8 no NaN/Inf, no shape violation, no future-frame leak
  9 peak VRAM, wall time, environment and exit code recorded
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for sub in ("training", "diagnostics", "code/xpolicylab/policy/OpenVLA_Pure"):
    sys.path.insert(0, os.path.join(REPO, sub))

RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
os.environ.setdefault("PUSHT_MODEL", RT + "/model_cache/openvla-7b-pristine")
os.environ.setdefault("PUSHT_HDF5", "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
os.environ.setdefault("PUSHT_SPLIT", RT + "/split80.json")
os.environ.setdefault("PUSHT_BOUNDS", RT + "/push_T_bounds.npz")
os.environ.setdefault("PUSHT_OFT", "/home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_OFT/openvla_oft")

import r6_data as D                      # noqa: E402
import action_codec as AC                # noqa: E402
import r8_contract as C                  # noqa: E402
import r8_vla_adapter as A               # noqa: E402
from r8_normalization import SIGNED_MOTION, NormalizationContract, LEGACY_POSITION  # noqa: E402

W0, BINS = 31744, 256
RESULTS = []


def check(name, ok, detail=None):
    RESULTS.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
    print("[%s] %-46s %s" % ("PASS" if ok else "FAIL", name,
                             json.dumps(detail) if detail is not None else ""), flush=True)
    return bool(ok)


# ------------------------------------------------------------------ preconditions
def load_frozen_contract(path, allow_legacy=False):
    meta = json.load(open(path))
    mode = meta.get("normalization_mode")
    if mode is None:
        # accept either a bare NormalizationContract.to_metadata() or a full adapter meta
        raise SystemExit("contract file lacks normalization_mode: %s" % path)
    if mode != SIGNED_MOTION and not allow_legacy:
        raise SystemExit("frozen contract is %r; R8 requires %r (use --allow-legacy only to "
                         "reproduce the known-lossy path)" % (mode, SIGNED_MOTION))
    contract = NormalizationContract.from_metadata(meta)
    return contract, meta


def real_frames(n=3):
    split = D.load_split()
    ep_name = split["val"][0]
    ep = D.load_episode(ep_name)
    frames = []
    for t in range(30, 30 + 40 * n, 40):
        if t + 1 >= len(ep["state"]):
            break
        frames.append({"episode": ep_name, "t": int(t),
                       "S": ep["state"][t], "Sp": ep["state"][t - 1], "Spp": ep["state"][t - 2],
                       "A": ep["action"][t]})
    return frames


def residual_quantiles(frames):
    split = D.load_split()
    S, A_, V = [], [], []
    for name in split["train"][:20]:
        ep = D.load_episode(name)
        t = D.valid_frames(len(ep["state"]))
        t = t[t >= 2]
        S.append(ep["state"][t]); A_.append(ep["action"][t])
        V.append(ep["state"][t] - ep["state"][t - 1])
    return C.velocity_residual_quantiles(np.concatenate(S), np.concatenate(A_),
                                         np.concatenate(V))


# ------------------------------------------------------------------- checks (CPU)

def frozen_quantiles(meta):
    """Fail closed: GPU/CPU decoder uses frozen 64-TRAIN-only quantiles."""
    provenance = meta.get("frozen_from", {})
    if (provenance.get("n_train_episodes_for_quantiles") != 64 or
            provenance.get("n_dev_episodes_excluded") != 16):
        raise ValueError("R8 frozen target quantiles require 64 TRAIN / 16 DEV; regenerate contract")
    a = np.asarray(meta.get("q01", []), dtype=np.float32)
    b = np.asarray(meta.get("q99", []), dtype=np.float32)
    if (a.shape != (14,) or b.shape != (14,) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not (b > a).all()):
        raise ValueError("frozen q01/q99 must be finite 14D ranges with q99 > q01")
    return a, b


def run_contract_checks(contract, out, q01, q99):
    nb = np.load(os.environ["PUSHT_BOUNDS"])
    pmin, pmax = nb["pmin"].astype(np.float32), nb["pmax"].astype(np.float32)
    frames = real_frames(2)
    ok = True
    feats = []
    for f in frames:
        p = A.build_proprio(f["S"], f["Sp"], f["Spp"], contract)
        feats.append(p)
        ok &= check("feature_shape_frame_%d" % f["t"], p.shape == (42,) and np.isfinite(p).all(),
                    {"shape": list(p.shape), "min": float(p.min()), "max": float(p.max())})
    # zero motion -> zero in the motion slots (signed contract)
    zero = np.zeros(14, np.float32)
    z = contract.transform(np.zeros(14, np.float32), zero, zero)
    ok &= check("zero_motion_is_zero_in_motion_slots",
                bool(np.allclose(z[14:], 0.0, atol=1e-7)), {"motion_slots_max_abs": float(np.abs(z[14:]).max())})
    # buffer path == offline path
    buf = A.ProprioHistory42(contract)
    ep = D.load_episode(D.load_split()["val"][0])
    same = True
    for t in range(40, 46):
        online, info = buf.observe(ep["state"][t])
        offline = (contract.transform(ep["state"][t], zero, zero) if t == 40 else
                   A.build_proprio(ep["state"][t], ep["state"][t - 1], None, contract) if t == 41 else
                   A.build_proprio(ep["state"][t], ep["state"][t - 1], ep["state"][t - 2], contract))
        same &= bool(np.allclose(online, offline, atol=1e-5))
    ok &= check("deploy_buffer_matches_offline_features", same)
    # target / command / decode round trip
    f = frames[0]
    v = f["S"] - f["Sp"]
    R = A.training_target(f["S"], f["A"], v)
    cmd = A.command_from_residual(f["S"], v, R)
    ok &= check("command_reconstruction_exact", bool(np.allclose(cmd, f["A"], atol=1e-6)),
                {"max_abs": float(np.abs(cmd - f["A"]).max())})
    idx = AC.encode_indices(AC.normalize(R, q01, q99), mode=AC.MODE_FIXED)
    cmd2, Rq, excluded = A.decode_tokens_to_command(idx, q01, q99, f["S"], v, AC,
                                                    cv_command=f["S"] + v)
    step = (np.asarray(q99) - np.asarray(q01)) / 255.0
    ok &= check("decoded_residual_within_one_step", bool(np.all(np.abs(Rq - R) <= step + 1e-6)),
                {"max_abs": float(np.abs(Rq - R).max()), "step_mean": float(step.mean())})
    ok &= check("gripper_policy_applied", excluded == [6, 13], {"excluded_dims": excluded})
    cmd_clipped = A.clip_command(cmd2, nb["amin"].astype(np.float32), nb["amax"].astype(np.float32))
    ok &= check("command_clipped_to_joint_bounds", bool(np.isfinite(cmd_clipped).all()))
    # metadata round trip and fail-closed
    meta = A.metadata(contract, q01, q99)
    A.save_metadata(os.path.join(out, "checkpoint_meta_draft.json"), meta)
    resolved = A.resolve_metadata(meta, projector_dim=42)
    ok &= check("metadata_roundtrip", resolved["contract"].mode == contract.mode)
    bad = dict(meta); bad["proprio_dim"] = 28
    try:
        A.resolve_metadata(bad, projector_dim=42)
        ok &= check("metadata_fail_closed_on_wrong_dim", False, {})
    except ValueError:
        ok &= check("metadata_fail_closed_on_wrong_dim", True, {"raised": "ValueError"})
    print("\nCPU contract checks:", "PASS" if ok else "FAIL")
    return ok, {"q01": q01.tolist(), "q99": q99.tolist(), "frames": [f["t"] for f in frames]}


# ------------------------------------------------------------------- checks (GPU)
def run_gpu_smoke(args, contract, out, q01, q99):
    """Memory-ordered smoke: generation checks first, the backward pass last.

    The GPU is shared with other jobs, so the phase order matters: cached decoding needs
    weights + KV cache only, while the backward pass needs activation memory. Doing the
    backward last (and emptying the cache between phases) keeps the peak affordable.
    """
    import gc
    import torch
    import h5py
    import train_pushT_pure2 as T
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder

    t_start = time.time()
    free, total = torch.cuda.mem_get_info()
    free_gb = free / 1e9
    ok = check("precondition_free_vram", free_gb >= args.min_free_gb,
               {"free_gb": round(free_gb, 2), "total_gb": round(total / 1e9, 2),
                "required_gb": args.min_free_gb})
    if not ok:
        print("[STOP] insufficient free VRAM; nothing was loaded", flush=True)
        return False, {"stopped": "insufficient_vram", "free_gb": free_gb}

    proc, vla, _ = T.build(3, False)      # real training build(); pristine module verified
    import inspect
    core = vla.base_model.model if hasattr(vla, "base_model") else vla
    llm_dim = int(vla.llm_dim)
    use_ckpt = free_gb < 30.0
    if use_ckpt:
        vla.gradient_checkpointing_enable()
        try:
            vla.enable_input_require_grads()
        except Exception:
            pass
    print("[smoke] gradient checkpointing =", use_ckpt, flush=True)
    proj = A.build_projector(llm_dim, device="cuda")
    n_params = sum(p.numel() for p in proj.parameters())
    ok &= check("real_weights_loaded", True,
                {"llm_dim": llm_dim, "projector_params": n_params,
                 "pristine_modeling": os.path.basename(inspect.getfile(type(core)))})
    ok &= check("projector_width_is_42", int(proj.proprio_dim) == 42,
                {"proprio_dim": int(proj.proprio_dim)})

    f = real_frames(1)[0]
    feats = A.build_proprio(f["S"], f["Sp"], f["Spp"], contract)
    v = f["S"] - f["Sp"]
    R = A.training_target(f["S"], f["A"], v)
    gt_idx = AC.encode_indices(AC.normalize(R, q01, q99), mode=AC.MODE_FIXED)
    b = np.load(os.environ["PUSHT_BOUNDS"])
    with h5py.File(os.path.join(os.environ["PUSHT_HDF5"], f["episode"]), "r") as h:
        pv, _unused = T.obs(h, f["t"], proc, 3, False, b["pmin"], b["pmax"])
    pr = torch.from_numpy(feats).unsqueeze(0).to("cuda")

    bld = PurePromptBuilder("openvla")
    bld.add_turn("human", "What action should the robot take to " + T.INSTR + "?")
    pids = proc.tokenizer(bld.get_prompt(), return_tensors="pt").input_ids.to("cuda")
    if int(pids[0, -1].item()) != 29871:
        pids = torch.cat([pids, torch.tensor([[29871]], dtype=pids.dtype, device="cuda")], dim=1)

    # ---------------------------------------------------- phase 1: inference only
    def first_logits(pr_in):
        vla.eval(); proj.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=pids, attention_mask=torch.ones_like(pids), pixel_values=pv,
                    labels=pids.clone(), proprio=pr_in, proprio_projector=proj, use_cache=True)
        lg = o.logits[0, -1][W0:W0 + BINS].float()
        del o
        return lg

    def gen(pr_in, use_cache=True):
        vla.eval(); proj.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=pids, attention_mask=torch.ones_like(pids), pixel_values=pv,
                    labels=pids.clone(), proprio=pr_in, proprio_projector=proj, use_cache=use_cache)
        if use_cache:
            past, lg = o.past_key_values, o.logits[0, -1]
            if past is None:
                raise RuntimeError("model returned no KV cache; gradient checkpointing must be "
                                   "disabled in eval mode (use_cache=False was forced)")
            toks = []
            for k in range(14):
                toks.append(int(torch.argmax(lg[W0:W0 + BINS]).item()))
                if k == 13:
                    break
                nxt = torch.tensor([[W0 + toks[-1]]], dtype=torch.long, device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o2 = vla(input_ids=nxt, past_key_values=past, labels=None, use_cache=True)
                past, lg = o2.past_key_values, o2.logits[0, -1]
                del o2
            del o
            return toks
        toks = []
        for k in range(14):
            inp = pids if not toks else torch.cat(
                [pids, torch.tensor([[W0 + t for t in toks]], dtype=pids.dtype, device="cuda")], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                        labels=inp.clone(), proprio=pr_in, proprio_projector=proj)
            toks.append(int(torch.argmax(o.logits[0, -1][W0:W0 + BINS]).item()))
            del o
        return toks

    base = first_logits(pr)
    slot_sens = {}
    for name, sl in (("state", slice(0, 14)), ("velocity", slice(14, 28)), ("acceleration", slice(28, 42))):
        pr2 = pr.clone()
        pr2[0, sl] = torch.clamp(pr2[0, sl] + 0.5, -1.0, 1.0)
        d = float((first_logits(pr2) - base).abs().max().item())
        slot_sens[name] = d
        ok &= check("slot_%s_moves_action_logits" % name, d > args.logit_gate, {"max_abs_dlogit": d})
        del pr2
    del base
    gc.collect(); torch.cuda.empty_cache()

    kv = gen(pr, True)
    full = gen(pr, False)
    ok &= check("kv_cache_equals_full_sequence", kv == full,
                {"matching_dims": sum(1 for a, c in zip(kv, full) if a == c), "kv": kv, "full": full})
    pr_pert = pr.clone()
    pr_pert[0, :14] = torch.clamp(pr[0, :14] + 0.5, -1, 1)
    perturbed = gen(pr_pert, True)
    check("input_perturbation_changes_tokens", perturbed != kv,
          {"dims_changed": sum(1 for a, c in zip(kv, perturbed) if a != c)})
    del pr_pert
    gc.collect(); torch.cuda.empty_cache()

    cmd, Rq, excluded = A.decode_tokens_to_command(np.asarray(kv), q01, q99, f["S"], v, AC,
                                                  cv_command=f["S"] + v)
    step = (np.asarray(q99) - np.asarray(q01)) / 255.0
    ok &= check("decoded_tokens_to_physical_command",
                bool(np.isfinite(cmd).all()) and excluded == [6, 13],
                {"excluded": excluded, "cmd_finite": bool(np.isfinite(cmd).all()),
                 "residual_within_step": bool(np.all(np.abs(Rq - R) <= step + 1e-6))})
    ok &= check("no_nan_inf_in_features", bool(np.isfinite(feats).all()) and feats.shape == (42,))
    ep = D.load_episode(f["episode"])
    S2 = ep["state"].copy(); S2[f["t"] + 1:] += 5.0
    A2 = ep["action"].copy(); A2[f["t"] + 1:] -= 5.0
    feats2 = A.build_proprio(S2[f["t"]], S2[f["t"] - 1], S2[f["t"] - 2], contract)
    tgt2 = A.training_target(S2[f["t"]], A2[f["t"]], S2[f["t"]] - S2[f["t"] - 1])
    ok &= check("no_future_frame_leak",
                bool(np.array_equal(feats, feats2)) and bool(np.array_equal(R, tgt2)))

    # ---------------------------------------------------- phase 2: backward last
    gc.collect(); torch.cuda.empty_cache()
    vla.train(); proj.train()
    aid = torch.tensor([[W0 + 1 + int(x) for x in gt_idx]], dtype=torch.long, device="cuda")
    stop = torch.full((1, 1), 2, dtype=torch.long, device="cuda")
    inp = torch.cat([pids, aid, stop], dim=1)
    lab = torch.cat([torch.full_like(pids, -100), aid, stop], dim=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_fwd = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                      labels=lab, proprio=pr, proprio_projector=proj)
    ok &= check("forward_loss_finite", bool(torch.isfinite(out_fwd.loss).item()),
                {"loss": float(out_fwd.loss.item())})
    vla.zero_grad(set_to_none=True); proj.zero_grad(set_to_none=True)
    out_fwd.loss.backward()
    pg = {n: (float(p.grad.norm()) if p.grad is not None else None) for n, p in proj.named_parameters()}
    lora = [float(p.grad.norm()) for n, p in vla.named_parameters()
            if p.requires_grad and p.grad is not None and "lora" in n]
    ok &= check("gradients_reach_projector", all(x not in (None, 0.0) for x in pg.values()), pg)
    ok &= check("gradients_reach_lora", bool(lora) and max(lora) > 0,
                {"n_lora": len(lora), "max_norm": max(lora) if lora else None})
    vla.zero_grad(set_to_none=True); proj.zero_grad(set_to_none=True)
    del out_fwd
    gc.collect(); torch.cuda.empty_cache()

    peak = float(torch.cuda.max_memory_allocated() / 1e9)
    wall = round(time.time() - t_start, 1)
    ok &= check("resources_recorded", True, {"peak_vram_gb": round(peak, 2),
                                             "wall_s": wall, "free_at_start_gb": round(free_gb, 2),
                                             "gradient_checkpointing": use_ckpt})
    return ok, {"peak_vram_gb": peak, "wall_s": wall, "slot_logit_sensitivity": slot_sens,
                "kv_tokens": kv, "full_tokens": full, "q01": q01.tolist(), "q99": q99.tolist(),
                "gradient_checkpointing": use_ckpt}


# --------------------------------------------------------- R10 KV probe helpers
NUMERIC_FLOOR_DLOGIT = 0.05      # documented heuristic, NOT a proof of bf16 causation
KV_SIDECAR_NAME = "r10_kv_logits.npz"


def kv_preflight(vla=None, free_bytes=None, min_free_gib=24.0):
    """Refuse to run the KV diagnostic unless the cache path is usable.

    A KV-cache diagnostic needs use_cache=True. Gradient checkpointing forces
    use_cache=False, which is what broke the first R8 attempt, so it is DISABLED here
    explicitly (no silent flag flips) and recorded in the summary.
    """
    info = {"min_free_gib": float(min_free_gib)}
    free_gib = None if free_bytes is None else float(free_bytes) / float(2 ** 30)
    info["free_gib"] = free_gib
    if free_gib is not None and free_gib < float(min_free_gib):
        raise RuntimeError("insufficient free VRAM for KV diagnose: %.2f GiB < %.2f GiB"
                           % (free_gib, float(min_free_gib)))
    if vla is not None:
        gc_before = bool(getattr(vla, "is_gradient_checkpointing", False))
        if gc_before:
            vla.gradient_checkpointing_disable()
        info["gradient_checkpointing_before"] = gc_before
        info["gradient_checkpointing_after"] = bool(getattr(vla, "is_gradient_checkpointing", False))
        try:
            vla.config.use_cache = True
        except Exception:  # noqa: BLE001 - config shape differs across wrappers
            pass
        info["use_cache_config"] = bool(getattr(getattr(vla, "config", None), "use_cache", True))
        if info["gradient_checkpointing_after"]:
            raise RuntimeError("gradient checkpointing must be disabled for a KV-cache diagnostic")
    return info


def save_kv_logits_sidecar(out_dir, logits_kv, logits_full, tokens_kv, tokens_full):
    """Persist the full 256-bin logits for every step (R10 requirement)."""
    kv = np.asarray(logits_kv, dtype=np.float32)
    fl = np.asarray(logits_full, dtype=np.float32)
    if kv.shape != fl.shape or kv.ndim != 2 or kv.shape[1] != BINS:
        raise ValueError("logits must be (steps, %d); got %s vs %s" % (BINS, kv.shape, fl.shape))
    if kv.shape[0] != len(tokens_kv) or kv.shape[0] != len(tokens_full):
        raise ValueError("token count does not match logits rows")
    if not (np.isfinite(kv).all() and np.isfinite(fl).all()):
        raise ValueError("non-finite logits")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, KV_SIDECAR_NAME)
    np.savez_compressed(path, logits_kv=kv, logits_full=fl,
                        token_kv=np.asarray(tokens_kv, dtype=np.int64),
                        token_full=np.asarray(tokens_full, dtype=np.int64),
                        step=np.arange(kv.shape[0], dtype=np.int64),
                        column_index=np.arange(BINS, dtype=np.int64))
    with open(path, "rb") as fh:
        data = fh.read()
    return {"path": os.path.abspath(path), "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "shape": list(kv.shape), "dtype": str(kv.dtype),
            "columns": "action-token window %d..%d (256 bins)" % (W0, W0 + BINS - 1)}

def kv_diagnose(args, contract, out):
    """R10 same-prefix KV diagnostic: teacher-forced full-path history, full logits archived.

    Differences from the R8 version (all evidence/plumbing, no model-math change):
      * per-step 256-bin logits for BOTH branches go to an .npz sidecar with sha256
      * cache preflight (gradient checkpointing disabled, use_cache=True, cache must exist)
      * free-VRAM gate in GiB (2^30), recorded before/after
      * per-step cache/prefix lengths and the step0 (prefix-only) boundary are recorded
    """
    import gc
    import torch
    import h5py
    import train_pushT_pure2 as T
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder

    free, total = torch.cuda.mem_get_info()
    pre = kv_preflight(None, free, args.min_free_gb)
    proc, vla, _ = T.build(3, False)
    llm_dim = int(vla.llm_dim)
    pre.update(kv_preflight(vla, free, args.min_free_gb))
    proj = A.build_projector(llm_dim, device="cuda")
    f = real_frames(1)[0]
    feats = A.build_proprio(f["S"], f["Sp"], f["Spp"], contract)
    b = np.load(os.environ["PUSHT_BOUNDS"])
    with h5py.File(os.path.join(os.environ["PUSHT_HDF5"], f["episode"]), "r") as h:
        pv, _ = T.obs(h, f["t"], proc, 3, False, b["pmin"], b["pmax"])
    pr = torch.from_numpy(feats).unsqueeze(0).to("cuda")
    bld = PurePromptBuilder("openvla")
    bld.add_turn("human", "What action should the robot take to " + T.INSTR + "?")
    pids = proc.tokenizer(bld.get_prompt(), return_tensors="pt").input_ids.to("cuda")
    if int(pids[0, -1].item()) != 29871:
        pids = torch.cat([pids, torch.tensor([[29871]], dtype=pids.dtype, device="cuda")], dim=1)

    vla.eval(); proj.eval()
    steps = []
    lg_kv_rows, lg_full_rows = [], []
    toks_kv, toks_full = [], []
    cached = None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o = vla(input_ids=pids, attention_mask=torch.ones_like(pids), pixel_values=pv,
                labels=pids.clone(), proprio=pr, proprio_projector=proj, use_cache=True)
    cached = o.past_key_values
    if cached is None:
        raise RuntimeError("no KV cache returned (use_cache=True): refusing to fake a comparison")
    cache_len = int(cached[0][0].shape[-2])
    prefix_len = int(pids.shape[1])
    lg_kv = o.logits[0, -1][W0:W0 + BINS].float()
    del o
    for k in range(14):
        inp_full = pids if k == 0 else torch.cat(
            [pids, torch.tensor([[W0 + t for t in toks_full]], dtype=pids.dtype, device="cuda")], dim=1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o_full = vla(input_ids=inp_full, attention_mask=torch.ones_like(inp_full),
                         pixel_values=pv, labels=inp_full.clone(), proprio=pr,
                         proprio_projector=proj)
        lg_full = o_full.logits[0, -1][W0:W0 + BINS].float()
        del o_full
        tk, tf = int(torch.argmax(lg_kv).item()), int(torch.argmax(lg_full).item())
        toks_kv.append(tk); toks_full.append(tf)
        lg_kv_rows.append(lg_kv.detach().float().cpu().numpy().copy())
        lg_full_rows.append(lg_full.detach().float().cpu().numpy().copy())
        srt = torch.sort(lg_full, descending=True).values
        steps.append({"step": k, "token_kv": tk, "token_full": tf, "match": tk == tf,
                      "max_abs_dlogit": float((lg_kv - lg_full).abs().max().item()),
                      "mean_abs_dlogit": float((lg_kv - lg_full).abs().mean().item()),
                      "top1_minus_top2_full": float((srt[0] - srt[1]).item()),
                      "prefix_len_full": int(inp_full.shape[1]),
                      "cache_len_kv": int(cache_len),
                      "step0_prefix_only": bool(k == 0)})
        if k == 13:
            break
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Teacher-force the SAME full-path token into the cached branch.
            o_next = vla(input_ids=torch.tensor([[W0 + tf]], dtype=torch.long, device="cuda"),
                         past_key_values=cached, labels=None, use_cache=True)
        cached = o_next.past_key_values
        if cached is None:
            raise RuntimeError("cache lost after step %d; refusing to continue" % k)
        cache_len = int(cached[0][0].shape[-2])
        lg_kv = o_next.logits[0, -1][W0:W0 + BINS].float()
        del o_next
        gc.collect(); torch.cuda.empty_cache()
    sidecar = save_kv_logits_sidecar(out, np.vstack(lg_kv_rows), np.vstack(lg_full_rows),
                                     toks_kv, toks_full)
    mism = [s for s in steps if not s["match"]]
    worst = max([s["max_abs_dlogit"] for s in mism], default=0.0)
    if not mism:
        cls = "equivalent"
    elif worst <= NUMERIC_FLOOR_DLOGIT:
        cls = "numeric_plausible"
    else:
        cls = "structural_suspect"
    summary = {"DIAG_EXECUTED": True, "same_prefix_teacher_forced": True,
               "reference_prefix": "full_sequence_greedy",
               "step0_semantics": "step0 uses the prefix only for BOTH branches",
               "steps": steps, "n_match": sum(1 for s in steps if s["match"]),
               "n_total": len(steps),
               "first_mismatch_step": next((s["step"] for s in steps if not s["match"]), None),
               "median_margin": float(np.median([s["top1_minus_top2_full"] for s in steps])),
               "median_max_dlogit": float(np.median([s["max_abs_dlogit"] for s in steps])),
               "max_abs_dlogit_at_mismatch": float(worst),
               "numeric_floor_dlogit": float(NUMERIC_FLOOR_DLOGIT),
               "mismatch_class": cls,
               "KV_EQUIVALENT": bool(not mism),
               "verdict": ("KV_EQUIVALENT: identical tokens under identical prefixes" if not mism
                           else "OPEN_UNRESOLVED: mismatch recorded, cause not established"),
               "logits_sidecar": sidecar,
               "precision": {"model_dtype": str(next(vla.parameters()).dtype),
                             "autocast": "bfloat16"},
               "preflight": pre, "free_bytes_before": int(free), "free_bytes_total": int(total),
               "model_reference": os.environ.get("PUSHT_MODEL", ""),
               "probe_sha256": __import__("hashlib").sha256(open(__file__, "rb").read()).hexdigest(),
               "contract_path": os.path.abspath(args.norm_contract),
               "out": out}
    free_after, _t = torch.cuda.mem_get_info()
    summary["free_bytes_after"] = int(free_after)
    with open(os.path.join(out, "r8_kv_diagnose.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("KV/full match %d/%d | first mismatch step %s | median margin %.6f | median "
          "max|dlogit| %.6f | class=%s | sidecar=%s (%d bytes)"
          % (summary["n_match"], summary["n_total"], summary["first_mismatch_step"],
             summary["median_margin"], summary["median_max_dlogit"], cls,
             sidecar["path"], sidecar["bytes"]), flush=True)
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm-contract", required=False, default="",
                    help="frozen normalization contract JSON (from workstream A)")
    ap.add_argument("--allow-legacy", action="store_true")
    ap.add_argument("--out", required=False, default="")
    ap.add_argument("--min-free-gb", type=float, default=24.0)
    ap.add_argument("--logit-gate", type=float, default=0.05)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--check-contract", action="store_true")
    ap.add_argument("--kv-diagnose", action="store_true")
    args = ap.parse_args()
    if args.plan:
        print(json.dumps({"preconditions": {
            "frozen contract file": "--norm-contract <path with normalization_mode>",
            "min free VRAM GB": args.min_free_gb,
            "no training": "forward/backward only, no optimizer step",
            "frozen code": "R6 smoke script is NOT modified; this is a separate harness"},
            "checks": [r["check"] for r in RESULTS] or "see module docstring"}, indent=1))
        return 0
    if not args.norm_contract:
        raise SystemExit("--norm-contract is required (freeze it in workstream A first)")
    out_dir = args.out or os.path.join("/home/dgut/pusht_runtime/r8_smoke", time.strftime("%Y%m%dT%H%M%SZ"))
    os.makedirs(out_dir, exist_ok=True)
    contract, meta = load_frozen_contract(args.norm_contract, args.allow_legacy)
    q01, q99 = frozen_quantiles(meta)
    print("[contract]", contract, "| source:", args.norm_contract, flush=True)
    if args.kv_diagnose:
        summary = kv_diagnose(args, contract, out_dir)
        return 0 if summary["n_match"] == summary["n_total"] else 6
    if args.check_contract:
        ok, extra = run_contract_checks(contract, out_dir, q01, q99)
        json.dump({"mode": "cpu_contract", "pass": ok, "contract": meta, "checks": RESULTS,
                   "extra": extra, "out": out_dir},
                  open(os.path.join(out_dir, "r8_smoke_cpu.json"), "w"), indent=1)
        return 0 if ok else 4
    ok, extra = run_gpu_smoke(args, contract, out_dir, q01, q99)
    summary = {"mode": "gpu", "pass": ok, "checks": RESULTS, "extra": extra,
               "contract": meta, "torch": __import__("torch").__version__,
               "cuda": __import__("torch").version.cuda, "out": out_dir}
    json.dump(summary, open(os.path.join(out_dir, "r8_smoke_gpu.json"), "w"), indent=1)
    n_pass = sum(1 for r in RESULTS if r["result"] == "PASS")
    print("\n%d/%d checks passed -> %s" % (n_pass, len(RESULTS), "PASS" if ok else "FAIL"))
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
