#!/usr/bin/env python3
"""R11 42D/R OpenVLA-7B entry: ZERO-UPDATE smoke (R13), bounded overfit (R14) and staged formal
training (R15) on a single A6000, with a machine-readable GPU budget ledger.

Guarantees (all fail-closed, all covered by tests/test_r11_p0_*.py):
  * the prismatic platform constants are forced to ARX *before* the model modules import, and
    r42_adapter.assert_runtime_contract() aborts if ACTION_DIM != 14 / window != 31743..31999,
  * 42D [S,V,Acc] from observed history only, R=(A-S)-V with A[t]==S[t+1],
  * one 14-token action block per sample, labels = -100*prompt + 14 tokens + STOP(2),
  * no prismatic 7D/8D/chunk default and no 14D projector is ever used,
  * --smoke does forward+backward+14-token inference+checkpoint roundtrip with optimizer_steps==0,
  * every GPU second is booked into the ledger bucket, and a negative balance aborts the job.

Never overwrites an existing run directory, never resumes a legacy adapter, never trains on DEV.
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("training", "diagnostics", "code/xpolicylab/policy/OpenVLA_Pure",
             "code/xpolicylab/policy/OpenVLA_R42"):
    _p = os.path.join(REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r42_adapter as R            # noqa: E402
import r42_train_contract as TC    # noqa: E402

RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
MODEL = os.environ.get("PUSHT_MODEL", RT + "/model_cache/openvla-7b-pristine")
HDF5_DIR = os.environ.get("PUSHT_HDF5",
                          "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
OFT = os.environ.get("PUSHT_OFT", RT + "/openvla_oft_src")
BOUNDS = os.environ.get("PUSHT_BOUNDS", os.path.join(REPO, "data", "push_T_bounds.npz"))
SPLIT = os.environ.get("PUSHT_SPLIT", os.path.join(REPO, "data", "split80.json"))
CONTRACT = os.environ.get("PUSHT_CONTRACT",
                          os.path.join(REPO, "evidence", "r9_gate_20261020",
                                       "frozen_contract_64train.json"))
LEDGER = os.environ.get("R11_LEDGER", os.path.join(REPO, "mission", "R11_R16_BUDGET_LEDGER.json"))
for _p in (OFT, os.path.join(OFT, "vla-scripts")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
INSTR = "Push the T-shaped block to align it precisely with the gray T-shaped pad."


def log(msg):
    print("[%s] %s" % (datetime.now(timezone.utc).strftime("%H:%M:%S"), msg), flush=True)


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------- ledger
def load_ledger(path=None):
    with open(path or LEDGER) as fh:
        return json.load(fh)


def ledger_budget(bucket, requested):
    led = load_ledger()
    b = led["buckets"][bucket]
    left_bucket = float(b["cap_s"]) - float(b.get("seconds_used", 0.0))
    left_total = float(led["aggregate"]["cap_s"]) - float(
        led["aggregate"].get("new_gpu_seconds_used", 0.0))
    return max(0.0, min(float(requested), left_bucket, left_total))


def ledger_book(bucket, seconds, event, note="", extra=None):
    """Only ever ADD to the ledger; a negative remaining balance aborts the job."""
    led = load_ledger()
    b = led["buckets"][bucket]
    b["seconds_used"] = float(b.get("seconds_used", 0.0)) + float(seconds)
    agg = led["aggregate"]
    agg["new_gpu_seconds_used"] = float(agg.get("new_gpu_seconds_used", 0.0)) + float(seconds)
    remaining_total = float(agg["cap_s"]) - agg["new_gpu_seconds_used"]
    remaining_bucket = float(b["cap_s"]) - b["seconds_used"]
    led.setdefault("events", []).append({"utc": utcnow(), "bucket": bucket, "event": event,
                                         "seconds": round(float(seconds), 3), "note": note,
                                         "extra": extra or {}})
    if len(led["events"]) > 4000:
        del led["events"][:-4000]
    with open(LEDGER, "w") as fh:
        json.dump(led, fh, indent=1)
    log("ledger %s %+.1fs -> bucket %s used %.1fs (left %.1fs), total left %.1fs"
        % (event, seconds, bucket, b["seconds_used"], remaining_bucket, remaining_total))
    if remaining_total < -1e-6 or remaining_bucket < -1e-6:
        raise SystemExit("BUDGET_EXCEEDED: refusing to continue (%s)" % bucket)
    return {"remaining_total": remaining_total, "remaining_bucket": remaining_bucket}


# -------------------------------------------------------------------- environment guard
def force_arx_platform():
    """prismatic picks ACTION_DIM from sys.argv; the R42 path needs the ARX 14D table."""
    joined = " ".join(sys.argv).lower()
    if "arx" not in joined and "goai" not in joined:
        sys.argv = list(sys.argv) + ["arx_x5_push_T"]
    return " ".join(sys.argv)


def gpu_snapshot():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()
    except Exception as exc:  # diagnostics only
        return ["nvidia-smi failed: %r" % (exc,)]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------------------------------------- data
def load_splits():
    import r9_trainer_contract as RC
    contract, meta = RC.load_frozen_contract(CONTRACT)
    splits = RC.select_splits(SPLIT)
    if set(splits["train"]) & set(splits["dev"]):
        raise SystemExit("TRAIN/DEV overlap")
    return contract, meta, splits


def read_episode(name):
    import h5py
    with h5py.File(os.path.join(HDF5_DIR, name), "r") as h:
        return TC.read14(h, "state"), TC.read14(h, "action")


def sample_row(state, action, t, contract):
    """Fail-closed row construction (P0-5), same assertions the CPU tests exercise."""
    row = TC.build_training_row(state, action, t, contract)
    TC.assert_no_future_leakage(state, action, t, contract)
    return row


# ------------------------------------------------------------------------------- model
def build_model(cams=3):
    import torch
    from transformers import (AutoModelForVision2Seq, AutoProcessor, AutoConfig,
                             AutoImageProcessor)
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    runtime = R.assert_runtime_contract()
    log("runtime contract: %s" % json.dumps(runtime, sort_keys=True))

    from peft import LoraConfig, get_peft_model
    from pure_autoreg_mode import enable_pure_autoreg
    import r8_vla_adapter as A
    import inspect

    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                low_cpu_mem_usage=True,
                                                trust_remote_code=True).to("cuda")
    reference = os.path.join(REPO, "code", "openvla_dynamic_modules", "modeling_prismatic.py")
    active = inspect.getfile(type(vla))
    if not os.path.isfile(reference):
        raise SystemExit("pristine reference source missing: %s" % reference)
    if sha256_file(active) != sha256_file(reference):
        raise SystemExit("R11 requires the pristine dynamic model source: %s" % active)
    log("pristine modeling sha256=%s" % sha256_file(active))
    vla.vision_backbone.set_num_images_in_input(cams)
    enable_pure_autoreg(vla)
    cfg = LoraConfig(r=32, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear",
                     init_lora_weights="gaussian")
    vla = get_peft_model(vla, cfg)
    proj = A.build_projector(int(vla.llm_dim), device="cuda")
    R.assert_projector_width(proj.fc1.weight.shape, expected=42)
    n_lora = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    n_proj = sum(p.numel() for p in proj.parameters())
    log("LoRA trainable=%.2fM | 42D projector=%.3fM | llm_dim=%d | modeling_sha256=%s"
        % (n_lora / 1e6, n_proj / 1e6, int(vla.llm_dim), sha256_file(active)))
    return proc, vla, proj, sha256_file(active)


def make_prompt(proc):
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    b = PurePromptBuilder("openvla")
    b.add_turn("human", "What action should the robot take to " + INSTR + "?")
    ids = proc.tokenizer(b.get_prompt(), return_tensors="np").input_ids
    return R.normalize_prompt_ids(ids.reshape(-1).tolist())


def episode_images(proc, name, t, cams=3):
    import h5py
    import cv2
    import torch
    from PIL import Image
    tiles = []
    with h5py.File(os.path.join(HDF5_DIR, name), "r") as h:
        for cam in R.CAMERA_ORDER[:cams]:
            buf = h["vision/" + cam + "/colors"][t]
            img = cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
            tiles.append(proc.image_processor.apply_transform(
                Image.fromarray(np.ascontiguousarray(img))))
    return torch.cat(tiles, dim=0).unsqueeze(0).to(torch.bfloat16).to("cuda")


def inference_14_tokens(vla, proj, pixel_values, prompt, proprio):
    """Greedy legal 14-token decode; the argmax is restricted to 31744..31999.

    `labels=input_ids` is required because enable_pure_autoreg() replaces
    _process_action_masks() with a version that needs labels; the returned loss is ignored and no
    gradient is ever taken here (torch.no_grad()).
    """
    import torch
    ids, bins = [], []
    with torch.no_grad():
        for _k in range(R.N_ACTIONS):
            inp = torch.tensor([prompt + ids], dtype=torch.long, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = vla(input_ids=inp, attention_mask=torch.ones_like(inp),
                          pixel_values=pixel_values, proprio=proprio, proprio_projector=proj,
                          labels=inp.clone())
            lg = out.logits[0, -1].float()
            cand = lg[R.TOKEN_BEGIN_IDX + 1: R.TOKEN_BEGIN_IDX + 1 + 256]
            k = int(torch.argmax(cand).item())
            bins.append(k)
            ids.append(R.TOKEN_BEGIN_IDX + 1 + k)
    return np.asarray(bins, np.int64), np.asarray(ids, np.int64)


def common_prefix_check(vla, proj, pixel_values, prompt, proprio, extra=3):
    """Full-path common prefix: appending tokens must not change earlier logits."""
    import torch
    with torch.no_grad():
        a = torch.tensor([prompt], dtype=torch.long, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            la = vla(input_ids=a, attention_mask=torch.ones_like(a), pixel_values=pixel_values,
                     proprio=proprio, proprio_projector=proj,
                     labels=a.clone()).logits[0, -1].float()
        b = torch.tensor([prompt + [31744] * extra], dtype=torch.long, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lb = vla(input_ids=b, attention_mask=torch.ones_like(b), pixel_values=pixel_values,
                     proprio=proprio, proprio_projector=proj,
                     labels=b.clone()).logits[0, -1 - extra].float()
    d = (la - lb).abs().max().item()
    win = la[R.TOKEN_BEGIN_IDX + 1: R.TOKEN_BEGIN_IDX + 1 + 256]
    win2 = lb[R.TOKEN_BEGIN_IDX + 1: R.TOKEN_BEGIN_IDX + 1 + 256]
    argmax_same = int(torch.argmax(win).item()) == int(torch.argmax(win2).item())
    return {"max_abs_logit_delta": float(d), "window_argmax_identical": bool(argmax_same),
            "n_extra_tokens": int(extra)}


# -------------------------------------------------------------------------------- smoke
def run_smoke(args):
    import torch
    bucket = args.bucket
    budget = ledger_budget(bucket, args.time_budget)
    if budget <= 30:
        raise SystemExit("no budget left in bucket %s" % bucket)
    ledger_book(bucket, 0.0, "start", note=args.out, extra={"budget_s": budget})
    t0 = time.time()
    gpu_before = gpu_snapshot()
    out = args.out
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit("output directory must be NEW and empty: %s" % out)
    os.makedirs(out + "/checkpoints", exist_ok=True)

    prompt_env = force_arx_platform()
    contract, meta, splits = load_splits()
    q01 = np.asarray(meta["q01"], np.float32)
    q99 = np.asarray(meta["q99"], np.float32)
    norm = R.normalizer_payload(
        normalization_mode=meta["normalization_mode"], pmin=meta["pmin"], pmax=meta["pmax"],
        q01=meta["q01"], q99=meta["q99"], v_scale=meta["v_scale"], a_scale=meta["a_scale"],
        motion_quantile=meta.get("motion_quantile"),
        n_train_episodes=meta["frozen_from"]["n_train_episodes_for_quantiles"],
        n_dev_episodes_excluded=meta["frozen_from"]["n_dev_episodes_excluded"])
    nb = np.load(BOUNDS)
    ev = {"mode": "smoke", "utc_start": utcnow(), "argv": prompt_env, "out": out,
          "bucket": bucket, "delta_contract_sha256": sha256_file(CONTRACT),
          "split_sha256": sha256_file(SPLIT), "bounds_sha256": sha256_file(BOUNDS),
          "hdf5_dir": HDF5_DIR, "optimizer_steps": 0, "gpu_before": gpu_before,
          "cuda_used": True, "base_model": MODEL}
    ledger_book(bucket, 0.0, "model_build_begin")

    proc, vla, proj, modeling_sha = build_model(args.cams)
    ev["modeling_sha256"] = modeling_sha
    prompt = make_prompt(proc)
    ev["prompt_len"] = len(prompt)
    ev["prompt_tail"] = prompt[-2:]
    ledger_book(bucket, 0.0, "model_build_done")

    # ---- real batch from 64 TRAIN only
    name = sorted(splits["train"])[args.train_index]
    state, action = read_episode(name)
    TC.assert_a_equals_next_state(state, action)
    t = 5
    row = sample_row(state, action, t, contract)
    pv = episode_images(proc, name, t, args.cams)
    pr = torch.from_numpy(row["features"]).reshape(1, 42).to("cuda")
    lab = R.build_labels(prompt, row["residual"], q01, q99)
    inp = torch.tensor([lab["input_ids"]], dtype=torch.long, device="cuda")
    tgt = torch.tensor([lab["labels"]], dtype=torch.long, device="cuda")
    ev["sample"] = {"episode": name, "t": int(t), "n_prompt": lab["n_prompt"],
                    "n_supervised": lab["n_supervised"], "input_len": int(inp.shape[1]),
                    "action_token_range": [int(min(lab["action_ids"])), int(max(lab["action_ids"]))],
                    "pixel_values_shape": list(pv.shape), "proprio_shape": list(pr.shape)}

    vla.train()
    proj.train()
    opt = torch.optim.AdamW(list(vla.parameters()) + list(proj.parameters()), lr=0.0)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                labels=tgt, proprio=pr, proprio_projector=proj)
        loss = o.loss
    if not torch.isfinite(loss):
        raise SystemExit("smoke loss is not finite: %r" % (float(loss),))
    loss.backward()
    lora_gn = float(torch.nn.utils.clip_grad_norm_(
        [p for p in vla.parameters() if p.requires_grad], 1.0))
    proj_gn = float(torch.nn.utils.clip_grad_norm_(list(proj.parameters()), 1.0))
    ev["teacher_forced"] = {"loss": float(loss), "lora_grad_norm": lora_gn,
                            "projector_grad_norm": proj_gn,
                            "lora_has_grad": lora_gn > 0.0, "projector_has_grad": proj_gn > 0.0}
    if not opt.state_dict()["state"]:
        ev["teacher_forced"]["optimizer_state_empty"] = True
    log("smoke loss=%.4f lora_gn=%.3f proj_gn=%.4f (optimizer_steps=0)"
        % (float(loss), lora_gn, proj_gn))
    # discard the gradients: this branch must not update anything
    opt.zero_grad(set_to_none=True)
    ev["optimizer_steps"] = 0
    ev["optimizer_stepped"] = False
    ledger_book(bucket, 0.0, "forward_backward_done")

    # ---- legal 14-token inference + physical decode
    vla.eval()
    proj.eval()
    bins, ids = inference_14_tokens(vla, proj, pv, prompt, pr)
    v = row["velocity"]
    decode_meta = R.build_metadata(
        "push_T", sha256_file(os.path.join(MODEL, "config.json")), "0" * 64, "0" * 64,
        sha256_file(CONTRACT), sha256_file(SPLIT), train_steps=0, seed=args.seed,
        gripper_policy=meta["gripper_policy"],
        safety_limits={"amin": nb["amin"].astype(np.float32).tolist(),
                       "amax": nb["amax"].astype(np.float32).tolist()},
        normalizer=norm)
    decode_meta["lora_sha256"] = decode_meta["base_model_sha256"]
    decode_meta["projector_sha256"] = decode_meta["base_model_sha256"]
    action_out, residual_hat, sinfo = R.decode_with_metadata(decode_meta, bins, row["state"], v)
    ev["inference"] = {"bins": [int(x) for x in bins], "ids": [int(x) for x in ids],
                       "unique_bins": int(len(set(int(x) for x in bins))),
                       "action": [float(x) for x in action_out],
                       "in_bounds": bool((action_out >= nb["amin"] - 1e-6).all()
                                         and (action_out <= nb["amax"] + 1e-6).all()),
                       "decode_info": sinfo}
    ev["common_prefix"] = common_prefix_check(vla, proj, pv, prompt, pr)
    ledger_book(bucket, 0.0, "inference_done")

    # ---- checkpoint save + reload roundtrip (same process, no second 7B load)
    ck = out + "/checkpoints/step0"
    os.makedirs(ck, exist_ok=True)
    vla.save_pretrained(ck + "/lora_adapter")
    torch.save(proj.state_dict(), ck + "/proprio_projector_42.pt")
    before = inference_14_tokens(vla, proj, pv, prompt, pr)[0]
    lora_file = os.path.join(ck, "lora_adapter", "adapter_model.safetensors")
    proj_file = os.path.join(ck, "proprio_projector_42.pt")
    md = R.build_metadata(
        "push_T", sha256_file(os.path.join(MODEL, "config.json")), sha256_file(lora_file),
        sha256_file(proj_file), sha256_file(CONTRACT), sha256_file(SPLIT), train_steps=0,
        seed=args.seed, gripper_policy=meta["gripper_policy"],
        safety_limits={"amin": nb["amin"].astype(np.float32).tolist(),
                       "amax": nb["amax"].astype(np.float32).tolist(),
                       "max_rate": [0.5] * 14}, normalizer=norm)
    md_sha = R.save_metadata(os.path.join(ck, "r42_metadata.json"), md)
    from safetensors.torch import load_file
    from peft.utils.save_and_load import set_peft_model_state_dict
    sd = load_file(lora_file)
    with torch.no_grad():
        for p in proj.parameters():
            p.zero_()
    set_peft_model_state_dict(vla, sd)
    proj.load_state_dict(torch.load(proj_file, map_location="cpu"))
    proj.to("cuda")
    R.assert_projector_width(proj.fc1.weight.shape, expected=42)
    after = inference_14_tokens(vla, proj, pv, prompt, pr)[0]
    reloaded = R.load_metadata(os.path.join(ck, "r42_metadata.json"), expect_task="push_T")
    ev["checkpoint"] = {"dir": ck, "lora_sha256": md["lora_sha256"],
                        "projector_sha256": md["projector_sha256"],
                        "metadata_sha256": md_sha,
                        "normalizer_sha256": md["normalizer_sha256"],
                        "tokens_after_reload_identical": bool(np.array_equal(before, after)),
                        "metadata_roundtrip_equal": bool(reloaded == md)}
    if not np.array_equal(before, after):
        raise SystemExit("checkpoint roundtrip changed the 14 action tokens")

    ev["utc_end"] = utcnow()
    ev["wall_clock_s"] = round(time.time() - t0, 1)
    ev["gpu_after"] = gpu_snapshot()
    ev["peak_mem_gb"] = round(float(torch.cuda.max_memory_allocated()) / 1e9, 3)
    ev["verdict"] = ("VERIFIED_GPU_ZERO_UPDATE"
                     if (ev["teacher_forced"]["lora_has_grad"]
                         and ev["teacher_forced"]["projector_has_grad"]
                         and ev["inference"]["in_bounds"]
                         and ev["checkpoint"]["tokens_after_reload_identical"])
                     else "FAILED_GPU_ZERO_UPDATE")
    with open(os.path.join(out, "smoke_evidence.json"), "w") as fh:
        json.dump(ev, fh, indent=1)
    ledger_book(bucket, ev["wall_clock_s"], "end", note=ev["verdict"],
                extra={"wall_clock_s": ev["wall_clock_s"]})
    log("SMOKE %s in %.1fs (peak %.2f GB)" % (ev["verdict"], ev["wall_clock_s"],
                                             ev["peak_mem_gb"]))
    return ev


# ------------------------------------------------------------------------------ training
def run_train(args):
    import torch
    bucket = args.bucket
    budget = ledger_budget(bucket, args.time_budget)
    if budget <= 30:
        raise SystemExit("no budget left in bucket %s" % bucket)
    out = args.out
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit("output directory must be NEW and empty: %s" % out)
    os.makedirs(out + "/checkpoints", exist_ok=True)
    ledger_book(bucket, 0.0, "start", note=out, extra={"budget_s": budget})
    t0 = time.time()
    gpu_before = gpu_snapshot()

    prompt_env = force_arx_platform()
    contract, meta, splits = load_splits()
    q01 = np.asarray(meta["q01"], np.float32)
    q99 = np.asarray(meta["q99"], np.float32)
    norm = R.normalizer_payload(
        normalization_mode=meta["normalization_mode"], pmin=meta["pmin"], pmax=meta["pmax"],
        q01=meta["q01"], q99=meta["q99"], v_scale=meta["v_scale"], a_scale=meta["a_scale"],
        motion_quantile=meta.get("motion_quantile"),
        n_train_episodes=meta["frozen_from"]["n_train_episodes_for_quantiles"],
        n_dev_episodes_excluded=meta["frozen_from"]["n_dev_episodes_excluded"])
    nb = np.load(BOUNDS)
    proc, vla, proj, modeling_sha = build_model(args.cams)
    prompt = make_prompt(proc)
    params = [p for p in vla.parameters() if p.requires_grad] + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[max(1, args.lr_milestone)],
                                                 gamma=0.1)
    run = {"mode": "train", "utc_start": utcnow(), "argv": prompt_env, "out": out,
           "bucket": bucket, "steps_requested": args.steps, "optimizer_steps": 0,
           "gpu_before": gpu_before, "modeling_sha256": modeling_sha,
           "delta_contract_sha256": sha256_file(CONTRACT), "split_sha256": sha256_file(SPLIT),
           "mode_train_only": True, "dev_used_for_gradients": False,
           "lr": args.lr, "batch": 1, "seed": args.seed, "cams": args.cams,
           "frame_pool": "t in [2, T-2]", "log": []}
    with open(os.path.join(out, "run_meta.json"), "w") as fh:
        json.dump(run, fh, indent=1)
    if args.init_lora:
        raise SystemExit("R11 must start from the pristine base (no legacy adapter resume)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    vla.train()
    proj.train()
    losses = []
    for step in range(1, args.steps + 1):
        elapsed = time.time() - t0
        if elapsed > budget - args.time_margin:
            log("stopping at step %d: %.0fs of a %.0fs bucket used" % (step, elapsed, budget))
            break
        name = random.choice(splits["train"])
        state, action = read_episode(name)
        t = int(random.randrange(2, len(state) - 1))
        row = sample_row(state, action, t, contract)
        pv = episode_images(proc, name, t, args.cams)
        pr = torch.from_numpy(row["features"]).reshape(1, 42).to("cuda")
        lab = R.build_labels(prompt, row["residual"], q01, q99)
        inp = torch.tensor([lab["input_ids"]], dtype=torch.long, device="cuda")
        tgt = torch.tensor([lab["labels"]], dtype=torch.long, device="cuda")
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                    labels=tgt, proprio=pr, proprio_projector=proj)
            loss = o.loss
        if not torch.isfinite(loss):
            with open(os.path.join(out, "FAILURES.md"), "w") as fh:
                fh.write("# training stopped\n\nnon-finite loss at step %d (%s t=%d)\n"
                         % (step, name, t))
            raise SystemExit("non-finite loss at step %d" % step)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        sched.step()
        losses.append(float(loss))
        if step <= 3 or step % args.log_every == 0:
            log("step=%d loss=%.4f gnorm=%.2f mem=%.2fGB %.0fs"
                % (step, float(loss), gn, torch.cuda.max_memory_allocated() / 1e9,
                   time.time() - t0))
            run["log"].append({"step": step, "loss": float(loss), "grad_norm": gn,
                               "episode": name, "t": t, "utc": utcnow()})
            with open(os.path.join(out, "run_meta.json"), "w") as fh:
                json.dump(run, fh, indent=1)
        run["optimizer_steps"] = step
        if args.save_every and (step % args.save_every == 0):
            ck = out + "/checkpoints/step%d" % step
            os.makedirs(ck, exist_ok=True)
            vla.save_pretrained(ck + "/lora_adapter")
            torch.save(proj.state_dict(), ck + "/proprio_projector_42.pt")
            md = R.build_metadata(
                "push_T", sha256_file(os.path.join(MODEL, "config.json")),
                sha256_file(os.path.join(ck, "lora_adapter", "adapter_model.safetensors")),
                sha256_file(os.path.join(ck, "proprio_projector_42.pt")),
                sha256_file(CONTRACT), sha256_file(SPLIT), train_steps=step, seed=args.seed,
                gripper_policy=meta["gripper_policy"],
                safety_limits={"amin": nb["amin"].astype(np.float32).tolist(),
                               "amax": nb["amax"].astype(np.float32).tolist(),
                               "max_rate": [0.5] * 14}, normalizer=norm)
            R.save_metadata(os.path.join(ck, "r42_metadata.json"), md)
            run.setdefault("checkpoints", []).append(
                {"step": step, "lora_sha256": md["lora_sha256"],
                 "projector_sha256": md["projector_sha256"], "dir": ck})
            log("saved %s" % ck)
    run["utc_end"] = utcnow()
    run["wall_clock_s"] = round(time.time() - t0, 1)
    run["gpu_after"] = gpu_snapshot()
    run["peak_mem_gb"] = round(float(torch.cuda.max_memory_allocated()) / 1e9, 3)
    run["first_loss"] = losses[0] if losses else None
    run["last_loss"] = losses[-1] if losses else None
    run["loss_head"] = losses[:5]
    run["loss_tail"] = losses[-5:]
    run["steps_per_second"] = round(len(losses) / max(1e-9, run["wall_clock_s"]), 4)
    with open(os.path.join(out, "run_meta.json"), "w") as fh:
        json.dump(run, fh, indent=1)
    ledger_book(bucket, run["wall_clock_s"], "end", note="steps=%d" % run["optimizer_steps"],
                extra={"steps": run["optimizer_steps"]})
    log("TRAIN_DONE steps=%d wall=%.1fs peak=%.2fGB first=%.4f last=%.4f"
        % (run["optimizer_steps"], run["wall_clock_s"], run["peak_mem_gb"],
           run["first_loss"] or -1, run["last_loss"] or -1))
    return run


# ------------------------------------------------------- R15 true cumulative continuation
def _r15_data_contract(meta, norm):
    import r9_trainer_contract as RC
    return {"split_sha256": sha256_file(SPLIT), "frozen_contract_sha256": sha256_file(CONTRACT),
            "bounds_sha256": sha256_file(BOUNDS), "hdf5_dir": os.path.abspath(HDF5_DIR),
            "codec": {"mode": "fixed", "window": list(R.CODEC_WINDOW), "stop": R.STOP_ID,
                      "pad": R.PAD_TOKEN_ID},
            "normalizer_sha256": R.normalizer_sha256(norm),
            "camera_order": list(R.CAMERA_ORDER),
            "gripper_dims": list(R.GRIPPER_DIMS),
            "target": R.TARGET, "alignment_version": R.ALIGNMENT_VERSION,
            "train_episodes": sorted(RC.select_splits(SPLIT)["train"])}


def r15_train_frames(splits):
    """Every history-safe TRAIN frame, as (episode, t) pairs, in a stable order."""
    import h5py
    frames = []
    for name in sorted(splits["train"]):
        with h5py.File(os.path.join(HDF5_DIR, name), "r") as h:
            T = len(h["action/left_arm_joint_states"])
        for t in range(TC.MIN_HISTORY_FRAMES, T - 1):
            frames.append((name, int(t)))
    return frames


def run_r15(args):
    """One cumulative stage of the formal R15 run: 0->300, 300->1000 or 1000->3000."""
    import torch
    import r15_ledger as L15
    import r15_resume as RS
    target = int(args.target_global_step)
    if target not in (300, 1000, 3000):
        raise SystemExit("--target-global-step must be 300, 1000 or 3000 (got %d)" % target)
    ledger = L15.Ledger(args.ledger_dir)
    with ledger:
        ledger.verify_projection()
        tot = ledger.totals()
    start_step = 0
    if args.resume_from:
        man, ts = RS.verify_and_load_manifest(args.resume_from)
        start_step = int(ts["global_step"])
        expected = {300: 0, 1000: 300, 3000: 1000}[target]
        if start_step != expected:
            raise SystemExit("to reach %d you must resume from step %d, but %s is step %d"
                             % (target, expected, args.resume_from, start_step))
    else:
        if target != 300:
            raise SystemExit("a fresh run may only target step 300; %d needs --resume-from" % target)
    need_steps = target - start_step
    if not os.path.exists(args.out) or not os.listdir(args.out):
        os.makedirs(args.out, exist_ok=True)
    else:
        raise SystemExit("output directory must be NEW and empty: %s" % args.out)
    run_id = args.run_id or ("r15-%d-%s" % (target, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    gpu_est = float(need_steps) * float(args.est_seconds_per_step) + float(args.time_margin)
    with ledger:
        guard = ledger.assert_can_start(args.bucket, gpu_est, requested_steps=need_steps,
                                        run_id=run_id)
        ledger.append(run_id, "job_begin", bucket=args.bucket, payload={
            "target_global_step": target, "start_global_step": start_step,
            "steps_requested": need_steps, "resume_from": args.resume_from,
            "gpu_estimate_s": round(gpu_est, 1), "argv": force_arx_platform(),
            "guard": guard, "gpu_snapshot": gpu_snapshot()})
    log("R15 %s: %d -> %d (%d steps), bucket %s, estimate %.0fs"
        % (run_id, start_step, target, need_steps, args.bucket, gpu_est))

    t0 = time.time()
    contract, meta, splits = load_splits()
    q01 = np.asarray(meta["q01"], np.float32)
    q99 = np.asarray(meta["q99"], np.float32)
    norm = R.normalizer_payload(
        normalization_mode=meta["normalization_mode"], pmin=meta["pmin"], pmax=meta["pmax"],
        q01=meta["q01"], q99=meta["q99"], v_scale=meta["v_scale"], a_scale=meta["a_scale"],
        motion_quantile=meta.get("motion_quantile"),
        n_train_episodes=meta["frozen_from"]["n_train_episodes_for_quantiles"],
        n_dev_episodes_excluded=meta["frozen_from"]["n_dev_episodes_excluded"])
    nb = np.load(BOUNDS)
    limits = {"amin": nb["amin"].astype(np.float32).tolist(),
              "amax": nb["amax"].astype(np.float32).tolist(), "max_rate": [0.5] * 14}
    dcontract = _r15_data_contract(meta, norm)
    base_id = RS.attest_base_model(MODEL)
    frames = r15_train_frames(splits)

    RS.seed_everything(args.seed)          # BEFORE any model/LoRA/projector construction
    proc, vla, proj, modeling_sha = build_model(args.cams)
    prompt = make_prompt(proc)
    params = [p for p in vla.parameters() if p.requires_grad] + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1000, 2500], gamma=0.1)
    sampler = RS.FrameSampler(frames, args.seed)
    if args.save_every <= 0:
        raise SystemExit("--save-every must be > 0: without milestone checkpoints a crash loses "
                         "the run and 'resume' would silently become a fresh restart")
    if args.resume_from:
        out = RS.load_full_state(args.resume_from, model=vla, proj=proj, opt=opt, sched=sched,
                                frames=frames, model_kind="peft",
                                expected_contract=dcontract, expected_base=base_id,
                                expected_global_step=start_step)
        sampler = out["sampler"]
        log("resumed from %s at global_step=%d (next frame %s)"
            % (args.resume_from, start_step, sampler.next()))
        sampler = out["sampler"]
    ev = {"mode": "r15_stage", "run_id": run_id, "utc_start": utcnow(), "target": target,
          "start_global_step": start_step, "steps": need_steps, "bucket": args.bucket,
          "out": args.out, "resume_from": args.resume_from, "seed": args.seed,
          "lr": args.lr, "sched_milestones": [1000, 2500], "cams": args.cams,
          "n_train_frames": len(frames), "sampler_order_hash": sampler.order_hash(),
          "data_contract": dcontract, "base_model": base_id, "modeling_sha256": modeling_sha,
          "train_only": True, "dev_used_for_gradients": False,
          "new_final_used": False, "gpu_snapshot_start": gpu_snapshot(), "log": []}
    with open(os.path.join(args.out, "stage_evidence.json"), "w") as fh:
        json.dump(ev, fh, indent=1)

    vla.train()
    proj.train()
    global_step = start_step
    step_times = []
    while global_step < target:
        elapsed = time.time() - t0
        if elapsed > gpu_est - args.time_margin:
            log("stopping at global_step=%d: bucket/time guard (%.0fs used of %.0fs)"
                % (global_step, elapsed, gpu_est))
            break
        name, t = sampler.next()
        st = time.time()
        state, action = read_episode(name)
        row = sample_row(state, action, t, contract)
        pv = episode_images(proc, name, t, args.cams)
        pr = torch.from_numpy(row["features"]).reshape(1, 42).to("cuda")
        lab = R.build_labels(prompt, row["residual"], q01, q99)
        inp = torch.tensor([lab["input_ids"]], dtype=torch.long, device="cuda")
        tgt = torch.tensor([lab["labels"]], dtype=torch.long, device="cuda")
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                    labels=tgt, proprio=pr, proprio_projector=proj)
            loss = o.loss
        if not torch.isfinite(loss):
            with open(os.path.join(args.out, "FAILURES.md"), "w") as fh:
                fh.write("# R15 stage stopped\n\nnon-finite loss at global_step %d (%s t=%d)\n"
                         % (global_step + 1, name, t))
            with ledger:
                ledger.append(run_id, "abort", bucket=args.bucket,
                              payload={"global_step": global_step, "reason": "non_finite_loss"},
                              gpu_seconds=time.time() - t0, formal_steps=0)
            raise SystemExit("non-finite loss at global_step %d" % (global_step + 1))
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        sched.step()
        global_step += 1
        step_times.append(time.time() - st)
        with ledger:
            ledger.append(run_id, "step_commit", bucket=args.bucket,
                          payload={"global_step": global_step, "loss": round(float(loss), 6),
                                   "grad_norm": round(gn, 4), "lr": sched.get_last_lr()[0],
                                   "episode": name, "t": int(t)},
                          formal_steps=1, gpu_seconds=max(0.0, time.time() - st))
        if global_step % 10 == 0 or global_step == target:
            ev["log"].append({"global_step": global_step, "loss": float(loss),
                              "grad_norm": gn, "lr": sched.get_last_lr()[0],
                              "seconds": round(sum(step_times[-10:]) / len(step_times[-10:]), 3),
                              "utc": utcnow()})
            log("step=%d loss=%.4f gnorm=%.2f lr=%.2e mem=%.2fGB"
                % (global_step, float(loss), gn, sched.get_last_lr()[0],
                   torch.cuda.max_memory_allocated() / 1e9))
            with open(os.path.join(args.out, "stage_evidence.json"), "w") as fh:
                json.dump(ev, fh, indent=1)
        if global_step % args.save_every == 0 or global_step == target:
            ck = RS.save_full_state(
                os.path.join(args.out, "checkpoints", "step%d" % global_step),
                model=vla, proj=proj, opt=opt, sched=sched, sampler=sampler,
                global_step=global_step, run_id=run_id, data_contract=dcontract,
                model_kind="peft", extra={"base_model": base_id, "modeling_sha256": modeling_sha,
                                          "target": target})
            log("saved full-state checkpoint step=%d sha=%s"
                % (global_step, ck["manifest"]["files"]["proprio_projector_42.pt"][:16]))

    ev["global_step_reached"] = global_step
    ev["utc_end"] = utcnow()
    ev["wall_clock_s"] = round(time.time() - t0, 1)
    ev["seconds_per_step"] = round(float(np.mean(step_times)), 3) if step_times else None
    ev["peak_mem_gb"] = round(float(torch.cuda.max_memory_allocated()) / 1e9, 3)
    ev["gpu_snapshot_end"] = gpu_snapshot()
    ev["verdict"] = "REACHED_TARGET" if global_step == target else "PARTIAL"
    with open(os.path.join(args.out, "stage_evidence.json"), "w") as fh:
        json.dump(ev, fh, indent=1)
    with ledger:
        led_tot = ledger.totals()
        ledger.append(run_id, "job_end", bucket=args.bucket,
                      payload={"global_step": global_step, "verdict": ev["verdict"],
                               "formal_steps_booked_here": global_step - start_step,
                               "wall_clock_s": ev["wall_clock_s"],
                               "totals_after": led_tot},
                      gpu_seconds=0.0, formal_steps=0)
        if ledger.totals()["formal_steps_used"] > L15.FORMAL_STEPS_CAP:
            raise SystemExit("FORMAL STEP CAP EXCEEDED")
    log("R15 %s reached step %d in %.1fs (%.3fs/step, peak %.2f GB)"
        % (ev["verdict"], global_step, ev["wall_clock_s"], ev["seconds_per_step"] or -1,
           ev["peak_mem_gb"]))
    return ev


# ------------------------------------------------------------------- R14 overfit mode
def fixed_overfit_frames(splits, n_frames, seed=0):
    """Deterministically pick one frame from each of n distinct TRAIN episodes."""
    eps = sorted(splits["train"])
    if n_frames > len(eps):
        raise SystemExit("cannot take %d diverse frames from %d TRAIN episodes"
                         % (n_frames, len(eps)))
    rs = np.random.RandomState(seed)
    picks = []
    for i in range(n_frames):
        name = eps[i]
        state, _action = read_episode(name)
        pool = np.arange(2, len(state) - 1)
        picks.append((name, int(pool[rs.randint(len(pool))])))
    return picks


def prompt_only_window(vla, proj, pv, prompt, pr):
    """Logits for the FIRST supervised position (the last prompt token), 256-wide window."""
    import torch
    with torch.no_grad():
        inp = torch.tensor([prompt], dtype=torch.long, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                      proprio=pr, proprio_projector=proj, labels=inp.clone())
    lg = out.logits[0, -1].float()
    return lg[R.TOKEN_BEGIN_IDX + 1: R.TOKEN_BEGIN_IDX + 1 + 256].cpu().numpy()


def teacher_forced_ce(vla, proj, pv, prompt, residual, pr, q01, q99):
    import torch
    lab = R.build_labels(prompt, residual, q01, q99)
    inp = torch.tensor([lab["input_ids"]], dtype=torch.long, device="cuda")
    tgt = torch.tensor([lab["labels"]], dtype=torch.long, device="cuda")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                    labels=tgt, proprio=pr, proprio_projector=proj)
    return float(o.loss)


def run_overfit(args):
    """R14: <=100 TRAIN-only updates on a fixed diverse frame set, with milestone archives."""
    import torch
    bucket = args.bucket
    if args.steps > 100:
        raise SystemExit("R14 is capped at 100 optimizer steps, requested %d" % args.steps)
    budget = ledger_budget(bucket, args.time_budget)
    if budget <= 60:
        raise SystemExit("no budget left in bucket %s" % bucket)
    out = args.out
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit("output directory must be NEW and empty: %s" % out)
    os.makedirs(out + "/checkpoints", exist_ok=True)
    ledger_book(bucket, 0.0, "start", note=out, extra={"budget_s": budget})
    t0 = time.time()
    gpu_before = gpu_snapshot()

    prompt_env = force_arx_platform()
    contract, meta, splits = load_splits()
    q01 = np.asarray(meta["q01"], np.float32)
    q99 = np.asarray(meta["q99"], np.float32)
    norm = R.normalizer_payload(
        normalization_mode=meta["normalization_mode"], pmin=meta["pmin"], pmax=meta["pmax"],
        q01=meta["q01"], q99=meta["q99"], v_scale=meta["v_scale"], a_scale=meta["a_scale"],
        motion_quantile=meta.get("motion_quantile"),
        n_train_episodes=meta["frozen_from"]["n_train_episodes_for_quantiles"],
        n_dev_episodes_excluded=meta["frozen_from"]["n_dev_episodes_excluded"])
    nb = np.load(BOUNDS)
    limits = {"amin": nb["amin"].astype(np.float32).tolist(),
              "amax": nb["amax"].astype(np.float32).tolist(), "max_rate": [0.5] * 14}
    frames = fixed_overfit_frames(splits, args.overfit_frames, seed=args.seed)
    proc, vla, proj, modeling_sha = build_model(args.cams)
    prompt = make_prompt(proc)
    ev = {"mode": "overfit", "utc_start": utcnow(), "argv": prompt_env, "out": out,
          "bucket": bucket, "steps_cap": 100, "steps_requested": args.steps,
          "n_fixed_frames": len(frames), "fixed_frames": [[e, t] for e, t in frames],
          "fixed_frames_distinct_episodes": len({e for e, _t in frames}),
          "train_only": True, "dev_used_for_gradients": False,
          "modeling_sha256": modeling_sha, "delta_contract_sha256": sha256_file(CONTRACT),
          "split_sha256": sha256_file(SPLIT), "gpu_before": gpu_before, "lr": args.lr,
          "seed": args.seed, "milestones": [], "log": []}
    with open(os.path.join(out, "overfit_evidence.json"), "w") as fh:
        json.dump(ev, fh, indent=1)

    params = [p for p in vla.parameters() if p.requires_grad] + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def snapshot(step, loss_curve):
        ck = out + "/checkpoints/step%d" % step
        os.makedirs(ck, exist_ok=True)
        vla.save_pretrained(ck + "/lora_adapter")
        torch.save(proj.state_dict(), ck + "/proprio_projector_42.pt")
        md = R.build_metadata(
            "push_T", sha256_file(os.path.join(MODEL, "config.json")),
            sha256_file(os.path.join(ck, "lora_adapter", "adapter_model.safetensors")),
            sha256_file(os.path.join(ck, "proprio_projector_42.pt")),
            sha256_file(CONTRACT), sha256_file(SPLIT), train_steps=step, seed=args.seed,
            gripper_policy=meta["gripper_policy"], safety_limits=limits, normalizer=norm)
        R.save_metadata(os.path.join(ck, "r42_metadata.json"), md)
        ces, windows = [], []
        for name, t in frames:
            state, action = read_episode(name)
            row = TC.build_training_row(state, action, t, contract)
            pv = episode_images(proc, name, t, args.cams)
            pr = torch.from_numpy(row["features"]).reshape(1, 42).to("cuda")
            ces.append(teacher_forced_ce(vla, proj, pv, prompt, row["residual"], pr, q01, q99))
            if len(windows) < args.pred_frames:
                windows.append(prompt_only_window(vla, proj, pv, prompt, pr).tolist())
        rec = {"step": step, "lora_sha256": md["lora_sha256"],
               "projector_sha256": md["projector_sha256"],
               "ce_mean": float(np.mean(ces)), "ce_per_frame": [round(c, 5) for c in ces],
               "first_position_argmax_bins": [int(np.argmax(w)) for w in windows],
               "first_position_window_logits": windows,
               "loss_tail": [round(x, 5) for x in loss_curve[-10:]],
               "utc": utcnow()}
        if step == 0 or step == args.steps:
            name, t = frames[0]
            state, action = read_episode(name)
            row = TC.build_training_row(state, action, t, contract)
            pv = episode_images(proc, name, t, args.cams)
            pr = torch.from_numpy(row["features"]).reshape(1, 42).to("cuda")
            bins, _ids = inference_14_tokens(vla, proj, pv, prompt, pr)
            rec["greedy_bins"] = [int(x) for x in bins]
            rec["greedy_unique"] = int(len(set(int(x) for x in bins)))
        ev["milestones"].append(rec)
        with open(os.path.join(out, "overfit_evidence.json"), "w") as fh:
            json.dump(ev, fh, indent=1)
        log("milestone step=%d CE=%.4f" % (step, rec["ce_mean"]))

    vla.train()
    proj.train()
    milestones = [int(x) for x in str(args.milestones).split(",") if str(x).strip() != ""]
    loss_curve = []
    if 0 in milestones:
        snapshot(0, loss_curve)
    for step in range(1, args.steps + 1):
        if time.time() - t0 > budget - args.time_margin:
            log("stopping at step %d: bucket budget reached" % step)
            break
        name, t = frames[(step - 1) % len(frames)]
        state, action = read_episode(name)
        row = sample_row(state, action, t, contract)
        pv = episode_images(proc, name, t, args.cams)
        pr = torch.from_numpy(row["features"]).reshape(1, 42).to("cuda")
        lab = R.build_labels(prompt, row["residual"], q01, q99)
        inp = torch.tensor([lab["input_ids"]], dtype=torch.long, device="cuda")
        tgt = torch.tensor([lab["labels"]], dtype=torch.long, device="cuda")
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                    labels=tgt, proprio=pr, proprio_projector=proj)
            loss = o.loss
        if not torch.isfinite(loss):
            with open(os.path.join(out, "FAILURES.md"), "w") as fh:
                fh.write("# R14 stopped\n\nnon-finite loss at step %d (%s t=%d)\n"
                         % (step, name, t))
            raise SystemExit("non-finite loss at step %d" % step)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        loss_curve.append(float(loss))
        ev["log"].append({"step": step, "loss": round(float(loss), 5), "grad_norm": round(gn, 4),
                          "episode": name, "t": t, "utc": utcnow()})
        if step <= 3 or step % 10 == 0 or step == args.steps:
            log("step=%d loss=%.4f gnorm=%.2f mem=%.2fGB %.0fs"
                % (step, float(loss), gn, torch.cuda.max_memory_allocated() / 1e9,
                   time.time() - t0))
            with open(os.path.join(out, "overfit_evidence.json"), "w") as fh:
                json.dump(ev, fh, indent=1)
        if step in milestones:
            snapshot(step, loss_curve)
    ev["optimizer_steps"] = len(loss_curve)
    ev["loss_first"] = loss_curve[0] if loss_curve else None
    ev["loss_last"] = loss_curve[-1] if loss_curve else None
    ev["loss_head_mean"] = float(np.mean(loss_curve[:10])) if loss_curve else None
    ev["loss_tail_mean"] = float(np.mean(loss_curve[-10:])) if loss_curve else None
    ev["utc_end"] = utcnow()
    ev["wall_clock_s"] = round(time.time() - t0, 1)
    ev["gpu_after"] = gpu_snapshot()
    ev["peak_mem_gb"] = round(float(torch.cuda.max_memory_allocated()) / 1e9, 3)
    ms = {m["step"]: m for m in ev["milestones"]}
    ev["ce_decreased"] = bool(ms and max(ms) > 0 and ms[max(ms)]["ce_mean"] < ms[min(ms)]["ce_mean"])
    last = ms.get(max(ms)) if ms else None
    ev["decodable_nonzero"] = bool(last and last.get("greedy_unique", 0) > 1)
    ev["verdict"] = ("OVERFIT_SIGNAL_PASS"
                     if (ev["ce_decreased"] and ev["decodable_nonzero"]
                         and ev["loss_tail_mean"] < ev["loss_head_mean"])
                     else "OVERFIT_SIGNAL_FAIL")
    with open(os.path.join(out, "overfit_evidence.json"), "w") as fh:
        json.dump(ev, fh, indent=1)
    led = load_ledger()
    led["results"]["overfit_optimizer_steps_used"] = int(
        led["results"].get("overfit_optimizer_steps_used", 0)) + ev["optimizer_steps"]
    with open(LEDGER, "w") as fh:
        json.dump(led, fh, indent=1)
    ledger_book(bucket, ev["wall_clock_s"], "end", note=ev["verdict"],
                extra={"steps": ev["optimizer_steps"], "peak_mem_gb": ev["peak_mem_gb"]})
    log("OVERFIT %s steps=%d wall=%.1fs head=%.4f tail=%.4f"
        % (ev["verdict"], ev["optimizer_steps"], ev["wall_clock_s"],
           ev["loss_head_mean"] or -1, ev["loss_tail_mean"] or -1))
    return ev


def run_dev_preds(args):
    """Greedy 14-token DEV predictions for the frozen gate frame list (no gradient, no update)."""
    import torch
    import r15_resume as RS
    sys.path.insert(0, os.path.join(REPO, "diagnostics"))
    import r15_dev_eval as DE
    contract, meta, splits = load_splits()
    q01 = np.asarray(meta["q01"], np.float32)
    q99 = np.asarray(meta["q99"], np.float32)
    norm = R.normalizer_payload(
        normalization_mode=meta["normalization_mode"], pmin=meta["pmin"], pmax=meta["pmax"],
        q01=meta["q01"], q99=meta["q99"], v_scale=meta["v_scale"], a_scale=meta["a_scale"],
        motion_quantile=meta.get("motion_quantile"),
        n_train_episodes=meta["frozen_from"]["n_train_episodes_for_quantiles"],
        n_dev_episodes_excluded=meta["frozen_from"]["n_dev_episodes_excluded"])
    force_arx_platform()          # MUST precede every prismatic import (fail-closed otherwise)
    RS.seed_everything(args.seed)
    proc, vla, proj, modeling_sha = build_model(args.cams)
    prompt = make_prompt(proc)
    man, ts = RS.verify_and_load_manifest(args.resume_from)
    frames = r15_train_frames(splits)
    from safetensors.torch import load_file
    from peft.utils.save_and_load import set_peft_model_state_dict
    set_peft_model_state_dict(vla, load_file(os.path.join(
        args.resume_from, "lora_adapter", "adapter_model.safetensors")))
    proj.load_state_dict(torch.load(os.path.join(args.resume_from, "proprio_projector_42.pt"),
                                    map_location="cpu"))
    proj.to("cuda")
    vla.eval()
    proj.eval()
    gate_frames = DE.frozen_frames(sorted(splits["dev"]), args.dev_per_episode, seed=args.dev_seed)
    if set(e for e, _t in gate_frames) & set(e for e, _t in frames):
        raise SystemExit("DEV gate frames overlap TRAIN")
    actions, rows = [], []
    for name, t in gate_frames:
        state, action = read_episode(name)
        row = TC.build_training_row(state, action, t, contract)
        pv = episode_images(proc, name, t, args.cams)
        pr = torch.from_numpy(row["features"]).reshape(1, 42).to("cuda")
        bins, _ids = inference_14_tokens(vla, proj, pv, prompt, pr)
        cmd, _r, _i = R.decode_command(bins, q01, q99, row["state"], row["velocity"])
        actions.append(cmd)
        rows.append([name, int(t)])
    out = args.dev_preds
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    a = np.stack(actions).astype(np.float32)
    np.savez_compressed(out, action=a, frames=np.array(rows),
                        frames_sha256=np.array(DE.frames_sha256(gate_frames)),
                        model_sha256=np.array(
                            RS.sha256_file(os.path.join(args.resume_from, "lora_adapter",
                                                        "adapter_model.safetensors"))),
                        global_step=np.array(int(ts["global_step"])),
                        modeling_sha256=np.array(modeling_sha))
    log("DEV predictions (%d frames, global_step=%d) -> %s"
        % (len(actions), int(ts["global_step"]), out))
    return {"n": len(actions), "out": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bucket", default="R13_7B_ZERO_STEP")
    ap.add_argument("--time_budget", type=float, default=1800.0)
    ap.add_argument("--time_margin", type=float, default=90.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_milestone", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--cams", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_index", type=int, default=0)
    ap.add_argument("--init_lora", default="")
    ap.add_argument("--overfit_frames", type=int, default=0,
                    help="R14: fixed diverse TRAIN frame count (0 = plain random sampling)")
    ap.add_argument("--milestones", default="0,20,40,60,80,100")
    ap.add_argument("--pred_frames", type=int, default=8)
    ap.add_argument("--r15", action="store_true", help="cumulative R15 stage")
    ap.add_argument("--target-global-step", type=int, default=0, choices=[0, 300, 1000, 3000])
    ap.add_argument("--resume-from", default="")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--ledger-dir", default=None)
    ap.add_argument("--est-seconds-per-step", type=float, default=3.0)
    ap.add_argument("--dev-preds", default="")
    ap.add_argument("--dev-per-episode", type=int, default=2)
    ap.add_argument("--dev-seed", type=int, default=0)
    args = ap.parse_args()
    t_start = time.time()
    try:
        if args.dev_preds:
            run_dev_preds(args)
            return 0
        if args.r15:
            run_r15(args)
            return 0
        if args.smoke:
            run_smoke(args)
            return 0
        if args.steps <= 0:
            raise SystemExit("--steps must be > 0 (or pass --smoke)")
        if args.overfit_frames > 0:
            run_overfit(args)
            return 0
        run_train(args)
        return 0
    except BaseException as exc:
        # A crash still consumed real GPU seconds: book them instead of leaving a silent debt.
        ledger_book(args.bucket, time.time() - t_start, "abort", note=repr(exc)[:300])
        raise


if __name__ == "__main__":
    sys.exit(main())
