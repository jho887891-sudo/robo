#!/usr/bin/env python3
"""R16 first-divergent-token REAL 7B differential probe.

No optimizer updates. No writes to checkpoint or historical evidence.
Use one frozen DEV frame and one model load; compare multiple k=1 paths against
the existing full multimodal forward. This is a DIAGNOSTIC, not a fix.

Run in the exact remote R15/R16 7B environment with the same flags as
tools/r16_kv_equivalence.py, e.g.
python tools/r16_kv_first_divergence.py \
  --checkpoint runs/m2_two_2000_20261021T123314Z/checkpoints/step2000 \
  --out evidence/r16_kv_first_divergence_$(date -u +%Y%m%dT%H%M%SZ).json
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import r16_kv_equivalence as KV
M2, MD, R, S = KV.M2, KV.MD, KV.R, KV.S


def actual_cache_length(cache):
    """Actual KV sequence length, never the TEXT input length."""
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    if isinstance(cache, (tuple, list)) and len(cache):
        return int(cache[0][0].shape[-2])
    raise RuntimeError("unknown KV class: %s" % type(cache).__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--normalizer", default="data/m2_task_normalizer_2task.json")
    ap.add_argument("--splits", default="data/m2_splits.json")
    ap.add_argument("--hdf5-root", default=None)
    ap.add_argument("--task", default="push_T", choices=["push_T", "stack_bowls"])
    ap.add_argument("--cams", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    tasks = ["push_T", "stack_bowls"]
    if args.hdf5_root:
        # Current HDF5 path is injected only through the normal index helper.
        indexes = {t: MD.load_task_index(t, args.hdf5_root) for t in tasks}
    else:
        indexes = {t: MD.load_task_index(t) for t in tasks}
    _doc, split, payload, _sp, _np = M2.load_m2_artifacts(
        tasks, args.splits, args.normalizer)
    episode, tick = M2.dev_frames_for_task(
        indexes[args.task], split[args.task], args.task, 1, 0)[0]
    state, action = M2._read_state_action(indexes[args.task]["hdf5_dir"], episode)
    row = M2.row_for(payload, args.task, state, action, tick, episode=episode)
    proc, vla, proj, ts, modeling_sha = KV.load_verified(args.checkpoint, args.cams)
    backend = S._TorchBackend(vla, proj)
    prompt = M2._make_prompt(proc, args.task, False)
    pv = M2._episode_images(proc, episode, tick, args.cams, indexes[args.task]["hdf5_dir"])
    feat = np.asarray(row["features"], np.float32)
    lo, hi = R.TOKEN_BEGIN_IDX + 1, R.TOKEN_BEGIN_IDX + 1 + 256

    def t_ids(ids):
        return torch.tensor([list(ids)], dtype=torch.long, device="cuda")

    def full_forward(ids, use_cache=None):
        inp = t_ids(ids)
        pr = torch.from_numpy(feat).reshape(1, 42).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(input_ids=inp, attention_mask=torch.ones_like(inp),
                      pixel_values=pv, proprio=pr, proprio_projector=proj,
                      labels=inp.clone(), use_cache=use_cache)
        return out.logits[0, -1].float().cpu().numpy()

    def compare(name, fn, ref, output):
        try:
            vec = fn()
            x = np.asarray(vec, np.float32)[lo:hi]
            output["cases"][name] = {
                "max_abs_256": float(np.max(np.abs(x - ref[lo:hi]))),
                "p99_abs_256": float(np.percentile(np.abs(x - ref[lo:hi]), 99)),
                "argmax": int(np.argmax(x)),
                "same_argmax": bool(int(np.argmax(x)) == int(np.argmax(ref[lo:hi]))),
                "logits_finite": bool(np.isfinite(x).all())}
        except Exception as e:
            output["cases"][name] = {"error": repr(e),
                                     "traceback": traceback.format_exc()[-4000:]}

    ref_k0 = full_forward(prompt)
    bin0 = int(np.argmax(ref_k0[lo:hi]))
    tok0 = lo + bin0
    ref_k1 = full_forward(prompt + [tok0])
    output = {
        "schema": "r16-kv-first-divergence-v1",
        "source": "pristine OpenVLA full multimodal vs incremental Llama",
        "checkpoint": os.path.abspath(args.checkpoint),
        "modeling_sha256": modeling_sha,
        "task": args.task,
        "episode": episode,
        "t": int(tick),
        "prompt_text_tokens": len(prompt),
        "expected_multimodal_extra_tokens": 769,
        "k0_full_argmax_bin": bin0,
        "k1_ref_argmax_bin": int(np.argmax(ref_k1[lo:hi])),
        "cases": {}}

    compare("full_multimodal_use_cache_true", lambda: full_forward(
        prompt + [tok0], use_cache=True), ref_k1, output)

    def new_prefix():
        lg, cache, n_text = backend.prefill_cached(prompt, pv, feat)
        true_n = actual_cache_length(cache)
        return lg, cache, n_text, true_n

    lg0, cache0, n_text, true_n = new_prefix()
    output["actual_cache_length"] = true_n
    output["backend_reported_n_past"] = n_text
    output["cache_type"] = type(cache0).__name__
    output["text_vs_real_cache_delta"] = true_n - n_text
    output["cache_expected_length"] = len(prompt) + 769
    output["cache_length_correct"] = bool(true_n == len(prompt) + 769)
    output["k0_prefill_max_abs"] = float(np.max(
        np.abs(np.asarray(lg0)[lo:hi] - ref_k0[lo:hi])))

    def step(method):
        _lg, cache, _ntext, L = new_prefix()
        tok = t_ids([tok0])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if method == "direct_old":
                out = vla.language_model(input_ids=tok, past_key_values=cache,
                                          use_cache=True)
            elif method == "direct_full_mask":
                out = vla.language_model(
                    input_ids=tok, past_key_values=cache, use_cache=True,
                    attention_mask=torch.ones((1, L + 1), device="cuda", dtype=torch.long),
                    position_ids=torch.tensor([[L]], device="cuda", dtype=torch.long),
                    cache_position=torch.tensor([L], device="cuda", dtype=torch.long))
            elif method == "wrapper_generation":
                # Exercises custom Prismatic one-token cached branch rather than
                # jumping directly to PeftModel.language_model.
                out = vla(input_ids=tok, past_key_values=cache, use_cache=True,
                          labels=None)
            else:
                raise ValueError(method)
        return out.logits[0, -1].float().cpu().numpy()

    for method in ("direct_old", "direct_full_mask", "wrapper_generation"):
        compare(method, lambda method=method: step(method), ref_k1, output)

    # LM-only control: does the same LM have standard causal cache parity
    # when no multimodal embeddings are inserted?
    def lm_text_full():
        tok = t_ids(prompt + [tok0])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla.language_model(input_ids=tok, use_cache=False)
        return out.logits[0, -1].float().cpu().numpy()

    lm_ref = lm_text_full()

    def lm_text_cached():
        text = t_ids(prompt)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out0 = vla.language_model(input_ids=text, use_cache=True)
            c = out0.past_key_values
            out1 = vla.language_model(input_ids=t_ids([tok0]),
                                       past_key_values=c, use_cache=True)
        return out1.logits[0, -1].float().cpu().numpy()

    compare("text_only_cache_vs_full", lm_text_cached, lm_ref, output)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, args.out)
    print(json.dumps({k: v for k, v in output.items() if k != "cases"}, indent=2))
    print(json.dumps(output["cases"], indent=2))
    # Exits nonzero when a test errors, while preserving all evidence.
    return 5 if any("error" in v for v in output["cases"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
