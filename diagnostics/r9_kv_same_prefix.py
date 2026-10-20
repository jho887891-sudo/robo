#!/usr/bin/env python3
"""R9 Gate C: same-prefix KV-cache probe (CPU contract + guarded GPU diagnose).

Why this exists: the R8 GPU smoke compared a *cached* decode against a *full-sequence*
decode; once the two branches picked different tokens their inputs diverged, so later logits
came from different contexts and could not identify a root cause. Here the FULL path token is
teacher-forced into the cached branch at every step: both branches always see the same history.

Verdicts are conservative by construction:
  * 14/14 match -> SAME_PREFIX_EQUIVALENT
  * mismatch with margin <= 2*max|dlogit| -> OPEN (flip is compatible with the measured delta,
    which is NOT proof that bf16 caused it)
  * any other mismatch -> OPEN (structural: cache/position/mask/token)
No root cause is declared here; a precision-controlled A/B is still required.
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

W0 = 31744
BINS = 256
N_STEPS = 14


def _window(logits, window):
    start, size = window
    v = np.asarray(logits, dtype=np.float64).reshape(-1)
    return v[start:start + size]


def same_prefix_probe(prefill_fn, cached_step_fn, full_fn, prefix_ids, n_steps=N_STEPS,
                      window=(W0, BINS)):
    """Teacher-forced comparison of cached vs full-sequence logits."""
    lg_kv = _window(prefill_fn(prefix_ids), window)   # prefill returns last-prefix logits
    ref_tokens = []
    steps = []
    for k in range(n_steps):
        ids_full = list(prefix_ids) + ref_tokens
        lg_full = _window(full_fn(ids_full), window)
        tk = int(np.argmax(lg_kv))
        tf = int(np.argmax(lg_full))
        d = np.abs(lg_kv - lg_full)
        srt = np.sort(lg_full)[::-1]
        steps.append({"step": k, "token_kv": tk, "token_full": tf, "match": tk == tf,
                      "max_abs_dlogit": float(d.max()), "mean_abs_dlogit": float(d.mean()),
                      "top1_minus_top2_full": float(srt[0] - srt[1]),
                      "prefix_len": len(ids_full)})
        if k == n_steps - 1:
            break
        ref_tokens.append(tf)
        lg_kv = _window(cached_step_fn(tf), window)
    return steps


def summarize(steps, precision_note=None, numeric_floor=0.05):
    """Classify WITHOUT declaring a root cause.

    The classic test `margin <= 2*max|dlogit|` is *necessarily* true for any genuine argmax
    flip, so it cannot discriminate numeric from structural. The discriminating quantity is
    the measured logit delta compared with a documented numeric-noise floor.
    """
    mism = [s for s in steps if not s["match"]]
    worst = max([s["max_abs_dlogit"] for s in mism], default=0.0)
    compatible = all(s["top1_minus_top2_full"] <= 2.0 * s["max_abs_dlogit"] + 1e-12 for s in mism)
    if not mism:
        verdict = "SAME_PREFIX_EQUIVALENT"
        cls = "none"
    elif worst <= numeric_floor:
        verdict = "OPEN_NUMERIC_PLAUSIBLE: flip within measured delta (<= %.3f); bf16 plausible NOT proven" % numeric_floor
        cls = "numeric_plausible"
    else:
        verdict = "OPEN_STRUCTURAL_SUSPECT: logit delta %.3f far above numeric scale; inspect cache/position/mask/token" % worst
        cls = "structural_suspect"
    return {"n_match": sum(1 for s in steps if s["match"]), "n_total": len(steps),
            "first_mismatch_step": next((s["step"] for s in mism), None),
            "mismatches_compatible_with_delta": bool(compatible),
            "median_margin": float(np.median([s["top1_minus_top2_full"] for s in steps])),
            "median_max_dlogit": float(np.median([s["max_abs_dlogit"] for s in steps])),
            "verdict": verdict, "mismatch_class": cls,
            "max_abs_dlogit_at_mismatch": float(worst), "numeric_floor": float(numeric_floor),
            "precision_note": precision_note, "steps": steps,
            "same_prefix_teacher_forced": True}


def exit_code(summary):
    """Non-zero when tokens differ: a KV mismatch must never look like a pass."""
    return 0 if summary["n_match"] == summary["n_total"] else 6


REQUIRED_PATTERNS = {
    "cache_is_threaded": r"past_key_values\s*=\s*cached",
    "teacher_forces_full_token": r"\[\[W0 \+ tf\]\]",
    "step0_uses_prefix_only": r"if k == 0",
    "window_is_256_action_tokens": r"W0:W0 \+ BINS",
    "nonzero_exit_on_mismatch": r"else 6",
}
FORBIDDEN_PATTERNS = {
    "advance_cache_with_cached_token": r"\[\[W0 \+ tk\]\]",
}


def static_audit(path):
    src = open(path).read()
    res = {"file": os.path.abspath(path),
           "sha256": hashlib.sha256(src.encode()).hexdigest(),
           "required": {}, "forbidden": {}}
    for name, pat in REQUIRED_PATTERNS.items():
        res["required"][name] = {"found": bool(re.search(pat, src)), "pattern": pat}
    for name, pat in FORBIDDEN_PATTERNS.items():
        res["forbidden"][name] = {"found": bool(re.search(pat, src)), "pattern": pat}
    res["pass"] = (all(v["found"] for v in res["required"].values())
                   and not any(v["found"] for v in res["forbidden"].values()))
    return res


def cpu_selfcheck(window_start=0, window_size=8, n_steps=8, out_dir=None):
    """Same-prefix equivalence on a tiny CPU Llama plus protocol-violation controls."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
                      attn_implementation="eager")
    model = LlamaForCausalLM(cfg).eval()
    window = (window_start, window_size)
    results = {}
    prefix_ids = [3, 9, 14, 27, 31, 5]

    def make(shift_positions=0, extra_cache_token=False):
        state = {"cache": None, "poisoned": False}

        def prefill(ids_prefix):
            ids = torch.tensor([ids_prefix], dtype=torch.long)
            with torch.no_grad():
                o = model(input_ids=ids, use_cache=True)
            state["cache"] = o.past_key_values
            return o.logits[0, -1].float().numpy()

        def cached_step(token_id):
            ids = torch.tensor([[token_id]], dtype=torch.long)
            cache = state["cache"]
            pos = None
            if shift_positions:
                base = state["cache"][0][0].shape[-2]
                pos = torch.tensor([[base + shift_positions]], dtype=torch.long)
            if extra_cache_token and not state["poisoned"]:
                new = []
                for layer in cache:
                    k, v = layer[0], layer[1]
                    pad_k = torch.zeros_like(k[:, :, :1, :])
                    pad_v = torch.zeros_like(v[:, :, :1, :])
                    new.append((torch.cat([k, pad_k], dim=2), torch.cat([v, pad_v], dim=2)))
                cache = tuple(new)
                state["poisoned"] = True
            with torch.no_grad():
                o = model(input_ids=ids, past_key_values=cache, use_cache=True,
                          position_ids=pos)
            state["cache"] = o.past_key_values
            return o.logits[0, -1].float().numpy()

        def full(seq):
            ids = torch.tensor([seq], dtype=torch.long)
            with torch.no_grad():
                o = model(input_ids=ids, use_cache=False)
            return o.logits[0, -1].float().numpy()

        return prefill, cached_step, full

    pre, step, full = make()
    steps = same_prefix_probe(pre, step, full, prefix_ids, n_steps=n_steps, window=window)
    results["same_prefix_fp32"] = summarize(steps, "fp32 CPU tiny Llama, identical prefix")

    for name, kwargs in (("shifted_position_ids", {"shift_positions": 3}),
                         ("cache_offset_injection", {"extra_cache_token": True})):
        pre, step, full = make(**kwargs)
        s = same_prefix_probe(pre, step, full, prefix_ids, n_steps=n_steps, window=window)
        results[name] = summarize(s, "negative control (protocol violation)")

    with torch.no_grad():
        o32 = model(input_ids=torch.tensor([prefix_ids], dtype=torch.long), use_cache=False)
        m16 = model.to(torch.bfloat16)
        o16 = m16(input_ids=torch.tensor([prefix_ids], dtype=torch.long), use_cache=False)
    lg32 = _window(o32.logits[0, -1].float().numpy(), window)
    lg16 = _window(o16.logits[0, -1].float().numpy(), window)
    d = np.abs(lg32 - lg16)
    srt = np.sort(lg16)[::-1]
    results["bf16_vs_fp32_precision"] = {
        "max_abs_dlogit": float(d.max()), "mean_abs_dlogit": float(d.mean()),
        "top1_minus_top2": float(srt[0] - srt[1]),
        "flip_possible": bool(srt[0] - srt[1] <= 2.0 * float(d.max())),
        "verdict": "compatibility only - NOT a root cause", "steps": []}

    clean = results["same_prefix_fp32"]
    clean_ok = (clean["n_match"] == clean["n_total"] and clean["median_max_dlogit"] < 1e-6)
    controls = {}
    for name in ("shifted_position_ids", "cache_offset_injection"):
        r = results[name]
        # a control is "detected" if either the argmax flips or the logits move far beyond
        # the clean-path noise floor (the probe must not be blind to corruption)
        detected = bool(r["n_match"] < r["n_total"]
                        or r["median_max_dlogit"] > 100.0 * max(clean["median_max_dlogit"], 1e-12))
        controls[name] = {"detected": detected, "n_match": r["n_match"],
                          "median_max_dlogit": r["median_max_dlogit"],
                          "noise_floor": clean["median_max_dlogit"]}
    results["controls"] = controls
    results["pass"] = bool(clean_ok and all(v["detected"] for v in controls.values()))
    results["window"] = {"start": window_start, "size": window_size}
    results["torch"] = torch.__version__
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "r9_kv_cpu_selfcheck.json"), "w") as fh:
            json.dump(results, fh, indent=1)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-selfcheck", action="store_true")
    ap.add_argument("--static-audit", default="")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--gpu-diagnose", action="store_true")
    ap.add_argument("--norm-contract", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.plan:
        info = {"cpu": "same-prefix probe + protocol controls + static audit of the GPU probe",
                "gpu": "requires R9_KV_GPU_AUTHORIZED=1 and >= 24 GB free VRAM, one bounded window",
                "gpu_command": "python diagnostics/r8_smoke.py --kv-diagnose --norm-contract <frozen> --min-free-gb 24 --out <dir>",
                "never": "no training, no optimizer step"}
        print(json.dumps(info, indent=1))
        return 0
    rc = 0
    if args.static_audit:
        audit = static_audit(args.static_audit)
        print(json.dumps(audit, indent=1))
        rc |= 0 if audit["pass"] else 3
    if args.cpu_selfcheck:
        res = cpu_selfcheck(out_dir=args.out or None)
        brief = {}
        for k, v in res.items():
            brief[k] = ({kk: vv for kk, vv in v.items() if kk != "steps"}
                        if isinstance(v, dict) else v)
        print(json.dumps(brief, indent=1)[:2500])
        rc |= 0 if res["pass"] else 4
    if args.gpu_diagnose:
        if os.environ.get("R9_KV_GPU_AUTHORIZED") != "1":
            print("refusing GPU KV diagnosis: R9_KV_GPU_AUTHORIZED is not set by the research lead",
                  file=sys.stderr)
            return 7
        print("authorised: run diagnostics/r8_smoke.py --kv-diagnose (bounded, one window)",
              file=sys.stderr)
        return 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
