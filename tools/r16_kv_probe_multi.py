#!/usr/bin/env python3
"""R16 multi-frame, multi-method probe: which call builds an INCOMPATIBLE cache?

The single-frame probe (PR #28) found every k=1 path agreeing (0.031 = bf16 noise) on the frame it
happened to pick, while the 32-input parity run diverged on 32/32 of the same DEV frames. So the question
is no longer "is it the mask/position" - it is "WHICH cache is incompatible". This probe answers that by
building the cache with different callers and stepping each with the SAME incremental call:

  A  direct_old        prefill via vla() [reduced], step via language_model, no explicit mask/positions
  B  direct_full_mask  same cache, step with explicit full attention mask + position_ids + cache_position
  C  via_vla_cache     cache produced by vla() over prompt+tok0 (i.e. the FULL multimodal forward),
                       then ONE language_model step for tok1   <- isolates caller-vs-caller mismatch
  D  layer audit       per-decoder-layer input hidden state, full(prompt+tok0) vs prefill+step

Every method is compared against the full forward at the same forced token sequence, so a difference is
attributable to the cache path and not to token drift.
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import r16_kv_equivalence as KV  # noqa: E402

M2, MD, R, S = KV.M2, KV.MD, KV.R, KV.S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--normalizer", default="data/m2_task_normalizer_2task.json")
    ap.add_argument("--splits", default="data/m2_splits.json")
    ap.add_argument("--cases", required=True, help="JSON list of [frame_task, prompt_task, episode, t]")
    ap.add_argument("--layer-audit-case", type=int, default=0)
    ap.add_argument("--cams", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    cases = json.load(open(a.cases))
    tasks = ["push_T", "stack_bowls"]
    indexes = {t: MD.load_task_index(t) for t in tasks}
    _doc, split, payload, _sp, _np = M2.load_m2_artifacts(tasks, a.splits, a.normalizer)
    proc, vla, proj, ts, modeling_sha = KV.load_verified(a.checkpoint, a.cams)
    backend = S._TorchBackend(vla, proj)
    prompt_ids = {t: M2._make_prompt(proc, t, False) for t in tasks}
    lo = R.TOKEN_BEGIN_IDX + 1
    hi = lo + 256

    def win(x):
        w = np.asarray(x, np.float32).reshape(-1)[lo:hi]
        if w.size != 256:
            raise SystemExit("logits do not cover the 256 legal bins")
        return w

    def prep(frame_task, episode, t):
        state, action = M2._read_state_action(indexes[frame_task]["hdf5_dir"], episode)
        row = M2.row_for(payload, frame_task, state, action, t, episode=episode)
        pv = M2._episode_images(proc, episode, t, a.cams, indexes[frame_task]["hdf5_dir"])
        return pv, np.asarray(row["features"], np.float32)

    def full(ids, pv, feat, cache_flag=None, want_hidden=False):
        inp = torch.tensor([list(ids)], dtype=torch.long, device="cuda")
        pr = torch.from_numpy(feat).reshape(1, 42).to("cuda")
        kw = {} if cache_flag is None else {"use_cache": cache_flag}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                      proprio=pr, proprio_projector=proj, labels=inp.clone(),
                      output_hidden_states=want_hidden, **kw)
        return out

    results = {"schema": "r16-kv-multiprobe-v1", "checkpoint": os.path.abspath(a.checkpoint),
               "modeling_sha256": modeling_sha, "cases": []}

    for ci, (ftask, ptask, ep, t) in enumerate(cases):
        pv, feat = prep(ftask, ep, int(t))
        prompt = prompt_ids[ptask]
        o0 = full(prompt, pv, feat)
        w0 = win(o0.logits[0, -1].float().cpu().numpy())
        tok0 = lo + int(np.argmax(w0))
        o1 = full(prompt + [tok0], pv, feat)
        ref1 = win(o1.logits[0, -1].float().cpu().numpy())
        tok1 = lo + int(np.argmax(ref1))
        o2 = full(prompt + [tok0, tok1], pv, feat)
        ref2 = win(o2.logits[0, -1].float().cpu().numpy())
        rec = {"case": ci, "frame_task": ftask, "prompt_task": ptask, "episode": ep, "t": int(t),
               "prompt_text_tokens": len(prompt), "tok0": tok0, "ref1_bin": int(tok1 - lo),
               "ref2_bin": int(np.argmax(ref2)), "methods": {}}

        def add(name, vec, ref):
            x = win(vec)
            rec["methods"][name] = {"max_abs": float(np.max(np.abs(x - ref))),
                                    "p99_abs": float(np.percentile(np.abs(x - ref), 99)),
                                    "same_argmax": bool(int(np.argmax(x)) == int(np.argmax(ref)))}

        # A / B share the same prefill cache
        try:
            lg0, cache0, n_text = backend.prefill_cached(prompt, pv, feat)
            L = int(cache0[0][0].shape[-2]) if isinstance(cache0, tuple) else int(cache0.get_seq_length())
            rec["cache_len"] = L
            rec["cache_expected"] = len(prompt) + 769
            tok = torch.tensor([[tok0]], dtype=torch.long, device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                oA = vla.language_model(input_ids=tok, past_key_values=cache0, use_cache=True)
            add("direct_old", oA.logits[0, -1].float().cpu().numpy(), ref1)
            cache0b = cache0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                oB = vla.language_model(
                    input_ids=tok, past_key_values=cache0b, use_cache=True,
                    attention_mask=torch.ones((1, L + 1), device="cuda", dtype=torch.long),
                    position_ids=torch.tensor([[L]], device="cuda", dtype=torch.long),
                    cache_position=torch.tensor([L], device="cuda", dtype=torch.long))
            add("direct_full_mask", oB.logits[0, -1].float().cpu().numpy(), ref1)
        except Exception:
            rec["methods"]["A_B_error"] = traceback.format_exc()[-1500:]

        # C: cache BUILT BY THE MULTIMODAL FORWARD over prompt+tok0, then one LM step for tok1
        try:
            oC0 = full(prompt + [tok0], pv, feat, cache_flag=True)
            cacheC = oC0.past_key_values
            LC = int(cacheC[0][0].shape[-2]) if isinstance(cacheC, tuple) else int(cacheC.get_seq_length())
            rec["vla_cache_len"] = LC
            tokB = torch.tensor([[tok1]], dtype=torch.long, device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                oC = vla.language_model(input_ids=tokB, past_key_values=cacheC, use_cache=True)
            add("via_vla_cache", oC.logits[0, -1].float().cpu().numpy(), ref2)
        except Exception:
            rec["methods"]["C_error"] = traceback.format_exc()[-1500:]

        results["cases"].append(rec)
        print("[probe] %s/%s %s t=%s  A=%s B=%s C=%s" % (
            ftask, ptask, ep, t,
            rec["methods"].get("direct_old", {}).get("max_abs"),
            rec["methods"].get("direct_full_mask", {}).get("max_abs"),
            rec["methods"].get("via_vla_cache", {}).get("max_abs")), flush=True)

    # ---- layer audit on the requested case: full(prompt+tok0) vs prefill+step at the last position
    ci = int(a.layer_audit_case)
    ftask, ptask, ep, t = cases[ci]
    pv, feat = prep(ftask, ep, int(t))
    prompt = prompt_ids[ptask]
    w0 = win(full(prompt, pv, feat).logits[0, -1].float().cpu().numpy())
    tok0 = lo + int(np.argmax(w0))
    caps = {"full": [], "step": []}
    layers = vla.language_model.model.layers

    def make_hook(tag):
        def hook(_m, args, kwargs, output):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is not None:
                caps[tag].append(hs.detach().float().cpu().numpy()[:, -1, :].copy())
        return hook
    hs_handle = [l.register_forward_hook(make_hook("full"), with_kwargs=True) for l in layers]
    with torch.no_grad():
        full(prompt + [tok0], pv, feat)
    for h in hs_handle:
        h.remove()
    hs_handle = [l.register_forward_hook(make_hook("step"), with_kwargs=True) for l in layers]
    lg0, cache, _n = backend.prefill_cached(prompt, pv, feat)
    tok = torch.tensor([[tok0]], dtype=torch.long, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vla.language_model(input_ids=tok, past_key_values=cache, use_cache=True)
    for h in hs_handle:
        h.remove()
    audit = []
    for i in range(min(len(caps["full"]), len(caps["step"]))):
        d = np.abs(caps["full"][i] - caps["step"][i])
        audit.append({"layer": i, "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                      "p99_abs": float(np.percentile(d, 99))})
    first = next((x["layer"] for x in audit if x["max_abs"] > 0.05), None)
    results["layer_audit"] = {"case": ci, "first_layer_over_0.05": first, "layers": audit}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, a.out)
    print(json.dumps({"layer_audit": {"first_layer_over_0.05": first,
                                      "first3": audit[:3]}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
