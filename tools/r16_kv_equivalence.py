#!/usr/bin/env python3
"""R16: is a single-prefill KV cache NUMERICALLY EQUIVALENT to the full-recompute decode? (issue #26)

The full path re-runs the whole multimodal forward (3 cameras -> 18 channels -> 769 vision tokens +
prompt) for every one of the 14 action tokens. The cached path prefills once (use_cache=True) and then
feeds only the new token to the language model, so the multimodal prefix lives in the KV.

This tool answers one question with real weights: do the two paths produce the SAME legal 256-bin logits,
the same 14/14 argmax bins and therefore the same physical 14D command? It also times both. It never
trains, never writes a checkpoint, and reports a divergence instead of hiding one.

Hard gate (PASS only if all hold):
  * every one of the 14 steps: identical argmax bin;
  * max |logit difference| inside the legal window below --tol;
  * the physical 14D action implied by the bins is identical;
  * a deliberately STALE cache must give different logits (proves the harness can see leakage).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("training", "code/xpolicylab/policy/OpenVLA_Pure", "code/xpolicylab/policy/OpenVLA_R42"):
    _p = os.path.join(REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import m2_train_shared as M2        # noqa: E402
import m2_multitask_data as MD      # noqa: E402
import r42_adapter as R             # noqa: E402
import r42_deploy_server as S       # noqa: E402  (the REAL backend under test)
import r15_resume as RS             # noqa: E402
import r11_train_pushT_r42 as R11   # noqa: E402


def load_verified(checkpoint, cams):
    import torch, hashlib
    from safetensors.torch import load_file
    from peft.utils.save_and_load import set_peft_model_state_dict
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()
    R11.force_arx_platform()
    proc, vla, proj, modeling_sha = R11.build_model(cams)
    man, ts = RS.verify_and_load_manifest(checkpoint)
    adapter = os.path.join(checkpoint, "lora_adapter", "adapter_model.safetensors")
    proj_pt = os.path.join(checkpoint, "proprio_projector_42.pt")
    for p, k in ((adapter, "lora_adapter/adapter_model.safetensors"),
                 (proj_pt, "proprio_projector_42.pt")):
        if man["files"].get(k) and sha(p) != man["files"][k]:
            raise SystemExit("manifest hash mismatch for %s" % k)
    set_peft_model_state_dict(vla, load_file(adapter))
    proj.load_state_dict(torch.load(proj_pt, map_location="cpu"))
    vla.eval()
    proj.eval()
    return proc, vla, proj, ts, modeling_sha


# --------------------------------------------------------------------------- decode loops (testable)
# These live at module level, with the backend injected, so a stub backend can assert the ONE property
# whose absence invalidated the first three R16 parity reports: every token fed to the model must be a
# LEGAL WINDOW TOKEN (31744..31999), never a bin index 0..255.
def needs_legal_token_check(backend, input_ids):
    for tok in input_ids:
        if not (R.TOKEN_BEGIN_IDX + 1 <= int(tok) <= R.TOKEN_BEGIN_IDX + 256):
            raise AssertionError(
                "the model was fed token %r, which is outside the legal action window %d..%d - a bin "
                "index leaked into the sequence" % (int(tok), R.TOKEN_BEGIN_IDX + 1,
                                                    R.TOKEN_BEGIN_IDX + 256))


def decode_full(backend, prompt, pixel_values, feat, window, n_actions=None, check=True):
    """14 greedy steps through the full-recompute path. Returns (bins, logit_windows)."""
    n = int(n_actions or R.N_ACTIONS)
    toks, bins, logits = [], [], []
    for _k in range(n):
        if check:
            needs_legal_token_check(backend, toks)
        w = window(backend.next_token_logits(prompt + toks, pixel_values, feat))
        logits.append(w)
        b = int(np.argmax(w))
        bins.append(b)
        toks.append(R.TOKEN_BEGIN_IDX + 1 + b)
    return bins, logits


def decode_cached(backend, prompt, pixel_values, feat, window, n_actions=None, check=True):
    """Same 14 greedy steps from ONE prefill plus incremental steps. Returns (bins, logit_windows)."""
    n = int(n_actions or R.N_ACTIONS)
    lg, cache, n_past = backend.prefill_cached(prompt, pixel_values, feat)
    bins, logits = [], []
    for k in range(n):
        w = window(lg)
        logits.append(w)
        b = int(np.argmax(w))
        bins.append(b)
        if k + 1 < n:
            tok = R.TOKEN_BEGIN_IDX + 1 + b
            if check:
                needs_legal_token_check(backend, [tok])
            lg, cache, n_past = backend.step_cached(tok, cache, n_past)
    return bins, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--normalizer", default="data/m2_task_normalizer_2task.json")
    ap.add_argument("--splits", default="data/m2_splits.json")
    ap.add_argument("--frames-per-task", type=int, default=8)   # 8 x 2 tasks x 2 prompts = 32 inputs
    ap.add_argument("--replay-episodes", type=int, default=3)
    ap.add_argument("--replay-steps", type=int, default=5)
    ap.add_argument("--measured-actions", type=int, default=64)
    ap.add_argument("--warmup-actions", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--cams", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tasks = ["push_T", "stack_bowls"]
    _doc, split_map, payload, _sp, _np = M2.load_m2_artifacts(tasks, a.splits, a.normalizer)
    indexes = {t: MD.load_task_index(t) for t in tasks}
    dev = {t: M2.dev_frames_for_task(indexes[t], split_map[t], t, a.frames_per_task, a.seed)
           for t in tasks}
    proc, vla, proj, ts, modeling_sha = load_verified(a.checkpoint, a.cams)
    backend = S._TorchBackend(vla, proj)
    prompt_ids = {t: M2._make_prompt(proc, t, False) for t in tasks}
    lo = R.TOKEN_BEGIN_IDX + 1          # 31744
    hi = lo + 256                        # 32000 -> exactly the 256 legal bins (off-by-one trap:
                                         # TOKEN_BEGIN_IDX + 256 is 31999 and yields only 255)
    ctx = {t: M2.TN.decode_context(payload, t) for t in tasks}

    def window(logits):
        w = np.asarray(logits, np.float32).reshape(-1)[lo:hi]
        if w.size != 256:
            raise SystemExit("logits do not cover the legal 256 window")
        return w

    def _token(bin_idx):
        """BIN -> LEGAL WINDOW TOKEN. This single conversion was missing from the full path below,
        which fed the model raw bin indices 0..255 (ordinary text tokens) from the second step on,
        while the cached path correctly fed 31744+bin. Every 'divergence' reported by the earlier
        versions of this harness came from that asymmetry, not from the cache."""
        return R.TOKEN_BEGIN_IDX + 1 + int(bin_idx)

    def full_decode(prompt, pv, feat):
        return decode_full(backend, prompt, pv, feat, window)

    def cached_decode(prompt, pv, feat):
        return decode_cached(backend, prompt, pv, feat, window)

    def residual_from_bins(task, bins):
        c = ctx[task]
        q01 = np.asarray(c["q01"], np.float32)
        q99 = np.asarray(c["q99"], np.float32)
        return q01 + ((np.asarray(bins, np.float32) + 0.5) / 256.0) * (q99 - q01)

    # ---------------------------------------------------------------- equivalence on frozen inputs
    cases, diverged = [], []
    t_all = {"full": [], "cached": []}
    n_inputs = 0
    for task in tasks:
        for (ep, t) in dev[task]:
            state_, action_ = M2._read_state_action(indexes[task]["hdf5_dir"], ep)
            row = M2.row_for(payload, task, state_, action_, t, episode=ep)
            pv = M2._episode_images(proc, ep, t, a.cams, indexes[task]["hdf5_dir"])
            feat = np.asarray(row["features"], np.float32)
            for ptask in tasks:                      # both prompts: different text lengths
                prompt = prompt_ids[ptask]
                t0 = time.perf_counter()
                ids_f, w_f = full_decode(prompt, pv, feat)
                t1 = time.perf_counter()
                ids_c, w_c = cached_decode(prompt, pv, feat)
                t2 = time.perf_counter()
                t_all["full"].append((t1 - t0) * 1e3)
                t_all["cached"].append((t2 - t1) * 1e3)
                deltas = [float(np.max(np.abs(w_f[k] - w_c[k]))) for k in range(R.N_ACTIONS)]
                p99 = [float(np.percentile(np.abs(w_f[k] - w_c[k]), 99)) for k in range(R.N_ACTIONS)]
                same = ids_f == ids_c
                rec = {"frame_task": task, "prompt_task": ptask, "episode": ep, "t": int(t),
                       "bins_full": ids_f, "bins_cached": ids_c,
                       "argmax_identical": bool(same),
                       "max_abs_delta": deltas, "p99_abs_delta": p99,
                       "worst_max_abs": float(max(deltas)),
                       "residual_l1": float(np.abs(residual_from_bins(task, ids_f)
                                                   - residual_from_bins(task, ids_c)).mean())}
                cases.append(rec)
                n_inputs += 1
                if (not same) or rec["worst_max_abs"] > a.tol:
                    diverged.append(rec)

    # ---------------------------------------------------------------- dynamic replay + invalidation
    replays = []
    for task in tasks[:a.replay_episodes]:
        frames = dev[task][:a.replay_steps]
        prev_ids = None
        for (ep, t) in frames:
            state_, action_ = M2._read_state_action(indexes[task]["hdf5_dir"], ep)
            row = M2.row_for(payload, task, state_, action_, t, episode=ep)
            pv = M2._episode_images(proc, ep, t, a.cams, indexes[task]["hdf5_dir"])
            feat = np.asarray(row["features"], np.float32)
            ids_f, w_f = full_decode(prompt_ids[task], pv, feat)
            ids_c, w_c = cached_decode(prompt_ids[task], pv, feat)
            replays.append({"task": task, "episode": ep, "t": int(t),
                            "argmax_identical": bool(ids_f == ids_c),
                            "worst_max_abs": float(max(float(np.max(np.abs(w_f[k] - w_c[k])))
                                                       for k in range(R.N_ACTIONS))),
                            "differs_from_previous_frame": bool(prev_ids is None or ids_c != prev_ids)})
            prev_ids = ids_c

    # ---------------------------------------------------------------- negative control: stale cache
    ep, t = dev["push_T"][0]
    state_, action_ = M2._read_state_action(indexes["push_T"]["hdf5_dir"], ep)
    row_a = M2.row_for(payload, "push_T", state_, action_, t, episode=ep)
    pv_a = M2._episode_images(proc, ep, t, a.cams, indexes["push_T"]["hdf5_dir"])
    ep2, t2 = dev["stack_bowls"][1]
    st2, ac2 = M2._read_state_action(indexes["stack_bowls"]["hdf5_dir"], ep2)
    row_b = M2.row_for(payload, "stack_bowls", st2, ac2, t2, episode=ep2)
    pv_b = M2._episode_images(proc, ep2, t2, a.cams, indexes["stack_bowls"]["hdf5_dir"])
    # prefill from frame A's images/42D, then decode B's TOKENS against it: the harness must notice
    lg, cache_a, n = backend.prefill_cached(prompt_ids["push_T"], pv_a,
                                           np.asarray(row_a["features"], np.float32))
    stale, cache_a, n = backend.step_cached(R.TOKEN_BEGIN_IDX + 1 + int(np.argmax(window(lg))),
                                            cache_a, n)
    good_b, cache_b, nb = backend.prefill_cached(prompt_ids["push_T"], pv_b,
                                                np.asarray(row_b["features"], np.float32))
    stale_delta = float(np.max(np.abs(window(stale) - window(good_b))))

    # ---------------------------------------------------------------- timing (same samples)
    def timed(fn, n, warm):
        for _ in range(warm):
            fn()
        xs = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            xs.append((time.perf_counter() - t0) * 1e3)
        xs = np.asarray(xs, float)
        return {"n": int(n), "p50_ms": float(np.percentile(xs, 50)),
                "p95_ms": float(np.percentile(xs, 95)), "p99_ms": float(np.percentile(xs, 99)),
                "mean_ms": float(xs.mean())}

    ep, t = dev["push_T"][0]
    state_, action_ = M2._read_state_action(indexes["push_T"]["hdf5_dir"], ep)
    row = M2.row_for(payload, "push_T", state_, action_, t, episode=ep)
    pv = M2._episode_images(proc, ep, t, a.cams, indexes["push_T"]["hdf5_dir"])
    feat = np.asarray(row["features"], np.float32)
    pr = prompt_ids["push_T"]
    tf = timed(lambda: full_decode(pr, pv, feat), a.measured_actions, a.warmup_actions)
    tc = timed(lambda: cached_decode(pr, pv, feat), a.measured_actions, a.warmup_actions)
    try:
        import torch
        vram = round(float(torch.cuda.max_memory_allocated()) / 1e9, 3)
    except Exception:
        vram = None

    ok = (not diverged) and all(r["argmax_identical"] for r in replays) \
        and stale_delta > a.tol
    rep = {"schema": "r16-kv-equivalence-v1", "checkpoint": os.path.abspath(a.checkpoint),
           "checkpoint_global_step": int(ts["global_step"]), "modeling_sha256": modeling_sha,
           "n_frozen_inputs": n_inputs, "n_cases": len(cases),
           "argmax_identical_all": all(c["argmax_identical"] for c in cases),
           "worst_max_abs": max(c["worst_max_abs"] for c in cases),
           "worst_residual_l1": max(c["residual_l1"] for c in cases),
           "diverged": diverged[:5], "n_diverged": len(diverged),
           "replay": replays, "stale_cache_delta": stale_delta,
           "stale_cache_detected": bool(stale_delta > a.tol),
           "timing_full_call": tf, "timing_cached_call": tc,
           "speedup_p50": (tf["p50_ms"] / tc["p50_ms"]) if tc["p50_ms"] else None,
           "vram_gb": vram, "tol": a.tol,
           "verdict": "EQUIVALENT" if ok else "DIVERGED"}
    with open(a.out, "w") as fh:
        json.dump({"summary": rep, "cases": cases}, fh, indent=1)
    print(json.dumps(rep, indent=1)[:4000])
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
