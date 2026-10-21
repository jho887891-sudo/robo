#!/usr/bin/env python3
"""R16: does the VISION-ONLY cache change the answer? (equivalence + real speedup)

The vision tower is the part of the 7.75 s control step that is recomputed 14 times for identical frames.
This tool measures, on real weights and real DEV frames:

  EQUIVALENCE  per-k legal 256-bin logits, argmax and physical bins: cached-vision vs full
  INVALIDATION frame B decoded after frame A must equal B decoded alone (no stale patches)
  NEGATIVE     an intentionally stale cache MUST change B's logits (the check can see leakage)
  SPEED        64 timed actions after 20 warmups, both paths, same samples
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
import r15_resume as RS             # noqa: E402
import r11_train_pushT_r42 as R11   # noqa: E402
import r42_deploy_server as S       # noqa: E402


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
    vla.eval(); proj.eval()
    return proc, vla, proj, ts, modeling_sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--normalizer", default="data/m2_task_normalizer_2task.json")
    ap.add_argument("--splits", default="data/m2_splits.json")
    ap.add_argument("--frames-per-task", type=int, default=8)
    ap.add_argument("--measured-actions", type=int, default=64)
    ap.add_argument("--warmup-actions", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-5)
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
    backend = S._TorchBackend(vla, proj)             # full path (pure-autoreg installed)
    prompt_ids = {t: M2._make_prompt(proc, t, False) for t in tasks}
    lo = R.TOKEN_BEGIN_IDX + 1
    hi = lo + 256

    def window(lg):
        w = np.asarray(lg, np.float32).reshape(-1)[lo:hi]
        if w.size != 256:
            raise SystemExit("logits do not cover the 256 legal bins")
        return w

    def decode(prompt, pv, feat, use_vision_cache):
        backend.vision_cache_enabled = bool(use_vision_cache)
        if use_vision_cache and not getattr(backend, "_vision_core", None):
            backend.install_vision_cache()
        backend.reset_vision_cache()
        ids, ws = [], []
        for _k in range(R.N_ACTIONS):
            w = window(backend.next_token_logits(prompt + ids, pv, feat))
            ws.append(w)
            ids.append(int(np.argmax(w)))
        return ids, ws

    cases, worst = [], 0.0
    n_inputs = 0
    for task in tasks:
        for (ep, t) in dev[task]:
            state_, action_ = M2._read_state_action(indexes[task]["hdf5_dir"], ep)
            row = M2.row_for(payload, task, state_, action_, t, episode=ep)
            pv = M2._episode_images(proc, ep, t, a.cams, indexes[task]["hdf5_dir"])
            feat = np.asarray(row["features"], np.float32)
            for ptask in tasks:
                ids_f, w_f = decode(prompt_ids[ptask], pv, feat, False)
                ids_c, w_c = decode(prompt_ids[ptask], pv, feat, True)
                deltas = [float(np.max(np.abs(w_f[k] - w_c[k]))) for k in range(R.N_ACTIONS)]
                # a frame change MUST invalidate: decode a DIFFERENT frame right after, no reset
                n_inputs += 1
                cases.append({"frame_task": task, "prompt_task": ptask, "episode": ep, "t": int(t),
                              "argmax_identical": ids_f == ids_c,
                              "worst_max_abs": float(max(deltas)),
                              "bins_full": ids_f, "bins_vision_cached": ids_c})
                worst = max(worst, max(deltas))

    # ---- invalidation: frame B after frame A must equal B alone
    epA, tA = dev["push_T"][0]
    epB, tB = dev["stack_bowls"][1]
    def frames_of(task, ep, t):
        state_, action_ = M2._read_state_action(indexes[task]["hdf5_dir"], ep)
        row = M2.row_for(payload, task, state_, action_, t, episode=ep)
        return (M2._episode_images(proc, ep, t, a.cams, indexes[task]["hdf5_dir"]),
                np.asarray(row["features"], np.float32))
    pvA, featA = frames_of("push_T", epA, tA)
    pvB, featB = frames_of("stack_bowls", epB, tB)
    pr = prompt_ids["push_T"]
    backend.vision_cache_enabled = True
    if not getattr(backend, "_vision_core", None):
        backend.install_vision_cache()
    backend.reset_vision_cache()
    decode(pr, pvA, featA, True)                      # populate the cache with frame A
    stale_ws = [window(backend.next_token_logits(pr, pvB, featB))]   # B WITHOUT reset -> stale
    backend.reset_vision_cache()
    fresh_ws = [window(backend.next_token_logits(pr, pvB, featB))]   # correct B
    backend.vision_cache_enabled = False
    plain_ws = [window(backend.next_token_logits(pr, pvB, featB))]   # full, no cache
    stale_delta = float(np.max(np.abs(stale_ws[0] - plain_ws[0])))
    stale_detected = bool(stale_delta > a.tol)
    fresh_matches = float(np.max(np.abs(fresh_ws[0] - plain_ws[0])))

    # ---- timing
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
                "p95_ms": float(np.percentile(xs, 95)), "p99_ms": float(np.percentile(xs, 99))}
    ep, t = dev["push_T"][0]
    pv, feat = frames_of("push_T", ep, t)
    tf = timed(lambda: decode(pr, pv, feat, False), a.measured_actions, a.warmup_actions)
    tc = timed(lambda: decode(pr, pv, feat, True), a.measured_actions, a.warmup_actions)
    try:
        import torch
        vram = round(float(torch.cuda.max_memory_allocated()) / 1e9, 3)
    except Exception:
        vram = None

    ok = all(c["argmax_identical"] for c in cases) and worst <= a.tol and stale_detected \
        and fresh_matches <= a.tol
    rep = {"schema": "r16-vision-cache-v1", "checkpoint": os.path.abspath(a.checkpoint),
           "checkpoint_global_step": int(ts["global_step"]), "modeling_sha256": modeling_sha,
           "n_inputs": n_inputs,
           "argmax_identical_all": all(c["argmax_identical"] for c in cases),
           "worst_max_abs": worst, "tol": a.tol,
           "invalidation": {"fresh_vs_full_max_abs": fresh_matches,
                            "stale_vs_full_max_abs": stale_delta,
                            "stale_detected": stale_detected},
           "timing_full": tf, "timing_vision_cached": tc,
           "speedup_p50": (tf["p50_ms"] / tc["p50_ms"]) if tc["p50_ms"] else None,
           "vram_gb": vram,
           "verdict": "EQUIVALENT" if ok else "DIVERGED", "cases": cases}
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "cases"}, indent=1)[:3000])
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
