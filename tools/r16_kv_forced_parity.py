#!/usr/bin/env python3
"""R16 teacher-forced per-k parity: is the remaining difference bf16 noise or semantics?

Free-running decode lets each path pick its own token, so ONE borderline argmax early in the sequence
cascades into a large late divergence and tells you nothing about the cache. This tool removes the
cascade: the cache path is FORCED to consume the token sequence the full path chose, and the two are
compared per position k = 0..13.

Then, separately, it quantifies what a free-running flip actually costs in PHYSICS: the 14D residual
implied by each path's own bins.

Verdict:
  NOISE_LEVEL_PARITY  every forced k within --tol of the full path AND the physical delta is sub-micron
  SEMANTIC_DIVERGENCE at least one forced k exceeds --tol (a real mechanism error, not rounding)
"""
import argparse
import json
import os
import sys

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
    ap.add_argument("--frames-per-task", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.5, help="absolute logit tolerance for bf16 parity")
    ap.add_argument("--cams", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tasks = ["push_T", "stack_bowls"]
    _doc, split, payload, _sp, _np = M2.load_m2_artifacts(tasks, a.splits, a.normalizer)
    indexes = {t: MD.load_task_index(t) for t in tasks}
    dev = {t: M2.dev_frames_for_task(indexes[t], split[t], t, a.frames_per_task, a.seed)
           for t in tasks}
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

    def ctx(task):
        c = M2.TN.decode_context(payload, task)
        return np.asarray(c["q01"], np.float32), np.asarray(c["q99"], np.float32)

    def residual_bins(task, bins):
        q01, q99 = ctx(task)
        return q01 + ((np.asarray(bins, np.float32) + 0.5) / 256.0) * (q99 - q01)

    per_k = {k: {"max_abs": [], "p99_abs": [], "argmax_same": []} for k in range(R.N_ACTIONS)}
    cases = []
    for task in tasks:
        for (ep, t) in dev[task]:
            state_, action_ = M2._read_state_action(indexes[task]["hdf5_dir"], ep)
            row = M2.row_for(payload, task, state_, action_, t, episode=ep)
            pv = M2._episode_images(proc, ep, t, a.cams, indexes[task]["hdf5_dir"])
            feat = np.asarray(row["features"], np.float32)
            for ptask in tasks:
                prompt = prompt_ids[ptask]
                # 1) free-running: the full path decides the tokens
                bins_f, w_f = KV.decode_full(backend, prompt, pv, feat, win)
                toks = [R.TOKEN_BEGIN_IDX + 1 + b for b in bins_f]
                # 2) teacher-forced cache path: SAME tokens, one prefill + 13 steps
                lg, cache, n = backend.prefill_cached(prompt, pv, feat)
                bins_c, w_c = [], []
                for k in range(R.N_ACTIONS):
                    w = win(lg)
                    bins_c.append(int(np.argmax(w)))
                    w_c.append(w)
                    if k + 1 < R.N_ACTIONS:
                        lg, cache, n = backend.step_cached(toks[k], cache, n)
                deltas = [float(np.max(np.abs(w_f[k] - w_c[k]))) for k in range(R.N_ACTIONS)]
                p99s = [float(np.percentile(np.abs(w_f[k] - w_c[k]), 99)) for k in range(R.N_ACTIONS)]
                for k in range(R.N_ACTIONS):
                    per_k[k]["max_abs"].append(deltas[k])
                    per_k[k]["p99_abs"].append(p99s[k])
                    per_k[k]["argmax_same"].append(bool(bins_c[k] == bins_f[k]))
                # 3) free-running cache path, for the physical cost of any flip
                bins_cf, _ = KV.decode_cached(backend, prompt, pv, feat, win)
                res = float(np.abs(residual_bins(task, bins_f) - residual_bins(task, bins_cf)).max())
                cases.append({"frame_task": task, "prompt_task": ptask, "episode": ep, "t": int(t),
                              "forced_max_abs_per_k": deltas,
                              "forced_argmax_same_per_k": [bool(bins_c[k] == bins_f[k])
                                                           for k in range(R.N_ACTIONS)],
                              "free_run_tokens_equal": bool(bins_cf == bins_f),
                              "free_run_token_flips": int(sum(1 for x, y in zip(bins_cf, bins_f)
                                                              if x != y)),
                              "physical_residual_max_abs": res})
            print("[parity] %s t=%s forced worst=%.4f free flips=%d physical=%.2e"
                  % (task, t, max(cases[-1]["forced_max_abs_per_k"]),
                     cases[-1]["free_run_token_flips"], cases[-1]["physical_residual_max_abs"]),
                  flush=True)

    stats = {str(k): {"max_abs": float(np.max(v["max_abs"])),
                      "mean_max_abs": float(np.mean(v["max_abs"])),
                      "p99_of_p99": float(np.percentile(v["p99_abs"], 99)),
                      "argmax_same_all": bool(all(v["argmax_same"])),
                      "argmax_same_count": int(sum(v["argmax_same"])),
                      "of": len(v["argmax_same"])}
             for k, v in per_k.items()}
    worst_forced = max(s["max_abs"] for s in stats.values())
    worst_phys = max(c["physical_residual_max_abs"] for c in cases)
    flips = sum(c["free_run_token_flips"] for c in cases)
    ok = worst_forced <= a.tol
    rep = {"schema": "r16-kv-forced-parity-v1", "checkpoint": os.path.abspath(a.checkpoint),
           "modeling_sha256": modeling_sha, "n_inputs": len(cases), "tol": a.tol,
           "per_k": stats, "worst_forced_max_abs": worst_forced,
           "worst_physical_residual_max_abs": worst_phys,
           "free_running_token_flips_total": flips, "of_tokens": len(cases) * R.N_ACTIONS,
           "verdict": ("NOISE_LEVEL_PARITY" if ok else "SEMANTIC_DIVERGENCE"),
           "cases": cases}
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "cases"}, indent=1)[:3000])
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
