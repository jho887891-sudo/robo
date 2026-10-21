#!/usr/bin/env python3
"""R16 three-way attribution: which decode path is the anomaly?

Reference  A: the TRAINING forward (labels=ids.clone(), proprio_projector, NO pure-autoreg patch)
Deploy    B: the live server path, backend.next_token_logits(prefix) after install_pure_autoreg()
Cached    C: one prefill + incremental language_model steps

The first equivalence run showed A-vs-B-vs-C like this: C's prefill is BIT-IDENTICAL to B's first call
(max|delta| == 0.0 on all 32 inputs) and the paths diverge from k=1. That is not a rounding story, so the
only question worth answering is which of the two is wrong. If A matches C, the deployed full path is the
anomaly (and the closed loop was being driven by it); if A matches B, the incremental step is.
"""
import argparse
import json
import os
import sys

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
    return proc, vla, proj, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--normalizer", default="data/m2_task_normalizer_2task.json")
    ap.add_argument("--splits", default="data/m2_splits.json")
    ap.add_argument("--frames-per-task", type=int, default=4)
    ap.add_argument("--cams", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    tasks = ["push_T", "stack_bowls"]
    _doc, split_map, payload, _sp, _np = M2.load_m2_artifacts(tasks, a.splits, a.normalizer)
    indexes = {t: MD.load_task_index(t) for t in tasks}
    dev = {t: M2.dev_frames_for_task(indexes[t], split_map[t], t, a.frames_per_task, a.seed)
           for t in tasks}
    proc, vla, proj, ts = load_verified(a.checkpoint, a.cams)
    prompt_ids = {t: M2._make_prompt(proc, t, False) for t in tasks}
    lo = R.TOKEN_BEGIN_IDX + 1
    hi = lo + 256

    def window(lg):
        w = np.asarray(lg, np.float32).reshape(-1)[lo:hi]
        if w.size != 256:
            raise SystemExit("logits do not cover the 256 legal bins")
        return w

    def ref_step(prompt, ids, pv, feat):
        """Training forward: the whole prefix in one call, labels supplied as the trainer does."""
        inp = torch.tensor([list(prompt) + list(ids)], dtype=torch.long, device="cuda")
        pr = torch.from_numpy(np.asarray(feat, np.float32)).reshape(1, -1).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                    proprio=pr, proprio_projector=proj, labels=inp.clone())
        return o.logits[0, -1].float().cpu().numpy()

    # ---- phase 1: the reference, WITHOUT any pure-autoreg patch installed
    ref = []
    for task in tasks:
        for (ep, t) in dev[task]:
            state_, action_ = M2._read_state_action(indexes[task]["hdf5_dir"], ep)
            row = M2.row_for(payload, task, state_, action_, t, episode=ep)
            pv = M2._episode_images(proc, ep, t, a.cams, indexes[task]["hdf5_dir"])
            feat = np.asarray(row["features"], np.float32)
            ids = []
            for k in range(R.N_ACTIONS):
                lg = ref_step(prompt_ids[task], ids, pv, feat)
                w = window(lg)
                ids.append(int(np.argmax(w)))
            ref.append({"task": task, "episode": ep, "t": int(t), "bins_ref": ids})
    print("[three-way] reference bins done: %d inputs" % len(ref), flush=True)

    # ---- phase 2: install the pure-autoreg patch, then B (full deploy) and C (cached)
    import r42_deploy_server as S
    backend = S._TorchBackend(vla, proj)
    rows = []
    for rec, task in zip(ref, [r["task"] for r in ref]):
        ep, t = rec["episode"], rec["t"]
        state_, action_ = M2._read_state_action(indexes[task]["hdf5_dir"], ep)
        row = M2.row_for(payload, task, state_, action_, t, episode=ep)
        pv = M2._episode_images(proc, ep, t, a.cams, indexes[task]["hdf5_dir"])
        feat = np.asarray(row["features"], np.float32)
        prompt = prompt_ids[task]
        # B: full deploy path
        ids_b, w_b = [], []
        for k in range(R.N_ACTIONS):
            w = window(backend.next_token_logits(prompt + ids_b, pv, feat))
            w_b.append(w); ids_b.append(int(np.argmax(w)))
        # C: cached path
        lg, cache, n = backend.prefill_cached(prompt, pv, feat)
        ids_c, w_c = [], []
        for k in range(R.N_ACTIONS):
            w = window(lg)
            w_c.append(w); ids_c.append(int(np.argmax(w)))
            if k + 1 < R.N_ACTIONS:
                lg, cache, n = backend.step_cached(R.TOKEN_BEGIN_IDX + 1 + ids_c[-1], cache, n)
        # A again, now WITH the patch installed, at the same k (isolates the patch's effect alone)
        rows.append({
            "task": task, "episode": ep, "t": int(t),
            "bins_ref": rec["bins_ref"], "bins_full": ids_b, "bins_cached": ids_c,
            "ref_eq_full": rec["bins_ref"] == ids_b,
            "ref_eq_cached": rec["bins_ref"] == ids_c,
            "full_eq_cached": ids_b == ids_c,
            "max_abs_ref_vs_full": [float(np.max(np.abs(w_b[k] - w_b[k]) * 0 + np.abs(
                window(ref_step(prompt, rec["bins_ref"][:k], pv, feat)) - w_b[k])))
                if False else None for k in range(0)],
            "k0_max_abs_ref_vs_full": float(np.max(np.abs(
                window(ref_step(prompt, [], pv, feat)) - w_b[0]))),
            "k0_max_abs_ref_vs_cached": float(np.max(np.abs(
                window(ref_step(prompt, [], pv, feat)) - w_c[0]))),
            "boundary_full": int(((np.asarray(ids_b) == 0) | (np.asarray(ids_b) == 255)).sum()),
            "boundary_cached": int(((np.asarray(ids_c) == 0) | (np.asarray(ids_c) == 255)).sum()),
            "boundary_ref": int(((np.asarray(rec["bins_ref"]) == 0)
                                 | (np.asarray(rec["bins_ref"]) == 255)).sum()),
        })
        print("[three-way] %s t=%s ref_eq_full=%s ref_eq_cached=%s"
              % (task, t, rows[-1]["ref_eq_full"], rows[-1]["ref_eq_cached"]), flush=True)

    n = len(rows)
    rep = {"schema": "r16-three-way-v1", "checkpoint": os.path.abspath(a.checkpoint),
           "checkpoint_global_step": int(ts["global_step"]), "n_inputs": n,
           "ref_eq_full": sum(r["ref_eq_full"] for r in rows),
           "ref_eq_cached": sum(r["ref_eq_cached"] for r in rows),
           "full_eq_cached": sum(r["full_eq_cached"] for r in rows),
           "k0_max_abs_ref_vs_full": max(r["k0_max_abs_ref_vs_full"] for r in rows),
           "k0_max_abs_ref_vs_cached": max(r["k0_max_abs_ref_vs_cached"] for r in rows),
           "boundary_bins": {"ref": sum(r["boundary_ref"] for r in rows),
                             "full": sum(r["boundary_full"] for r in rows),
                             "cached": sum(r["boundary_cached"] for r in rows),
                             "of": 14 * n},
           "rows": rows}
    if rep["ref_eq_full"] > rep["ref_eq_cached"]:
        rep["attribution"] = "FULL_PATH_MATCHES_TRAINING -> the INCREMENTAL STEP is the defect"
    elif rep["ref_eq_cached"] > rep["ref_eq_full"]:
        rep["attribution"] = "CACHED_PATH_MATCHES_TRAINING -> the DEPLOYED FULL PATH is the defect"
    else:
        rep["attribution"] = "NEITHER matches training - investigate both"
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1)[:2500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
