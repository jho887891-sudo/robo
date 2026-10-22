#!/usr/bin/env python3
"""R24 Stage B: freeze a stratified frame manifest BEFORE any model is evaluated.

The R17 audit's warning is the design constraint here: the 8 DEV frames per task that have been reused
across several rounds are EXPLORATORY evidence, and repeatedly selecting on them (or on the 20 held-out
episodes) makes any later "result" a selection artefact. So this tool:

  * picks DEV episodes by a hash of the episode name, NOT by any model score;
  * stratifies every task's DEV frames into motion strata computed from the DEMONSTRATION ONLY
    (state velocity and residual magnitude - never a model output);
  * also draws a TRAIN frame set from the same episodes the optimiser saw, for the fit test;
  * writes the whole selection with per-stratum counts and a single manifest sha256;
  * refuses to run twice with a different selection (the file is a freeze, not a sample).

Honest limitation recorded in the output: the "contact" phase cannot be identified from the HDF5 alone
(that needs privileged simulator object poses), so the strata are IMITATION-KINEMATIC: still (|V| ~ 0,
the state a real episode starts in), slow-motion, fast-motion and large-residual. Contact/approach phases
must come from a privileged trace in the closed-loop instrument (Stage D), not from here.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("training", "code/xpolicylab/policy/OpenVLA_Pure", "code/xpolicylab/policy/OpenVLA_R42"):
    _p = os.path.join(REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import m2_train_shared as M2       # noqa: E402
import m2_multitask_data as MD     # noqa: E402

# Read the training contract's own limit instead of assuming one: [S, V, Acc] needs TWO past states,
# so frames t < 2 cannot be turned into a training row at all. A first manifest used t >= 1, which the
# model loader then rejected ("frame 1 has no [S,V,Acc] history"); that is a selector bug, not a
# selection-on-results change, and the manifest was re-frozen before any model was evaluated.
import r42_train_contract as _TC  # noqa: E402

MIN_HISTORY = int(_TC.MIN_HISTORY_FRAMES)


def ep_hash(ep):
    return int(hashlib.sha256(ep.encode()).hexdigest()[:8], 16)


def stratify(recs):
    """Label each frame by IMITATION-KINEMATIC stratum, from demonstration statistics only."""
    speeds = np.array([r["speed"] for r in recs])
    resids = np.array([r["resid_mag"] for r in recs])
    s_med = float(np.median(speeds)) if len(speeds) else 0.0
    r_med = float(np.median(resids)) if len(resids) else 0.0
    for r in recs:
        if r["speed"] <= 0.2 * s_med:
            r["stratum"] = "still"
        elif r["speed"] <= s_med:
            r["stratum"] = "slow_motion"
        else:
            r["stratum"] = "fast_motion"
        if r["resid_mag"] >= 2.0 * r_med:
            r["stratum"] += "+large_residual"
    return {"speed_median": s_med, "resid_median": r_med}


def pick(recs, per_stratum, seed=0):
    """Deterministic, model-free selection: sort by a hash of (episode, t) and take the first N."""
    rs = np.random.RandomState(seed)
    buckets = {}
    for r in recs:
        buckets.setdefault(r["stratum"], []).append(r)
    chosen = []
    for name in sorted(buckets):
        items = sorted(buckets[name], key=lambda r: (ep_hash(r["episode"]), r["t"]))
        take = min(per_stratum, len(items))
        chosen.extend(items[:take])
    return sorted(chosen, key=lambda r: (r["episode"], r["t"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="push_T,stack_bowls")
    ap.add_argument("--normalizer", default="data/m2_task_normalizer_2task.json")
    ap.add_argument("--splits", default="data/m2_splits.json")
    ap.add_argument("--dev-frames", type=int, default=12, help="frames per stratum per task (DEV)")
    ap.add_argument("--train-frames", type=int, default=12, help="frames per stratum per task (TRAIN)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "mission", "r24", "frame_manifest.json"))
    a = ap.parse_args()

    import h5py
    import r42_train_contract as TC

    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    _doc, split_map, payload, _sp, _np = M2.load_m2_artifacts(tasks, a.splits, a.normalizer)
    out = {"schema": "r24-frame-manifest-v1", "tasks": tasks,
           "why": ("frozen BEFORE any model evaluation so that the R24 comparison cannot be a selection "
                   "artefact; DEV episodes are chosen by hash of the episode name, strata from the "
                   "demonstration alone, and the same manifest is used for every checkpoint"),
           "selection": {"dev_frames_per_stratum_per_task": a.dev_frames,
                         "train_frames_per_stratum_per_task": a.train_frames,
                         "seed": a.seed,
                         "dev_episode_rule": "sorted by sha256(episode)[:8] - never by model score",
                         "strata_definition": "still (|V| <= 0.2*median), slow_motion (<=median), "
                                              "fast_motion (>median), '+large_residual' when "
                                              "|R| >= 2*median"},
           "limitation": ("contact and approach phases are NOT identifiable from the HDF5 alone; they need "
                          "privileged simulator object poses (Stage D trace). These strata are "
                          "imitation-kinematic, and are reported as such."),
           "frames": {}}

    for t in tasks:
        idx = MD.load_task_index(t)
        smap = split_map[t]
        dev_eps = [str(e).split("/")[-1] for e in smap["dev"]]
        dev_eps = sorted(dev_eps, key=ep_hash)
        train_eps = [str(e).split("/")[-1] for e in smap["train"]]
        block = {}
        for split_name, eps in (("dev", dev_eps), ("train", train_eps[:12])):
            recs = []
            for ep in eps:
                p = os.path.join(idx["hdf5_dir"], ep)
                if not os.path.isfile(p):
                    continue
                with h5py.File(p, "r") as h:
                    S = np.asarray(TC.read14(h, "state"), np.float32)
                    A = np.asarray(TC.read14(h, "action"), np.float32)
                for tick in range(MIN_HISTORY, len(S) - 1):
                    v = S[tick] - S[tick - 1]
                    r = (A[tick] - S[tick]) - v
                    recs.append({"episode": ep, "t": int(tick),
                                 "speed": float(np.abs(v).mean()),
                                 "resid_mag": float(np.abs(r).mean())})
                if split_name == "train" and len(recs) > 4000:
                    break
            stats = stratify(recs)
            per = a.dev_frames if split_name == "dev" else a.train_frames
            chosen = pick(recs, per, a.seed)
            counts = {}
            for r in chosen:
                counts[r["stratum"]] = counts.get(r["stratum"], 0) + 1
            block[split_name] = [{"episode": r["episode"], "t": r["t"], "stratum": r["stratum"],
                                  "speed": round(r["speed"], 8),
                                  "resid_mag": round(r["resid_mag"], 8)} for r in chosen]
            block[split_name + "_strata"] = {"counts": counts, "median_speed": stats["speed_median"],
                                             "median_resid": stats["resid_median"],
                                             "pool_size": len(recs)}
            print("[manifest] %-12s %-5s n=%d %s" % (t, split_name, len(chosen), counts), flush=True)
        out["frames"][t] = block

    # The freeze FLAG itself must be inside the hashed body, otherwise a post-hoc flip of
    # frozen_before_model_run would not invalidate the manifest (a regression test caught this).
    out["frozen_before_model_run"] = True
    body = json.dumps({k: v for k, v in out.items() if k != "manifest_sha256"}, sort_keys=True)
    out["manifest_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, a.out)
    print("manifest_sha256 = %s" % out["manifest_sha256"])
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
