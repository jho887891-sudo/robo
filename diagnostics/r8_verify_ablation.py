#!/usr/bin/env python3
"""R8: independent verification of the coverage ablation + compact evidence export.

Recomputes every reported number from the stored predictions, checks that the three
dataset sizes are genuinely NESTED prefixes of one sampling order, and writes small
per-run per-frame error vectors so the evidence can be audited without the 51 MB of
prediction archives.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "training"))
import r6_data as D                                   # noqa: E402
from r8_coverage_ablation import (pool_from_episodes, stratified_order,   # noqa: E402
                                  take_prefix, split_episodes)

RESULTS = []


def check(name, ok, detail):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print("[%s] %-46s %s" % ("PASS" if ok else "FAIL", name,
                             json.dumps(detail) if detail is not None else ""), flush=True)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--err-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.err_dir, exist_ok=True)
    d = json.load(open(args.json))
    cv_ref = float(d["copy_V_holdout_L1"])
    id_ref = float(d["identity_holdout_L1"])
    ok = True

    # ---- nesting of the training subsets (same sampling order, true prefixes)
    split = D.load_split()
    tr_eps, _dev = split_episodes(split["train"], n_dev=16)
    rows = pool_from_episodes(tr_eps)
    order, edges = stratified_order(rows, seed=0)
    sets = {}
    for size in (200, 2000):
        blob = take_prefix(rows, order, size)
        sets[size] = set(zip(blob["episode"].tolist(), blob["frame"].tolist()))
    blob_full = take_prefix(rows, order, None)
    sets["full"] = set(zip(blob_full["episode"].tolist(), blob_full["frame"].tolist()))
    ok &= check("dataset_200_is_subset_of_2000", sets[200] <= sets[2000],
                {"n200": len(sets[200]), "overlap": len(sets[200] & sets[2000])})
    ok &= check("dataset_2000_is_subset_of_full", sets[2000] <= sets["full"],
                {"n2000": len(sets[2000]), "overlap": len(sets[2000] & sets["full"])})
    for size in (200, 2000):
        man = d["manifests"][str(size)]
        ok &= check("manifest_size_%d_matches_prefix" % size,
                    man["n_frames"] == size and man["n_frames"] == len(sets[size]), man)

    # ---- recompute every reported aggregate from the prediction archives
    max_dl1 = 0.0
    per_run = {}
    for key in sorted(d["runs"]):
        agg = d["runs"][key]["aggregate"]
        l1s, r2s, coss, signs, wins = [], [], [], [], []
        for seed_rec in d["runs"][key]["per_seed"]:
            seed = seed_rec["seed"]
            path = os.path.join(args.run_dir, "pred_cov_%s_seed%d.npz" % (key, seed))
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            z = np.load(path)
            err = np.abs(z["cmd"] - z["gt"])
            cv = np.abs(z["state"] + z["V"] - z["gt"])
            l1 = float(err.mean())
            l1s.append(l1)
            eps = z["episode"]
            d_ep = [err[eps == e].mean() - cv[eps == e].mean() for e in sorted(set(eps.tolist()))]
            wins.append(int(sum(1 for x in d_ep if x < 0)))
            num = ((z["R_hat"] - z["R_gt"]) ** 2).sum()
            var = ((z["R_gt"] - z["R_gt"].mean(axis=0)) ** 2).sum()
            r2s.append(float(1.0 - num / var))
            td, gd = z["cmd"] - z["state"], z["gt"] - z["state"]
            denom = np.linalg.norm(td, axis=1) * np.linalg.norm(gd, axis=1)
            coss.append(float(np.nanmean(np.where(denom > 0, (td * gd).sum(1) / np.maximum(denom, 1e-12), np.nan))))
            m = np.abs(z["R_gt"]) >= 1e-4
            signs.append(float(np.mean(np.sign(z["R_hat"][m]) == np.sign(z["R_gt"][m]))) if m.any() else None)
            np.save(os.path.join(args.err_dir, "err_%s_seed%d.npy" % (key, seed)),
                    err.mean(axis=1).astype(np.float32))
            max_dl1 = max(max_dl1, abs(l1 - seed_rec["L1"]))
        per_run[key] = {"L1_mean_recomputed": float(np.mean(l1s)),
                        "L1_mean_logged": agg["L1"]["mean"],
                        "R2_mean_recomputed": float(np.mean(r2s)),
                        "cosine_mean_recomputed": float(np.mean(coss)),
                        "R_sign_mean_recomputed": float(np.nanmean(signs)),
                        "wins_per_seed": wins,
                        "steps": agg["steps"], "epochs": agg["epochs"]}
        ok &= check("rerun_%s_seed_L1_matches" % key, abs(per_run[key]["L1_mean_recomputed"]
                                                         - agg["L1"]["mean"]) < 1e-9,
                    {"recomputed": per_run[key]["L1_mean_recomputed"], "logged": agg["L1"]["mean"]})
    ok &= check("cv_and_identity_recomputed", abs(cv_ref - 0.001795) < 5e-6 and id_ref > 0,
                {"cv": cv_ref, "identity": id_ref})
    ok &= check("full_data_beats_cv_all_protocols",
                all(d["runs"]["19127_" + p]["aggregate"]["L1"]["mean"] < cv_ref
                    for p in ("dev_selected", "equal_epochs", "equal_updates")),
                {p: d["runs"]["19127_" + p]["aggregate"]["L1"]["mean"]
                 for p in ("dev_selected", "equal_epochs", "equal_updates")})
    ok &= check("two_thousand_frames_loses_to_cv",
                all(d["runs"]["2000_" + p]["aggregate"]["L1"]["mean"] > cv_ref
                    for p in ("dev_selected", "equal_epochs", "equal_updates")),
                {p: d["runs"]["2000_" + p]["aggregate"]["L1"]["mean"]
                 for p in ("dev_selected", "equal_epochs", "equal_updates")})
    ok &= check("two_hundred_frames_worse_than_identity",
                all(d["runs"]["200_" + p]["aggregate"]["L1"]["mean"] > id_ref
                    for p in ("dev_selected", "equal_epochs", "equal_updates")),
                {p: d["runs"]["200_" + p]["aggregate"]["L1"]["mean"]
                 for p in ("dev_selected", "equal_epochs", "equal_updates")})
    ok &= check("full_data_wins_all_episodes_all_seeds",
                all(w == 20 for w in per_run["19127_dev_selected"]["wins_per_seed"]),
                per_run["19127_dev_selected"]["wins_per_seed"])
    ok &= check("seed_stability_full_data",
                (max(d["runs"]["19127_dev_selected"]["aggregate"]["L1"]["max"]
                     for _ in (0,)) / d["runs"]["19127_dev_selected"]["aggregate"]["L1"]["min"]) < 1.10,
                {"min": d["runs"]["19127_dev_selected"]["aggregate"]["L1"]["min"],
                 "max": d["runs"]["19127_dev_selected"]["aggregate"]["L1"]["max"]})

    summary = {"checks": RESULTS, "all_pass": all(r["pass"] for r in RESULTS),
               "per_run": per_run, "max_abs_L1_discrepancy": max_dl1,
               "cv_holdout_L1": cv_ref, "identity_holdout_L1": id_ref,
               "err_vector_dir": os.path.abspath(args.err_dir)}
    json.dump(summary, open(args.out, "w"), indent=1)
    print("\n%d/%d checks pass -> %s" % (sum(1 for r in RESULTS if r["pass"]), len(RESULTS),
                                         "PASS" if summary["all_pass"] else "FAIL"))
    return 0 if summary["all_pass"] else 4


if __name__ == "__main__":
    sys.exit(main())
