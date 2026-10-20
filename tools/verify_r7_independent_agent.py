"""Independent recomputation of the corrected R7 evidence (revision 2).

Reads the archived per-seed prediction files (*.npz) and the JSON reports and
recomputes the headline numbers from scratch, so a reporting bug cannot hide
behind the pipeline's own aggregation.

New in this revision: every sidecar file is first *resolved* to the JSON it
actually corresponds to (by matching its recomputed L1 against the logged L1 of
each invocation, within 1e-9).  r7_state_mlp is invoked twice per run - once on
all legal frames and once with --restrict-t2 - and both invocations write
pred_SV_R_*.npz / pred_pilot200_R_*.npz, so the second one overwrites the first.
Without this resolution step a verifier silently compares stale files against
fresh numbers.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

LEFT, RIGHT = slice(0, 7), slice(7, 14)
SIGN_THRESHOLD = 1e-3
R2_SIGN_MASK = 1e-4
RESOLVE_TOL = 1e-9


def _corr(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _cos_rows(a, b):
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def correction_metrics(R_hat, R_gt):
    resid = R_hat - R_gt
    var_gt = float(R_gt.var())
    mask = np.abs(R_gt) >= R2_SIGN_MASK
    cos = _cos_rows(R_hat, R_gt)
    return {
        "L1": float(np.abs(resid).mean()),
        "zero_correction_L1": float(np.abs(R_gt).mean()),
        "R2_variance_explained": (1.0 - float(resid.var()) / var_gt) if var_gt > 0 else None,
        "corr": _corr(R_hat, R_gt),
        "cosine_mean": float(np.nanmean(cos)) if np.isfinite(cos).any() else None,
        "sign_agreement": float(np.mean(np.sign(R_hat[mask]) == np.sign(R_gt[mask])))
        if mask.any() else None,
        "sign_n": int(mask.sum()),
    }


def direction_metrics(cmd, gt, state):
    pd, gd = cmd - state, gt - state
    mask = np.abs(gd) >= SIGN_THRESHOLD
    cos = _cos_rows(pd, gd)
    return {
        "corr_delta_pooled": _corr(pd, gd),
        "cosine_mean": float(np.nanmean(cos)) if np.isfinite(cos).any() else None,
        "sign_agreement_at_1e-3": float(np.mean(np.sign(pd[mask]) == np.sign(gd[mask])))
        if mask.any() else None,
        "sign_n_at_1e-3": int(mask.sum()),
    }


def rebuild_split(split_json, n_dev=16):
    data = json.load(open(split_json))
    train = sorted(data["train"])
    holdout = sorted(data["val"])
    step = max(1, len(train) // n_dev)
    dev = sorted(set(train[::step][:n_dev]))
    train_only = [e for e in train if e not in set(dev)]
    return {
        "train": train_only,
        "dev": dev,
        "holdout": holdout,
        "counts": {"train": len(train_only), "dev": len(dev), "holdout": len(holdout)},
        "overlap_train_dev": sorted(set(train_only) & set(dev)),
        "overlap_train_holdout": sorted(set(train_only) & set(holdout)),
        "overlap_dev_holdout": sorted(set(dev) & set(holdout)),
    }


def logged_L1(doc, key, seed):
    if key == "pilot200":
        vals = [r["L1"] for r in doc.get("pilot200", {}).get("per_seed", []) if r["seed"] == seed]
    else:
        vals = [r["L1"] for r in doc.get("runs", {}).get(key, {}).get("per_seed", [])
                if r["seed"] == seed]
    return vals[0] if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split-json", default="/home/dgut/pusht_runtime/split80.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args()

    report = {"run_dir": args.run_dir, "checks": {}, "recomputed": {}, "sidecars": []}
    main_json = json.load(open(os.path.join(args.run_dir, "state_mlp.json")))
    docs = {"main": main_json}
    acc_path = os.path.join(args.run_dir, "state_mlp_acc.json")
    if os.path.isfile(acc_path):
        docs["acc"] = json.load(open(acc_path))

    # ---- split reconstruction
    split = rebuild_split(args.split_json)
    report["splits"] = {
        "counts": split["counts"],
        "overlap": {
            "train_dev": split["overlap_train_dev"],
            "train_holdout": split["overlap_train_holdout"],
            "dev_holdout": split["overlap_dev_holdout"],
        },
        "reported": {k: len(main_json["splits"][k + "_episodes"])
                     for k in ("train", "dev", "holdout")},
        "matches_pipeline": (
            split["train"] == sorted(main_json["splits"]["train_episodes"])
            and split["dev"] == sorted(main_json["splits"]["dev_episodes"])
            and split["holdout"] == sorted(main_json["splits"]["holdout_episodes"])
        ),
    }
    report["checks"]["split_is_64_16_20"] = split["counts"] == {"train": 64, "dev": 16, "holdout": 20}
    report["checks"]["no_episode_overlap"] = not any(report["splits"]["overlap"].values())
    report["checks"]["split_matches_pipeline"] = report["splits"]["matches_pipeline"]
    holdout = set(split["holdout"])

    # ---- identities + metric recomputation, with sidecar resolution
    rows = []
    for path in sorted(glob.glob(os.path.join(args.run_dir, "pred_*.npz"))):
        name = os.path.basename(path)
        stem = name[len("pred_"):-len(".npz")]
        key, seed = stem.rsplit("_", 1)[0], int(stem.rsplit("_", 1)[1])
        if key.startswith("pilot200"):
            key = "pilot200"  # the sidecar is pred_pilot200_R_<seed>.npz
        z = np.load(path)
        cmd = z["cmd"].astype(np.float64)
        gt = z["gt"].astype(np.float64)
        state = z["state"].astype(np.float64)
        V = z["V"].astype(np.float64)
        R_hat = z["R_hat"].astype(np.float64)
        R_gt = z["R_gt"].astype(np.float64)
        episodes = z["episode"]

        errA = float(np.abs(R_gt - ((gt - state) - V)).max())
        errB = float(np.abs(cmd - (state + V + R_hat)).max())
        L1 = float(np.abs(cmd - gt).mean())

        resolved = None
        for tag, doc in docs.items():
            val = logged_L1(doc, key, seed)
            if val is not None and abs(val - L1) <= RESOLVE_TOL:
                resolved = tag
                break

        corr = correction_metrics(R_hat, R_gt)
        direc = direction_metrics(cmd, gt, state)
        cv_cmd = state + V
        eps = sorted(set(episodes.tolist()))
        per_ep = {ep: float(np.abs(cmd[episodes == ep] - gt[episodes == ep]).mean()) for ep in eps}
        beating = sum(
            1 for ep in eps
            if np.abs(cmd[episodes == ep] - gt[episodes == ep]).mean()
            < np.abs(cv_cmd[episodes == ep] - gt[episodes == ep]).mean()
        )
        rows.append({
            "file": name, "run_key": key, "seed": seed, "n": int(len(cmd)),
            "episodes": len(eps),
            "episodes_outside_holdout": sorted(set(eps) - holdout),
            "L1": L1, "constant_velocity_L1": float(np.abs(cv_cmd - gt).mean()),
            "macro_L1": float(np.mean(list(per_ep.values()))) if per_ep else None,
            "episodes_beating_cv": beating,
            "correction": corr, "direction": direc,
            "fixA_max_err": errA, "fixB_max_err": errB,
            "fixA_ok": errA <= args.atol, "fixB_ok": errB <= args.atol,
            "resolved_against": resolved,
            "logged_main": logged_L1(docs["main"], key, seed),
            "logged_acc": logged_L1(docs.get("acc", {}), key, seed) if "acc" in docs else None,
        })

    report["sidecars"] = rows
    collisions = [r for r in rows if r["logged_main"] is not None and r["logged_acc"] is not None
                  and r["resolved_against"] == "acc" and abs(r["logged_main"] - r["logged_acc"]) > RESOLVE_TOL]
    report["sidecar_collisions"] = [
        {"file": r["file"], "resolved_against": r["resolved_against"],
         "logged_main": r["logged_main"], "logged_acc": r["logged_acc"],
         "L1_on_disk": r["L1"], "n_rows": r["n"]}
        for r in collisions
    ]

    report["checks"]["fixA_R_gt_is_A_minus_S_minus_V"] = all(r["fixA_ok"] for r in rows)
    report["checks"]["fixB_cmd_is_S_plus_V_plus_R_hat"] = all(r["fixB_ok"] for r in rows)
    report["checks"]["all_predictions_from_holdout"] = all(not r["episodes_outside_holdout"] for r in rows)
    report["checks"]["every_sidecar_resolves_to_a_reported_run"] = all(
        r["resolved_against"] is not None for r in rows
    )

    # ---- per-run comparison against the JSON the sidecar belongs to
    comparison = []
    for tag, doc in docs.items():
        for run_name, run in doc.get("runs", {}).items():
            for seed_entry in run.get("per_seed", []):
                seed = seed_entry["seed"]
                mine = [r for r in rows
                        if r["run_key"] == run_name and r["seed"] == seed]
                if not mine:
                    continue
                mine = mine[0]
                reported_R2 = (seed_entry.get("correction") or {}).get("R2_variance_explained")
                comparison.append({
                    "invocation": tag, "run": run_name, "seed": seed,
                    "sidecar": mine["file"],
                    "resolved_against": mine["resolved_against"],
                    "reported_L1": seed_entry["L1"], "recomputed_L1": mine["L1"],
                    "L1_matches": abs(seed_entry["L1"] - mine["L1"]) <= RESOLVE_TOL,
                    "reported_R2": reported_R2,
                    "recomputed_R2": mine["correction"]["R2_variance_explained"],
                    "R2_matches": (reported_R2 is None and mine["correction"]["R2_variance_explained"] is None)
                    or (reported_R2 is not None and mine["correction"]["R2_variance_explained"] is not None
                        and abs(reported_R2 - mine["correction"]["R2_variance_explained"]) <= 1e-9),
                })
    report["recomputed"]["comparison"] = comparison
    report["checks"]["reported_metrics_match_recomputation_for_resolved_sidecars"] = all(
        row["L1_matches"] and row["R2_matches"]
        for row in comparison if row["resolved_against"] is not None
    ) if comparison else None

    # ---- fixC
    pilot = main_json.get("pilot200", {})
    fixed = pilot.get("fixed_epochs_from_full_run")
    executed = [p.get("fixed_epochs_executed") for p in pilot.get("per_seed", [])]
    sels = [p.get("checkpoint_selection") for p in pilot.get("per_seed", [])]
    report["pilot200"] = {
        "fixed_epochs_from_full_run": fixed, "executed": executed, "selection": sels,
        "L1": [p.get("L1") for p in pilot.get("per_seed", [])],
        "n_train_frames": [p.get("n_train_frames") for p in pilot.get("per_seed", [])],
    }
    report["checks"]["fixC_pilot_uses_fixed_epoch_budget"] = bool(
        fixed is not None and executed and all(x == fixed for x in executed)
    )
    report["checks"]["fixC_pilot_does_not_select_on_train"] = bool(
        sels and all(s not in (True, "best", "dev_best") for s in sels)
    )

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)

    print(json.dumps(report["checks"], indent=1))
    print("splits:", report["splits"]["counts"], "| matches pipeline:", report["checks"]["split_matches_pipeline"])
    print("sidecar collisions:", len(report["sidecar_collisions"]))
    for c in report["sidecar_collisions"]:
        print("   %-26s rows=%-5d disk L1=%.6f | main %.6f | acc %.6f"
              % (c["file"], c["n_rows"], c["L1_on_disk"], c["logged_main"], c["logged_acc"]))
    print("comparison rows:", len(comparison),
          "| L1 mismatches:", sum(1 for r in comparison if not r["L1_matches"]),
          "| R2 mismatches:", sum(1 for r in comparison if not r["R2_matches"]))
    for r in comparison:
        print("   %-4s %-10s seed=%d sidecar=%-24s resolved=%-5s L1 %.6f vs %.6f (%s) R2 %s vs %s (%s)"
              % (r["invocation"], r["run"], r["seed"], r["sidecar"], r["resolved_against"],
                 r["recomputed_L1"], r["reported_L1"], r["L1_matches"],
                 None if r["recomputed_R2"] is None else round(r["recomputed_R2"], 4),
                 None if r["reported_R2"] is None else round(r["reported_R2"], 4),
                 r["R2_matches"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
