#!/usr/bin/env python3
"""R6: replication table for the 300-update pilot (noise floor vs A/B effect).

Reads any number of run directories and reports, per evaluation step, the per-run
validation L1 with the identical-configuration spread and the arm-vs-arm separation.
This is what decides whether a single-pair A/B difference means anything.
"""
import argparse
import json
import os
import statistics


def load(run_dir):
    ev = {}
    for line in open(os.path.join(run_dir, "metrics.jsonl")):
        rec = json.loads(line)
        if rec.get("stage") == "eval" and rec.get("split") == "val":
            ev[int(rec["step"])] = rec
    status = json.load(open(os.path.join(run_dir, "status.json")))
    return ev, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="label=path pairs, repeatable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    arms = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        ev, status = load(path)
        if status.get("result") != "DONE":
            raise SystemExit("%s did not finish (%s)" % (label, status.get("result")))
        last = ev[max(ev)]
        arm = last["arm"]
        arms.setdefault(arm, {})[label] = {
            "path": os.path.abspath(path),
            "curve": {str(k): float(v["gen_L1"]) for k, v in sorted(ev.items())},
            "final": {"val_L1": float(last["gen_L1"]),
                      "identity_L1": float(last["identity_L1"]),
                      "corr_delta": last["r6"]["delta_correlation"]["pooled"],
                      "sign_at_1e-3": next(r["sign_agreement"] for r in last["r6"]["sign_agreement"]
                                           if r["basis"] == "expert_delta" and r["threshold"] == 1e-3),
                      "magnitude_ratio": last["r6"]["magnitude"]["ratio"],
                      "unique_token_rows": last["r5"]["unique_generated_token_rows"],
                      "n": last["n"], "tf_tok_acc": last.get("tf_tok_acc")},
        }
    steps = sorted({int(s) for a in arms.values() for r in a.values() for s in r["curve"]})
    table = {}
    for step in steps:
        row = {arm: {lab: r["curve"].get(str(step)) for lab, r in runs.items()}
               for arm, runs in arms.items()}
        stats = {}
        for arm, vals in row.items():
            xs = [v for v in vals.values() if v is not None]
            if not xs:
                continue
            stats[arm] = {"n": len(xs), "mean": statistics.mean(xs), "min": min(xs), "max": max(xs),
                          "spread_pct": (max(xs) - min(xs)) / (statistics.mean(xs) + 1e-12) * 100}
        sep = None
        if "duplicate" in stats and "true" in stats:
            sep = {"true_max_lt_duplicate_min": bool(stats["true"]["max"] < stats["duplicate"]["min"]),
                   "mean_improvement_pct": (stats["duplicate"]["mean"] - stats["true"]["mean"])
                   / (stats["duplicate"]["mean"] + 1e-12) * 100,
                   "cross_pair_orderings_consistent": bool(stats["true"]["max"] < stats["duplicate"]["min"])}
        table[str(step)] = {"per_run": row, "per_arm": stats, "separation": sep}
    out = {"runs": arms, "by_step": table,
           "note": ("identical-configuration runs are the noise floor; an A/B effect must be read "
                    "against that spread, not against a single pair")}
    json.dump(out, open(args.out, "w"), indent=1)
    for step in steps:
        row = table[str(step)]
        parts = []
        for arm in ("duplicate", "true"):
            if arm in row["per_arm"]:
                s = row["per_arm"][arm]
                parts.append("%s mean=%.5f [%.5f, %.5f] spread=%.1f%%" % (
                    arm, s["mean"], s["min"], s["max"], s["spread_pct"]))
        sep = row["separation"]
        print("step %-4s %s | separated=%s improvement=%.1f%%" % (
            step, " | ".join(parts), sep and sep["true_max_lt_duplicate_min"],
            sep["mean_improvement_pct"] if sep else float("nan")))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
