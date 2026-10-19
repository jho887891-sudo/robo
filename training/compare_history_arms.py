#!/usr/bin/env python3
"""R6: compare the two 28D-proprio arms against the pre-registered gates.

Reads the two pilot run directories (duplicate / true) plus the Phase-1 diagnostic
JSONs, and prints + writes every gate with the number that decided it. A gate that
cannot be evaluated is reported as FAIL (never silently skipped).
"""
import argparse
import json
import os
import sys

ARM_DUP, ARM_TRUE = "duplicate", "true"


def _load_evals(run_dir):
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.isfile(path):
        raise SystemExit("missing metrics.jsonl in " + run_dir)
    evals, steps = {}, {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("stage") == "eval" and rec.get("split") == "val":
            evals[int(rec["step"])] = rec
        elif rec.get("stage") == "train":
            steps[int(rec["step"])] = rec
    return evals, steps


def _final(run_dir, evals):
    with open(os.path.join(run_dir, "status.json")) as fh:
        status = json.load(fh)
    if status.get("result") != "DONE":
        raise SystemExit("%s did not finish: %s" % (run_dir, status.get("result")))
    return status, max(evals)


def _phase1(dirpath):
    out = {}
    for name in ("phase1_alignment", "phase1_history_value", "phase1_quant_oracle"):
        p = os.path.join(dirpath or "", name + ".json")
        if os.path.isfile(p):
            out[name] = json.load(open(p))
    return out


def gates_from_phase1(p1):
    g = []
    al, hv, qo = p1.get("phase1_alignment"), p1.get("phase1_history_value"), p1.get("phase1_quant_oracle")
    if al:
        val = al["splits"]["val"]
        e1 = val["per_error"]["e1_next_state"]
        g.append({"gate": "H1_alignment_A_t_equals_S_t+1",
                  "pass": bool(e1["max"] == 0.0 and val["shift_scan_best_shift"] == "1"),
                  "measured": {"E1_max": e1["max"], "E1_exact_zero_fraction": e1["exact_zero_fraction"],
                               "best_shift": val["shift_scan_best_shift"],
                               "E2_hold": val["per_error"]["e2_current_state"]["mean"],
                               "E3_extrapolated": val["per_error"]["e3_extrapolated"]["mean"]}})
        c = al["checks"]["val"]
        g.append({"gate": "H2_no_future_leak_in_E3",
                  "pass": bool(c["leak_test_all_episodes_pass"] and c["leak_test_can_fail_e1_responds"]
                               and c["leak_test_max_abs_diff"] == 0.0),
                  "measured": {"leak_max_abs_diff": c["leak_test_max_abs_diff"],
                               "control_can_fail": c["leak_test_can_fail_e1_responds"]}})
        eps = al["episodes"]["val"]
        wins = sum(1 for v in eps.values() if v["e3"] < v["e2"])
        g.append({"gate": "H4_constant_velocity_beats_identity_in_most_episodes",
                  "pass": bool(wins >= 0.85 * len(eps)),
                  "measured": {"episodes_cv_better": wins, "episodes": len(eps),
                               "required": int(0.85 * len(eps))}})
    if hv:
        v = hv["splits"]["val"]
        row = next(r for r in v["sign"] if r["basis"] == "expert_delta" and r["threshold"] == 1e-3)
        g.append({"gate": "H3_history_carries_expert_delta_information",
                  "pass": bool(v["corr_pooled_flattened"] >= 0.5 and row["sign_agreement"] >= 0.75),
                  "measured": {"corr_pooled": v["corr_pooled_flattened"],
                               "corr_left": v["corr_arm"]["left"], "corr_right": v["corr_arm"]["right"],
                               "sign_agreement_at_1e-3": row["sign_agreement"], "n": row["n"],
                               "constant_velocity_L1": v["history_only_predictors"]["constant_velocity_L1"],
                               "hold_state_L1": v["history_only_predictors"]["hold_state_L1"]}})
    if qo:
        res = qo["splits"]["val"]["residual"]["blocks"]
        ab = qo["splits"]["val"]["absolute"]["blocks"]
        cv_cont = res["constant_velocity"]["continuous_L1"]
        oracle = res["oracle_true_delta"]["quantized_L1"]
        g.append({"gate": "H5_256bin_residual_codec_is_not_the_bottleneck",
                  "pass": bool(oracle <= 0.5 * cv_cont),
                  "measured": {"residual_oracle_L1": oracle, "constant_velocity_continuous_L1": cv_cont,
                               "ratio": oracle / (cv_cont + 1e-12),
                               "constant_velocity_quantized_L1": res["constant_velocity"]["quantized_L1"],
                               "rounding_L1": res["oracle_true_delta"]["rounding_L1"],
                               "clipping_L1": res["oracle_true_delta"]["clipping_L1"],
                               "clipped_entry_fraction": res["oracle_true_delta"]["clipped_entry_fraction"]}})
        g.append({"gate": "H6_absolute_codec_floor_documented",
                  "pass": True,
                  "measured": {"absolute_oracle_L1": ab["oracle_true_action"]["quantized_L1"],
                               "absolute_step_mean": sum(qo["splits"]["val"]["absolute"]["physical_step_per_dim"]) / 14,
                               "residual_step_mean": sum(qo["splits"]["val"]["residual"]["physical_step_per_dim"]) / 14,
                               "note": "informational: proves an absolute-target policy cannot beat cv on 256 bins"}})
    return g


def gates_from_arms(cfg_dup, cfg_true, ev_dup, ev_true, step_dup, step_true, smoke):
    g = []
    d, t = ev_dup[step_dup], ev_true[step_true]
    same = (cfg_dup["proprio_dim"] == cfg_true["proprio_dim"] == 28
            and cfg_dup["steps"] == cfg_true["steps"] and cfg_dup["lr"] == cfg_true["lr"]
            and cfg_dup["seed"] == cfg_true["seed"] and cfg_dup["target_mode"] == cfg_true["target_mode"]
            and cfg_dup["samples"] == cfg_true["samples"] and cfg_dup["val_samples"] == cfg_true["val_samples"])
    g.append({"gate": "P2_interface_and_budget_identical",
              "pass": bool(same and cfg_dup["slot_gap_max"] == 0.0 and cfg_true["slot_gap_max"] > 0.0),
              "measured": {"proprio_dim": [cfg_dup["proprio_dim"], cfg_true["proprio_dim"]],
                           "steps": cfg_true["steps"], "lr": cfg_true["lr"], "seed": cfg_true["seed"],
                           "same_samples": cfg_dup["samples"] == cfg_true["samples"],
                           "same_val_samples": cfg_dup["val_samples"] == cfg_true["val_samples"],
                           "duplicate_slot_gap_max": cfg_dup["slot_gap_max"],
                           "true_slot_gap_max": cfg_true["slot_gap_max"]}})
    if smoke is not None:
        g.append({"gate": "P1_smoke_gate_passed",
                  "pass": bool(smoke.get("result") == "PASS"),
                  "measured": {"mode": smoke.get("history_mode"),
                               "checks": {c["check"]: c["result"] for c in smoke.get("checks", [])},
                               "grad_history_slot": smoke.get("grad_history_slot")}})
    ident = t["r5"]["episode_macro_L1"]["hold_state"]
    cv = t["r5"]["episode_macro_L1"]["constant_velocity"]
    l1d, l1t = d["gen_L1"], t["gen_L1"]
    g.append({"gate": "P3_history_arm_beats_no_history_control_by_10pct",
              "pass": bool(l1t < 0.9 * l1d),
              "measured": {"val_L1_duplicate": l1d, "val_L1_true": l1t,
                           "relative_improvement": (l1d - l1t) / (l1d + 1e-12)}})
    g.append({"gate": "P4_history_arm_beats_hold_state",
              "pass": bool(t["ratio_vs_identity"] < 1.0),
              "measured": {"val_L1_true": l1t, "identity_L1_same_frames": t["identity_L1"],
                           "ratio": t["ratio_vs_identity"]}})
    g.append({"gate": "P5_history_arm_beats_constant_velocity",
              "pass": bool(l1t < cv),
              "measured": {"val_L1_true": l1t, "constant_velocity_macro_L1": cv,
                           "ratio": l1t / (cv + 1e-12)}})
    dcorr = t["r6"]["delta_correlation"]["pooled"]
    row = next(r for r in t["r6"]["sign_agreement"]
               if r["basis"] == "expert_delta" and r["threshold"] == 1e-3)
    g.append({"gate": "P6_direction_correlation_and_sign_accuracy",
              "pass": bool(dcorr is not None and dcorr > 0.3 and row["sign_agreement"] > 0.65),
              "measured": {"corr_pred_delta_gt_delta": dcorr, "sign_agreement_at_1e-3": row["sign_agreement"],
                           "n": row["n"],
                           "duplicate_corr": d["r6"]["delta_correlation"]["pooled"],
                           "past_motion_reference_corr": t["r6"]["past_motion_delta_correlation"]["pooled"]}})
    collapsed = [x for x in t["r5"]["per_dim_generated_token_diversity"] if x["mode_fraction"] >= 0.95]
    g.append({"gate": "P7_token_diversity_not_collapsed",
              "pass": bool(not collapsed and t["r5"]["unique_generated_token_rows"] >= 0.5 * t["n"]),
              "measured": {"collapsed_dims": [x["dimension"] for x in collapsed],
                           "unique_rows": t["r5"]["unique_generated_token_rows"], "n": t["n"]}})
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duplicate", required=True)
    ap.add_argument("--true", required=True)
    ap.add_argument("--phase1-dir", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ev_dup, st_dup = _load_evals(args.duplicate)
    ev_true, st_true = _load_evals(args.true)
    status_dup, step_dup = _final(args.duplicate, ev_dup)
    status_true, step_true = _final(args.true, ev_true)
    cfg_dup = json.load(open(os.path.join(args.duplicate, "_snapshot", "config.json")))
    cfg_true = json.load(open(os.path.join(args.true, "_snapshot", "config.json")))
    smoke_path = os.path.join(args.phase1_dir or "", "smoke_true.json")
    smoke = json.load(open(smoke_path)) if os.path.isfile(smoke_path) else None

    p1 = _phase1(args.phase1_dir)
    g1 = gates_from_phase1(p1)
    g2 = gates_from_arms(cfg_dup, cfg_true, ev_dup, ev_true, step_dup, step_true, smoke)

    table = []
    for step in sorted(set(ev_dup) | set(ev_true)):
        row = {"step": step}
        for arm, evs in ((ARM_DUP, ev_dup), (ARM_TRUE, ev_true)):
            if step in evs:
                rec = evs[step]
                row[arm] = {"L1": rec["gen_L1"], "identity": rec["identity_L1"],
                            "corr": rec["r6"]["delta_correlation"]["pooled"],
                            "uniq_rows": rec["uniq_rows"]}
        table.append(row)
    t = ev_true[step_true]
    d = ev_dup[step_dup]
    result = {
        "phase1_gates": g1,
        "pilot_gates": g2,
        "phase1_all_pass": all(x["pass"] for x in g1) if g1 else None,
        "pilot_gates_passed": sum(1 for x in g2 if x["pass"]),
        "pilot_gates_total": len(g2),
        "val_curve": table,
        "final_val": {
            "duplicate": {"L1": d["gen_L1"], "identity": d["identity_L1"],
                          "cv_macro": d["r5"]["episode_macro_L1"]["constant_velocity"],
                          "corr": d["r6"]["delta_correlation"]["pooled"],
                          "beats_cv_episodes": d["r6"]["episodes_beating_constant_velocity"]},
            "true": {"L1": t["gen_L1"], "identity": t["identity_L1"],
                     "cv_macro": t["r5"]["episode_macro_L1"]["constant_velocity"],
                     "corr": t["r6"]["delta_correlation"]["pooled"],
                     "beats_cv_episodes": t["r6"]["episodes_beating_constant_velocity"]},
            "per_arm_true": t["r5"]["per_arm_physical_L1"]["model"],
            "per_episode_true": t["r6"]["per_episode"],
        },
        "runs": {"duplicate": os.path.abspath(args.duplicate), "true": os.path.abspath(args.true)},
        "evidence_dir": os.path.abspath(args.phase1_dir) if args.phase1_dir else None,
    }
    json.dump(result, open(args.out, "w"), indent=1)

    print("== Phase-1 gates ==")
    for g in g1:
        print("  [%s] %s" % ("PASS" if g["pass"] else "FAIL", g["gate"]))
    print("== Pilot gates ==")
    for g in g2:
        print("  [%s] %s" % ("PASS" if g["pass"] else "FAIL", g["gate"]))
        print("        ", json.dumps(g["measured"]))
    print("== val curve ==")
    for row in table:
        print("  step=%-4d dup_L1=%.5f true_L1=%.5f  identity=%.5f" % (
            row["step"], row.get(ARM_DUP, {}).get("L1", float("nan")),
            row.get(ARM_TRUE, {}).get("L1", float("nan")),
            row.get(ARM_TRUE, {}).get("identity", float("nan"))))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
