#!/usr/bin/env python3
"""R5 结果分析：把两组 status.json 汇总成报告用的表格与验收判定（纯 CPU，只读）。

用法：
  python diagnostics/r5_analyze.py --run-dir <r5_ab_jobs/stamp> --out <json>
"""
import argparse
import json
import os

DIMS = (["left_arm_j%d" % i for i in range(6)] + ["left_gripper"] +
        ["right_arm_j%d" % i for i in range(6)] + ["right_gripper"])
METHODS = ("model", "hold_state", "constant_velocity", "train_median_delta")


def load(path):
    return json.load(open(path))


def token_stats(rows):
    """rows: list of 14-length token lists -> per-dim diversity and boundary fractions."""
    out = []
    n = len(rows)
    for d in range(14):
        col = [int(r[d]) for r in rows]
        uniq = sorted(set(col))
        mode_n = max(col.count(v) for v in uniq)
        out.append({"dim": d, "unique": len(uniq),
                    "mode_fraction": mode_n / n,
                    "frac0": col.count(0) / n,
                    "frac255": col.count(255) / n})
    return out


def summarize_run(path):
    st = load(os.path.join(path, "status.json"))
    cfg = load(os.path.join(path, "_snapshot", "config.json"))
    res = {"target_mode": cfg["action_target_mode"], "steps": cfg["steps"],
           "n_train": len(cfg["samples"]), "n_val": len(cfg["val_samples"]),
           "lr": cfg["lr"], "mechanism": cfg["mechanism"],
           "train_episodes": len({e for e, _ in cfg["samples"]}),
           "val_episodes": len({e for e, _ in cfg["val_samples"]}),
           "result": st["result"], "grad_checkpoint": st.get("grad_checkpoint")}
    for key, split in (("final_eval", "train"), ("final_val_eval", "val")):
        e = st[key]
        r5 = e["r5"]
        res[split] = {
            "n": e["n"], "gen_L1": e["gen_L1"], "identity_L1": e["identity_L1"],
            "repeat_previous_L1": e["repeat_previous_L1"],
            "pred_delta": e["pred_delta"], "gt_delta": e["gt_delta"],
            "tf_tok_acc": e["tf_tok_acc"], "uniq_rows": e["uniq_rows"],
            "delta_ratio": r5["predicted_delta_to_expert_delta_ratio"],
            "per_arm": r5["per_arm_physical_L1"],
            "per_dim": r5["per_dim_physical_L1"],
            "macro": r5["episode_macro_L1"],
            "per_episode": r5["per_episode_physical_L1"],
            "tokens": token_stats(e["gen_tokens_per_sample"]),
        }
    return res


def curve(path):
    out = []
    for line in open(os.path.join(path, "metrics.jsonl")):
        r = json.loads(line)
        if r["stage"] == "eval":
            out.append({"split": r["split"], "step": r["step"], "gen_L1": r["gen_L1"],
                        "identity_L1": r["identity_L1"],
                        "constant_velocity": r["r5"]["per_arm_physical_L1"] and
                        r["r5"]["episode_macro_L1"]["constant_velocity"],
                        "tf": r["tf_tok_acc"],
                        "ratio": r["r5"]["predicted_delta_to_expert_delta_ratio"],
                        "macro": r["r5"]["episode_macro_L1"],
                        "episodes": r["r5"]["evaluation_episode_count"]})
    return out


def episodes_better(model_ep, base_ep):
    wins = [e for e in model_ep if model_ep[e]["model"] < model_ep[e][base_ep]]
    return len(wins), sorted(set(model_ep) - set(wins))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="")
    args = _ = ap.parse_args() if False else ap.parse_args()
    a = summarize_run(os.path.join(args.run_dir, "absolute_keep"))
    b = summarize_run(os.path.join(args.run_dir, "residual_keep"))
    res = {"run_dir": args.run_dir, "absolute": a, "residual": b,
           "curves": {"absolute": curve(os.path.join(args.run_dir, "absolute_keep")),
                      "residual": curve(os.path.join(args.run_dir, "residual_keep"))}}

    v = b["val"]
    ident = v["identity_L1"]
    cv = v["macro"]["constant_velocity"]
    win_i, lose_i = episodes_better(v["per_episode"], "hold_state")
    win_cv, _ = episodes_better(v["per_episode"], "constant_velocity")
    collapsed = [t["dim"] for t in v["tokens"] if t["mode_fraction"] == 1.0]
    low_div = [t["dim"] for t in v["tokens"] if t["unique"] < 3]
    boundary = max(max(t["frac0"], t["frac255"]) for t in v["tokens"])
    right_better = [d for d in range(7, 14)
                    if v["per_dim"]["model"][d] < v["per_dim"]["hold_state"][d]]
    checks = {
        "1_residual_beats_identity": bool(v["gen_L1"] < ident),
        "2_residual_beats_constant_velocity": bool(v["gen_L1"] < cv),
        "3_residual_beats_median_delta": bool(v["gen_L1"] < v["macro"]["train_median_delta"]),
        "4a_right_arm_not_worse": bool(v["per_arm"]["right"]["model"] < v["per_arm"]["right"]["hold_state"]),
        "4b_right_6of7_dims_better": bool(len(right_better) >= 6),
        "5a_episodes_win_over_identity_ge_10": bool(win_i >= 10),
        "5b_episodes_win_over_cv_ge_10": bool(win_cv >= 10),
        "6a_no_collapsed_dims": bool(not collapsed),
        "6b_unique_tokens_ge_3": bool(not low_div),
        "6c_boundary_fraction_under_10pct": bool(boundary < 0.10),
    }
    res["acceptance"] = {"checks": checks,
                         "all_pass": all(checks.values()),
                         "val_residual_L1": v["gen_L1"], "val_identity_L1": ident,
                         "val_constant_velocity_L1": cv,
                         "val_median_delta_L1": v["macro"]["train_median_delta"],
                         "episodes_won_over_identity": win_i,
                         "episodes_lost_to_identity": lose_i,
                         "episodes_won_over_cv": win_cv,
                         "collapsed_dims": collapsed, "low_diversity_dims": low_div,
                         "max_boundary_fraction": boundary,
                         "right_dims_better_than_identity": right_better}

    print("== run:", args.run_dir)
    for name, r in (("absolute", a), ("residual", b)):
        print("%-9s result=%s steps=%d train(n=%d,eps=%d) val(n=%d,eps=%d) ckpt=%s" % (
            name, r["result"], r["steps"], r["n_train"], r["train_episodes"],
            r["n_val"], r["val_episodes"], r["grad_checkpoint"]))
        for split in ("train", "val"):
            s = r[split]
            print("   %-5s n=%-3d model=%.5f ident=%.5f cv=%.5f med=%.5f | L/R model=%.5f/%.5f ident=%.5f/%.5f cv=%.5f/%.5f | mac(model/cv)=%.5f/%.5f | delta_ratio=%.3f" % (
                split, s["n"], s["gen_L1"], s["identity_L1"], s["macro"]["constant_velocity"],
                s["macro"]["train_median_delta"],
                s["per_arm"]["left"]["model"], s["per_arm"]["right"]["model"],
                s["per_arm"]["left"]["hold_state"], s["per_arm"]["right"]["hold_state"],
                s["per_arm"]["left"]["constant_velocity"], s["per_arm"]["right"]["constant_velocity"],
                s["macro"]["model"], s["macro"]["constant_velocity"], s["delta_ratio"]))
    print("\n== val per-dim (model / identity / cv) ==")
    for d, name in enumerate(DIMS):
        print("   %-14s %.5f / %.5f / %.5f" % (name, b["val"]["per_dim"]["model"][d],
                                               b["val"]["per_dim"]["hold_state"][d],
                                               b["val"]["per_dim"]["constant_velocity"][d]))
    print("\n== val token diversity (residual) ==")
    print("   ", [(t["dim"], t["unique"], round(t["mode_fraction"], 3), round(t["frac0"], 3),
                   round(t["frac255"], 3)) for t in b["val"]["tokens"]])
    print("\n== acceptance ==")
    for k, val in checks.items():
        print("   %-40s %s" % (k, "PASS" if val else "FAIL"))
    print("   all_pass:", all(checks.values()))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
