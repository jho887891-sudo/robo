#!/usr/bin/env python3
"""Compare completed R4 absolute / residual GPU runs using the same samples.

NO training is launched here. Fails closed on differing datasets/configuration.
Both primary metrics are in absolute physical joint units: deployment translates
residual predictions back to absolute commands before scoring.
"""
import argparse
import json
from pathlib import Path


def load_run(path):
    root = Path(path)
    with open(root / "_snapshot" / "config.json") as fh:
        config = json.load(fh)
    with open(root / "status.json") as fh:
        status = json.load(fh)
    if status["result"] != "DONE" or status.get("final_val_eval") is None:
        raise ValueError("missing completed train+held-out validation result: " + str(root))
    return config, status


def compare(absolute_dir, residual_dir):
    a, sa = load_run(absolute_dir)
    r, sr = load_run(residual_dir)
    if a["action_target_mode"] != "absolute" or r["action_target_mode"] != "residual":
        raise ValueError("wrong experiment target mode")
    for key in ("samples", "val_samples", "steps", "lr", "mechanism"):
        if a[key] != r[key]:
            raise ValueError("unmatched A/B experimental configuration: " + key)
    if a["mechanism"] != "keep":
        raise ValueError("A/B requires the proven causal autoregressive mechanism")
    if a.get("sample_source") != "train" or r.get("sample_source") != "train":
        raise ValueError("A/B must fit train-only samples, never held-out samples")
    train_episodes = {e for e, _ in a["samples"]}
    val_episodes = {e for e, _ in a["val_samples"]}
    if train_episodes & val_episodes:
        raise ValueError("train/validation episode leakage in A/B manifest")
    if a.get("hashes") and a.get("hashes") != r.get("hashes"):
        raise ValueError("A/B source snapshots differ")
    for split_key in ("final_eval", "final_val_eval"):
        x, y = sa[split_key], sr[split_key]
        if x.get("n") != y.get("n"):
            raise ValueError("unmatched evaluation sample counts")
        if abs(x["identity_L1"] - y["identity_L1"]) > 1e-6:
            raise ValueError("identity baseline mismatch across target modes")
        if abs(x["repeat_previous_L1"] - y["repeat_previous_L1"]) > 1e-6:
            raise ValueError("repeat-previous baseline mismatch across target modes")
        expected_split = "train" if split_key == "final_eval" else "val"
        if x.get("split") != expected_split or y.get("split") != expected_split:
            raise ValueError("incorrectly labeled evaluation split")
    out = {
        "comparison": "absolute_position_vs_residual_position",
        "samples": len(a["samples"]),
        "validation_samples": len(a["val_samples"]),
        "updates": a["steps"],
        "mechanism": "keep",
        "metric": "mean absolute error of 14D absolute joint positions (physical units)",
    }
    for name, key in (("train", "final_eval"), ("val", "final_val_eval")):
        x, y = sa[key], sr[key]
        out[name] = {
            "absolute_L1": float(x["gen_L1"]),
            "residual_L1": float(y["gen_L1"]),
            "identity_L1": float(x["identity_L1"]),
            "repeat_previous_L1": float(x["repeat_previous_L1"]),
            "residual_minus_absolute_L1": float(y["gen_L1"] - x["gen_L1"]),
            "residual_improves": bool(y["gen_L1"] < x["gen_L1"]),
            "absolute_beats_identity": bool(x["gen_L1"] < x["identity_L1"]),
            "residual_beats_identity": bool(y["gen_L1"] < y["identity_L1"]),
            "absolute_beats_repeat_previous": bool(x["gen_L1"] < x["repeat_previous_L1"]),
            "residual_beats_repeat_previous": bool(y["gen_L1"] < y["repeat_previous_L1"]),
            "absolute_pred_delta": float(x["pred_delta"]),
            "residual_pred_delta": float(y["pred_delta"]),
            "expert_delta": float(x["gt_delta"]),
            "absolute_per_dim": x["per_dim_gen"],
            "residual_per_dim": y["per_dim_gen"],
            "identity_per_dim": x["per_dim_identity"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--absolute", required=True, help="completed absolute run output directory")
    ap.add_argument("--residual", required=True, help="completed residual run output directory")
    ap.add_argument("--out", default="", help="optional output JSON")
    args = ap.parse_args()
    result = compare(args.absolute, args.residual)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(payload + "\n")


if __name__ == "__main__":
    main()
