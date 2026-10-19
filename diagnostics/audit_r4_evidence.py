#!/usr/bin/env python3
"""Read-only audit of prior R4 A/B evidence; don't conflate action states with token diversity."""
import argparse
import collections
import json
from pathlib import Path


def audit(path):
    root = Path(path)
    config = json.loads((root / "residual_keep_config.json").read_text())
    ab = json.loads((root / "target_comparison.json").read_text())
    a = json.loads((root / "absolute_keep_status.json").read_text())
    r = json.loads((root / "residual_keep_status.json").read_text())
    if a.get("result") != "DONE" or r.get("result") != "DONE":
        raise ValueError("R4 runs did not complete")
    tokens = r["final_val_eval"]["gen_tokens_per_sample"]
    counts = [collections.Counter(row[j] for row in tokens) for j in range(14)]
    train_eps = collections.Counter(e for e, _ in config["samples"])
    val_eps = collections.Counter(e for e, _ in config["val_samples"])
    if set(train_eps) & set(val_eps):
        raise ValueError("training and validation overlap")
    if len(tokens) != len(config["val_samples"]):
        raise ValueError("R4 token row count != validation manifest")
    return {
        "source": str(root),
        "R4_train_frames": len(config["samples"]),
        "R4_train_episodes": len(train_eps),
        "R4_train_per_episode": dict(sorted(train_eps.items())),
        "R4_val_frames": len(config["val_samples"]),
        "R4_val_episodes": len(val_eps),
        "R4_val_per_episode": dict(sorted(val_eps.items())),
        "residual_val_model_L1": float(ab["val"]["residual_L1"]),
        "absolute_val_model_L1": float(ab["val"]["absolute_L1"]),
        "val_hold_position_L1": float(ab["val"]["identity_L1"]),
        "residual_val_unique_token_rows": len({tuple(x) for x in tokens}),
        "residual_val_unique_bins_by_dim": [len(x) for x in counts],
        "residual_val_mode_fraction_by_dim": [max(x.values()) / len(tokens) for x in counts],
        "caveats": [
            "20 unique absolute joint commands does not prove 20 meaningful residual policies",
            "20 frames are not 20 independent episodes",
            "residual validation still loses to identity baseline",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", default="evidence/r4_ab_run_20261019T044749Z")
    p.add_argument("--out", default="")
    args = p.parse_args()
    result = audit(args.evidence)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if args.out:
        Path(args.out).write_text(payload + "\n")


if __name__ == "__main__":
    main()
