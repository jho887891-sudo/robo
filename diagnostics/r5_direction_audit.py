#!/usr/bin/env python3
"""R5 post-hoc audit from committed, already-generated token evidence only.

Absolute-token Pearson correlation is for absolute joint POSITION, not
velocity / action delta. Residual-token Pearson is for quantized DELTA.
This tool cannot establish correlation against unquantized physical targets:
the original continuous GT/state arrays are not committed to GitHub.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def pearson_or_none(x, y):
    """Undefined, not zero, when either variable has zero variance."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        raise ValueError("Pearson requires paired vectors of at least two elements")
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.sqrt(np.sum(xc * xc) * np.sum(yc * yc)))
    if denom < 1e-12:
        return None
    return float(np.sum(xc * yc) / denom)


def decode_tokens(token_rows, q01, q99):
    bins = np.asarray(token_rows, np.int64)
    lo, hi = np.asarray(q01, np.float64), np.asarray(q99, np.float64)
    if bins.ndim != 2 or bins.shape[1] != 14 or lo.shape != (14,) or hi.shape != (14,):
        raise ValueError("require (n,14) token rows and 14D q01/q99")
    if np.any((bins < 0) | (bins > 255)) or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("invalid tokens or nonfinite statistics")
    if np.any(hi < lo):
        raise ValueError("q99 less than q01")
    return lo[None, :] + (bins / 255.0) * (hi - lo)[None, :]


def direction_report(config, val_eval, threshold=0.002):
    """Compare predicted vs GT QUANTIZED targets in their physical units.

    Sign agreement is restricted to |quantized_GT| > threshold to avoid
    counting near-zero movement quantized to a slightly negative center bin
    as a meaningful correct direction. Missing-dimension correlation => null.
    """
    mode = config["action_target_mode"]
    if mode not in ("absolute", "residual"):
        raise ValueError("unknown action target mode")
    p = decode_tokens(val_eval["gen_tokens_per_sample"], config["q01"], config["q99"])
    g = decode_tokens(val_eval["gt_tokens_per_sample"], config["q01"], config["q99"])
    if p.shape != g.shape or len(p) != len(config["val_samples"]):
        raise ValueError("pred/GT token rows and validation sample manifest differ")
    corr = [pearson_or_none(p[:, j], g[:, j]) for j in range(14)]
    good = [x for x in corr if x is not None]
    if not good:
        raise ValueError("all dimensions constant; no Pearson correlation is defined")
    # Directional sign is meaningful only for DELTA targets. Signs of
    # absolute joint positions measure position sign, not motion direction.
    if mode == "residual":
        active = np.abs(g) > threshold
        matched_sign = np.sign(p) == np.sign(g)
        active_count = int(active.sum())
        if not active_count:
            raise ValueError("no active quantized residual targets at chosen threshold")
        pred_confident = np.abs(p) > threshold
        active_fraction = active_count / g.size
        sign_accuracy = float(np.mean(matched_sign[active]))
        confident_accuracy = float(np.mean((matched_sign & pred_confident)[active]))
        near_zero_pred = int(np.sum(active & ~pred_confident))
    else:
        active_count = None
        active_fraction = None
        sign_accuracy = None
        confident_accuracy = None
        near_zero_pred = None
    return {
        "target_mode": mode,
        "correlation_quantity": "absolute_position" if mode == "absolute" else "quantized_residual_increment",
        "corr_per_dim": corr,
        "undefined_corr_dims": [j for j, v in enumerate(corr) if v is None],
        "mean_corr_defined_dims_only": float(np.mean(good)),
        "quantized_target_bin_widths_physical_units": ((np.asarray(config["q99"])-np.asarray(config["q01"]))/255.0).tolist(),
        "sign_threshold_physical_units": threshold,
        "all_dim_frames": int(g.size),
        "active_gt_dim_frames": active_count,
        "active_gt_fraction": active_fraction,
        "active_gt_sign_accuracy": sign_accuracy,
        "active_gt_sign_accuracy_confident_prediction": confident_accuracy,
        "active_gt_predicted_near_zero_count": near_zero_pred,
        "n_val_frames": int(len(p)),
        "notes": [
            "quantized labels are an approximation to physical ground-truth targets",
            "absolute-target correlation cannot be interpreted as motion-direction correlation",
            "sign accuracy on near-zero values is unstable and depends on quantization centering",
            "a gripper dimension can have different physical units from a joint dimension",
        ],
    }


def audit(evidence_dir, threshold=0.002):
    root = Path(evidence_dir)
    result = {"evidence": str(root), "threshold": threshold, "modes": {}}
    for mode in ("absolute", "residual"):
        cfg = json.loads((root / (mode + "_keep_config.json")).read_text())
        status = json.loads((root / (mode + "_keep_status.json")).read_text())
        if status.get("result") != "DONE" or status.get("final_val_eval", {}).get("split") != "val":
            raise ValueError("experiment incomplete or incorrectly labeled: " + mode)
        result["modes"][mode] = direction_report(cfg, status["final_val_eval"], threshold)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="evidence/r5_ab_run_20261019T103523Z")
    ap.add_argument("--threshold", type=float, default=0.002)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.threshold < 0 or not np.isfinite(args.threshold):
        raise SystemExit("invalid threshold")
    result = audit(args.evidence, args.threshold)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
