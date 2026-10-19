"""R6 Phase 1 / Task C: 256-bin action quantization ORACLE error.

Encodes the *true continuous expert target* with the R2-fixed codec
(ACTION_TOKEN_BEGIN_IDX+1+idx, idx in 0..255, LEVELS=linspace(-1,1,256)) and the
train-only q01/q99 that R5 actually used, decodes it back, and measures the
physical L1 that no predictor can go below. Quantization is decomposed into a
grid (rounding/floor) part and a range-clipping part, so clipping is never hidden
inside the "quantization error" figure.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_CODE = os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure")
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
import r6_data as D          # noqa: E402
import action_codec as AC    # noqa: E402
import action_target as AT   # noqa: E402


def quantiles(split, mode):
    vals = [AT.training_target(ep["state"], ep["action"], mode) for ep in D.load_episodes(split["train"])]
    allt = np.concatenate(vals, axis=0)
    return (np.quantile(allt, 0.01, axis=0).astype(np.float32),
            np.quantile(allt, 0.99, axis=0).astype(np.float32), allt)


def quantize(target, q01, q99):
    """Fixed codec: clip -> grid index -> LEVELS, plus the pre-quantization value."""
    t = np.asarray(target, np.float32)
    raw = 2.0 * (t - q01) / (q99 - q01 + 1e-8) - 1.0        # unclipped normalized value
    clipped = np.clip(raw, -1.0, 1.0)
    idx = AC.encode_indices(clipped, mode=AC.MODE_FIXED)
    rec = AC.denormalize(AC.decode_indices(idx, mode=AC.MODE_FIXED), q01, q99)
    inv = AC.denormalize(clipped, q01, q99)                  # exact value of the clipped input
    return {"idx": idx, "rec": rec, "inv": inv, "raw": raw,
            "clipped_mask": np.abs(raw) > 1.0,
            "step": (np.asarray(q99) - np.asarray(q01)) / (AC.N_BINS - 1)}


def block(pred_target, gt_target, q01, q99, name, thresholds=(0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3)):
    """Continuous vs quantized physical L1 for one predictor, with error decomposition."""
    q = quantize(pred_target, q01, q99)
    gt = np.asarray(gt_target, np.float64)
    e_total = np.abs(q["rec"] - gt)
    e_round = np.abs(q["rec"] - q["inv"])
    e_clip = np.abs(q["inv"] - gt)
    cont = np.abs(np.asarray(pred_target, np.float64) - gt)
    dirs = []
    for thr in thresholds:
        m = np.abs(gt) >= thr
        n = int(m.sum())
        if n == 0:
            dirs.append({"threshold": thr, "n": 0})
            continue
        flip = np.sign(q["rec"][m]) != np.sign(gt[m])
        dirs.append({"threshold": thr, "n": n, "fraction_of_entries": float(n / gt.size),
                     "quantized_sign_flip_rate": float(flip.mean())})
    return {
        "predictor": name,
        "continuous_L1": float(cont.mean()),
        "quantized_L1": float(e_total.mean()),
        "quantization_penalty": float(e_total.mean() - cont.mean()),
        "rounding_L1": float(e_round.mean()),
        "clipping_L1": float(e_clip.mean()),
        "clipped_entry_fraction": float(q["clipped_mask"].mean()),
        "clipped_dims": sorted({D.DIM_NAMES[d] for d in np.where(q["clipped_mask"].any(axis=0))[0]}),
        "mean_signed_error_per_dim": (q["rec"] - gt).mean(axis=0).tolist(),
        "per_dim_quantized_L1": e_total.mean(axis=0).tolist(),
        "per_arm_quantized_L1": {"left": float(e_total[:, D.LEFT].mean()),
                                 "right": float(e_total[:, D.RIGHT].mean())},
        "direction": dirs,
        "quantization_step_per_dim": q["step"].astype(float).tolist(),
        "max_normalized_quant_error": AC.MAX_QUANT_ERROR,
    }


def run():
    split = D.load_split()
    res = {"codec": AC.summary(), "dim_names": D.DIM_NAMES,
           "quantile_source": "TRAIN episodes only (all rows), identical to action_stats()",
           "splits": {}}
    train_eps = D.load_episodes(split["train"])
    val_eps = D.load_episodes(split["val"])

    # ---- Task B's best history-only linear map, refitted here so the quantization
    #      comparison for "history linear" is self-contained (TRAIN-fitted).
    Vtr = np.concatenate([(e["state"][1:-1] - e["state"][:-2]) for e in train_eps])
    Dtr = np.concatenate([(e["action"][1:-1] - e["state"][1:-1]) for e in train_eps])
    slope = np.zeros(14); inter = np.zeros(14)
    for d in range(14):
        x = Vtr[:, d]
        if x.var() <= 0:
            inter[d] = Dtr[:, d].mean()
            continue
        slope[d] = float(np.cov(x, Dtr[:, d], bias=True)[0, 1] / x.var())
        inter[d] = float(Dtr[:, d].mean() - slope[d] * x.mean())

    for split_name in ("val", "train"):
        eps = train_eps if split_name == "train" else val_eps
        res["splits"][split_name] = {}
        for mode in (AT.RESIDUAL, AT.ABSOLUTE):
            q01, q99, _allt = quantiles(split, mode)
            S, A, V = [], [], []
            for e in eps:
                t = D.valid_frames(len(e["state"]))
                S.append(e["state"][t]); A.append(e["action"][t])
                V.append(e["state"][t] - e["state"][t - 1])
            S, A, V = np.concatenate(S), np.concatenate(A), np.concatenate(V)
            gt_target = AT.training_target(S, A, mode)
            if mode == AT.RESIDUAL:
                preds = {"oracle_true_delta": gt_target,
                         "identity_zero_delta": np.zeros_like(gt_target),
                         "constant_velocity": V,
                         "train_linear_history": slope[None, :] * V + inter[None, :]}
            else:
                preds = {"oracle_true_action": gt_target,
                         "identity_current_state": S,
                         "constant_velocity": S + V,
                         "train_linear_history": S + slope[None, :] * V + inter[None, :]}
            blocks = {k: block(v, gt_target, q01, q99, k) for k, v in preds.items()}
            extra = {}
            for k, v in preds.items():
                q = quantize(v, q01, q99)
                cmd = q["rec"] if mode == AT.ABSOLUTE else S + q["rec"]
                extra[k] = float(np.abs(cmd - A).mean())
            res["splits"][split_name][mode] = {
                "q01": q01.tolist(), "q99": q99.tolist(),
                "target_range_per_dim": (q99 - q01).astype(float).tolist(),
                "physical_step_per_dim": ((q99 - q01) / (AC.N_BINS - 1)).astype(float).tolist(),
                "n_frames": int(len(S)),
                "blocks": blocks,
                "physical_L1_sent_to_robot": extra,
            }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = run()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    for split_name, modes in res["splits"].items():
        for mode, b in modes.items():
            print("[%s/%s] frames=%d  step(mean)=%.6f" % (
                split_name, mode, b["n_frames"],
                float(np.mean(b["physical_step_per_dim"]))))
            for k, blk in b["blocks"].items():
                print("   %-22s cont=%.6f quant=%.6f (+%.6f) round=%.6f clip=%.6f clipped=%.4f" % (
                    k, blk["continuous_L1"], blk["quantized_L1"], blk["quantization_penalty"],
                    blk["rounding_L1"], blk["clipping_L1"], blk["clipped_entry_fraction"]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
