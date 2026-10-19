#!/usr/bin/env python3
"""R6: verify the 28D checkpoint contract end to end (CPU only).

Checks that each pilot checkpoint declares its proprio interface, that the declared
width matches the projector file on disk, that the legacy 14D path still resolves to
mode=none, and that the two arms are genuinely different models.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))
import proprio_history as PH  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def inspect(run_dir):
    meta_path = os.path.join(run_dir, "run_meta.json")
    proj_path = os.path.join(run_dir, "checkpoints", "final", "proprio_projector--pure.pt")
    if not (os.path.isfile(meta_path) and os.path.isfile(proj_path)):
        raise SystemExit("missing checkpoint artifacts in " + run_dir)
    meta = json.load(open(meta_path))
    sd = torch.load(proj_path, map_location="cpu")
    width = int(sd["fc1.weight"].shape[1])
    mode = PH.resolve_mode(meta)
    PH.verify_projector_width(meta, width, mode)
    checks = {
        "run": os.path.abspath(run_dir),
        "declared_proprio_dim": meta.get("proprio_dim"),
        "declared_history_mode": meta.get("proprio_history_mode"),
        "resolved_mode": mode,
        "projector_fc1_width": width,
        "projector_params": int(sum(v.numel() for v in sd.values())),
        "first_frame_policy": meta.get("first_frame_policy"),
        "adapter_sha256": meta.get("adapter_sha256"),
        "projector_sha256": meta.get("proprio_projector_sha256"),
        "projector_sha256_matches_file": (sha(proj_path) == meta.get("proprio_projector_sha256")),
        "pmin_len": len(meta.get("pmin", [])) or None, "pmax_len": len(meta.get("pmax", [])) or None,
        "action_target_mode": meta.get("action_target_mode"),
        "codec_mode": meta.get("codec_mode"),
        "width_matches_declaration": width == PH.proprio_dim(mode),
    }
    # pmin/pmax live in push_T_bounds.npz, not necessarily in checkpoint metadata:
    # require them only when the checkpoint does carry them.
    optional_bounds_ok = ((checks["pmin_len"] in (None, 14)) and (checks["pmax_len"] in (None, 14)))
    checks["PASS"] = bool(checks["width_matches_declaration"] and checks["projector_sha256_matches_file"]
                          and optional_bounds_ok)
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duplicate", required=True)
    ap.add_argument("--true", required=True)
    ap.add_argument("--legacy", default="", help="optional R5 14D run directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dup = inspect(args.duplicate)
    tru = inspect(args.true)
    legacy = inspect(args.legacy) if args.legacy else None
    out = {
        "duplicate_arm": dup, "true_arm": tru, "legacy_14d_arm": legacy,
        "arms_are_different_models": dup["adapter_sha256"] != tru["adapter_sha256"],
        "legacy_resolves_to": None if legacy is None else legacy["resolved_mode"],
        "same_proprio_dim_across_arms": dup["declared_proprio_dim"] == tru["declared_proprio_dim"],
        "same_projector_param_count": dup["projector_params"] == tru["projector_params"],
        "note": ("legacy 14D checkpoints carry no proprio metadata and MUST resolve to mode=none; "
                 "28D checkpoints must declare proprio_dim and proprio_history_mode"),
    }
    out["PASS"] = bool(dup["PASS"] and tru["PASS"] and out["arms_are_different_models"]
                       and out["same_proprio_dim_across_arms"] and out["same_projector_param_count"]
                       and (legacy is None or (legacy["PASS"] and legacy["resolved_mode"] == PH.NONE)))
    json.dump(out, open(args.out, "w"), indent=1)
    for k in ("duplicate_arm", "true_arm", "legacy_14d_arm"):
        if out[k]:
            print("[%s] mode=%s dim=%s fc1_width=%s params=%d sha_ok=%s PASS=%s" % (
                k, out[k]["resolved_mode"], out[k]["declared_proprio_dim"], out[k]["projector_fc1_width"],
                out[k]["projector_params"], out[k]["projector_sha256_matches_file"], out[k]["PASS"]))
    print("arms_are_different_models=%s same_param_count=%s -> PASS=%s" % (
        out["arms_are_different_models"], out["same_projector_param_count"], out["PASS"]))
    return 0 if out["PASS"] else 4


if __name__ == "__main__":
    sys.exit(main())
