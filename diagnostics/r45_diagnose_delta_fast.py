#!/usr/bin/env python3
"""Fast CPU-only, read-only diagnosis of R44 frozen delta-statistic mismatch.

Scans 300 episodes' 14D state/action and metadata, NOT large JPEG/HDF5 file
bytes. It CANNOT produce or attest a training manifest. Exact payload SHA
match is required; tiny numerical differences remain BLOCKED for training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "training", ROOT / "diagnostics"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from r45_delta_payload_diff import compare_delta_payloads


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def scan_delta_only(split, root, read_episode=None):
    """Full row coverage, no full-file hash; returns an explicitly weak inventory."""
    import numpy as np
    import m2_multitask_data as M2
    import r44_prepare_delta as P

    reader = read_episode or P.load_14d
    tasks = ("push_T", "stack_bowls", "sweep_blocks")
    if set(split) != set(tasks):
        raise ValueError("expected exactly 3 frozen tasks")
    deltas = {}
    stat_inventory = []
    n_train, n_dev = 0, 0
    for task in tasks:
        seen = set()
        train_parts = []
        for part in ("train", "dev"):
            for rel in split[task][part]:
                prefix, episode = M2.split_qualified(rel)
                if prefix != task or episode in seen:
                    raise ValueError(f"{task}/{episode}: invalid or duplicate split entry")
                seen.add(episode)
                path = Path(root) / task / "arx_x5" / "data" / episode
                before = path.stat()
                record = reader(path)
                after = path.stat()
                a = (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev)
                b = (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev)
                if a != b or before.st_size <= 0:
                    raise ValueError(f"{task}/{episode}: file identity changed during read")
                state, action = record["state"], record["action"]
                if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 14:
                    raise ValueError(f"{task}/{episode}: state/action must both be (T,14)")
                if not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise ValueError(f"{task}/{episode}: non-finite 14D observations")
                if state.shape[0] < 4 or not np.array_equal(
                        action[:-1], state[1:]):
                    raise ValueError(f"{task}/{episode}: action is not exact next state")
                if set(record["cameras"]) != set(P.CAMERAS):
                    raise ValueError(f"{task}/{episode}: not exactly 3 cameras")
                if record["instruction"] != M2.TASK_PROMPTS[task]:
                    raise ValueError(f"{task}/{episode}: task instruction mismatch")
                rows = int(len(state) - 3)
                if part == "train":
                    train_parts.append(action[2:-1] - state[2:-1])
                    n_train += rows
                else:
                    n_dev += rows
                stat_inventory.append({
                    "episode": f"{task}/{episode}",
                    "subset": part, "rows": rows, "size_bytes": before.st_size,
                    "mtime_ns": before.st_mtime_ns,
                    "note": "stat identity only, not HDF5 content SHA256",
                })
        if len(split[task]["train"]) != 80 or len(split[task]["dev"]) != 20:
            raise ValueError(f"{task}: expected 80 TRAIN / 20 DEV episodes")
        deltas[task] = np.concatenate(train_parts, axis=0)
    if len(stat_inventory) != 300 or n_train != 108780 or n_dev != 28268:
        raise ValueError(f"frozen dataset count mismatch: files={len(stat_inventory)}, "
                         f"TRAIN={n_train}, DEV={n_dev}")
    return deltas, stat_inventory, {"train_samples": n_train, "dev_samples": n_dev}


def diagnose(splits, root, feature, delta, scan_fn=None, quantile_forensics=False):
    import r44_delta_v1 as D
    import m2_multitask_data as M2
    import m2_task_normalizer as TN
    import numpy as np
    split = json.loads(Path(splits).read_text(encoding="utf-8"))["splits"]
    frozen = json.loads(Path(delta).read_text(encoding="utf-8"))
    D.validate_normalizer(frozen)
    split_sha = sha_file(splits)
    if frozen["fit"]["split_sha256"] != split_sha:
        raise ValueError("frozen delta normalizer does not bind this split")
    dev_lists = {t: [M2.split_qualified(e)[1] for e in v["dev"]]
                 for t, v in split.items()}
    TN.load_payload(str(feature), tasks=list(split), dev_lists=dev_lists)
    scanned = (scan_fn or scan_delta_only)(split, root)
    deltas, inventory, counts = scanned
    derived = D.fit_normalizer(deltas, split_sha256=split_sha)
    difference = compare_delta_payloads(frozen, derived)
    quantile_fingerprint = None
    if quantile_forensics:
        from r45_quantile_fingerprint import fingerprint_quantiles
        quantile_fingerprint = fingerprint_quantiles(deltas, frozen)
    outcome = "DELTA_ONLY_COMPATIBLE_NOT_FULL_ATTESTED" if (
        frozen["sha256"] == derived["sha256"]) else "BLOCKED_DELTA_MISMATCH"
    try:
        source_sha = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
            timeout=10).strip()
    except Exception:
        source_sha = "UNKNOWN_RUNTIME_SHA"
    report = {
        "schema": "r45-fast-delta-diagnosis/v1",
        "status": outcome,
        "gpu_used": False, "formal_manifest_created": False,
        "full_hdf5_sha256_checked": False,
        "train_rows_and_dev_rows": counts,
        "episode_count": len(inventory),
        "dataset_stat_inventory_only": inventory,
        "split_sha256": split_sha,
        "feature_normalizer_file_sha256": sha_file(feature),
        "delta_normalizer_file_sha256": sha_file(delta),
        "difference": difference,
        "quantile_arithmetic_fingerprint": quantile_fingerprint,
        "source_sha": source_sha,
        "script_sha256": sha_file(Path(__file__)),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "limitations": "No image data/full HDF5 SHA256 read; never use this report as r44-dataset-manifest/v1. "
                       "Even a matching delta payload does not authorize CPU Gate or GPU.",
    }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", type=Path, default=ROOT / "data/m2_splits.json")
    p.add_argument("--hdf5-root", type=Path, required=True)
    p.add_argument("--feature-normalizer", type=Path, required=True)
    p.add_argument("--delta-normalizer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="new directory only")
    p.add_argument("--quantile-forensics", action="store_true",
                   help="compare float32/float64 quantile interpolation variants on the same TRAIN rows")
    args = p.parse_args()
    if args.out.exists():
        print("BLOCKED_OUTPUT_EXISTS", file=sys.stderr)
        return 2
    try:
        report = diagnose(args.splits, args.hdf5_root,
                          args.feature_normalizer, args.delta_normalizer,
                          quantile_forensics=args.quantile_forensics)
        args.out.mkdir(parents=True, exist_ok=False)
        f = args.out / "report.json"
        f.write_text(json.dumps(report, indent=2, sort_keys=True,
                                allow_nan=False) + "\n", encoding="utf-8")
        (args.out / "SHA256SUMS").write_text(
            f"{sha_file(f)}  report.json\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "out": str(args.out),
                          "changed_fields": report["difference"]["changed_field_count"]},
                         sort_keys=True))
        return 0 if report["status"].startswith("DELTA_ONLY_COMPATIBLE") else 3
    except Exception as exc:
        print(f"BLOCKED_FAST_DELTA_DIAGNOSIS: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
