#!/usr/bin/env python3
"""Read-only R45 dataset root discovery from explicit or repository-documented paths.

Does not search the entire host, modify mounts, access CUDA, download HDF5, or
infer that missing paths mean the dataset was deleted. Emits auditable JSON;
a complete file inventory is still NOT an HDF5 content/contract validation.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HISTORICAL_ROOT = Path("/home/T7/ojh/robot_actual/data/data/hdf5")
REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def inspect_root(root, splits):
    """Stat all TRAIN/DEV episode files without loading any HDF5 contents."""
    root = root.expanduser()
    record = {"candidate": str(root), "exists": root.is_dir(),
              "is_symlink": root.is_symlink(), "complete": False}
    if not record["exists"]:
        return record
    train = []
    dev = []
    absent = []
    for task in sorted(splits):
        for split in ("train", "dev"):
            for entry in splits[task].get(split, []):
                # Match the exact frozen R45.build_index() path mapping.
                name = Path(entry).name
                filename = root / task / "arx_x5" / "data" / name
                try:
                    stat = filename.stat()
                    if not filename.is_file() or stat.st_size <= 0:
                        absent.append((task, split, name))
                    elif split == "train":
                        train.append((task, name, int(stat.st_size)))
                    else:
                        dev.append((task, name, int(stat.st_size)))
                except OSError:
                    absent.append((task, split, name))
    record["expected_train"] = sum(len(v.get("train", [])) for v in splits.values())
    record["expected_dev"] = sum(len(v.get("dev", [])) for v in splits.values())
    record["present_train"] = len(train)
    record["present_dev"] = len(dev)
    record["missing_count"] = len(absent)
    record["missing_examples"] = ["/".join(k) for k in absent[:5]]
    record["complete"] = (not absent and
                          len(train) == record["expected_train"] and
                          len(dev) == record["expected_dev"])
    # Path-independent identity for inventory, not a substitute for file SHA256s.
    record["inventory_sha256"] = hashlib.sha256(json.dumps(
        sorted(train + dev), separators=(",", ":")).encode()).hexdigest()
    return record


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, default=REPO / "data/m2_splits.json")
    ap.add_argument("--root-candidate", action="append", default=[],
                    help="explicitly supplied real HDF5 mount root; can repeat")
    ap.add_argument("--out", type=Path,
                    help="optional, new JSON evidence path (will not overwrite)")
    args = ap.parse_args(argv)

    if not args.splits.is_file():
        print("BLOCKED_SPLITS: frozen split manifest is not available", file=sys.stderr)
        return 2
    payload = json.loads(args.splits.read_text(encoding="utf-8"))
    splits = payload["splits"]
    candidates = []
    if os.environ.get("M2_HDF5_ROOT"):
        candidates.append((os.environ["M2_HDF5_ROOT"], "M2_HDF5_ROOT"))
    candidates.append((str(HISTORICAL_ROOT), "data/README.md / R44 script historical path"))
    candidates += [(p, "explicit --root-candidate") for p in args.root_candidate]
    unique = {}
    for path, provenance in candidates:
        p = os.path.abspath(os.path.expanduser(path))
        unique.setdefault(p, provenance)
    inspected = [dict(inspect_root(Path(path), splits), source=prov)
                 for path, prov in unique.items()]
    complete = [r for r in inspected if r["complete"]]
    status = "DISCOVERED_CANDIDATE_NEEDS_HDF5_GATE" if len(complete) == 1 else (
        "BLOCKED_AMBIGUOUS_ROOT" if len(complete) > 1 else "BLOCKED_NO_COMPLETE_DATASET")
    report = {"schema": "r45-dataset-path-discovery/v1",
              "status": status,
              "splits_sha256": sha256(args.splits),
              "complete_candidate_count": len(complete),
              "candidates": inspected,
              "chosen_root": complete[0]["candidate"] if len(complete) == 1 else None,
              "limitations": ("File presence/size are checked read-only. Contents, HDF5 "
                              "metadata, SHA256 identity and correct mount provenance "
                              "remain unverified. Missing mount does not prove deletion.")}
    if args.out is not None:
        path = args.out
        if path.exists():
            print(f"BLOCKED_OUTPUT_EXISTS: {path}", file=sys.stderr)
            return 2
        path.parent.mkdir(parents=True, exist_ok=True)
        # O_EXCL prevents overwriting even under a concurrent race.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if len(complete) == 1 else 2


if __name__ == "__main__":
    raise SystemExit(run())
