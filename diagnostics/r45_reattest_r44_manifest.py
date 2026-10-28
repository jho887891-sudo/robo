#!/usr/bin/env python3
"""Re-attest R44 frozen dataset manifest from original HDF5 without CUDA or refitting for use.

The complete TRAIN/DEV HDF5 payload is inspected and stable-SHA256 hashed.
The existing 42D and delta normalizers are preserved byte-for-byte.
A TRAIN-only delta re-fit is used *only* as a proof of compatibility and its
payload hash must match the supplied frozen delta normalizer exactly.
Do not confuse this with recovering the missing historical manifest bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "training", REPO / "diagnostics"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def sha_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def validated_manifest(*, splits_path, root, feature_path, delta_path,
                       scan_fn=None):
    import r44_delta_v1 as D
    import r44_prepare_delta as P
    import m2_multitask_data as M2
    import m2_task_normalizer as TN
    from r44_preflight_contract import make_training_manifest

    split_doc = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    splits = split_doc["splits"]
    tasks = ("push_T", "stack_bowls", "sweep_blocks")
    if set(splits) != set(tasks):
        raise ValueError("frozen split must cover exactly three tasks")
    episode_keys = []
    for task in tasks:
        train, dev = splits[task]["train"], splits[task]["dev"]
        if not train or not dev:
            raise ValueError(f"{task}: both TRAIN and DEV must exist")
        train_names = [M2.split_qualified(e)[1] for e in train]
        dev_names = [M2.split_qualified(e)[1] for e in dev]
        if len(set(train_names)) != len(train_names) or len(set(dev_names)) != len(dev_names):
            raise ValueError(f"{task}: split has duplicate episode names")
        if set(train_names) & set(dev_names):
            raise ValueError(f"{task}: TRAIN/DEV overlap")
        episode_keys.extend(f"{task}/{name}" for name in train_names + dev_names)
    if len(episode_keys) != 300 or len(set(episode_keys)) != 300:
        raise ValueError(f"frozen episode count must be exactly 300, got {len(episode_keys)}")

    dev_lists = {t: [M2.split_qualified(x)[1] for x in splits[t]["dev"]] for t in tasks}
    TN.load_payload(str(feature_path), tasks=list(tasks), dev_lists=dev_lists)
    frozen = json.loads(Path(delta_path).read_text(encoding="utf-8"))
    D.validate_normalizer(frozen)
    split_sha = sha_file(splits_path)
    if frozen.get("fit", {}).get("split_sha256") != split_sha:
        raise ValueError("existing delta normalizer split hash differs from frozen TRAIN/DEV")

    # The established R44 scanner verifies all 300 original episode files,
    # HDF5 14D state/action, three cameras, instruction, and A_t==S_(t+1),
    # and SHA256 hashes every HDF5 with a stat identity check.
    scanner = scan_fn or P.scan
    audit, deltas = scanner(splits, str(Path(root).resolve()))
    blocks = audit["tasks"]
    if set(blocks) != set(tasks):
        raise ValueError("scanner did not return all three tasks")
    if any(b["excluded"] for b in blocks.values()) or audit["hash_errors"]:
        raise ValueError("missing/invalid HDF5 episodes or stable hash errors")
    if audit["contract_violations"] or audit["max_abs_action_vs_next_state"] != 0.0:
        raise ValueError("A_t != S_(t+1) for one or more HDF5 episodes")
    if audit["instruction_violations"]:
        raise ValueError("HDF5 instruction mismatch")
    if set(audit["instructions"]) != set(tasks):
        raise ValueError("HDF5 task instruction set incomplete")
    for task in tasks:
        if audit["instructions"][task]["text"] != M2.TASK_PROMPTS[task]:
            raise ValueError(f"{task}: actual HDF5 instruction differs from frozen prompt")
        if blocks[task]["train_episodes"] != len(splits[task]["train"]):
            raise ValueError(f"{task}: TRAIN episodes not fully attested")
        if blocks[task]["dev_episodes"] != len(splits[task]["dev"]):
            raise ValueError(f"{task}: DEV episodes not fully attested")
        if int(blocks[task]["train_samples"]) != int(frozen["tasks"][task]["n_rows"]):
            raise ValueError(f"{task}: frozen delta row count differs from scanned TRAIN")
    if int(audit["totals"]["train_samples"]) != 108780:
        raise ValueError("TRAIN sample total != 108780")
    if int(audit["totals"]["dev_samples"]) != 28268:
        raise ValueError("DEV sample total != 28268")
    if set(audit["files"]) != set(episode_keys):
        raise ValueError("file hashes do not cover exactly the frozen 300 episodes")

    reconstructed = D.fit_normalizer(deltas, split_sha256=split_sha)
    if reconstructed["sha256"] != frozen["sha256"]:
        raise ValueError(
            "BLOCKED_DELTA_MISMATCH: TRAIN-derived delta normalizer does not "
            "match supplied frozen payload, refuse to re-fit or replace it")
    manifest = make_training_manifest(
        split_sha256=split_sha,
        feature_normalizer_sha256=sha_file(feature_path),
        delta_normalizer_sha256=sha_file(delta_path),
        delta_normalizer_payload_sha256=frozen["sha256"],
        files=audit["files"],
        instructions=audit["instructions"],
    )
    return manifest, {
        "schema": "r45-r44-manifest-reattest/v1",
        "status": "CPU_MANIFEST_REATTEST_PASS",
        "source": "new re-attestation of original frozen HDF5, NOT recovered historical bytes",
        "file_hashes": len(audit["files"]),
        "train_samples": audit["totals"]["train_samples"],
        "dev_samples": audit["totals"]["dev_samples"],
        "split_sha256": split_sha,
        "feature_normalizer_sha256": sha_file(feature_path),
        "delta_normalizer_file_sha256": sha_file(delta_path),
        "delta_normalizer_payload_sha256": frozen["sha256"],
        "recomputed_delta_payload_sha256": reconstructed["sha256"],
        "hdf5_root": str(Path(root).resolve()),
        "lora_training_config_approved": False,
        "gpu_smoke_authorized": False,
        "explanation": "No model loaded, no optimizer step, no new normalizer applied.",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, default=REPO / "data/m2_splits.json")
    ap.add_argument("--hdf5-root", type=Path, required=True)
    ap.add_argument("--feature-normalizer", type=Path, required=True)
    ap.add_argument("--delta-normalizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="must not exist")
    args = ap.parse_args(argv)
    if args.out.exists():
        print("BLOCKED_OUTPUT_EXISTS: never overwrite frozen evidence", file=sys.stderr)
        return 2
    for path in (args.splits, args.feature_normalizer, args.delta_normalizer):
        if not path.is_file():
            print(f"BLOCKED_MISSING_FILE: {path}", file=sys.stderr)
            return 2
    if not args.hdf5_root.is_dir():
        print(f"BLOCKED_MISSING_HDF5_ROOT: {args.hdf5_root}", file=sys.stderr)
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
    try:
        manifest, report = validated_manifest(
            splits_path=args.splits, root=args.hdf5_root,
            feature_path=args.feature_normalizer, delta_path=args.delta_normalizer)
        # Full validation happens before new evidence directory is created.
        out = args.out.resolve()
        out.mkdir(parents=True, exist_ok=False)
        files = {
            "r44_dataset_manifest.json": manifest,
            "report.json": report,
        }
        for name, payload in files.items():
            target = out / name
            target.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                         allow_nan=False) + "\n", encoding="utf-8")
        with (out / "SHA256SUMS").open("w", encoding="utf-8") as fh:
            for name in sorted(files):
                fh.write(f"{sha_file(out / name)}  {name}\n")
        print(json.dumps({"status": "CPU_MANIFEST_REATTEST_PASS",
                          "manifest_path": str(out / "r44_dataset_manifest.json"),
                          "evidence": str(out)}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"BLOCKED_MANIFEST_REATTEST: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
