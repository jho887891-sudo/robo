#!/usr/bin/env python3
"""Read-only CPU attestation of the fixed R46 GPU Smoke inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for name in ("repo", "splits", "hdf5_root", "manifest", "sample_plan",
                 "model_identity", "base_model", "out"):
        ap.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    sys.path.insert(0, str(args.repo.resolve()))
    from diagnostics.r46_smoke_contract import (SELECTION_SOURCE_SHA,
                                                validate_sample_plan)
    from diagnostics.r46_smoke_supervisor import (model_identity_sha_list,
                                                  verify_model_identity)
    split = json.loads(args.splits.read_text())
    manifest = json.loads(args.manifest.read_text())
    samples = json.loads(args.sample_plan.read_text())
    identity = json.loads(args.model_identity.read_text())
    sample_check = validate_sample_plan(
        samples, split_doc=split, manifest=manifest, split_sha256=sha(args.splits),
        manifest_sha256=sha(args.manifest), hdf5_root=args.hdf5_root,
        source_sha=SELECTION_SOURCE_SHA, verify_files=True,
    )
    shards = verify_model_identity(identity, args.base_model)
    args.out.mkdir(parents=True, mode=0o700)
    report = {"schema": "r46-gpu-inputs-cpu/v1", "status": "PASS_INPUT_ATTESTATION_CPU",
              "sample_plan_sha256": sha(args.sample_plan),
              "manifest_sha256": sha(args.manifest),
              "model_identity_sha256": sha(args.model_identity),
              "approved_model_sha_list": model_identity_sha_list(identity),
              "model_shards": [x["filename"] for x in shards],
              "sample_check": sample_check,
              "limits": "CPU file/content checks only; no CUDA, weight load or GPU Smoke"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (args.out / "SHA256SUMS").write_text(f"{sha(args.out / 'report.json')}  report.json\n")
    print(json.dumps({k: report[k] for k in ("status", "sample_plan_sha256",
                                           "manifest_sha256", "model_identity_sha256",
                                           "model_shards")}, sort_keys=True))


if __name__ == "__main__":
    main()
