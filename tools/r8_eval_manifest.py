#!/usr/bin/env python3
"""R8-C: dataset manifest, content hashing and freeze tooling for an INDEPENDENT eval set.

The R2-R8 rounds reused the same 20 Holdout episodes, so that set can no longer serve as
an untouched final test. This tool does not create data; it makes a future independent set
auditable before any model sees it:

  build    hash every episode (state+action content, length, instruction) into a manifest
  verify   re-hash the data and prove the manifest still matches (tamper/duplication check)
  freeze   bind a manifest + evaluation protocol + git SHA + UTC timestamp into one file
  disjoint refuse a set that overlaps the training/holdout episodes already used

Content hash definition (deterministic, order independent per episode):
  sha256( float64_le(state) || float64_le(action) || utf8(instruction) || int64(length) )
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

BASE_DIM = 14


def _c14(h, group):
    return np.concatenate([
        h[group + "/left_arm_joint_states"][:], h[group + "/left_ee_joint_states"][:],
        h[group + "/right_arm_joint_states"][:], h[group + "/right_ee_joint_states"][:],
    ], axis=1).astype(np.float64)


def episode_record(path):
    import h5py
    with h5py.File(path, "r") as h:
        state = _c14(h, "state")
        action = _c14(h, "action")
        ins = h["instruction"][()]
        if isinstance(ins, bytes):
            ins = ins.decode("utf-8", "replace")
        freq = int(h["additional_info/frequency"][()]) if "additional_info/frequency" in h else None
    if state.shape != action.shape or state.shape[1] != BASE_DIM:
        raise ValueError("%s: bad state/action shape %s / %s" % (path, state.shape, action.shape))
    if not (np.isfinite(state).all() and np.isfinite(action).all()):
        raise ValueError("%s: non-finite state/action" % path)
    hsh = hashlib.sha256()
    hsh.update(np.ascontiguousarray(state, dtype="<f8").tobytes())
    hsh.update(np.ascontiguousarray(action, dtype="<f8").tobytes())
    hsh.update(str(ins).encode("utf-8"))
    hsh.update(np.int64(len(state)).tobytes())
    return {"file": os.path.basename(path), "length": int(len(state)),
            "instruction": str(ins), "frequency": freq,
            "content_sha256": hsh.hexdigest(),
            "state_action_sha256": hashlib.sha256(
                np.ascontiguousarray(state, dtype="<f8").tobytes()
                + np.ascontiguousarray(action, dtype="<f8").tobytes()).hexdigest()}


def build(data_dir, task_filter=None):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".hdf5"))
    records = []
    for f in files:
        rec = episode_record(os.path.join(data_dir, f))
        if task_filter and task_filter.lower() not in rec["instruction"].lower():
            continue
        records.append(rec)
    if not records:
        raise ValueError("no episodes found in %s" % data_dir)
    hashes = [r["content_sha256"] for r in records]
    dataset_hash = hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest()
    return {"data_dir": os.path.abspath(data_dir), "task_filter": task_filter,
            "n_episodes": len(records), "n_unique_content": len(set(hashes)),
            "n_duplicate_content": len(records) - len(set(hashes)),
            "total_frames": int(sum(r["length"] for r in records)),
            "frequencies": sorted({r["frequency"] for r in records if r["frequency"]}),
            "instructions": sorted({r["instruction"] for r in records}),
            "dataset_sha256": dataset_hash, "episodes": records}


def verify(manifest, data_dir=None):
    data_dir = data_dir or manifest["data_dir"]
    current = build(data_dir, manifest.get("task_filter"))
    problems = []
    if current["n_episodes"] != manifest["n_episodes"]:
        problems.append("episode count %d != %d" % (current["n_episodes"], manifest["n_episodes"]))
    by_file = {r["file"]: r for r in current["episodes"]}
    for rec in manifest["episodes"]:
        cur = by_file.get(rec["file"])
        if cur is None:
            problems.append("missing episode: " + rec["file"])
        elif cur["content_sha256"] != rec["content_sha256"]:
            problems.append("content changed: " + rec["file"])
    if current["dataset_sha256"] != manifest["dataset_sha256"]:
        problems.append("dataset hash changed")
    return {"ok": not problems, "problems": problems,
            "dataset_sha256": current["dataset_sha256"],
            "n_episodes": current["n_episodes"]}


def check_disjoint(manifest, split_path):
    split = json.load(open(split_path))
    used = set(split["train"]) | set(split["val"])
    overlap = sorted({r["file"] for r in manifest["episodes"]} & used)
    dup = manifest["n_episodes"] - manifest["n_unique_content"]
    return {"ok": not overlap and dup == 0, "n_overlap_with_used": len(overlap),
            "overlapping_files": overlap[:10], "n_duplicate_content": dup,
            "used_episode_count": len(used)}


def freeze(manifest, protocol, git_sha, out_path, protocol_path=None):
    frozen = {
        "frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha,
        "protocol": protocol,
        "protocol_source": os.path.abspath(protocol_path) if protocol_path else None,
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "rule": "this file must be committed BEFORE any model is evaluated on the set; "
                "any later change invalidates the evaluation",
    }
    with open(out_path, "w") as fh:
        json.dump(frozen, fh, indent=1)
    return frozen


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data-dir", required=True)
    b.add_argument("--task-filter", default=None)
    b.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True)
    v.add_argument("--data-dir", default=None)
    d = sub.add_parser("disjoint")
    d.add_argument("--manifest", required=True)
    d.add_argument("--split", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--manifest", required=True)
    f.add_argument("--protocol", required=True)
    f.add_argument("--git-sha", required=True)
    f.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd == "build":
        m = build(args.data_dir, args.task_filter)
        json.dump(m, open(args.out, "w"), indent=1)
        print("episodes=%d unique=%d frames=%d dataset_sha256=%s"
              % (m["n_episodes"], m["n_unique_content"], m["total_frames"],
                 m["dataset_sha256"][:16]))
        print("wrote", args.out)
        return 0
    if args.cmd == "verify":
        m = json.load(open(args.manifest))
        res = verify(m, args.data_dir)
        json.dump(res, open(args.manifest + ".verify.json", "w"), indent=1)
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 3
    if args.cmd == "disjoint":
        m = json.load(open(args.manifest))
        res = check_disjoint(m, args.split)
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 4
    if args.cmd == "freeze":
        m = json.load(open(args.manifest))
        protocol = json.load(open(args.protocol))
        frozen = freeze(m, protocol, args.git_sha, args.out, args.protocol)
        print("frozen_utc=%s manifest_sha256=%s" % (frozen["frozen_utc"],
                                                    frozen["manifest_sha256"][:16]))
        return 0
    raise SystemExit("unknown command")


if __name__ == "__main__":
    sys.exit(main())
