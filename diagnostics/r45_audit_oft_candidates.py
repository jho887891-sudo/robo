#!/usr/bin/env python3
"""Read-only static fingerprint for explicit OFT source candidates.

Presence of modules is not evidence of historical training provenance.
This script never chooses an OFT tree or imports prismatic/CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HISTORICAL_DEFAULT = "/home/dgut/pusht_runtime/openvla_oft_src"
REQUIRED = ("prismatic/extern/hf/configuration_prismatic.py",
            "prismatic/extern/hf/processing_prismatic.py",
            "prismatic/extern/hf/modeling_prismatic.py")
OPTIONAL = ("prismatic/__init__.py", "vla-scripts")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_sha(path):
    try:
        r = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False)
        return r.stdout.strip() if r.returncode == 0 and len(r.stdout.strip()) == 40 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def inspect_candidate(raw):
    p = Path(raw).expanduser()
    resolved = p.resolve()
    rec = {"path": str(p), "resolved_path": str(resolved),
           "exists": p.is_dir(), "symlink": p.is_symlink(),
           "historical_default": str(resolved) == HISTORICAL_DEFAULT,
           "required": {}, "optional": {}, "git_head": None,
           "structurally_viable": False}
    if not rec["exists"]:
        return rec
    rec["git_head"] = git_sha(resolved)
    for entry in REQUIRED:
        f = resolved / entry
        rec["required"][entry] = {
            "exists": f.is_file(),
            "sha256": sha_file(f) if f.is_file() else None,
        }
    for entry in OPTIONAL:
        f = resolved / entry
        rec["optional"][entry] = "directory" if f.is_dir() else (
            "file" if f.is_file() else "missing")
    rec["structurally_viable"] = all(v["exists"] for v in rec["required"].values())
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", action="append", required=True,
                    help="an explicitly identified OFT source root, repeatable")
    ap.add_argument("--out", type=Path, required=True, help="new evidence JSON")
    args = ap.parse_args(argv)
    if args.out.exists():
        print("BLOCKED_OUTPUT_EXISTS", file=sys.stderr)
        return 2
    rows = [inspect_candidate(path) for path in dict.fromkeys(args.candidate)]
    viable = [r for r in rows if r["structurally_viable"]]
    report = {"schema": "r45-oft-source-candidate-audit/v1",
              "status": "CANDIDATES_NEED_PROVENANCE" if viable
                        else "BLOCKED_NO_VIABLE_OFT_CANDIDATE",
              "candidate_count": len(rows), "viable_count": len(viable),
              "candidates": rows, "selected": None,
              "limitations": ("Static source existence and SHA are not evidence of "
                              "correct model runtime, import compatibility, or the "
                              "historical R44/OFT checkout. A matching run log or "
                              "explicitly approved source identity is required. "
                              "No CUDA was accessed.") }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "viable_count": len(viable),
                      "out": str(args.out)}, sort_keys=True))
    return 0 if viable else 2


if __name__ == "__main__":
    raise SystemExit(main())
