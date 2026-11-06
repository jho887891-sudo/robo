#!/usr/bin/env python3
"""Formal two-task OpenVLA training entry; defaults to replacement sampling.

This entry reuses the R44 model, action, checkpoint, safety and full-manifest
attestation implementation. Training still requires explicit GPU authorization.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training import r44_train


def main(argv=None):
    forwarded = list(sys.argv[1:] if argv is None else argv)
    if "--r51-two-task" in forwarded:
        raise SystemExit("R51 mode is implicit in this entry")
    return r44_train.main(["--r51-two-task", *forwarded])


if __name__ == "__main__":
    raise SystemExit(main())
