"""Pure-stdlib fail-closed helpers for R44 CPU preflight and dataset identity."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


class R44PreflightError(ValueError):
    pass


def sha256_file_stable(path: os.PathLike[str] | str, chunk_size: int = 1 << 20) -> dict:
    """Hash one file and refuse a result if size/mtime/inode/device changed while reading."""
    p = Path(path).expanduser().resolve()
    before = p.stat()
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    after = p.stat()
    identity_before = (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev)
    identity_after = (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev)
    if identity_before != identity_after:
        raise R44PreflightError(f"file changed while hashing: {p}")
    return {
        "path": str(p),
        "sha256": h.hexdigest(),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "inode": int(after.st_ino),
        "device": int(after.st_dev),
    }


def final_preflight_status(checks: list[dict], budget_ok: bool) -> tuple[str, list[dict]]:
    """Data/contract failures outrank budget status; budget is not a data-contract check."""
    failed = [
        c for c in checks
        if c.get("name") != "gpu_budget_covers_the_approved_plan"
        and c.get("result") != "PASS"
    ]
    if failed:
        return "FAIL_PREFLIGHT", failed
    if not budget_ok:
        return "BLOCKED_BUDGET", []
    return "READY", []


def make_training_manifest(
    *,
    split_sha256: str,
    feature_normalizer_sha256: str | None,
    delta_normalizer_sha256: str,
    delta_normalizer_payload_sha256: str,
    files: dict[str, dict[str, Any]],
    instructions: dict[str, dict[str, str]],
) -> dict:
    """Build the exact strict manifest schema consumed by training/r44_train.py."""
    if not split_sha256 or not delta_normalizer_sha256 or not delta_normalizer_payload_sha256:
        raise R44PreflightError("split and delta-normalizer identities are required")
    if not files:
        raise R44PreflightError("training manifest has no episode file records")
    if not instructions:
        raise R44PreflightError("training manifest has no instruction records")
    for key, rec in files.items():
        if not key or not rec.get("path") or not rec.get("sha256") or int(rec.get("size_bytes", -1)) < 0:
            raise R44PreflightError(f"incomplete file identity for {key!r}")
    for task, rec in instructions.items():
        text = rec.get("text")
        sha = rec.get("sha256")
        if not isinstance(text, str) or not text or hashlib.sha256(text.encode("utf-8")).hexdigest() != sha:
            raise R44PreflightError(f"instruction text/hash mismatch for {task!r}")
    return {
        "schema": "r44-dataset-manifest/v1",
        "split_sha256": split_sha256,
        "feature_normalizer_sha256": feature_normalizer_sha256,
        "delta_normalizer_sha256": delta_normalizer_sha256,
        "delta_normalizer_payload_sha256": delta_normalizer_payload_sha256,
        "files": files,
        "instructions": instructions,
    }
