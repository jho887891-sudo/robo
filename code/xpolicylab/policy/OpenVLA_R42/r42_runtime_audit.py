#!/usr/bin/env python3
"""Fail-closed, append-only JSONL trace for the R42 policy-server boundary.

This module uses only the standard library. It never stores image pixels; callers may store
per-camera hashes and the state/action vectors needed for post-run diagnosis.
"""
import atexit
import datetime
import json
import os
import re
import threading
import time


SCHEMA_VERSION = "r42-runtime-audit/v2"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RuntimeAuditError(RuntimeError):
    """Raised when an enabled audit trace cannot be safely persisted."""


class RuntimeAuditWriter(object):
    """Create one exclusive JSONL file and durably append one JSON object per record."""

    def __init__(self, path, run_id, run_metadata=None):
        if not isinstance(path, str) or not path.strip():
            raise RuntimeAuditError("audit path must be a non-empty string")
        if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
            raise RuntimeAuditError(
                "run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
        if run_metadata is not None and not isinstance(run_metadata, dict):
            raise RuntimeAuditError("run_metadata must be a dict")

        self.path = os.path.abspath(os.path.expanduser(path))
        self.run_id = run_id
        self._record_index = 0
        self._closed = False
        self._lock = threading.RLock()
        parent = os.path.dirname(self.path)
        if not os.path.isdir(parent):
            raise RuntimeAuditError(
                "audit parent directory does not exist: %s" % parent)

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self._fd = os.open(self.path, flags, 0o600)
        except FileExistsError:
            raise RuntimeAuditError(
                "audit log already exists; refusing to overwrite: %s" % self.path)
        except OSError as exc:
            raise RuntimeAuditError(
                "cannot create audit log %s: %s" % (self.path, exc))

        try:
            self.write({
                "event": "run_start",
                "path": self.path,
                "metadata": dict(run_metadata or {}),
            })
        except Exception:
            self.close()
            raise
        atexit.register(self.close)

    @staticmethod
    def _utc_now():
        return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

    def write(self, record):
        """Append and fsync a single event. Any serialization or I/O failure is fatal."""
        with self._lock:
            if self._closed:
                raise RuntimeAuditError("audit writer is closed")
            if not isinstance(record, dict):
                raise RuntimeAuditError("audit record must be a dict")
            event = record.get("event")
            if not isinstance(event, str) or not event.strip():
                raise RuntimeAuditError("audit record requires a non-empty event")

            payload = dict(record)
            payload.update({
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "record_index": self._record_index,
                "recorded_at_utc": self._utc_now(),
                "monotonic_ns": time.monotonic_ns(),
            })
            try:
                raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False) + "\n").encode("utf-8")
            except (TypeError, ValueError) as exc:
                self.close()
                raise RuntimeAuditError(
                    "audit record is not JSON-safe: %s" % exc)

            try:
                offset = 0
                while offset < len(raw):
                    written = os.write(self._fd, raw[offset:])
                    if written <= 0:
                        raise OSError("write returned no bytes")
                    offset += written
                os.fsync(self._fd)
            except OSError as exc:
                self.close()
                raise RuntimeAuditError(
                    "audit write/fsync failed for %s: %s" % (self.path, exc))
            self._record_index += 1
            return payload

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                os.close(self._fd)
            except OSError:
                # Preserve the original write exception if close is part of failure handling.
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
