#!/usr/bin/env python3
"""R15 Lane T / P0-LEDGER-2: append-only, single-writer GPU + optimizer-step ledger.

Current state of the art in this repo was `ledger_book()`, which rewrote the whole ledger JSON
with `open(path, "w")`.  That is not crash safe (a killed process leaves a truncated file), it is
not tamper evident (any external writer silently resets the counters) and it never guarded the
3000-formal-step cap.  This module replaces it with:

  * an immutable append-only WAL of chained-hash events (`events.jsonl`), written under an
    exclusive `flock` by a single writer, with `O_APPEND` + `fsync`;
  * a projection file (`projection.json`) that is rewritten atomically (tmp -> fsync -> rename)
    from the WAL and carries `wal_last_seq` + `wal_sha256`, so any external overwrite of the
    projection is detected on the next open;
  * strict accounting of BOTH GPU seconds and cumulative formal optimizer steps, checked
    fail-closed before any job starts;
  * duplicate (run_id, kind, global_step) rejection and monotonic-sequence enforcement;
  * a read-only `import_history` event so the already-spent 1027.5 s is carried over instead of
    being reset, with reconstructed timestamps kept separate from real log timestamps.

The ledger directory MUST live outside any git worktree
(default `/home/T7/ojh/robot_actual/r15_ledger`) so that a `git checkout`/`git push` or a mirror
sync of the repository can never overwrite it.
"""
import errno
import fcntl
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone

SCHEMA_VERSION = 2
LOCK_TIMEOUT_S = 30.0

DEFAULT_LEDGER_DIR = os.environ.get("R15_LEDGER_DIR", "/home/T7/ojh/robot_actual/r15_ledger")

# Bucket caps from mission/R11_R16_DELEGATED_BUDGET_AND_GATES.json@83b5794
BUCKET_CAPS_S = {
    "COLLECTOR": 14400.0,
    "R13_7B_ZERO_STEP": 1800.0,
    "R14_TRAIN_OVERFIT": 3600.0,
    "R15_TRAIN_FORMAL": 50400.0,
    "R16_SIM_DEV": 14400.0,
    "R16_SIM_FINAL": 21600.0,
    "TWO_TASK_PILOT": 7200.0,
    "RESERVE": 1800.0,
}
TOTAL_CAP_S = 115200.0
FORMAL_STEPS_CAP = 3000
OVERFIT_STEPS_CAP = 100


class LedgerError(RuntimeError):
    pass


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    try:
        dfd = os.open(d, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except (OSError, AttributeError):
        pass
    return _sha256_file(path)


class Ledger:
    """Single-writer WAL ledger.  Every public method re-validates the whole WAL chain."""

    def __init__(self, ledger_dir=None):
        self.dir = os.path.abspath(ledger_dir or DEFAULT_LEDGER_DIR)
        os.makedirs(self.dir, exist_ok=True)
        self.wal_path = os.path.join(self.dir, "events.jsonl")
        self.proj_path = os.path.join(self.dir, "projection.json")
        self.lock_path = os.path.join(self.dir, "writer.lock")
        self._fd = None

    # ------------------------------------------------------------------ locking
    def __enter__(self):
        self._fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.time() + LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.time() > deadline:
                    os.close(self._fd)
                    self._fd = None
                    raise LedgerError("another writer holds %s (waited %.0fs)"
                                      % (self.lock_path, LOCK_TIMEOUT_S))
                time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None
        return False

    # --------------------------------------------------------------- WAL read
    def read_events(self):
        """Parse the WAL, verify the hash chain and return the events."""
        if not os.path.exists(self.wal_path):
            return []
        events = []
        prev = ""
        with open(self.wal_path, "r") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    raise LedgerError("corrupt WAL at line %d: not JSON" % lineno)
                if ev.get("prev_hash") != prev:
                    raise LedgerError("WAL chain broken at line %d (seq %r)"
                                      % (lineno, ev.get("seq")))
                if int(ev.get("seq", -1)) != len(events) + 1:
                    raise LedgerError("WAL seq out of order at line %d: %r != %d"
                                      % (lineno, ev.get("seq"), len(events) + 1))
                body = {k: v for k, v in ev.items() if k != "hash"}
                want = _sha256_bytes(json.dumps(body, sort_keys=True).encode())
                if ev.get("hash") != want:
                    raise LedgerError("WAL hash mismatch at line %d (seq %r)"
                                      % (lineno, ev.get("seq")))
                prev = ev["hash"]
                events.append(ev)
        return events

    def verify_projection(self, events=None):
        """The projection must agree with the WAL; otherwise someone overwrote it."""
        events = self.read_events() if events is None else events
        if not os.path.exists(self.proj_path):
            if events:
                raise LedgerError("projection missing while the WAL has %d events" % len(events))
            return None
        proj = json.load(open(self.proj_path))
        last_seq = events[-1]["seq"] if events else 0
        if int(proj.get("wal_last_seq", -1)) != last_seq:
            raise LedgerError(
                "projection is out of sync with the WAL (projection seq %r, WAL seq %d): the "
                "ledger was overwritten or truncated outside this writer"
                % (proj.get("wal_last_seq"), last_seq))
        truth = self.totals(events)
        got = proj.get("totals") or {}
        for key in ("gpu_seconds_used", "formal_steps_used"):
            if abs(float(got.get(key, -1)) - float(truth[key])) > 1e-6:
                raise LedgerError(
                    "projection %s=%r contradicts the WAL-derived %r: the ledger file was edited "
                    "outside this writer" % (key, got.get(key), truth[key]))
        return proj

    # ---------------------------------------------------------------- append
    def append(self, run_id, kind, payload=None, bucket=None, gpu_seconds=0.0,
               formal_steps=0, actual_ts=None, reconstructed_ts=None):
        events = self.read_events()
        for ev in events:
            if (ev["run_id"], ev["kind"], (ev["payload"] or {}).get("global_step")) == \
                    (run_id, kind, (payload or {}).get("global_step")) and kind in (
                        "job_begin", "step_commit", "job_end", "run_start"):
                raise LedgerError("duplicate event (%s, %s, %r) - refusing to book twice"
                                  % (run_id, kind, (payload or {}).get("global_step")))
        prev = events[-1]["hash"] if events else ""
        ev = {"seq": len(events) + 1, "utc": actual_ts or utcnow(), "run_id": str(run_id),
              "kind": str(kind), "bucket": bucket, "gpu_seconds": round(float(gpu_seconds), 3),
              "formal_steps": int(formal_steps), "payload": payload or {},
              "prev_hash": prev}
        if reconstructed_ts:
            ev["reconstructed_ts"] = reconstructed_ts
            ev["timestamp_provenance"] = "reconstructed"
        else:
            ev["timestamp_provenance"] = "actual"
        ev["hash"] = _sha256_bytes(json.dumps(ev, sort_keys=True).encode())
        line = json.dumps(ev, sort_keys=True) + "\n"
        with open(self.wal_path, "a") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self._write_projection(self.read_events())
        return ev

    # ------------------------------------------------------------- projection
    def totals(self, events=None):
        events = self.read_events() if events is None else events
        buckets = {k: {"cap_s": v, "used_s": 0.0, "formal_steps": 0} for k, v in BUCKET_CAPS_S.items()}
        gpu_total = 0.0
        formal = 0
        for ev in events:
            g = float(ev.get("gpu_seconds") or 0.0)
            s = int(ev.get("formal_steps") or 0)
            gpu_total += g
            formal += s
            b = ev.get("bucket")
            if b in buckets:
                buckets[b]["used_s"] += g
                buckets[b]["formal_steps"] += s
        return {"gpu_seconds_used": gpu_total, "gpu_seconds_remaining": TOTAL_CAP_S - gpu_total,
                "formal_steps_used": formal, "formal_steps_remaining": FORMAL_STEPS_CAP - formal,
                "buckets": buckets, "n_events": len(events)}

    def _write_projection(self, events):
        last = events[-1] if events else None
        proj = {
            "schema_version": SCHEMA_VERSION, "written_utc": utcnow(),
            "wal_last_seq": last["seq"] if last else 0,
            "wal_last_hash": last["hash"] if last else "",
            "wal_sha256": _sha256_file(self.wal_path) if os.path.exists(self.wal_path) else "",
            "totals": self.totals(events),
            "wal_path": self.wal_path,
        }
        _atomic_write_json(self.proj_path, proj)
        return proj

    # ------------------------------------------------------------------ guards
    def assert_can_start(self, bucket, requested_seconds, requested_steps=0, run_id=None):
        """Fail closed before a job: bucket, aggregate seconds and formal-step caps."""
        self.verify_projection()
        t = self.totals()
        if bucket not in BUCKET_CAPS_S:
            raise LedgerError("unknown bucket %r (known: %s)" % (bucket, sorted(BUCKET_CAPS_S)))
        left_bucket = BUCKET_CAPS_S[bucket] - t["buckets"][bucket]["used_s"]
        left_total = t["gpu_seconds_remaining"]
        left_steps = t["formal_steps_remaining"]
        if left_total <= 0 or left_bucket <= 0:
            raise LedgerError("no GPU budget left: bucket %s %.1fs, aggregate %.1fs"
                              % (bucket, left_bucket, left_total))
        if float(requested_seconds) > left_bucket + 1e-6 or float(requested_seconds) > left_total + 1e-6:
            raise LedgerError("requested %.1fs exceeds the remaining budget (bucket %.1fs, "
                              "aggregate %.1fs)" % (requested_seconds, left_bucket, left_total))
        if int(requested_steps) > left_steps:
            raise LedgerError("requested %d formal steps exceeds the remaining %d of %d"
                              % (requested_steps, left_steps, FORMAL_STEPS_CAP))
        if run_id is not None:
            for ev in self.read_events():
                if ev["run_id"] == run_id and ev["kind"] == "job_end":
                    raise LedgerError("run_id %r already finished; never rebook a run" % (run_id,))
        return {"left_bucket": left_bucket, "left_total": left_total, "left_steps": left_steps}

    def import_history(self, source_path, gpu_seconds, formal_steps=0, bucket_mix=None,
                       note=""):
        """Carry already-spent GPU seconds in WITHOUT touching the historical evidence file."""
        if self.read_events():
            raise LedgerError("import_history must be the first event of a fresh ledger")
        rec = {"source_file": os.path.abspath(source_path),
               "source_sha256": _sha256_file(source_path),
               "seconds": float(gpu_seconds), "formal_steps": int(formal_steps),
               "bucket_mix": bucket_mix or {"R13_7B_ZERO_STEP": 169.7,
                                            "R14_TRAIN_OVERFIT": 857.8},
               "note": note}
        # bucket=None: the aggregate is booked here, the per-bucket split lives in the markers
        ev = self.append("history-import", "import_history", payload=rec,
                         bucket=None, gpu_seconds=gpu_seconds, formal_steps=formal_steps)
        # attribute the seconds to their real buckets as separate, non-counting markers
        for b, s in rec["bucket_mix"].items():
            self.append("history-import", "import_history_bucket",
                        payload={"bucket": b, "seconds": s, "counts_toward_total": False,
                                 "counted_upstream_by": "import_history"},
                        bucket=b, gpu_seconds=0.0, formal_steps=0)
        return ev

    def bucket_used_reported(self):
        """Bucket usage including the imported history markers (for reporting only)."""
        t = self.totals()
        mix = {}
        for ev in self.read_events():
            if ev["kind"] == "import_history_bucket":
                mix[ev["payload"]["bucket"]] = mix.get(ev["payload"]["bucket"], 0.0) + \
                    float(ev["payload"]["seconds"])
        out = {k: dict(v) for k, v in t["buckets"].items()}
        for b, s in mix.items():
            if b in out:
                out[b]["used_s"] += s
        return out
