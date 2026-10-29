#!/usr/bin/env python3
"""Fail-closed R46 GPU smoke parent. CPU tests are safe; execution needs a real approval."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
MAX_WALL_SECONDS = 1800.0
FROZEN_CONTRACT = {
    "model": "OpenVLA-7B",
    "lora_r": 32,
    "lora_alpha": 16,
    "lora_target_modules": ["q_proj", "v_proj"],
    "proprio_dim": 42,
    "camera_order": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
    "action_target": "joint_delta_v1",
    "optimizer_steps": 5,
}
HASH_KEYS = ("approved_original_manifest_sha256", "approved_smoke_samples_sha256")
ALLOWED_SMOKE_FLAGS = {
    "--splits", "--hdf5-root", "--feature-normalizer", "--delta-normalizer",
    "--base-model", "--oft-root",
}


class ApprovalError(ValueError):
    pass


def _parse_utc(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ApprovalError(f"{field} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ApprovalError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_approval(doc: dict, *, source_sha: str, requested_wall_seconds: float,
                      expected_hashes: dict, now: datetime | None = None) -> dict:
    """Validate a human-provided R47 authorization without inventing any approval fields."""
    if not isinstance(doc, dict):
        raise ApprovalError("approval document must be a JSON object")
    required = (
        "schema", "template_only", "approved", "approval_id", "approver",
        "approved_source_sha", "approved_gpu_uuid", "approved_window_start_iso8601",
        "approved_window_end_iso8601", "approved_max_wall_seconds",
        "approved_min_total_gib", "approved_min_free_gib", "require_exclusive_gpu",
        "approved_model_sha_list", *HASH_KEYS, "allowed_operation",
        "formal_training_approved", "extra_benchmark_approved",
    )
    missing = [key for key in required if key not in doc]
    if missing:
        raise ApprovalError("missing approval fields: " + ", ".join(missing))
    unknown = set(doc) - set(required) - {"note"}
    if unknown:
        raise ApprovalError("unsupported approval fields: " + ", ".join(sorted(unknown)))
    if doc["schema"] != "r47-gpu-approval-template/v1":
        raise ApprovalError("approval schema must match the published R47 approval template")
    if doc["template_only"] is not False or doc["approved"] is not True:
        raise ApprovalError("approval must be a completed, explicitly approved document")
    if not isinstance(doc["approval_id"], str) or not doc["approval_id"].strip():
        raise ApprovalError("approval_id must be a nonempty human-issued identifier")
    if not isinstance(doc["approver"], str) or not doc["approver"].strip():
        raise ApprovalError("approver must identify the research lead")
    if len(source_sha) != 40 or any(ch not in "0123456789abcdef" for ch in source_sha.lower()) \
            or doc["approved_source_sha"] != source_sha:
        raise ApprovalError("approved source SHA does not exactly match requested source SHA")
    if not isinstance(doc["approved_gpu_uuid"], str) or not doc["approved_gpu_uuid"].startswith("GPU-"):
        raise ApprovalError("approved_gpu_uuid must be a concrete GPU UUID")

    start = _parse_utc(doc["approved_window_start_iso8601"], "approved_window_start_iso8601")
    end = _parse_utc(doc["approved_window_end_iso8601"], "approved_window_end_iso8601")
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if end <= start:
        raise ApprovalError("approved GPU window end must be later than its start")
    if not start <= now < end:
        raise ApprovalError("current time is outside the approved GPU window")
    try:
        approved_cap = float(doc["approved_max_wall_seconds"])
        requested = float(requested_wall_seconds)
    except (TypeError, ValueError) as exc:
        raise ApprovalError("wall-clock limits must be numbers") from exc
    if not math.isfinite(approved_cap) or not math.isfinite(requested) \
            or approved_cap <= 0 or requested <= 0 or approved_cap > MAX_WALL_SECONDS \
            or requested > approved_cap or requested > MAX_WALL_SECONDS:
        raise ApprovalError("requested wall limit exceeds approval or the 1800 second hard ceiling")
    if requested > (end - now).total_seconds():
        raise ApprovalError("approved window cannot fit the full requested wall limit")
    if doc["require_exclusive_gpu"] is not True:
        raise ApprovalError("exclusive GPU mode must be explicitly true")
    if doc["allowed_operation"] != "FIVE_SAMPLE_SMOKE_ONLY":
        raise ApprovalError("approval operation must be FIVE_SAMPLE_SMOKE_ONLY")
    if doc["formal_training_approved"] is not False:
        raise ApprovalError("formal training must remain unapproved")
    if doc["extra_benchmark_approved"] is not False:
        raise ApprovalError("extra benchmark must remain unapproved")
    for key in ("approved_min_total_gib", "approved_min_free_gib"):
        try:
            value = float(doc[key])
        except (TypeError, ValueError) as exc:
            raise ApprovalError(f"{key} must be numeric") from exc
        if not math.isfinite(value) or value <= 0:
            raise ApprovalError(f"{key} must be finite and positive")
    if float(doc["approved_min_free_gib"]) > float(doc["approved_min_total_gib"]):
        raise ApprovalError("approved minimum free memory cannot exceed total memory")
    model_sha_list = doc["approved_model_sha_list"]
    if not isinstance(model_sha_list, list) or not model_sha_list:
        raise ApprovalError("approved_model_sha_list must contain the frozen model shard hashes")
    if model_sha_list != expected_hashes.get("approved_model_sha_list"):
        raise ApprovalError("approved model shard SHA list differs from the validated model identity")
    for key in HASH_KEYS:
        value = doc[key]
        if not isinstance(value, str) or len(value) != 64 \
                or any(ch not in "0123456789abcdef" for ch in value.lower()):
            raise ApprovalError(f"{key} must be a SHA256 hex digest")
        if expected_hashes.get(key) != value:
            raise ApprovalError(f"{key} does not match the validated local input")
    return {
        **doc,
        "approved_window_start_iso8601": start.isoformat(),
        "approved_window_end_iso8601": end.isoformat(),
        "approved_max_wall_seconds": approved_cap,
        "requested_wall_seconds": requested,
    }


def sha256_file(path: Path, *, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"wall-clock budget expired while hashing {path.name}")
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def model_identity_sha_list(identity: dict) -> list[dict]:
    """Canonical approval list for every file in the frozen standard OpenVLA model identity."""
    if identity.get("schema") != "r45-static-base-identity/v1" \
            or identity.get("shard_count") != 3 or identity.get("indexed") is not True:
        raise ApprovalError("model identity must attest three indexed OpenVLA shards")
    file_rows = identity.get("files")
    if not isinstance(file_rows, list):
        raise ApprovalError("model identity files must be a list")
    records = {row.get("name"): row for row in file_rows
               if isinstance(row, dict) and isinstance(row.get("name"), str)}
    if len(records) != len(file_rows):
        raise ApprovalError("model identity contains malformed or duplicate file records")
    all_files = sorted(({"filename": name, "sha256": record.get("sha256")}
                        for name, record in records.items()), key=lambda row: row["filename"])
    shards = [row for row in all_files if row["filename"].endswith(".safetensors")]
    if len(shards) != 3 or "config.json" not in records or "model.safetensors.index.json" not in records:
        raise ApprovalError("model identity must include config, index and exactly three weight shards")
    if any(Path(row["filename"]).name != row["filename"]
           or not isinstance(row["sha256"], str) or len(row["sha256"]) != 64
           or any(ch not in "0123456789abcdef" for ch in row["sha256"].lower())
           for row in all_files):
        raise ApprovalError("model identity contains an invalid filename or SHA256")
    return all_files


def verify_model_identity(identity: dict, model_root: Path, *, deadline: float | None = None) -> list[dict]:
    """Hash every identity file, then verify the indexed standard OpenVLA shard set."""
    all_files = model_identity_sha_list(identity)
    records = {row["filename"]: row for row in all_files}
    shards = [row for row in all_files if row["filename"].endswith(".safetensors")]
    root = Path(model_root).expanduser().resolve()
    for row in all_files:
        path = root / row["filename"]
        if not path.is_file():
            raise ApprovalError(f"OpenVLA model identity file is missing: {row['filename']}")
        before = path.stat()
        if sha256_file(path, deadline=deadline) != row["sha256"]:
            raise ApprovalError(f"OpenVLA model identity SHA mismatch: {row['filename']}")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ApprovalError(f"OpenVLA model identity file changed during hashing: {row['filename']}")
    index_path = root / "model.safetensors.index.json"
    index_doc = json.loads(index_path.read_text(encoding="utf-8"))
    if set((index_doc.get("weight_map") or {}).values()) != {row["filename"] for row in shards}:
        raise ApprovalError("OpenVLA shard set differs from model.safetensors.index.json")
    return shards


def smoke_base_model_path(argv: list[str]) -> Path:
    """Extract one unambiguous base-model path from the frozen smoke arguments."""
    values = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--base-model":
            if index + 1 >= len(argv) or argv[index + 1].startswith("-"):
                raise ApprovalError("smoke arguments require a value for --base-model")
            values.append(argv[index + 1])
            index += 2
            continue
        if token.startswith("--base-model="):
            values.append(token.split("=", 1)[1])
        index += 1
    if len(values) != 1 or not values[0].strip():
        raise ApprovalError("smoke arguments must contain exactly one --base-model path")
    return Path(values[0]).expanduser().resolve()


def validate_smoke_args(argv: list[str]) -> list[str]:
    """Allow only frozen-input paths; callers cannot alter the smoke or trainer contract."""
    for token in argv:
        if token.startswith("-"):
            flag = token.split("=", 1)[0]
            if flag not in ALLOWED_SMOKE_FLAGS:
                raise ApprovalError(f"smoke argument is not permitted: {flag}")
    return list(argv)


def probe_gpu(gpu_uuid: str, min_total_gib: float, min_free_gib: float,
              *, timeout_seconds: float = 15.0,
              allowed_compute_pids: set[int] | frozenset[int] = frozenset()) -> dict:
    """Read GPU inventory using nvidia-smi only; never imports torch or creates CUDA."""
    command = ["nvidia-smi", "--query-gpu=uuid,memory.total,memory.free,utilization.gpu",
               "--format=csv,noheader,nounits"]
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise TimeoutError("no wall-clock budget remains for GPU inventory")
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=timeout_seconds, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"GPU inventory unavailable (exit {result.returncode}): {result.stderr.strip()}")
    rows = list(csv.reader(line for line in result.stdout.splitlines() if line.strip()))
    parsed = []
    for row in rows:
        if len(row) != 4:
            raise RuntimeError("nvidia-smi returned an unparseable GPU inventory")
        try:
            utilization = float(row[3].strip().replace("%", ""))
        except ValueError as exc:
            raise RuntimeError("nvidia-smi GPU utilization is unavailable") from exc
        if not math.isfinite(utilization) or not 0 <= utilization <= 100:
            raise RuntimeError("nvidia-smi returned invalid GPU utilization")
        parsed.append({"uuid": row[0].strip(), "total_gib": float(row[1]) / 1024,
                       "free_gib": float(row[2]) / 1024, "utilization_percent": utilization})
    matches = [row for row in parsed if row["uuid"] == gpu_uuid]
    if len(matches) != 1:
        raise RuntimeError("approved GPU UUID is absent or ambiguous")
    gpu = matches[0]
    if gpu["total_gib"] < min_total_gib or gpu["free_gib"] < min_free_gib:
        raise RuntimeError("approved GPU does not meet the approved memory thresholds")

    proc = subprocess.run(
        ["nvidia-smi", "--id=" + gpu_uuid, "--query-compute-apps=pid,process_name",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=timeout_seconds, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"GPU process inventory unavailable (exit {proc.returncode})")
    if any(not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
           for pid in allowed_compute_pids):
        raise ValueError("allowed compute PID set contains an invalid PID")
    observed = []
    allowed = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        row = next(csv.reader([line]))
        if len(row) < 1:
            raise RuntimeError("nvidia-smi returned an unparseable compute process row")
        try:
            pid = int(row[0].strip())
        except ValueError as exc:
            raise RuntimeError("nvidia-smi returned an unparseable compute process PID") from exc
        record = {"pid": pid, "process_name": row[1].strip() if len(row) > 1 else ""}
        observed.append(record)
        if pid in allowed_compute_pids:
            allowed.append(pid)
    active = [row for row in observed if row["pid"] not in allowed_compute_pids]
    if active:
        details = "; ".join(f"{row['pid']}, {row['process_name']}" for row in active)
        raise RuntimeError("exclusive GPU admission refused; compute processes exist: " + details)
    return {"gpu": gpu, "active_compute_processes": [],
            "allowed_compute_processes": sorted(allowed),
            "observed_compute_processes": observed, "nvidia_smi_exit_code": 0}


def _append_event(stream, events_path: Path, record: dict) -> None:
    record = {"utc": datetime.now(timezone.utc).isoformat(), **record}
    stream.write(json.dumps(record, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _atomic_json(path: Path, value: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _signal_owned_group(process: subprocess.Popen, sig: int) -> bool:
    """Signal only our live child leader's verified session/process group."""
    if process.poll() is not None:
        return False
    try:
        if os.getpgid(process.pid) != process.pid or os.getsid(process.pid) != process.pid:
            return False
        os.killpg(process.pid, sig)
        return True
    except (OSError, ProcessLookupError):
        return False


def _live_session_members(pgid: int, session_id: int) -> list[int] | None:
    """Return live members of this run's session; None means identity cannot be proven."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None
    members = []
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="ascii")
            fields = stat.rsplit(")", 1)[1].split()
            state, pgrp, sid = fields[0], int(fields[2]), int(fields[3])
            if pgrp == pgid and sid == session_id and state != "Z":
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def _signal_verified_session(pgid: int, session_id: int, sig: int) -> bool:
    members = _live_session_members(pgid, session_id)
    if members is None or not members:
        return False
    try:
        os.killpg(pgid, sig)
        return True
    except OSError:
        return False


def _set_parent_death_signal(expected_parent: int) -> None:
    """Linux child setup: if the supervisor dies, terminate this smoke process too."""
    import ctypes
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        os._exit(126)
    if os.getppid() != expected_parent:
        os.kill(os.getpid(), signal.SIGKILL)


def run_child(argv: list[str], output_dir: Path, max_wall_seconds: float, *,
              poll_interval: float = 0.05, grace_seconds: float = 10.0,
              started_monotonic: float | None = None, env: dict | None = None,
              precreated: bool = False) -> dict:
    """Run one owned process group under a parent monotonic deadline."""
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        allowed = precreated and output_dir.is_dir() and {p.name for p in output_dir.iterdir()} == {"supervisor.jsonl"}
        if not allowed:
            raise FileExistsError(f"output directory already exists: {output_dir}")
    if not argv or not math.isfinite(float(max_wall_seconds)) \
            or max_wall_seconds <= 0 or max_wall_seconds > MAX_WALL_SECONDS:
        raise ValueError("child command is required and wall limit must be in (0, 1800]")
    if not math.isfinite(float(poll_interval)) or poll_interval <= 0 \
            or not math.isfinite(float(grace_seconds)) or grace_seconds < 0:
        raise ValueError("poll interval must be positive and signal grace must be finite/non-negative")
    if started_monotonic is not None and not math.isfinite(float(started_monotonic)):
        raise ValueError("supervisor monotonic start must be finite")
    if sys.platform != "linux":
        raise RuntimeError("parent-supervised hard stop requires Linux process groups and PDEATHSIG")

    if not precreated:
        output_dir.mkdir(parents=True, mode=0o700)
    events_path = output_dir / "supervisor.jsonl"
    final_path = output_dir / "supervisor_final.json"
    started = time.monotonic() if started_monotonic is None else float(started_monotonic)
    deadline = started + float(max_wall_seconds)
    grace = min(float(grace_seconds), max(0.05, float(max_wall_seconds) * 0.10))
    interrupt_at = deadline - grace
    kill_reserve = min(0.1, float(max_wall_seconds) * 0.05)
    hard_kill_at = deadline - kill_reserve
    report = {
        "schema": "r46-supervisor/v1", "status": "RUNNING",
        "command": argv, "parent_pid": os.getpid(), "child_pid": None,
        "child_pgid": None, "deadline_monotonic_seconds": deadline,
        "max_wall_seconds": float(max_wall_seconds), "signals_sent": [],
        "stop_reason": None, "child_exit_code": None,
        "hard_stop_confirmed": None,
    }
    stop_reason = "CHILD_EXITED"
    interrupted = False
    process = None
    with events_path.open("a" if precreated else "x", encoding="utf-8", buffering=1) as events:
        _append_event(events, events_path, {"event": "parent_started", "parent_pid": os.getpid(),
                                             "deadline_monotonic_seconds": deadline})
        if time.monotonic() >= deadline:
            report.update({"status": "STOP_TIMEOUT_BEFORE_CHILD",
                           "stop_reason": "STOP_TIMEOUT_BEFORE_CHILD",
                           "elapsed_wall_seconds": time.monotonic() - started,
                           "hard_stop_confirmed": True})
            _append_event(events, events_path, {"event": "parent_finished",
                                                 "stop_reason": "STOP_TIMEOUT_BEFORE_CHILD",
                                                 "child_exit_code": None})
            _atomic_json(final_path, report)
            return report
        with (output_dir / "stdout.log").open("xb") as stdout, \
                (output_dir / "stderr.log").open("xb") as stderr:
            try:
                parent_pid = os.getpid()
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                           start_new_session=True, close_fds=True, env=env,
                                           preexec_fn=lambda p=parent_pid: _set_parent_death_signal(p))
            except OSError as exc:
                report.update({"status": "CHILD_START_FAILED", "stop_reason": "CHILD_START_FAILED",
                               "error": repr(exc), "elapsed_wall_seconds": time.monotonic() - started,
                               "hard_stop_confirmed": True})
                _append_event(events, events_path, {"event": "child_start_failed", "error": repr(exc)})
                _atomic_json(final_path, report)
                return report
            report.update({"child_pid": process.pid, "child_pgid": process.pid})
            _append_event(events, events_path, {"event": "child_started", "child_pid": process.pid,
                                                 "child_pgid": process.pid})
            sigint_sent = False
            sigkill_sent = False
            try:
                while True:
                    code = process.poll()
                    if code is not None:
                        report["child_exit_code"] = code
                        if sigint_sent:
                            stop_reason = "STOP_TIMEOUT_SIGINT" if not interrupted else "STOP_MANUAL_SIGINT"
                        members = _live_session_members(process.pid, process.pid)
                        if members is None:
                            stop_reason = "HARD_STOP_UNCONFIRMED"
                        elif members:
                            if _signal_verified_session(process.pid, process.pid, signal.SIGKILL):
                                report["signals_sent"].append("SIGKILL_ORPHAN_GROUP")
                                cleanup_deadline = min(deadline, time.monotonic() + 0.2)
                                while time.monotonic() < cleanup_deadline:
                                    remaining = _live_session_members(process.pid, process.pid)
                                    if remaining == []:
                                        break
                                    if remaining is None:
                                        break
                                    time.sleep(min(poll_interval, max(0.001, cleanup_deadline - time.monotonic())))
                                remaining = _live_session_members(process.pid, process.pid)
                                stop_reason = "CHILD_EXITED_WITH_ORPHANS_CLEANED" if remaining == [] \
                                    else "HARD_STOP_UNCONFIRMED"
                            else:
                                stop_reason = "HARD_STOP_UNCONFIRMED"
                        break
                    now = time.monotonic()
                    if not sigint_sent and now >= interrupt_at:
                        sent = _signal_owned_group(process, signal.SIGINT)
                        sigint_sent = sent
                        if sent:
                            report["signals_sent"].append("SIGINT")
                            _append_event(events, events_path, {"event": "signal_sent", "signal": "SIGINT",
                                                                 "child_pgid": process.pid})
                        else:
                            stop_reason = "HARD_STOP_UNCONFIRMED"
                            break
                    if now >= hard_kill_at:
                        stop_reason = "STOP_TIMEOUT_KILLED"
                        sent = _signal_owned_group(process, signal.SIGKILL)
                        if sent:
                            sigkill_sent = True
                            report["signals_sent"].append("SIGKILL")
                            _append_event(events, events_path, {"event": "signal_sent", "signal": "SIGKILL",
                                                                 "child_pgid": process.pid})
                            try:
                                report["child_exit_code"] = process.wait(
                                    timeout=max(0.001, deadline - time.monotonic()))
                            except subprocess.TimeoutExpired:
                                stop_reason = "HARD_STOP_UNCONFIRMED"
                            members = _live_session_members(process.pid, process.pid)
                            if members is None or members:
                                stop_reason = "HARD_STOP_UNCONFIRMED"
                        else:
                            stop_reason = "HARD_STOP_UNCONFIRMED"
                        break
                    time.sleep(min(poll_interval, max(0.001, hard_kill_at - now)))
            except KeyboardInterrupt:
                interrupted = True
                stop_reason = "STOP_MANUAL_SIGINT"
                sent = _signal_owned_group(process, signal.SIGINT)
                if sent:
                    report["signals_sent"].append("SIGINT")
                    end_grace = min(hard_kill_at, time.monotonic() + 10.0)
                    while process.poll() is None and time.monotonic() < end_grace:
                        time.sleep(min(poll_interval, max(0.001, end_grace - time.monotonic())))
                    if process.poll() is None:
                        stop_reason = "STOP_MANUAL_SIGKILL"
                        if _signal_owned_group(process, signal.SIGKILL):
                            report["signals_sent"].append("SIGKILL")
                            try:
                                report["child_exit_code"] = process.wait(
                                    timeout=max(0.001, deadline - time.monotonic()))
                            except subprocess.TimeoutExpired:
                                stop_reason = "HARD_STOP_UNCONFIRMED"
                            members = _live_session_members(process.pid, process.pid)
                            if members is None or members:
                                stop_reason = "HARD_STOP_UNCONFIRMED"
                        else:
                            stop_reason = "HARD_STOP_UNCONFIRMED"
                    else:
                        report["child_exit_code"] = process.returncode
                else:
                    stop_reason = "HARD_STOP_UNCONFIRMED"

        report["elapsed_wall_seconds"] = time.monotonic() - started
        report["stop_reason"] = stop_reason
        report["hard_stop_confirmed"] = stop_reason != "HARD_STOP_UNCONFIRMED"
        report["status"] = ("PASS_CHILD_EXITED" if stop_reason == "CHILD_EXITED"
                            and report["child_exit_code"] == 0 else stop_reason)
        _append_event(events, events_path, {"event": "parent_finished", "stop_reason": stop_reason,
                                             "child_exit_code": report["child_exit_code"],
                                             "elapsed_wall_seconds": report["elapsed_wall_seconds"]})
        _atomic_json(final_path, report)
    return report


def _read_json(path: str, label: str, *, deadline: float | None = None) -> dict:
    try:
        source = Path(path)
        if source.stat().st_size > 4 * 1024 * 1024:
            raise ApprovalError(f"{label} exceeds 4 MiB input limit")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"wall-clock budget expired before reading {label}")
        with source.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ApprovalError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ApprovalError(f"{label} must be a JSON object")
    return value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--approval-file", required=True)
    ap.add_argument("--source-sha", required=True)
    ap.add_argument("--dataset-manifest", required=True)
    ap.add_argument("--sample-plan", required=True)
    ap.add_argument("--model-identity", required=True)
    ap.add_argument("--output-dir", "--out", dest="out", required=True)
    ap.add_argument("--max-wall-seconds", required=True, type=float)
    ap.add_argument("smoke_args", nargs=argparse.REMAINDER,
                    help="arguments forwarded to the fixed R46 smoke child after --")
    args = ap.parse_args(argv)
    supervisor_started = time.monotonic()

    out = Path(args.out).expanduser().resolve()
    if out.exists():
        print("BLOCKED_OUTPUT: output directory already exists", file=sys.stderr)
        return 2
    if sys.platform != "linux":
        print("BLOCKED_PLATFORM: R46 execution requires Linux process-group semantics", file=sys.stderr)
        return 2
    if not math.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= MAX_WALL_SECONDS:
        print("BLOCKED_BUDGET: max wall seconds must be finite and in (0,1800]", file=sys.stderr)
        return 2
    try:
        actual_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                    capture_output=True, text=True,
                                    timeout=max(0.001, min(5.0, supervisor_started + args.max_wall_seconds
                                                           - time.monotonic())),
                                    check=True).stdout.strip()
        approval = _read_json(args.approval_file, "approval",
                              deadline=supervisor_started + args.max_wall_seconds)
    except (ApprovalError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"BLOCKED_AUTHORIZATION: {exc}", file=sys.stderr)
        return 2
    try:
        if actual_sha != args.source_sha:
            raise ApprovalError("requested source SHA differs from current checkout HEAD")
        deadline = supervisor_started + args.max_wall_seconds
        manifest_path = Path(args.dataset_manifest).expanduser().resolve()
        sample_path = Path(args.sample_plan).expanduser().resolve()
        identity_path = Path(args.model_identity).expanduser().resolve()
        identity = _read_json(str(identity_path), "model identity", deadline=deadline)
        model_file_list = model_identity_sha_list(identity)
        hashes = {
            "approved_original_manifest_sha256": sha256_file(manifest_path, deadline=deadline),
            "approved_smoke_samples_sha256": sha256_file(sample_path, deadline=deadline),
            "approved_model_sha_list": model_file_list,
        }
        normalized = validate_approval(approval, source_sha=actual_sha,
                                       requested_wall_seconds=args.max_wall_seconds,
                                       expected_hashes=hashes)
        disk_free = shutil.disk_usage(out.parent).free
        if disk_free < 1024**3:
            raise ApprovalError("output filesystem requires at least 1 GiB free for bounded logs")
        child_args = list(args.smoke_args)
        if child_args and child_args[0] == "--":
            child_args = child_args[1:]
        if "--" in child_args or any(arg in ("--gpu-smoke-authorized", "--gpu-window-confirmed")
                                      for arg in child_args):
            raise ApprovalError("caller cannot pass legacy authorization flags to the smoke child")
        child_args = validate_smoke_args(child_args)
        model_root = smoke_base_model_path(child_args)
        verify_model_identity(identity, model_root, deadline=deadline)
        if time.monotonic() >= deadline:
            raise TimeoutError("wall-clock budget expired while attesting the OpenVLA model")
        child = [sys.executable, str(REPO / "diagnostics/r46_openvla_smoke.py"),
                 "--approval-file", str(Path(args.approval_file).resolve()),
                 "--source-sha", actual_sha,
                 "--sample-plan", str(Path(args.sample_plan).resolve()), "--dataset-manifest",
                 str(Path(args.dataset_manifest).resolve()), "--model-identity",
                 str(identity_path), "--approved-gpu-uuid",
                 normalized["approved_gpu_uuid"], "--out", str(out / "smoke"),
                 "--max-wall-seconds", str(args.max_wall_seconds), "--", *child_args]
    except (ApprovalError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"BLOCKED_APPROVAL_CONTRACT: {exc}", file=sys.stderr)
        return 2

    out.mkdir(parents=True, mode=0o700)
    events_path = out / "supervisor.jsonl"
    with events_path.open("x", encoding="utf-8") as events:
        _append_event(events, events_path, {"event": "approval_validated",
                                            "approval_id": normalized["approval_id"],
                                            "source_sha": actual_sha,
                                            "deadline_monotonic_seconds": deadline})
    try:
        if time.monotonic() >= deadline:
            raise TimeoutError("wall-clock budget expired before GPU inventory")
        admission = probe_gpu(normalized["approved_gpu_uuid"],
                              float(normalized["approved_min_total_gib"]),
                              float(normalized["approved_min_free_gib"]),
                              timeout_seconds=min(15.0, deadline - time.monotonic()))
        if time.monotonic() >= deadline:
            raise TimeoutError("wall-clock budget expired during GPU admission")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError, TimeoutError) as exc:
        blocked = {"schema": "r46-supervisor/v1", "status": "BLOCKED_GPU_ADMISSION",
                   "stop_reason": "BLOCKED_GPU_ADMISSION", "approval_id": normalized["approval_id"],
                   "source_sha": actual_sha, "error": repr(exc),
                   "elapsed_wall_seconds": time.monotonic() - supervisor_started,
                   "hard_stop_confirmed": True, "child_pid": None, "child_exit_code": None}
        with events_path.open("a", encoding="utf-8") as events:
            _append_event(events, events_path, {"event": "gpu_admission_blocked", "error": repr(exc)})
        _atomic_json(out / "supervisor_final.json", blocked)
        print(f"BLOCKED_GPU_ADMISSION: {exc}", file=sys.stderr)
        return 2

    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = normalized["approved_gpu_uuid"]
    child_env["R46_SUPERVISOR_PID"] = str(os.getpid())
    result = run_child(child, out, args.max_wall_seconds,
                       started_monotonic=supervisor_started, env=child_env, precreated=True)
    result["approval_id"] = normalized["approval_id"]
    result["admission"] = admission
    result["input_sha256"] = hashes
    _atomic_json(out / "supervisor_final.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS_CHILD_EXITED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
