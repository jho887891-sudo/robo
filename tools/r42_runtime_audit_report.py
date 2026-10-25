#!/usr/bin/env python3
"""Validate R42 runtime JSONL evidence without repairing or rewriting the source trace.

Exit codes:
  0: trace is structurally valid and every observation has an action or recorded action error
  2: evidence is partial (e.g. interrupted JSONL tail or an unresolved observation)
  3: trace is invalid (malformed records, schema violations, or broken references)
  4: report output path already exists or cannot be written safely
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone


TRACE_SCHEMA = "r42-runtime-audit/v1"
REPORT_SCHEMA = "r42-runtime-audit-report/v1"
EVENTS = {
    "run_start", "episode_reset", "observation_received", "backend_ready",
    "action_error", "action_step",
}
CAMERA_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
LATENCY_KEYS = ("preprocess", "forward_total", "decode", "total")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _err(errors, index, message):
    errors.append("record %s: %s" % (index if index is not None else "?", message))


def _vector(value, size, label, index, errors):
    if not isinstance(value, list) or len(value) != size:
        _err(errors, index, "%s must be a list of length %d" % (label, size))
        return False
    if not all(_is_finite_number(x) for x in value):
        _err(errors, index, "%s contains a non-finite or non-numeric value" % label)
        return False
    return True


def _int_field(record, key, index, errors, minimum=0):
    value = record.get(key)
    if not _is_int(value) or value < minimum:
        _err(errors, index, "%s must be an integer >= %d" % (key, minimum))
        return False
    return True


def _validate_camera_inputs(record, camera_order, index, errors):
    cameras = record.get("camera_inputs")
    if not isinstance(cameras, dict):
        _err(errors, index, "camera_inputs must be an object")
        return
    if camera_order and set(cameras) != set(camera_order):
        _err(errors, index, "camera_inputs keys do not match run_start metadata camera_order")
    for name, camera in cameras.items():
        if not isinstance(camera, dict):
            _err(errors, index, "camera_inputs.%s must be an object" % name)
            continue
        if not isinstance(camera.get("sha256"), str) or not CAMERA_HASH_RE.fullmatch(camera["sha256"]):
            _err(errors, index, "camera_inputs.%s.sha256 must be 64 lowercase hex chars" % name)
        shape = camera.get("shape")
        if (not isinstance(shape, list) or len(shape) != 3
                or not all(_is_int(x) and x > 0 for x in shape)):
            _err(errors, index, "camera_inputs.%s.shape must be three positive integers" % name)
        if camera.get("dtype") != "uint8":
            _err(errors, index, "camera_inputs.%s.dtype must be uint8" % name)


def _validate_record(record, index, camera_order, dim_order, errors):
    event = record.get("event")
    if event not in EVENTS:
        _err(errors, index, "unknown event %r" % event)
        return

    if event == "run_start":
        if not isinstance(record.get("path"), str) or not record["path"]:
            _err(errors, index, "run_start.path must be a non-empty string")
        if not isinstance(record.get("metadata"), dict):
            _err(errors, index, "run_start.metadata must be an object")
        return

    if event == "episode_reset":
        for key in ("episode_index", "reset_count", "global_step_count"):
            _int_field(record, key, index, errors)
        return

    if event == "backend_ready":
        return

    if event == "observation_received":
        for key in ("observation_index", "episode_index", "episode_step_index",
                    "global_step_index", "env_idx"):
            _int_field(record, key, index, errors)
        _vector(record.get("state14"), 14, "state14", index, errors)
        dims = record.get("state_dim_order")
        if not isinstance(dims, list) or len(dims) != 14:
            _err(errors, index, "state_dim_order must contain 14 names")
        elif dim_order and dims != dim_order:
            _err(errors, index, "state_dim_order differs from run_start metadata")
        _validate_camera_inputs(record, camera_order, index, errors)
        if record.get("raw_images_persisted") is not False:
            _err(errors, index, "raw_images_persisted must be false")
        if "images" in record or "raw_image_bytes" in record:
            _err(errors, index, "raw image payload must not be persisted")
        return

    if event == "action_error":
        for key in ("episode_index", "episode_step_index", "global_step_index", "env_idx",
                    "observation_record_index", "observation_index"):
            _int_field(record, key, index, errors)
        for key in ("stage", "error_type", "error_message"):
            if not isinstance(record.get(key), str):
                _err(errors, index, "%s must be a string" % key)
        return

    if event == "action_step":
        for key in ("observation_record_index", "observation_index", "episode_index",
                    "episode_step_index", "global_step_index", "env_idx"):
            _int_field(record, key, index, errors)
        _vector(record.get("state14"), 14, "state14", index, errors)
        _vector(record.get("velocity14"), 14, "velocity14", index, errors)
        _vector(record.get("features42"), 42, "features42", index, errors)
        dims = record.get("state_dim_order")
        if not isinstance(dims, list) or len(dims) != 14:
            _err(errors, index, "state_dim_order must contain 14 names")
        elif dim_order and dims != dim_order:
            _err(errors, index, "state_dim_order differs from run_start metadata")
        _validate_camera_inputs(record, camera_order, index, errors)

        prediction = record.get("prediction")
        if not isinstance(prediction, dict):
            _err(errors, index, "prediction must be an object")
        else:
            for key in ("token_ids14", "bins14"):
                values = prediction.get(key)
                if (not isinstance(values, list) or len(values) != 14
                        or not all(_is_int(x) for x in values)):
                    _err(errors, index, "prediction.%s must contain 14 integers" % key)
            _vector(prediction.get("residual14"), 14, "prediction.residual14", index, errors)

        response = record.get("server_response")
        if not isinstance(response, dict):
            _err(errors, index, "server_response must be an object")
        else:
            _vector(response.get("action14"), 14, "server_response.action14", index, errors)
            if not isinstance(response.get("action_dict"), dict):
                _err(errors, index, "server_response.action_dict must be an object")
            if not isinstance(response.get("decoder_info"), dict):
                _err(errors, index, "server_response.decoder_info must be an object")
            semantics = response.get("semantics")
            if not isinstance(semantics, str) or "server-generated" not in semantics:
                _err(errors, index, "server_response semantics must say server-generated")

        for key in ("previous_server_response_action14",
                    "observed_state_minus_previous_server_response14"):
            value = record.get(key)
            if value is not None:
                _vector(value, 14, key, index, errors)

        latency = record.get("latency_ms")
        if not isinstance(latency, dict):
            _err(errors, index, "latency_ms must be an object")
        else:
            for key in LATENCY_KEYS:
                value = latency.get(key)
                if not _is_finite_number(value) or float(value) < 0:
                    _err(errors, index, "latency_ms.%s must be finite and >= 0" % key)
            values = [latency.get(key) for key in LATENCY_KEYS]
            if all(_is_finite_number(x) and float(x) >= 0 for x in values):
                if float(latency["total"]) + 1e-9 < max(float(latency[k]) for k in LATENCY_KEYS[:-1]):
                    _err(errors, index, "latency_ms.total is smaller than an individual phase")
        if record.get("n_action_tokens") != 14:
            _err(errors, index, "n_action_tokens must be 14")
        if not isinstance(record.get("codec_mode"), str):
            _err(errors, index, "codec_mode must be a string")
        if record.get("raw_images_persisted") is True or "raw_image_bytes" in record:
            _err(errors, index, "raw image payload must not be persisted")


def audit_trace(path):
    """Return a machine-readable report. The input trace is never modified."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    with open(abs_path, "rb") as fh:
        raw = fh.read()
    digest = hashlib.sha256(raw).hexdigest()
    errors = []
    warnings = []
    records = []
    partial_tail = False

    if not raw:
        errors.append("trace file is empty")
    else:
        lines = raw.splitlines(keepends=True)
        for line_number, raw_line in enumerate(lines, 1):
            has_newline = raw_line.endswith(b"\\n")
            payload = raw_line.rstrip(b"\\r\\n")
            if not payload:
                _err(errors, line_number, "blank JSONL line")
                continue
            try:
                record = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if line_number == len(lines) and not has_newline:
                    partial_tail = True
                    warnings.append("incomplete trailing JSONL tail at line %d: %s"
                                    % (line_number, str(exc)))
                    continue
                _err(errors, line_number, "invalid JSONL record: %s" % exc)
                continue
            if not has_newline:
                partial_tail = True
                warnings.append("last JSONL record at line %d has no newline terminator" % line_number)
            if not isinstance(record, dict):
                _err(errors, line_number, "JSONL record must be an object")
                continue
            records.append(record)

    if records:
        run_ids = {r.get("run_id") for r in records}
        if len(run_ids) != 1 or not isinstance(next(iter(run_ids)), str) or not next(iter(run_ids)):
            errors.append("all records must share one non-empty run_id")
        for position, record in enumerate(records):
            index = record.get("record_index")
            if not _is_int(index) or index != position:
                _err(errors, index if _is_int(index) else position,
                     "record_index must be contiguous from zero (expected %d)" % position)
            if record.get("schema_version") != TRACE_SCHEMA:
                _err(errors, index if _is_int(index) else position,
                     "schema_version must be %s" % TRACE_SCHEMA)
        starts = [i for i, r in enumerate(records) if r.get("event") == "run_start"]
        if starts != [0]:
            errors.append("exactly one run_start must be the first record")
        metadata = records[0].get("metadata", {}) if records[0].get("event") == "run_start" else {}
        camera_order = metadata.get("camera_order", []) if isinstance(metadata, dict) else []
        dim_order = metadata.get("state_dim_order", []) if isinstance(metadata, dict) else []
        if not isinstance(camera_order, list) or not camera_order:
            errors.append("run_start metadata must include camera_order")
            camera_order = []
        if not isinstance(dim_order, list) or len(dim_order) != 14:
            errors.append("run_start metadata must include 14 state_dim_order names")
            dim_order = []
        for position, record in enumerate(records):
            idx = record.get("record_index", position)
            _validate_record(record, idx, camera_order, dim_order, errors)
    else:
        camera_order = []
        dim_order = []

    observations = {r.get("record_index"): r for r in records
                    if r.get("event") == "observation_received"}
    action_steps = [r for r in records if r.get("event") == "action_step"]
    action_errors = [r for r in records if r.get("event") == "action_error"]
    resets = [r for r in records if r.get("event") == "episode_reset"]
    backend_ready = [r for r in records if r.get("event") == "backend_ready"]
    terminal_by_observation = {}
    linked_actions = {}

    for record in action_steps + action_errors:
        idx = record.get("record_index")
        ref = record.get("observation_record_index")
        observation = observations.get(ref)
        if observation is None:
            _err(errors, idx, "observation_record_index does not reference an observation_received record")
            continue
        if not _is_int(ref) or ref >= idx:
            _err(errors, idx, "observation_record_index must point to an earlier record")
        if record.get("observation_index") != observation.get("observation_index"):
            _err(errors, idx, "observation_index does not match referenced observation")
        if record.get("episode_index") != observation.get("episode_index"):
            _err(errors, idx, "episode_index does not match referenced observation")
        if record.get("env_idx") != observation.get("env_idx"):
            _err(errors, idx, "env_idx does not match referenced observation")
        if record.get("event") == "action_step":
            if record.get("state14") != observation.get("state14"):
                _err(errors, idx, "action_step state14 differs from referenced observation")
            if record.get("camera_inputs") != observation.get("camera_inputs"):
                _err(errors, idx, "action_step camera_inputs differ from referenced observation")
        terminal_by_observation.setdefault(ref, set()).add(record.get("event"))
        linked_actions.setdefault(ref, []).append(record)

    unresolved = []
    for ref, observation in observations.items():
        if ref not in terminal_by_observation:
            unresolved.append(ref)
            warnings.append("observation record %s has no linked action_step or action_error" % ref)
        linked = linked_actions.get(ref, [])
        if sum(1 for r in linked if r.get("event") == "action_step") > 1:
            warnings.append("observation record %s has multiple action_step records" % ref)

    first_observation = next(iter(observations.values()), None)
    first_ref = first_observation.get("record_index") if first_observation else None
    first_linked = terminal_by_observation.get(first_ref, set()) if first_observation else set()
    reset_episodes = {r.get("episode_index") for r in resets}
    post_reset = bool(first_observation and
                      first_observation.get("episode_index") in reset_episodes)
    first_response = "PASS" if "action_step" in first_linked else "BLOCKED"
    gates = {
        "first_observation": "PASS" if first_observation else "BLOCKED",
        "post_reset_observation": "PASS" if post_reset else "BLOCKED",
        "first_server_response": first_response,
        "client_acknowledgement": "UNKNOWN",
        "camera_state_sync": "UNKNOWN",
        "robot_actuation": "UNKNOWN",
        "reset_vs_demo_S0": "BLOCKED",
    }
    gate_reasons = {
        "first_observation": ("validated 14D state and camera fingerprints recorded"
                              if first_observation else "no observation_received event"),
        "post_reset_observation": ("first observation belongs to a recorded reset episode"
                                   if post_reset else "no matching episode_reset before first observation"),
        "first_server_response": ("first observation has a linked server-generated action_step"
                                  if first_response == "PASS"
                                  else "first observation has no linked successful action_step"),
        "client_acknowledgement": "the server trace cannot observe client acceptance",
        "camera_state_sync": "camera/state wall-clock synchronization is not established by this schema",
        "robot_actuation": "server-generated actions do not prove simulator or robot execution",
        "reset_vs_demo_S0": "no demo-start S0 reference vector was supplied to this validator",
    }

    if errors:
        integrity = "FAIL"
    elif partial_tail:
        integrity = "PARTIAL"
    else:
        integrity = "PASS"
    if integrity == "FAIL":
        outcome = "INVALID"
    elif integrity == "PARTIAL" or unresolved:
        outcome = "PARTIAL"
    elif action_errors:
        outcome = "ACTION_ERRORS"
    else:
        outcome = "COMPLETE"

    report = {
        "schema_version": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input_path": abs_path,
        "input_sha256": digest,
        "run_id": records[0].get("run_id") if records else None,
        "trace_integrity": integrity,
        "run_outcome": outcome,
        "counts": {
            "records": len(records),
            "observations": len(observations),
            "actions": len(action_steps),
            "action_errors": len(action_errors),
            "resets": len(resets),
            "backend_ready": len(backend_ready),
        },
        "unresolved_observation_records": unresolved,
        "gates": gates,
        "gate_reasons": gate_reasons,
        "errors": errors,
        "warnings": warnings,
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="R42 runtime JSONL trace")
    parser.add_argument("--out", help="optional report JSON path; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        report = audit_trace(args.input)
    except OSError as exc:
        print(json.dumps({"trace_integrity": "FAIL", "errors": [str(exc)]}, indent=2), file=sys.stderr)
        return 3

    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\\n"
    if args.out:
        out_path = os.path.abspath(os.path.expanduser(args.out))
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(out_path, flags, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(rendered)
                fh.flush()
                os.fsync(fh.fileno())
        except FileExistsError:
            print("REFUSING to overwrite report: %s" % out_path, file=sys.stderr)
            return 4
        except OSError as exc:
            print("cannot write report %s: %s" % (out_path, exc), file=sys.stderr)
            return 4
    print(rendered, end="")
    if report["trace_integrity"] == "FAIL":
        return 3
    if report["trace_integrity"] == "PARTIAL" or report["run_outcome"] == "PARTIAL":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
