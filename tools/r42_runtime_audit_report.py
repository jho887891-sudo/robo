#!/usr/bin/env python3
"""Validate R42 runtime JSONL evidence without repairing or rewriting the source trace.

Exit codes:
  0: trace is valid, complete, and has no action errors; the first server response is present
  2: evidence is partial, an action failed, or a required first-action gate is blocked
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


TRACE_SCHEMA = "r42-runtime-audit/v2"
SUPPORTED_TRACE_SCHEMAS = {"r42-runtime-audit/v1", TRACE_SCHEMA}
REPORT_SCHEMA = "r42-runtime-audit-report/v2"
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


def _validate_record(record, index, camera_order, dim_order, joint_group_order, errors):
    event = record.get("event")
    if not isinstance(event, str) or event not in EVENTS:
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
        overhead = record.get("observation_audit_overhead_ms")
        if not _is_finite_number(overhead) or float(overhead) < 0:
            _err(errors, index, "observation_audit_overhead_ms must be finite and >= 0")
        return

    if event == "action_step":
        for key in ("observation_record_index", "observation_index", "episode_index",
                    "episode_step_index", "global_step_index", "env_idx"):
            _int_field(record, key, index, errors)
        state14_valid = _vector(record.get("state14"), 14, "state14", index, errors)
        _vector(record.get("velocity14"), 14, "velocity14", index, errors)
        if record.get("schema_version") == TRACE_SCHEMA:
            _vector(record.get("rate_reference_action14"), 14,
                    "rate_reference_action14", index, errors)
            rate_source = record.get("rate_limit_source")
            if rate_source != "safety_limits.max_rate":
                _err(errors, index, "rate_limit_source must be safety_limits.max_rate")
            reference_source = record.get("rate_reference_source")
            if reference_source not in ("measured_state", "previous_server_response"):
                _err(errors, index,
                     "rate_reference_source must be measured_state or previous_server_response")
            action_dt = record.get("control_dt")
            if not _is_finite_number(action_dt) or float(action_dt) <= 0:
                _err(errors, index, "control_dt must be finite and > 0")
        overhead = record.get("observation_audit_overhead_ms")
        if not _is_finite_number(overhead) or float(overhead) < 0:
            _err(errors, index, "observation_audit_overhead_ms must be finite and >= 0")
        pre_audit_time = record.get("pre_audit_return_wall_utc")
        if not isinstance(pre_audit_time, str) or not pre_audit_time:
            _err(errors, index, "pre_audit_return_wall_utc must be a non-empty string")
        span = record.get("server_span_pre_audit_ms")
        if not _is_finite_number(span) or float(span) < 0:
            _err(errors, index, "server_span_pre_audit_ms must be finite and >= 0")
        timing_semantics = record.get("timing_semantics")
        if not isinstance(timing_semantics, str) or "fsync" not in timing_semantics.lower():
            _err(errors, index, "timing_semantics must explain the fsync boundary")
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
            action14_valid = _vector(
                response.get("action14"), 14, "server_response.action14", index, errors)
            action_dict = response.get("action_dict")
            if not isinstance(action_dict, dict):
                _err(errors, index, "server_response.action_dict must be an object")
            elif joint_group_order:
                if set(action_dict) != set(joint_group_order):
                    _err(errors, index, "server_response.action_dict keys differ from joint_group_order")
                flattened = []
                flatten_valid = True
                for group in joint_group_order:
                    values = action_dict.get(group)
                    expected_dim = sum(
                        1 for name in dim_order
                        if isinstance(name, str) and name.startswith(group + "["))
                    if (not isinstance(values, list) or len(values) != expected_dim
                            or not all(_is_finite_number(value) for value in values)):
                        _err(errors, index,
                             "server_response.action_dict.%s must contain %d finite values"
                             % (group, expected_dim))
                        flatten_valid = False
                    else:
                        flattened.extend(float(value) for value in values)
                action14 = response.get("action14")
                if (flatten_valid and action14_valid and len(flattened) == 14
                        and any(abs(flattened[i] - float(action14[i])) > 1e-6
                                for i in range(14))):
                    _err(errors, index,
                         "server_response.action_dict flattened values differ from action14")
            decoder_info = response.get("decoder_info")
            if not isinstance(decoder_info, dict):
                _err(errors, index, "server_response.decoder_info must be an object")
            else:
                if "dt" not in decoder_info:
                    _err(errors, index, "server_response.decoder_info.dt is required (null means rate branch skipped)")
                rate_limited = decoder_info.get("rate_limited")
                if not _is_int(rate_limited) or rate_limited < 0:
                    _err(errors, index, "server_response.decoder_info.rate_limited must be an integer >= 0")
            semantics = response.get("semantics")
            if not isinstance(semantics, str) or "server-generated" not in semantics:
                _err(errors, index, "server_response semantics must say server-generated")

        previous_action = record.get("previous_server_response_action14")
        observed_delta = record.get("observed_state_minus_previous_server_response14")
        previous_step = record.get("previous_response_global_step_index")
        if previous_action is None:
            if observed_delta is not None:
                _err(errors, index, "delta must be null when no previous server response exists")
            if previous_step is not None:
                _err(errors, index, "previous response step index must be null without a previous response")
        else:
            previous_valid = _vector(
                previous_action, 14, "previous_server_response_action14", index, errors)
            delta_valid = _vector(
                observed_delta, 14, "observed_state_minus_previous_server_response14", index, errors)
            current_step = record.get("global_step_index")
            if (not _is_int(previous_step) or not _is_int(current_step)
                    or previous_step >= current_step):
                _err(errors, index, "previous_response_global_step_index must precede global_step_index")
            if state14_valid and previous_valid and delta_valid:
                expected_delta = [
                    float(state) - float(previous)
                    for state, previous in zip(record["state14"], previous_action)
                ]
                if any(abs(float(actual) - expected) > 1e-5
                       for actual, expected in zip(observed_delta, expected_delta)):
                    _err(errors, index,
                         "observed_state_minus_previous_server_response14 is inconsistent with state14 - previous response")

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
            has_newline = raw_line.endswith(b"\n")
            payload = raw_line.rstrip(b"\r\n")
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
        run_ids = [r.get("run_id") for r in records]
        if (not all(isinstance(value, str) and value for value in run_ids)
                or len(set(run_ids)) != 1):
            errors.append("all records must share one non-empty string run_id")
        for position, record in enumerate(records):
            index = record.get("record_index")
            if not _is_int(index) or index != position:
                _err(errors, index if _is_int(index) else position,
                     "record_index must be contiguous from zero (expected %d)" % position)
            if record.get("schema_version") not in SUPPORTED_TRACE_SCHEMAS:
                _err(errors, index if _is_int(index) else position,
                     "schema_version must be one of %s" % sorted(SUPPORTED_TRACE_SCHEMAS))
        starts = [i for i, r in enumerate(records) if r.get("event") == "run_start"]
        if starts != [0]:
            errors.append("exactly one run_start must be the first record")
        metadata = records[0].get("metadata", {}) if records[0].get("event") == "run_start" else {}
        camera_order = metadata.get("camera_order", []) if isinstance(metadata, dict) else []
        dim_order = metadata.get("state_dim_order", []) if isinstance(metadata, dict) else []
        joint_group_order = metadata.get("joint_group_order", []) if isinstance(metadata, dict) else []
        if (not isinstance(camera_order, list) or not camera_order
                or not all(isinstance(x, str) and x for x in camera_order)
                or len(set(camera_order)) != len(camera_order)):
            errors.append("run_start metadata must include unique camera_order names")
            camera_order = []
        if (not isinstance(dim_order, list) or len(dim_order) != 14
                or not all(isinstance(x, str) and x for x in dim_order)):
            errors.append("run_start metadata must include 14 state_dim_order names")
            dim_order = []
        if (not isinstance(joint_group_order, list) or not joint_group_order
                or not all(isinstance(x, str) and x for x in joint_group_order)
                or len(set(joint_group_order)) != len(joint_group_order)):
            errors.append("run_start metadata must include unique joint_group_order names")
            joint_group_order = []
        configured_rate = metadata.get("configured_max_rate14") if isinstance(metadata, dict) else None
        if configured_rate is not None and (
                not isinstance(configured_rate, list) or len(configured_rate) != 14
                or not all(_is_finite_number(x) and float(x) >= 0 for x in configured_rate)):
            errors.append("run_start configured_max_rate14 must be null or 14 finite non-negative values")
        control_dt = metadata.get("control_dt") if isinstance(metadata, dict) else None
        control_hz = metadata.get("control_hz") if isinstance(metadata, dict) else None
        if (not _is_finite_number(control_dt) or float(control_dt) <= 0
                or not _is_finite_number(control_hz) or float(control_hz) <= 0
                or abs(float(control_dt) * float(control_hz) - 1.0) > 1e-4):
            errors.append("run_start metadata must include consistent positive control_dt and control_hz")
        for position, record in enumerate(records):
            idx = record.get("record_index", position)
            _validate_record(record, idx, camera_order, dim_order, joint_group_order, errors)
    else:
        camera_order = []
        dim_order = []
        joint_group_order = []
        metadata = {}

    observations = {r.get("record_index"): r for r in records
                    if r.get("event") == "observation_received"
                    and _is_int(r.get("record_index"))}
    action_steps = [r for r in records if r.get("event") == "action_step"]
    action_errors = [r for r in records if r.get("event") == "action_error"]
    resets = [r for r in records if r.get("event") == "episode_reset"]
    backend_ready = [r for r in records if r.get("event") == "backend_ready"]
    terminal_by_observation = {}
    linked_actions = {}

    for record in action_steps + action_errors:
        idx = record.get("record_index")
        ref = record.get("observation_record_index")
        if not _is_int(idx):
            _err(errors, idx, "action/error record_index must be an integer")
            continue
        if not _is_int(ref):
            _err(errors, idx, "observation_record_index must be an integer")
            continue
        observation = observations.get(ref)
        if observation is None:
            _err(errors, idx, "observation_record_index does not reference an observation_received record")
            continue
        if ref >= idx:
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

    # Independently verify that each schema-v2 rate reference matches its declared source.
    # Per-environment history is used because the wrapper keeps separate previous commands by env.
    last_action_by_episode_env = {}
    for record in action_steps:
        if record.get("schema_version") != TRACE_SCHEMA:
            continue
        idx = record.get("record_index")
        key = (record.get("episode_index"), record.get("env_idx"))
        source = record.get("rate_reference_source")
        reference = record.get("rate_reference_action14")
        state = record.get("state14")
        if source == "measured_state":
            if (isinstance(reference, list) and isinstance(state, list)
                    and len(reference) == 14 and len(state) == 14
                    and all(_is_finite_number(x) for x in reference + state)
                    and any(abs(float(reference[i]) - float(state[i])) > 1e-5
                            for i in range(14))):
                _err(errors, idx, "measured_state rate reference differs from state14")
        elif source == "previous_server_response":
            previous = last_action_by_episode_env.get(key)
            if previous is None:
                _err(errors, idx,
                     "previous_server_response rate reference has no prior action for this episode/env")
            elif (isinstance(reference, list) and len(reference) == 14
                    and all(_is_finite_number(x) for x in reference)
                    and any(abs(float(reference[i]) - float(previous[i])) > 1e-5
                            for i in range(14))):
                _err(errors, idx,
                     "previous_server_response rate reference differs from prior action_step action14")
        response = record.get("server_response")
        action = response.get("action14") if isinstance(response, dict) else None
        if (isinstance(action, list) and len(action) == 14
                and all(_is_finite_number(x) for x in action)):
            last_action_by_episode_env[key] = [float(x) for x in action]

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
    post_reset = bool(first_observation and any(
        _is_int(r.get("record_index"))
        and r.get("record_index") < first_observation.get("record_index")
        and r.get("episode_index") == first_observation.get("episode_index")
        for r in resets))
    first_response = "PASS" if "action_step" in first_linked else "BLOCKED"
    configured_rate = metadata.get("configured_max_rate14") if isinstance(metadata, dict) else None
    control_dt = metadata.get("control_dt") if isinstance(metadata, dict) else None
    rate_config_valid = (
        isinstance(configured_rate, list) and len(configured_rate) == 14
        and all(_is_finite_number(x) and float(x) >= 0 for x in configured_rate)
        and _is_finite_number(control_dt) and float(control_dt) > 0)
    action_rate_info = [
        (r.get("server_response") or {}).get("decoder_info")
        for r in action_steps
    ]
    rate_steps_active = bool(rate_config_valid and action_rate_info) and all(
        isinstance(info, dict)
        and _is_finite_number(info.get("dt"))
        and float(info["dt"]) > 0
        and abs(float(info["dt"]) - float(control_dt)) <= 1e-6
        and _is_int(info.get("rate_limited")) and info["rate_limited"] >= 0
        for info in action_rate_info)
    rate_gate = "PASS" if rate_config_valid and rate_steps_active else "BLOCKED"
    if not rate_config_valid:
        rate_reason = "run_start metadata has no valid configured_max_rate14/control_dt"
    elif not action_rate_info:
        rate_reason = "no successful action_step exists to prove the decoder rate branch was entered"
    elif any(not isinstance(info, dict) or info.get("dt") is None for info in action_rate_info):
        rate_reason = ("one or more decoder_info.dt values are null; the decoder skips rate limiting "
                       "when prev_cmd and dt are both absent")
    elif not rate_steps_active:
        rate_reason = "decoder dt does not match the configured control_dt or rate metadata is invalid"
    else:
        rate_reason = "configured max_rate exists and every successful action reports the expected positive dt"

    rate_compliance_checks = []
    rate_compliance_problem = None
    if rate_config_valid and rate_steps_active and action_steps:
        max_step = [float(value) * float(control_dt) for value in configured_rate]
        for record in action_steps:
            response = record.get("server_response")
            action14 = response.get("action14") if isinstance(response, dict) else None
            reference14 = record.get("rate_reference_action14")
            record_dt = record.get("control_dt")
            vectors_valid = (
                isinstance(action14, list) and len(action14) == 14
                and all(_is_finite_number(value) for value in action14)
                and isinstance(reference14, list) and len(reference14) == 14
                and all(_is_finite_number(value) for value in reference14))
            if not vectors_valid:
                rate_compliance_checks.append(False)
                rate_compliance_problem = "an action lacks finite 14D action/reference vectors"
                continue
            if (not _is_finite_number(record_dt)
                    or abs(float(record_dt) - float(control_dt)) > 1e-6):
                rate_compliance_checks.append(False)
                rate_compliance_problem = "action control_dt differs from run_start control_dt"
                continue
            within_bound = all(
                abs(float(action14[i]) - float(reference14[i])) <= max_step[i] + 1e-5
                for i in range(14))
            rate_compliance_checks.append(within_bound)
            if not within_bound:
                rate_compliance_problem = (
                    "server_response.action14 exceeds max_rate * control_dt relative to "
                    "rate_reference_action14")
    rate_compliance_gate = (
        "PASS" if rate_config_valid and rate_steps_active and action_steps
        and len(rate_compliance_checks) == len(action_steps)
        and all(rate_compliance_checks) else "BLOCKED")
    if not rate_config_valid:
        rate_compliance_reason = "run_start metadata has no valid configured_max_rate14/control_dt"
    elif not rate_steps_active:
        rate_compliance_reason = "rate-limit decoder branch is not certified active"
    elif not action_steps:
        rate_compliance_reason = "no successful action_step exists to verify rate compliance"
    elif rate_compliance_problem:
        rate_compliance_reason = rate_compliance_problem
    elif rate_compliance_gate == "PASS":
        rate_compliance_reason = "every action delta is within configured max_rate * control_dt"
    else:
        rate_compliance_reason = "one or more action deltas could not be verified"

    gates = {
        "first_observation": "PASS" if first_observation else "BLOCKED",
        "post_reset_observation": "PASS" if post_reset else "BLOCKED",
        "first_server_response": first_response,
        "rate_limit_branch_active": rate_gate,
        "rate_limit_compliance": rate_compliance_gate,
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
        "rate_limit_branch_active": rate_reason,
        "rate_limit_compliance": rate_compliance_reason,
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
    if integrity != "PASS":
        for gate in ("first_observation", "post_reset_observation", "first_server_response",
                     "rate_limit_branch_active", "rate_limit_compliance"):
            gates[gate] = "BLOCKED"
            gate_reasons[gate] = (
                "trace integrity is %s; this gate cannot be certified" % integrity)
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
        "latency_caveat": (
            "The JSONL writer fsyncs each record. observation_audit_overhead_ms includes camera "
            "hashing and observation persistence. server_span_pre_audit_ms includes observation "
            "audit overhead but is sampled before action_step fsync; actual client receive time is "
            "not measured. Do not compare these timings directly with unaudited runs."),
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
        abs_path = os.path.abspath(os.path.expanduser(args.input))
        print("TRACE INPUT UNAVAILABLE: %s" % exc, file=sys.stderr)
        report = {
            "schema_version": REPORT_SCHEMA,
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "input_path": abs_path,
            "input_sha256": None,
            "run_id": None,
            "latency_caveat": (
                "This run has no readable trace, so no runtime timing can be certified. "
                "The audit writer is configured to fsync each record."),
            "trace_integrity": "FAIL",
            "run_outcome": "INVALID",
            "counts": {"records": 0, "observations": 0, "actions": 0,
                       "action_errors": 0, "resets": 0, "backend_ready": 0},
            "unresolved_observation_records": [],
            "gates": {
                "first_observation": "BLOCKED",
                "post_reset_observation": "BLOCKED",
                "first_server_response": "BLOCKED",
                "rate_limit_branch_active": "BLOCKED",
                "rate_limit_compliance": "BLOCKED",
                "client_acknowledgement": "UNKNOWN",
                "camera_state_sync": "UNKNOWN",
                "robot_actuation": "UNKNOWN",
                "reset_vs_demo_S0": "BLOCKED",
            },
            "gate_reasons": {
                "first_observation": "trace file was unavailable",
                "post_reset_observation": "trace file was unavailable",
                "first_server_response": "trace file was unavailable",
                "rate_limit_branch_active": "trace file was unavailable; rate limiting cannot be certified",
                "rate_limit_compliance": "trace file was unavailable; action deltas cannot be checked",
                "client_acknowledgement": "the server trace cannot observe client acceptance",
                "camera_state_sync": "camera/state wall-clock synchronization is not established",
                "robot_actuation": "server-generated actions do not prove execution",
                "reset_vs_demo_S0": "no demo-start S0 reference vector was supplied",
            },
            "errors": [str(exc)],
            "warnings": [],
        }

    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
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
    if (report["trace_integrity"] == "PARTIAL"
            or report["run_outcome"] == "PARTIAL"
            or report["counts"]["action_errors"] > 0
            or report["gates"]["first_observation"] != "PASS"
            or report["gates"]["post_reset_observation"] != "PASS"
            or report["gates"]["first_server_response"] != "PASS"
            or report["gates"]["rate_limit_branch_active"] != "PASS"
            or report["gates"]["rate_limit_compliance"] != "PASS"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
