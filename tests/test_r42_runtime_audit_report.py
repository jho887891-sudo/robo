import copy
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

from r42_runtime_audit_report import audit_trace, main


SCHEMA = "r42-runtime-audit/v2"
RUN_ID = "r30-report-test-001"
CAMERAS = ["cam0", "cam1", "cam2"]
JOINT_GROUPS = ["left_arm_joint_state", "left_ee_joint_state",
                "right_arm_joint_state", "right_ee_joint_state"]
DIM_NAMES = ["left_arm_joint_state[%d]" % i for i in range(6)]
DIM_NAMES += ["left_ee_joint_state[0]"]
DIM_NAMES += ["right_arm_joint_state[%d]" % i for i in range(6)]
DIM_NAMES += ["right_ee_joint_state[0]"]
STATE = [0.1 + i * 0.02 for i in range(14)]
ACTION = [value + 0.01 for value in STATE]


def action_dict_from14(vector):
    return {
        "left_arm_joint_state": list(vector[:6]),
        "left_ee_joint_state": list(vector[6:7]),
        "right_arm_joint_state": list(vector[7:13]),
        "right_ee_joint_state": list(vector[13:14]),
    }


def rec(event, index, **fields):
    return {
        "event": event,
        "schema_version": SCHEMA,
        "run_id": RUN_ID,
        "record_index": index,
        "recorded_at_utc": "2026-10-25T00:00:00Z",
        "monotonic_ns": index + 1,
        **fields,
    }


def camera_inputs():
    return {
        name: {"sha256": (hexchar * 64), "shape": [8, 8, 3], "dtype": "uint8"}
        for name, hexchar in zip(CAMERAS, ["a", "b", "c"])
    }


def valid_rows():
    return [
        rec("run_start", 0, path="/tmp/run.jsonl", metadata={
            "camera_order": CAMERAS,
            "state_dim_order": DIM_NAMES,
            "joint_group_order": JOINT_GROUPS,
            "configured_max_rate14": [0.5] * 14,
            "control_dt": 0.04,
            "control_hz": 25.0,
            "raw_images_persisted": False,
            "camera_state_wallclock_sync": "UNKNOWN",
        }),
        rec("episode_reset", 1, episode_index=1, reset_count=1, global_step_count=0),
        rec("observation_received", 2, observation_index=0, episode_index=1,
            episode_step_index=0, global_step_index=0, env_idx=0,
            source_timestamp=None, frequency_hz=25.0,
            observation_wall_utc="2026-10-25T00:00:01Z",
            state_dim_order=DIM_NAMES, state14=STATE,
            camera_inputs=camera_inputs(), camera_state_wallclock_sync="UNKNOWN",
            raw_images_persisted=False),
        rec("backend_ready", 3, backend_type="StubBackend", device="cpu",
            metadata_source="test", checkpoint_dir="/tmp/checkpoint"),
        rec("action_step", 4, observation_record_index=2, observation_index=0,
            episode_index=1, episode_step_index=0, global_step_index=0, env_idx=0,
            source_timestamp=None, frequency_hz=25.0,
            observation_wall_utc="2026-10-25T00:00:01Z",
            pre_audit_return_wall_utc="2026-10-25T00:00:01.1Z", server_span_pre_audit_ms=100.0,
            timing_semantics="sampled before action_step fsync; not client receive time",
            observation_audit_overhead_ms=3.5,
            state_dim_order=DIM_NAMES, state14=STATE,
            rate_reference_action14=STATE, rate_reference_source="measured_state",
            rate_limit_source="safety_limits.max_rate", control_dt=0.04,
            velocity14=[0.0] * 14,
            features42=[0.0] * 42, camera_inputs=camera_inputs(),
            prediction={"token_ids14": list(range(14)), "bins14": list(range(14)),
                        "residual14": [0.0] * 14},
            server_response={"action14": ACTION, "action_dict": action_dict_from14(ACTION),
                             "decoder_info": {"clipped_joint": 0, "rate_limited": 0, "dt": 0.04},
                             "semantics": "server-generated response; execution is not observed"},
            previous_server_response_action14=None, previous_response_global_step_index=None,
            observed_state_minus_previous_server_response14=None,
            comparison_semantics="not proof of actuation", camera_state_wallclock_sync="UNKNOWN",
            latency_ms={"preprocess": 1.0, "forward_total": 90.0, "decode": 2.0, "total": 95.0},
            n_action_tokens=14, codec_mode="fixed_256"),
    ]


def write_trace(path, rows, trailing_newline=True):
    data = "\n".join(json.dumps(row, allow_nan=False) for row in rows)
    if trailing_newline:
        data += "\n"
    path.write_text(data, encoding="utf-8")


def test_audit_report_accepts_complete_trace_but_keeps_execution_unknown(tmp_path):
    path = tmp_path / "runtime.jsonl"
    write_trace(path, valid_rows())

    report = audit_trace(str(path))

    assert report["trace_integrity"] == "PASS"
    assert report["run_outcome"] == "COMPLETE"
    assert report["counts"] == {
        "records": 5, "observations": 1, "actions": 1, "action_errors": 0,
        "resets": 1, "backend_ready": 1,
    }
    assert report["gates"]["first_observation"] == "PASS"
    assert report["gates"]["first_server_response"] == "PASS"
    assert report["gates"]["rate_limit_branch_active"] == "PASS"
    assert report["gates"]["rate_limit_compliance"] == "PASS"
    assert report["gates"]["client_acknowledgement"] == "UNKNOWN"
    assert report["gates"]["camera_state_sync"] == "UNKNOWN"
    assert report["gates"]["robot_actuation"] == "UNKNOWN"
    assert "fsync" in report["latency_caveat"].lower()
    assert "client" in report["latency_caveat"].lower()
    assert report["errors"] == []


def test_audit_report_links_action_error_without_calling_trace_corrupt(tmp_path):
    path = tmp_path / "runtime.jsonl"
    rows = valid_rows()[:3]
    rows.append(rec("action_error", 3, episode_index=1, episode_step_index=0,
                    global_step_index=0, env_idx=0, observation_record_index=2,
                    observation_index=0, observation_audit_overhead_ms=3.5,
                    stage="server_act", error_type="RuntimeError",
                    error_message="backend exploded"))
    write_trace(path, rows)

    report = audit_trace(str(path))

    assert report["trace_integrity"] == "PASS"
    assert report["run_outcome"] == "ACTION_ERRORS"
    assert report["counts"]["action_errors"] == 1
    assert report["gates"]["first_observation"] == "PASS"
    assert report["gates"]["first_server_response"] == "BLOCKED"
    assert report["errors"] == []


def test_audit_report_marks_observation_without_terminal_event_partial(tmp_path):
    path = tmp_path / "runtime.jsonl"
    write_trace(path, valid_rows()[:3])

    report = audit_trace(str(path))

    assert report["trace_integrity"] == "PASS"
    assert report["run_outcome"] == "PARTIAL"
    assert report["gates"]["first_server_response"] == "BLOCKED"
    assert report["warnings"]


def test_audit_report_rejects_noncontiguous_record_indices_and_blocks_gates(tmp_path):
    path = tmp_path / "runtime.jsonl"
    rows = valid_rows()
    rows[3]["record_index"] = 20
    write_trace(path, rows)

    report = audit_trace(str(path))

    assert report["trace_integrity"] == "FAIL"
    assert any("record_index" in error for error in report["errors"])
    assert report["gates"]["first_server_response"] == "BLOCKED"


def test_audit_report_detects_incomplete_jsonl_tail_without_rewriting_evidence(tmp_path):
    path = tmp_path / "runtime.jsonl"
    rows = valid_rows()[:3]
    write_trace(path, rows, trailing_newline=False)
    original = path.read_bytes()

    report = audit_trace(str(path))

    assert report["trace_integrity"] == "PARTIAL"
    assert report["run_outcome"] == "PARTIAL"
    assert path.read_bytes() == original
    assert any("newline" in warning.lower() or "tail" in warning.lower()
               for warning in report["warnings"])


def test_audit_report_rejects_action_link_to_unknown_observation(tmp_path):
    path = tmp_path / "runtime.jsonl"
    rows = valid_rows()
    rows[-1]["observation_record_index"] = 999
    write_trace(path, rows)

    report = audit_trace(str(path))

    assert report["trace_integrity"] == "FAIL"
    assert any("observation_record_index" in error for error in report["errors"])



def test_audit_report_returns_failure_for_unhashable_corrupt_identity_fields(tmp_path):
    path = tmp_path / "runtime.jsonl"
    rows = valid_rows()
    rows[0]["run_id"] = ["not-a-string"]
    rows[2]["record_index"] = {"not": "an integer"}
    write_trace(path, rows)

    report = audit_trace(str(path))

    assert report["trace_integrity"] == "FAIL"
    assert report["errors"]


def test_cli_refuses_to_overwrite_existing_report(tmp_path, capsys):
    trace = tmp_path / "runtime.jsonl"
    write_trace(trace, valid_rows())
    output = tmp_path / "report.json"
    output.write_text("preserve this report\\n", encoding="utf-8")

    result = main(["--input", str(trace), "--out", str(output)])

    assert result == 4
    assert output.read_text(encoding="utf-8") == "preserve this report\\n"
    assert "REFUSING to overwrite report" in capsys.readouterr().err



def test_cli_writes_failure_report_when_trace_file_is_missing(tmp_path, capsys):
    missing = tmp_path / "missing.jsonl"
    output = tmp_path / "missing_report.json"

    result = main(["--input", str(missing), "--out", str(output)])

    assert result == 3
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["trace_integrity"] == "FAIL"
    assert report["run_outcome"] == "INVALID"
    assert report["counts"]["records"] == 0
    assert report["gates"]["first_observation"] == "BLOCKED"
    assert report["errors"]
    assert "No such file" in capsys.readouterr().err



def test_cli_blocks_next_gate_when_first_action_is_an_error(tmp_path, capsys):
    path = tmp_path / "runtime.jsonl"
    rows = valid_rows()[:3]
    rows.append(rec("action_error", 3, episode_index=1, episode_step_index=0,
                    global_step_index=0, env_idx=0, observation_record_index=2,
                    observation_index=0, observation_audit_overhead_ms=3.5,
                    stage="backend_load", error_type="RuntimeError",
                    error_message="backend load failed"))
    write_trace(path, rows)

    result = main(["--input", str(path)])

    assert result == 2
    report = json.loads(capsys.readouterr().out)
    assert report["trace_integrity"] == "PASS"
    assert report["run_outcome"] == "ACTION_ERRORS"
    assert report["gates"]["first_server_response"] == "BLOCKED"



def test_post_reset_gate_requires_reset_record_before_first_observation(tmp_path):
    path = tmp_path / "runtime.jsonl"
    rows = valid_rows()
    observation = rows[2]
    reset = rows[1]
    observation["record_index"] = 1
    reset["record_index"] = 2
    rows[1] = observation
    rows[2] = reset
    rows[-1]["observation_record_index"] = observation["record_index"]
    write_trace(path, rows)

    report = audit_trace(str(path))

    assert report["trace_integrity"] == "PASS"
    assert report["gates"]["first_observation"] == "PASS"
    assert report["gates"]["post_reset_observation"] == "BLOCKED"



def test_audit_report_rejects_action_dict_that_differs_from_flat_14d_response(tmp_path):
    path = tmp_path / "runtime.jsonl"
    rows = valid_rows()
    rows[-1]["server_response"]["action_dict"]["left_arm_joint_state"][0] += 0.25
    write_trace(path, rows)

    report = audit_trace(str(path))

    assert report["trace_integrity"] == "FAIL"
    assert any("action_dict" in error and "action14" in error
               for error in report["errors"])


def test_audit_report_checks_previous_response_delta_math(tmp_path):
    path = tmp_path / "runtime.jsonl"
    rows = valid_rows()
    second_state = [value + 0.1 for value in STATE]
    second_observation = copy.deepcopy(rows[2])
    second_observation.update({
        "record_index": 5,
        "observation_index": 1,
        "episode_step_index": 1,
        "global_step_index": 1,
        "state14": second_state,
    })
    second_action = copy.deepcopy(rows[4])
    second_action.update({
        "record_index": 6,
        "observation_record_index": 5,
        "observation_index": 1,
        "episode_step_index": 1,
        "global_step_index": 1,
        "state14": second_state,
        "rate_reference_action14": ACTION,
        "rate_reference_source": "previous_server_response",
        "previous_server_response_action14": ACTION,
        "previous_response_global_step_index": 0,
        "observed_state_minus_previous_server_response14": [
            state - action for state, action in zip(second_state, ACTION)
        ],
    })
    rows.extend([second_observation, second_action])
    write_trace(path, rows)

    report = audit_trace(str(path))

    assert report["trace_integrity"] == "PASS"
    assert report["counts"]["observations"] == 2
    assert report["counts"]["actions"] == 2


def test_audit_report_rejects_incorrect_previous_response_delta(tmp_path):
    path = tmp_path / "runtime.jsonl"
    rows = valid_rows()
    second_state = [value + 0.1 for value in STATE]
    second_observation = copy.deepcopy(rows[2])
    second_observation.update({
        "record_index": 5,
        "observation_index": 1,
        "episode_step_index": 1,
        "global_step_index": 1,
        "state14": second_state,
    })
    second_action = copy.deepcopy(rows[4])
    second_action.update({
        "record_index": 6,
        "observation_record_index": 5,
        "observation_index": 1,
        "episode_step_index": 1,
        "global_step_index": 1,
        "state14": second_state,
        "rate_reference_action14": ACTION,
        "rate_reference_source": "previous_server_response",
        "previous_server_response_action14": ACTION,
        "previous_response_global_step_index": 0,
        "observed_state_minus_previous_server_response14": [
            state - action for state, action in zip(second_state, ACTION)
        ],
    })
    second_action["observed_state_minus_previous_server_response14"][0] += 0.5
    rows.extend([second_observation, second_action])
    write_trace(path, rows)

    report = audit_trace(str(path))

    assert report["trace_integrity"] == "FAIL"
    assert any("observed_state_minus_previous_server_response14" in error
               for error in report["errors"])



def test_rate_limit_gate_blocks_when_configured_rule_is_bypassed(tmp_path, capsys):
    path = tmp_path / "runtime.jsonl"
    rows = valid_rows()
    rows[-1]["server_response"]["decoder_info"]["dt"] = None
    rows[-1]["server_response"]["decoder_info"]["rate_limited"] = 0
    write_trace(path, rows)

    report = audit_trace(str(path))
    assert report["trace_integrity"] == "PASS"
    assert report["gates"]["rate_limit_branch_active"] == "BLOCKED"
    assert report["gates"]["rate_limit_compliance"] == "BLOCKED"
    assert "dt" in report["gate_reasons"]["rate_limit_branch_active"].lower()

    result = main(["--input", str(path)])
    assert result == 2
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["gates"]["rate_limit_branch_active"] == "BLOCKED"



def test_audit_report_blocks_action_that_exceeds_configured_rate_step(tmp_path, capsys):
    path = tmp_path / "runtime.jsonl"
    rows = valid_rows()
    action = list(rows[-1]["server_response"]["action14"])
    action[0] += 0.1
    rows[-1]["server_response"]["action14"] = action
    rows[-1]["server_response"]["action_dict"] = action_dict_from14(action)
    write_trace(path, rows)

    report = audit_trace(str(path))

    assert report["trace_integrity"] == "PASS"
    assert report["gates"]["rate_limit_branch_active"] == "PASS"
    assert report["gates"]["rate_limit_compliance"] == "BLOCKED"
    assert "max_rate * control_dt" in report["gate_reasons"]["rate_limit_compliance"]

    result = main(["--input", str(path)])
    assert result == 2
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["gates"]["rate_limit_compliance"] == "BLOCKED"


def test_schema_v1_trace_remains_readable_but_cannot_certify_rate_compliance(tmp_path):
    path = tmp_path / "legacy-v1.jsonl"
    rows = valid_rows()
    for row in rows:
        row["schema_version"] = "r42-runtime-audit/v1"
    rows[-1].pop("rate_reference_action14")
    rows[-1].pop("rate_reference_source")
    rows[-1].pop("rate_limit_source")
    rows[-1].pop("control_dt")
    write_trace(path, rows)

    report = audit_trace(str(path))

    assert report["trace_integrity"] == "PASS"
    assert report["gates"]["rate_limit_branch_active"] == "PASS"
    assert report["gates"]["rate_limit_compliance"] == "BLOCKED"
    assert "reference vectors" in report["gate_reasons"]["rate_limit_compliance"]



def test_audit_report_rejects_rate_reference_not_linked_to_prior_action(tmp_path):
    path = tmp_path / "runtime.jsonl"
    rows = valid_rows()
    second_state = [value + 0.1 for value in STATE]
    second_observation = copy.deepcopy(rows[2])
    second_observation.update({
        "record_index": 5,
        "observation_index": 1,
        "episode_step_index": 1,
        "global_step_index": 1,
        "state14": second_state,
    })
    second_action = copy.deepcopy(rows[4])
    second_action.update({
        "record_index": 6,
        "observation_record_index": 5,
        "observation_index": 1,
        "episode_step_index": 1,
        "global_step_index": 1,
        "state14": second_state,
        "rate_reference_action14": STATE,
        "rate_reference_source": "previous_server_response",
        "previous_server_response_action14": ACTION,
        "previous_response_global_step_index": 0,
        "observed_state_minus_previous_server_response14": [
            state - action for state, action in zip(second_state, ACTION)
        ],
    })
    rows.extend([second_observation, second_action])
    write_trace(path, rows)

    report = audit_trace(str(path))

    assert report["trace_integrity"] == "FAIL"
    assert any("rate reference differs from prior action_step" in error
               for error in report["errors"])
