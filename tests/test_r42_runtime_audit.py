import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_DIR = os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_R42")
sys.path.insert(0, POLICY_DIR)

from r42_runtime_audit import (
    RuntimeAuditError,
    RuntimeAuditWriter,
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writer_persists_ordered_jsonl_records_and_shared_run_id(tmp_path):
    path = tmp_path / "runtime.jsonl"
    writer = RuntimeAuditWriter(
        str(path), "r30-test-001",
        run_metadata={"checkpoint_dir": "/checkpoints/step1000",
                      "normalizer_sha256": "a" * 64},
    )
    writer.write({"event": "episode_reset", "episode_index": 1})
    writer.write({"event": "action_step", "state14": [0.0] * 14})
    writer.close()

    rows = read_jsonl(path)
    assert [row["event"] for row in rows] == [
        "run_start", "episode_reset", "action_step"
    ]
    assert [row["record_index"] for row in rows] == [0, 1, 2]
    assert {row["run_id"] for row in rows} == {"r30-test-001"}
    assert rows[0]["metadata"]["checkpoint_dir"] == "/checkpoints/step1000"
    assert rows[2]["state14"] == [0.0] * 14
    assert all(row["schema_version"] == "r42-runtime-audit/v2" for row in rows)
    assert all(row["monotonic_ns"] > 0 for row in rows)


def test_writer_refuses_existing_file_without_truncating_it(tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_text("historical evidence\n", encoding="utf-8")

    with pytest.raises(RuntimeAuditError, match="already exists"):
        RuntimeAuditWriter(str(path), "r30-test-002")

    assert path.read_text(encoding="utf-8") == "historical evidence\n"


def test_writer_requires_a_safe_nonempty_run_id(tmp_path):
    path = tmp_path / "runtime.jsonl"
    with pytest.raises(RuntimeAuditError, match="run_id"):
        RuntimeAuditWriter(str(path), "")
    with pytest.raises(RuntimeAuditError, match="run_id"):
        RuntimeAuditWriter(str(path), "unsafe run id")
    assert not path.exists()


def test_writer_rejects_nonfinite_values_without_writing_a_partial_record(tmp_path):
    path = tmp_path / "runtime.jsonl"
    writer = RuntimeAuditWriter(str(path), "r30-test-003")
    with pytest.raises(RuntimeAuditError, match="JSON-safe"):
        writer.write({"event": "action_step", "state14": [float("nan")]})

    rows = read_jsonl(path)
    assert [row["event"] for row in rows] == ["run_start"]


def test_writer_requires_event_and_existing_parent_directory(tmp_path):
    path = tmp_path / "missing" / "runtime.jsonl"
    with pytest.raises(RuntimeAuditError, match="parent directory"):
        RuntimeAuditWriter(str(path), "r30-test-004")

    valid = tmp_path / "valid.jsonl"
    writer = RuntimeAuditWriter(str(valid), "r30-test-005")
    with pytest.raises(RuntimeAuditError, match="event"):
        writer.write({"state14": [0.0] * 14})
    writer.close()
    assert [row["event"] for row in read_jsonl(valid)] == ["run_start"]
