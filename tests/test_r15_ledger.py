#!/usr/bin/env python3
"""R15 Lane T / P0-LEDGER-2 regression: append-only WAL, no-overwrite, double cap.

RED first: the old `ledger_book()` rewrote the whole JSON with open(path,"w"), had no WAL, no
lock, no duplicate detection and no formal-step cap, and it had already been silently reset to 0
by an external write once in this mission.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "training"))

import r15_ledger as L  # noqa: E402


@pytest.fixture()
def led(tmp_path):
    return L.Ledger(str(tmp_path / "ledger"))


def test_fresh_ledger_is_empty(led):
    with led:
        t = led.totals()
    assert t["gpu_seconds_used"] == 0.0
    assert t["formal_steps_used"] == 0
    assert t["gpu_seconds_remaining"] == L.TOTAL_CAP_S
    assert t["formal_steps_remaining"] == L.FORMAL_STEPS_CAP


def test_history_import_keeps_the_already_spent_seconds(led, tmp_path):
    src = tmp_path / "historical_ledger.json"
    src.write_text(json.dumps({"aggregate": {"new_gpu_seconds_used": 1027.5}}))
    with led:
        led.import_history(str(src), 1027.5, formal_steps=0,
                           note="R13 169.7 + R14 857.8")
        t = led.totals()
        rep = led.bucket_used_reported()
    assert t["gpu_seconds_used"] == pytest.approx(1027.5)
    assert t["gpu_seconds_remaining"] == pytest.approx(L.TOTAL_CAP_S - 1027.5)
    assert t["formal_steps_used"] == 0
    assert rep["R13_7B_ZERO_STEP"]["used_s"] == pytest.approx(169.7)
    assert rep["R14_TRAIN_OVERFIT"]["used_s"] == pytest.approx(857.8)
    with led:
        with pytest.raises(L.LedgerError):
            led.import_history(str(src), 1027.5)


def test_duplicate_event_is_refused(led):
    with led:
        led.append("run-a", "job_begin", payload={"global_step": 0}, bucket="R15_TRAIN_FORMAL")
        with pytest.raises(L.LedgerError):
            led.append("run-a", "job_begin", payload={"global_step": 0},
                       bucket="R15_TRAIN_FORMAL")


def test_external_overwrite_of_the_projection_is_detected(led):
    with led:
        led.append("run-a", "job_begin", payload={"global_step": 0}, bucket="R15_TRAIN_FORMAL")
        led.append("run-a", "step_commit", payload={"global_step": 1}, gpu_seconds=2.5,
                   formal_steps=1, bucket="R15_TRAIN_FORMAL")
    proj = json.load(open(led.proj_path))
    proj["totals"]["gpu_seconds_used"] = 0.0
    proj["totals"]["formal_steps_used"] = 0
    json.dump(proj, open(led.proj_path, "w"))
    with led:
        with pytest.raises(L.LedgerError):
            led.verify_projection()


def test_truncated_wal_is_detected(led):
    with led:
        for i in range(3):
            led.append("run-a", "step_commit", payload={"global_step": i + 1}, gpu_seconds=1.0,
                       formal_steps=1, bucket="R15_TRAIN_FORMAL")
    lines = open(led.wal_path).read().splitlines(True)
    open(led.wal_path, "w").write("".join(lines[:-1]))
    with led:
        with pytest.raises(L.LedgerError):
            led.verify_projection()


def test_tampered_wal_event_is_detected(led):
    with led:
        led.append("run-a", "step_commit", payload={"global_step": 1}, gpu_seconds=1.0,
                   formal_steps=1, bucket="R15_TRAIN_FORMAL")
    raw = open(led.wal_path).read()
    open(led.wal_path, "w").write(raw.replace('"gpu_seconds": 1.0', '"gpu_seconds": 0.0'))
    with led:
        with pytest.raises(L.LedgerError):
            led.read_events()


def test_out_of_order_seq_is_detected(led):
    with led:
        led.append("run-a", "step_commit", payload={"global_step": 1}, gpu_seconds=1.0,
                   formal_steps=1, bucket="R15_TRAIN_FORMAL")
    ev = json.loads(open(led.wal_path).read().strip())
    ev["seq"] = 5
    body = {k: v for k, v in ev.items() if k != "hash"}
    ev["hash"] = L._sha256_bytes(json.dumps(body, sort_keys=True).encode())
    open(led.wal_path, "w").write(json.dumps(ev, sort_keys=True) + "\n")
    with led:
        with pytest.raises(L.LedgerError):
            led.read_events()


def test_start_guard_checks_bucket_total_and_steps(led):
    with led:
        led.append("run-a", "job_begin", payload={"global_step": 0}, bucket="R15_TRAIN_FORMAL")
        led.append("run-a", "step_commit", payload={"global_step": 1}, gpu_seconds=100.0,
                   formal_steps=10, bucket="R15_TRAIN_FORMAL")
        ok = led.assert_can_start("R15_TRAIN_FORMAL", 3600.0, requested_steps=2990, run_id="run-b")
        assert ok["left_steps"] == L.FORMAL_STEPS_CAP - 10
        assert ok["over_budget"] is False
        # the research lead lifted the ceilings: the default path only reports, never blocks
        soft = led.assert_can_start("R15_TRAIN_FORMAL", L.BUCKET_CAPS_S["R15_TRAIN_FORMAL"] + 1)
        assert soft["over_budget"] is True and soft["over_budget_reasons"]
        # strict mode keeps the mechanism honest when a caller wants a hard gate
        with pytest.raises(L.LedgerError):
            led.assert_can_start("R15_TRAIN_FORMAL", L.BUCKET_CAPS_S["R15_TRAIN_FORMAL"] + 1,
                                 strict=True)
        with pytest.raises(L.LedgerError):
            led.assert_can_start("R15_TRAIN_FORMAL", 10.0, requested_steps=3000, strict=True)
        with pytest.raises(L.LedgerError):
            led.assert_can_start("NOT_A_BUCKET", 10.0)


def test_finished_run_id_cannot_be_rebooked(led):
    with led:
        led.append("run-a", "job_begin", payload={"global_step": 0}, bucket="R15_TRAIN_FORMAL")
        led.append("run-a", "job_end", payload={"global_step": 300}, gpu_seconds=50.0,
                   formal_steps=300, bucket="R15_TRAIN_FORMAL")
        with pytest.raises(L.LedgerError):
            led.assert_can_start("R15_TRAIN_FORMAL", 10.0, run_id="run-a")


def test_formal_step_cap_is_enforced_end_to_end(led):
    with led:
        led.append("run-a", "job_begin", payload={"global_step": 0}, bucket="R15_TRAIN_FORMAL")
        for s in range(3000):
            led.append("run-a", "step_commit", payload={"global_step": s + 1},
                       gpu_seconds=0.0, formal_steps=1, bucket="R15_TRAIN_FORMAL")
        t = led.totals()
    assert t["formal_steps_used"] == 3000
    with led:
        # reporting is still accurate past the nominal cap ...
        rep = led.assert_can_start("R15_TRAIN_FORMAL", 1.0, requested_steps=1)
        assert rep["left_steps"] == 0
        # ... but only strict mode refuses
        with pytest.raises(L.LedgerError):
            led.assert_can_start("R15_TRAIN_FORMAL", 1.0, requested_steps=1, strict=True)


def test_second_writer_is_blocked(led, monkeypatch):
    monkeypatch.setattr(L, "LOCK_TIMEOUT_S", 0.3)
    outer = L.Ledger(led.dir)
    with outer:
        inner = L.Ledger(led.dir)
        with pytest.raises(L.LedgerError):
            with inner:
                pass


def test_second_heavy_gpu_job_is_refused(led):
    """A duplicated launch must be refused outright, not discovered via CUDA OOM."""
    first = L.GpuJobLock(led.dir)
    first.__enter__()
    try:
        second = L.GpuJobLock(led.dir)
        with pytest.raises(L.LedgerError):
            second.__enter__()
        assert "pid=" in first.holder()
    finally:
        first.__exit__(None, None, None)
    third = L.GpuJobLock(led.dir)
    third.__enter__()
    third.__exit__(None, None, None)


def test_abort_is_booked_into_the_wal(led):
    with led:
        led.append("run-a", "job_begin", payload={"global_step": 0}, bucket="R15_TRAIN_FORMAL")
        led.append("run-a", "abort", bucket="R15_TRAIN_FORMAL",
                   payload={"reason": "OutOfMemoryError"}, gpu_seconds=55.0, formal_steps=0)
        t = led.totals()
    assert t["gpu_seconds_used"] == pytest.approx(55.0)
    assert t["formal_steps_used"] == 0
    kinds = [e["kind"] for e in led.read_events()]
    assert kinds == ["job_begin", "abort"]


def test_projection_is_atomic_and_hash_linked(led):
    with led:
        led.append("run-a", "step_commit", payload={"global_step": 1}, gpu_seconds=1.5,
                   formal_steps=1, bucket="R15_TRAIN_FORMAL")
        proj = json.load(open(led.proj_path))
        events = led.read_events()
    assert proj["wal_last_seq"] == events[-1]["seq"]
    assert proj["wal_last_hash"] == events[-1]["hash"]
    assert len(proj["wal_sha256"]) == 64
    assert not [f for f in os.listdir(led.dir) if f.startswith(".tmp-")]
