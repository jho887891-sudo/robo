#!/usr/bin/env python3
"""Regression for recovering an official verdict from a completed run.

The first bounded official DEV episode finished (600 control steps, three 601-frame videos, and an
official _result.json with success_rate 0.0) and the driver recorded NO_RESULT, because it looked for
_result.json on the HOST side of the mount while the container had written it into its own eval_result.
These tests pin the three pieces that fix that: the in-container discovery command, the object parser,
and the cross-check against the client's own stdout tally.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import r15_loop_dev as LD  # noqa: E402


OFFICIAL = {"success_rate": 0.0, "eval_time": 1, "score": 0.0,
            "details": {"0": {"layout_id": 0, "success": False, "score": 0.0}}}


def test_official_result_object_is_parsed_into_episodes():
    entries = LD.parse_result_obj(OFFICIAL)
    assert len(entries) == 1
    assert entries[0]["success"] is False


def test_success_is_read_when_it_actually_succeeds():
    d = {"success_rate": 0.5, "details": {"0": {"layout_id": 0, "success": True},
                                          "1": {"layout_id": 1, "success": False}}}
    entries = LD.parse_result_obj(d)
    assert [e["success"] for e in entries] == [True, False]
    k = sum(1 for e in entries if e["success"])
    assert k == 1


def test_tally_line_is_parsed_for_cross_checking():
    line = ("[INFO] ...\nSuccess nums: 0, Fail nums: 1, Unstable nums: 0\nNo more seeds to run")
    assert LD.parse_official_tally(line) == {"success": 0, "fail": 1, "unstable": 0}
    assert LD.parse_official_tally("nothing here") is None


def test_container_discovery_follows_the_eval_result_symlink():
    """eval_result is a SYMLINK inside the container; find must be given -L or it descends into nothing."""
    import inspect
    src = inspect.getsource(LD.newest_result_in_container)
    assert "find -L" in src, "the container discovery must follow symlinks"
    assert "xargs -r ls -t" in src, "newest-first selection"


def test_host_discovery_still_works(tmp_path):
    root = tmp_path / "RoboDojo"
    p = root / "eval_result" / "RoboDojo" / "push_T" / "x" / "_result.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(OFFICIAL))
    got = LD.newest_result(str(root), "push_T", "OpenVLA_R42")
    assert got is not None and os.path.isfile(got)
    assert LD.parse_result(got)[0]["success"] is False


def test_fresh_result_only_belongs_to_own_run(tmp_path):
    """A same-task/seed historical _result.json MUST NOT certify a new rollout."""
    import os
    p = tmp_path / "_result.json"
    p.write_text(json.dumps(OFFICIAL))
    os.utime(p, (100.0, 100.0))
    ok, mt = LD.official_result_belongs_to_run(
        str(p), started_at=200.0, ended_at=250.0, clock_slack_s=1.0)
    assert not ok and mt == pytest.approx(100.0)

    os.utime(p, (235.0, 235.0))
    ok, mt = LD.official_result_belongs_to_run(
        str(p), started_at=200.0, ended_at=250.0, clock_slack_s=1.0)
    assert ok and mt == pytest.approx(235.0)

    os.utime(p, (270.0, 270.0))
    ok, _ = LD.official_result_belongs_to_run(
        str(p), started_at=200.0, ended_at=250.0, clock_slack_s=1.0)
    assert not ok


def test_container_official_mtime_obtained_from_actual_container(monkeypatch):
    """A Docker result symlink may point to container /tmp, not the host tree."""
    from types import SimpleNamespace
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="207", stderr="")

    monkeypatch.setattr(LD.subprocess, "run", fake_run)
    ok, mt = LD.official_result_belongs_to_run(
        "/workspace/RoboDojo/eval_result/_result.json",
        started_at=200.0, ended_at=210.0, container="rd_setup")
    assert ok and mt == 207
    assert calls[0] == [
        "docker", "exec", "rd_setup", "stat", "-c", "%Y",
        "/workspace/RoboDojo/eval_result/_result.json"]


def test_failed_container_stat_refuses_unproven_result(monkeypatch):
    from types import SimpleNamespace

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="missing")

    monkeypatch.setattr(LD.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="cannot stat"):
        LD.official_result_belongs_to_run(
            "/tmp/missing/_result.json", started_at=10.0, ended_at=11.0,
            container="rd_setup")
