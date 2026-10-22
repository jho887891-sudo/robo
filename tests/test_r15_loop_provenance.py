#!/usr/bin/env python3
"""An official result may only certify the launch that WROTE it (audit PR#29).

'newest _result.json for this task' can hand an earlier seeded rollout to a later launch that shares task,
seed and checkpoint, and a --reparse-only recovery must never be presented as a fresh run.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import r15_loop_dev as LD  # noqa: E402


def test_fresh_result_inside_the_window_is_accepted(tmp_path):
    p = tmp_path / "_result.json"
    p.write_text(json.dumps({"success_rate": 0.0, "details": {}}))
    mtime = os.path.getmtime(p)
    ok, got = LD.official_result_belongs_to_run(str(p), mtime - 60, mtime + 60)
    assert ok is True and got == pytest.approx(mtime, abs=1.0)


def test_stale_result_is_refused(tmp_path):
    p = tmp_path / "_result.json"
    p.write_text("{}")
    mtime = os.path.getmtime(p)
    ok, got = LD.official_result_belongs_to_run(str(p), mtime + 3600, mtime + 7200)
    assert ok is False and got == pytest.approx(mtime, abs=1.0)


def test_clock_slack_covers_filesystem_timestamp_rounding(tmp_path):
    p = tmp_path / "_result.json"
    p.write_text("{}")
    mtime = os.path.getmtime(p)
    ok, _ = LD.official_result_belongs_to_run(str(p), mtime + 1.5, mtime + 100, clock_slack_s=2.0)
    assert ok is True


def test_container_stat_is_used_and_its_failure_propagates(monkeypatch):
    calls = {}

    class P:
        returncode = 0
        stdout = "1700000000\n"
        stderr = ""

    def fake_run(cmd, **kw):
        calls["cmd"] = cmd
        return P()
    monkeypatch.setattr(LD.subprocess, "run", fake_run)
    assert LD.official_result_mtime("/workspace/RoboDojo/eval_result/x/_result.json",
                                    container="rd_setup") == 1700000000.0
    assert calls["cmd"][:3] == ["docker", "exec", "rd_setup"], "must stat inside the container"

    class Bad(P):
        returncode = 1
        stderr = "no such file"
    monkeypatch.setattr(LD.subprocess, "run", lambda cmd, **kw: Bad())
    with pytest.raises(RuntimeError):
        LD.official_result_mtime("/nope", container="rd_setup")


def test_reparse_only_is_never_a_verified_rollout(tmp_path, monkeypatch):
    """The reparse branch must label its record PROVENANCE_PENDING."""
    src = open(os.path.join(REPO, "tools", "r15_loop_dev.py")).read()
    seg = src[src.find("if a.reparse_only:"):src.find("t0 = time.time()")]
    assert "PROVENANCE_PENDING" in seg, "reparse_only must not claim a fresh official verdict"
    assert "reparse_only" in seg
