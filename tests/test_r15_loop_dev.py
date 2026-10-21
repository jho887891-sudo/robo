#!/usr/bin/env python3
"""Regression for tools/r15_loop_dev.py: the closed-loop driver must read the OFFICIAL verdict only,
enforce the cumulative DEV budget, and never invent a success."""
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import r15_loop_dev as L  # noqa: E402


def test_command_uses_the_official_client_and_records_joint_actions(tmp_path):
    args = type("A", (), dict(root="/R", root_inner="/R", docker="", dataset="RoboDojo",
                              task="push_T", env_cfg="arx_x5",
                              policy_name="OpenVLA_R42", host="127.0.0.1", port=18765, episodes=1,
                              seed=0, env_gpu="0", ckpt_label="step1000",
                              action_type="joint"))()
    cmd = L.build_cmd(args)
    assert cmd[:3] == ["bash", "/R/scripts/robodojo.sh", "client"]
    assert "--eval-num" in cmd and cmd[cmd.index("--eval-num") + 1] == "1"
    assert cmd[cmd.index("--action-type") + 1] == "joint"
    assert cmd[cmd.index("--policy-port") + 1] == "18765"


def test_docker_runner_wraps_the_same_official_command():
    args = type("A", (), dict(root="/R", root_inner="/workspace/RoboDojo", docker="rd_setup",
                              dataset="RoboDojo", task="push_T", env_cfg="arx_x5",
                              policy_name="OpenVLA_R42", host="127.0.0.1", port=19000, episodes=1,
                              seed=0, env_gpu="0", ckpt_label="step1000",
                              action_type="joint"))()
    cmd = L.build_cmd(args)
    assert cmd[:5] == ["docker", "exec", "rd_setup", "bash", "-lc"]
    script = cmd[5]
    assert script.startswith("cd /workspace/RoboDojo && ")
    assert "robodojo.sh client" in script and "--eval-num 1" in script
    # without --docker the same command runs on the host, from the host path
    args.docker = ""
    assert L.build_cmd(args)[:3] == ["bash", "/R/scripts/robodojo.sh", "client"]   # root, not root_inner


def test_result_parsing_reads_the_official_success_flags(tmp_path):
    p = tmp_path / "_result.json"
    p.write_text(json.dumps({"details": {"0": {"success": True, "score": 1.0},
                                         "1": {"success": False, "score": 0.0}}}))
    entries = L.parse_result(str(p))
    assert len(entries) == 2
    assert sum(1 for e in entries if e["success"]) == 1


def test_dev_budget_is_cumulative_and_refused_past_the_cap(tmp_path):
    ledger = tmp_path / "l.jsonl"
    L.append_ledger(str(ledger), {"task": "push_T", "stage": 3, "episodes_requested": 20})
    recs = L.load_ledger(str(ledger))
    assert L.episodes_used(recs, "push_T") == 20
    assert L.episodes_used(recs, "stack_bowls") == 0
    # a 21st episode for that task must be refused, another task is unaffected
    out = tmp_path / "o.json"
    r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "r15_loop_dev.py"),
                        "--task", "push_T", "--port", "1", "--episodes", "1",
                        "--stage", "1", "--out", str(out), "--ledger", str(ledger)],
                       capture_output=True, text=True)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "REFUSED" in r.stdout


def test_dry_run_never_launches_the_simulator(tmp_path):
    out = tmp_path / "o.json"
    r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "r15_loop_dev.py"),
                        "--task", "push_T", "--port", "18765", "--episodes", "1",
                        "--out", str(out), "--ledger", str(tmp_path / "l.jsonl"), "--dry-run"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads(r.stdout)
    assert rec["dry_run"] is True
    assert "robodojo.sh client" in rec["cmd"]
    assert not (tmp_path / "l.jsonl").exists()      # nothing booked for a dry run
