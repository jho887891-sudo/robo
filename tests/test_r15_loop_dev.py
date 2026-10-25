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
                              docker_env=[], dataset="RoboDojo", task="push_T", env_cfg="arx_x5",
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


def test_dev_budget_counts_observed_not_requested(tmp_path):
    ledger = tmp_path / "l.jsonl"
    # a crashed launch observed nothing -> must not consume budget
    L.append_ledger(str(ledger), {"task": "push_T", "stage": 1, "episodes_requested": 1,
                                  "episodes_observed": 0, "verdict": "NO_RESULT"})
    assert L.episodes_used(L.load_ledger(str(ledger)), "push_T") == 0
    # a real run consumes exactly what the official result contained
    L.append_ledger(str(ledger), {"task": "push_T", "stage": 1, "episodes_requested": 1,
                                  "episodes_observed": 1, "verdict": "OFFICIAL_FAILURE"})
    assert L.episodes_used(L.load_ledger(str(ledger)), "push_T") == 1


def test_docker_env_is_passed_through():
    args = type("A", (), dict(root="/R", root_inner="/workspace/RoboDojo", docker="rd_setup",
                              docker_env=["SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0",
                                          "CUROBO_KERNEL_BACKEND=cuda_core"],
                              dataset="RoboDojo", task="push_T", env_cfg="arx_x5",
                              policy_name="OpenVLA_R42", host="127.0.0.1", port=19000, episodes=1,
                              seed=0, env_gpu="0", ckpt_label="step1000",
                              action_type="joint"))()
    cmd = L.build_cmd(args)
    assert cmd[:3] == ["docker", "exec", "-e"]
    assert "SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0" in cmd
    assert cmd[-4] == "rd_setup" and cmd[-3] == "bash" and cmd[-2] == "-lc"


def test_dev_budget_is_cumulative_and_refused_past_the_cap(tmp_path):
    ledger = tmp_path / "l.jsonl"
    L.append_ledger(str(ledger), {"task": "push_T", "stage": 3, "episodes_requested": 20,
                                  "episodes_observed": 20, "verdict": "OFFICIAL_SUCCESS"})
    recs = L.load_ledger(str(ledger))
    assert L.episodes_used(recs, "push_T") == 20  # observed, not requested
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



def test_dry_run_accepts_shared_run_id_for_server_trace_and_client_ledger(tmp_path):
    out = tmp_path / "o.json"
    run_id = "r15loop-push_T-s1-20261025T120000Z-port19000"
    r = subprocess.run([
        sys.executable, os.path.join(REPO, "tools", "r15_loop_dev.py"),
        "--task", "push_T", "--port", "19000", "--episodes", "1",
        "--out", str(out), "--ledger", str(tmp_path / "ledger.jsonl"),
        "--dry-run", "--docker", "rd_setup", "--run-id", run_id,
        "--docker-env", "ROBODOJO_RUN_ID=stale-run-id",
    ], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads(r.stdout)
    assert rec["robodojo_run_id"] == run_id
    assert "ROBODOJO_RUN_ID=" + run_id in rec["cmd"]
    assert "ROBODOJO_RUN_ID=stale-run-id" not in rec["cmd"]
    assert rec["cmd"].count("ROBODOJO_RUN_ID=") == 1
    assert rec["dry_run"] is True
    assert not (tmp_path / "ledger.jsonl").exists()


def test_explicit_run_id_rejects_shell_unsafe_characters(tmp_path):
    out = tmp_path / "o.json"
    r = subprocess.run([
        sys.executable, os.path.join(REPO, "tools", "r15_loop_dev.py"),
        "--task", "push_T", "--port", "19000", "--episodes", "1",
        "--out", str(out), "--ledger", str(tmp_path / "ledger.jsonl"),
        "--dry-run", "--run-id", "bad run;touch /tmp/pwned",
    ], capture_output=True, text=True)
    assert r.returncode != 0
    assert "run-id" in (r.stdout + r.stderr).lower()
    assert not (tmp_path / "ledger.jsonl").exists()



def test_pushT_launcher_persists_unique_trace_and_never_truncates_port_only_logs():
    launcher = os.path.join(REPO, "scripts", "r15_loop_pushT.sh")
    with open(launcher, encoding="utf-8") as fh:
        source = fh.read()

    # Seconds alone collide when two independent launchers start together on the same port/stage.
    assert 'RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)-pid' + chr(36) + '{BASHPID}"' in source
    assert 'SRVLOG="logs/r15_policyserver_${PORT}_${RUN_STAMP}.log"' in source
    assert 'AUDIT_LOG="$PWD/runs/r15_loop_dev/runtime_audit_push_T_stage${STAGE}_${RUN_STAMP}.jsonl"' in source
    assert 'AUDIT_REPORT="$PWD/runs/r15_loop_dev/runtime_audit_report_push_T_stage${STAGE}_${RUN_STAMP}.json"' in source
    assert 'R42_AUDIT_LOG_PATH="$AUDIT_LOG" R42_AUDIT_RUN_ID="$RUN_ID"' in source
    assert '--run-id "$RUN_ID"' in source
    assert '--docker-env "ROBODOJO_RUN_ID=$RUN_ID"' not in source
    assert 'for path in "$SRVLOG" "$AUDIT_LOG" "$AUDIT_REPORT" "$OUT"; do' in source
    assert 'CLIENT_RC=$?' in source
    assert 'trap - EXIT INT TERM' in source
    assert '"$PY" tools/r42_runtime_audit_report.py --input "$AUDIT_LOG" --out "$AUDIT_REPORT"' in source
    assert 'AUDIT_RC=$?' in source
    assert 'if [ "$CLIENT_RC" -ne 0 ]; then' in source
    assert 'elif [ "$AUDIT_RC" -ne 0 ]; then' in source
    assert 'SRVLOG=logs/r15_policyserver_${PORT}.log' not in source
