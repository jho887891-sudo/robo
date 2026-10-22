#!/usr/bin/env python3
"""Regression for tools/r15_gate_check.py: the gate is a program verdict, and it must fail closed.

The three cases below are the only ones that matter for honesty:
  * an archive that was not produced on the pre-registered frame list is refused outright;
  * a predictor that IS constant velocity gets FAIL (the gate cannot be passed by a baseline);
  * an oracle that reproduces the labels gets PASS (the gate is not impossible).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "tools", "r15_gate_check.py")
HDF5_DIR = os.environ.get("PUSHT_HDF5",
                          "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
SPLIT = os.path.join(REPO, "data", "split80.json")
HAVE_DATA = os.path.isdir(HDF5_DIR) and os.path.isfile(SPLIT)
sys.path.insert(0, os.path.join(REPO, "diagnostics"))
for _s in ("training", "code/xpolicylab/policy/OpenVLA_Pure"):
    sys.path.insert(0, os.path.join(REPO, _s))


def run_gate(tmp_path, pred, gate="G5_1000"):
    out = tmp_path / "gate.json"
    r = subprocess.run([sys.executable, TOOL, "--gate", gate, "--pred", str(pred),
                        "--out", str(out)],
                       capture_output=True, text=True, cwd=REPO)
    rep = json.loads(out.read_text()) if out.exists() else {}
    return r.returncode, rep, r


@pytest.mark.skipif(not HAVE_DATA, reason="Push-T HDF5 not available")
def test_refuses_archive_from_a_different_frame_list(tmp_path):
    np.savez_compressed(str(tmp_path / "pred.npz"),
                        action=np.zeros((32, 14), np.float32),
                        frames_sha256=np.array("0" * 64))
    rc, rep, r = run_gate(tmp_path, tmp_path / "pred.npz")
    assert rc == 4, r.stdout + r.stderr
    assert rep["verdict"] == "CANNOT_EVALUATE"
    assert rep["error"]


@pytest.mark.skipif(not HAVE_DATA, reason="Push-T HDF5 not available")
def test_constant_velocity_predictions_cannot_pass(tmp_path):
    import r15_dev_eval as E
    ev = E.run(None, 2, 0, str(tmp_path / "eval"), 256, True)
    smp = np.load(str(tmp_path / "eval" / "dev_samples.npz"))
    cv = (smp["state"] + smp["velocity"]).astype(np.float32)
    np.savez_compressed(str(tmp_path / "pred.npz"), action=cv,
                        frames_sha256=np.array(ev["dev_frames_sha256"]))
    rc, rep, r = run_gate(tmp_path, tmp_path / "pred.npz")
    assert rc == 3, r.stdout + r.stderr
    assert rep["verdict"] == "FAIL"
    by_id = {c["id"]: c for c in rep["checks"]}
    assert by_id["cohort.frozen_frames"]["passed"] is True     # the archive IS the right frames
    assert by_id["G5.macro_relative_5pct"]["passed"] is False  # but it is just the baseline
    assert by_id["G5.pass_rule"]["passed"] is False


@pytest.mark.skipif(not HAVE_DATA, reason="Push-T HDF5 not available")
def test_oracle_predictions_pass(tmp_path):
    import r15_dev_eval as E
    ev = E.run(None, 2, 0, str(tmp_path / "eval2"), 256, True)
    smp = np.load(str(tmp_path / "eval2" / "dev_samples.npz"))
    np.savez_compressed(str(tmp_path / "pred.npz"), action=smp["action"],
                        frames_sha256=np.array(ev["dev_frames_sha256"]))
    rc, rep, r = run_gate(tmp_path, tmp_path / "pred.npz")
    assert rc == 0, r.stdout + r.stderr
    assert rep["verdict"] == "PASS"
    # oracle: exact on RAW, epsilon on SAFE (a few expert labels touch the joint box edge)
    assert rep["vla_metrics"]["raw_physical_l1_macro"] == pytest.approx(0.0, abs=1e-9)
    assert rep["vla_metrics"]["physical_l1_macro"] <= 1e-5


@pytest.mark.skipif(not HAVE_DATA, reason="Push-T HDF5 not available")
def test_every_report_carries_the_metric_contract(tmp_path):
    rc, rep, _r = run_gate(tmp_path, tmp_path / "does_not_exist.npz")
    assert rc == 4
    assert rep["verdict"] == "CANNOT_EVALUATE"
