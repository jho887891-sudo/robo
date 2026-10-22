#!/usr/bin/env python3
"""R15 collection-gate tests -- pure CPU, fake environment double, no isaacsim.

The gate is exercised on a run directory produced by the real driver + the real HDF5
writer, then every tamper class the mission cares about is applied to that run:

  * a clean run passes (exit 0) and the report is machine-readable;
  * editing the official success flag in the ledger is caught (the flag is re-read from
    three places: ledger, HDF5 meta_json, HDF5 audit group);
  * swapping camera pixels is caught even though the writer's content_sha256 ignores them;
  * deleting provenance, truncating a file, editing the state array, adding an unledgered
    episode, or exceeding a mission cap all fail with exit 3.
"""
import importlib.util
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "training" / "r15_collect_episode.py"
TOOL_PATH = ROOT / "tools" / "r15_collect_check.py"

h5py = pytest.importorskip("h5py")

BASE_DIM = 14
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
JOINT_GROUPS = (("left_arm", 6), ("left_ee", 1), ("right_arm", 6), ("right_ee", 1))
INSTRUCTION = "Push the T-shaped block to align it precisely with the gray T-shaped pad."
PROV = {
    "block_pose": {"source": "layout_manager.get_instance_pose(label='t')",
                   "file": "RoboDojo/env/scene_manager/layout_manager.py", "line": 421},
    "target_pose": {"source": "layout_manager.get_instance_pose(label='target_t')",
                    "file": "RoboDojo/env/scene_manager/layout_manager.py", "line": 421},
    "ee_pose": {"source": "obs['state']['left_ee_pose'] (robot_manager.get_real_endpose)",
                "file": "RoboDojo/env/observation_manager/obs_manager.py", "line": 164},
}


def _load(path, name):
    assert path.is_file(), "missing module %s" % path
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def D():
    return _load(DRIVER_PATH, "r15_collect_episode_for_gate_tests")


@pytest.fixture(scope="module")
def C():
    return _load(TOOL_PATH, "r15_collect_check_under_test")


# --------------------------------------------------------------------------- doubles

def jpeg(w=96, h=96, filler=1024, tag=0):
    sof = (b"\xff\xc0" + struct.pack(">H", 17) + bytes([8]) + struct.pack(">HH", h, w)
           + bytes([3]) + bytes([1, 0x22, 0]) + bytes([2, 0x11, 1]) + bytes([3, 0x11, 1]))
    sos = b"\xff\xda" + struct.pack(">H", 12) + bytes([3, 1, 0, 2, 0x11, 3, 0x11, 0, 0x3f, 0])
    payload = bytes([(tag + i) % 251 for i in range(filler)])
    return b"\xff\xd8" + sof + sos + payload + b"\xff\xd9"


class FakeEnv(object):
    def __init__(self, steps_to_end=110, official_success=True):
        self.steps_to_end = int(steps_to_end)
        self.official_success = bool(official_success)
        self.reset(seed=[0])

    def reset(self, seed=None):
        self.seed = [int(s) for s in seed]
        self.count = 0
        self.end_flag = [False]
        self.success = [True]
        self.offset = float(self.seed[0] % 7) * 0.005
        self.sim = np.full(BASE_DIM, 0.05 + self.offset)
        return None

    def get_obs(self):
        state = {}
        col = 0
        for group, dim in JOINT_GROUPS:
            state[group + "_joint_state"] = self.sim[col:col + dim].copy()
            col += dim
        vision = {cam: {"color": jpeg(tag=7 * i + self.count + 13)}
                  for i, cam in enumerate(CAMERAS)}
        return {"state": state, "vision": vision, "instruction": INSTRUCTION}

    def take_action(self, action):
        self.count += 1
        self.sim = np.concatenate([np.asarray(action[g + "_joint_state"], dtype=np.float64)
                                   for g, _ in JOINT_GROUPS])
        self.is_episode_end()

    def is_episode_end(self):
        if self.end_flag[0]:
            return True
        if self.count >= self.steps_to_end:
            self.end_flag[0] = True
            self.success[0] = self.official_success
            return True
        return False


class FakeExpert(object):
    def __init__(self):
        self.t = 0

    def act(self, obs, privileged):
        self.t += 1
        return 0.01 + np.linspace(0.0, 0.5, BASE_DIM) * (0.001 * self.t)

    def success(self):
        return True

    def info(self):
        return {"name": "fake"}


def privileged_fn(env, obs):
    return {"block_pose": np.array([0.10, 0.10, 0.00, 1.0, 0.0, 0.0, 0.0]),
            "target_pose": np.array([0.20, 0.10, 0.00, 1.0, 0.0, 0.0, 0.0]),
            "ee_pose": np.array([0.05, 0.10, 0.02, 1.0, 0.0, 0.0, 0.0])}


def make_cfg(D, run_id="r15gate"):
    return D.CollectConfig(
        run_id=run_id, task="push_T", instruction=INSTRUCTION, step_lim=600,
        writer_task="Push the T-shaped block", expert_name="fake_expert",
        expert_version_sha="a" * 40, expert_status="IMPLEMENTED", provenance=PROV,
        privileged_keys=("block_pose", "target_pose", "ee_pose"),
        predicate_ref="RoboDojo/task/RoboDojo/tasks/push_T.py:19-29")


def build_run(D, tmp_path, verdicts=(True, False), steps=110):
    run_dir = str(tmp_path / "run")
    cfg = make_cfg(D)
    records = []
    for i, verdict in enumerate(verdicts):
        record = D.collect(cfg.task, FakeEnv(steps_to_end=steps,
                                             official_success=verdict),
                           FakeExpert(), 100 + i, run_dir, cfg=cfg,
                           privileged_fn=privileged_fn)
        assert record["status"] == "accepted", record
        records.append(record)
    return run_dir, records


def read_ledger(run_dir):
    path = os.path.join(run_dir, "ledger.jsonl")
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def write_ledger(run_dir, records):
    path = os.path.join(run_dir, "ledger.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for rec in records:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")


def codes_of(report):
    return sorted({failure["code"] for failure in report["failures"]})


# --------------------------------------------------------------------------- clean run

def test_clean_run_passes(C, D, tmp_path):
    run_dir, records = build_run(D, tmp_path)
    report = C.run_check(run_dir)
    assert report["verdict"] == C.PASS, report["failures"]
    assert report["exit_code"] == 0
    assert report["failures"] == []
    assert len(report["episodes"]) == 2
    assert report["counts"]["total"]["accepted"] == 2
    assert report["counts"]["total"]["successes"] == 1
    for episode in report["episodes"]:
        assert episode["codes"] == []
        assert episode["frames"] == 110
        assert episode["provenance_keys"] == ["block_pose", "ee_pose", "target_pose"]
    # the official verdict is present, and the failures were recorded as failures
    flags = sorted(e["official_success"] for e in report["episodes"])
    assert flags == [False, True]
    assert report["registry"]["task"] == "push_T"
    assert report["registry"]["status"] == "IMPLEMENTED"


def test_gate_exit_codes_and_report_file(tmp_path):
    run_dir = os.path.join(str(tmp_path), "run")
    D = _load(DRIVER_PATH, "r15_collect_episode_for_subprocess")
    build_run(D, tmp_path)
    pass_run = subprocess.run([sys.executable, str(TOOL_PATH), run_dir],
                              capture_output=True, text=True)
    assert pass_run.returncode == 0, pass_run.stdout + pass_run.stderr
    assert "PASS" in pass_run.stdout
    report_path = os.path.join(run_dir, "check_report.json")
    assert os.path.isfile(report_path)
    payload = json.load(open(report_path, encoding="utf-8"))
    assert payload["verdict"] == "PASS" and payload["exit_code"] == 0

    records = read_ledger(run_dir)
    records[0]["official_success"] = not records[0]["official_success"]
    write_ledger(run_dir, records)
    fail_run = subprocess.run([sys.executable, str(TOOL_PATH), run_dir],
                              capture_output=True, text=True)
    assert fail_run.returncode == 3, fail_run.stdout + fail_run.stderr
    assert "FAIL" in fail_run.stdout
    payload = json.load(open(report_path, encoding="utf-8"))
    assert payload["verdict"] == "FAIL"
    assert payload["failures"]


# --------------------------------------------------------------------------- tampering

def test_edited_official_flag_is_caught(C, D, tmp_path):
    run_dir, _ = build_run(D, tmp_path)
    records = read_ledger(run_dir)
    target = [r for r in records if r["official_success"] is False][0]
    target["official_success"] = True          # forge a success
    write_ledger(run_dir, records)
    report = C.run_check(run_dir)
    assert report["verdict"] == C.FAIL
    assert "verdict_mismatch" in codes_of(report)


def test_forged_success_termination_is_caught(C, D, tmp_path):
    run_dir, _ = build_run(D, tmp_path, verdicts=(False,))
    records = read_ledger(run_dir)
    records[0]["official_success"] = True
    records[0]["termination"] = "official_success"
    write_ledger(run_dir, records)
    report = C.run_check(run_dir)
    assert "verdict_mismatch" in codes_of(report)


def test_camera_pixels_are_covered_by_the_gate(C, D, tmp_path):
    run_dir, records = build_run(D, tmp_path, verdicts=(True,))
    path = os.path.join(run_dir, records[0]["episode_path"])
    with h5py.File(path, "r+") as handle:
        ds = handle["vision/cam_head/colors"]
        ds[0] = np.frombuffer(jpeg(tag=999), dtype=np.uint8)
        handle.flush()
    report = C.run_check(run_dir)
    assert report["verdict"] == C.FAIL
    codes = codes_of(report)
    assert "camera_sha256_mismatch" in codes
    assert "file_sha256_mismatch" in codes


def test_missing_provenance_is_caught(C, D, tmp_path):
    run_dir, records = build_run(D, tmp_path, verdicts=(True,))
    path = os.path.join(run_dir, records[0]["episode_path"])
    with h5py.File(path, "r+") as handle:
        del handle["prov/block_pose"]
        handle.flush()
    report = C.run_check(run_dir)
    assert report["verdict"] == C.FAIL
    assert "prov_missing" in codes_of(report)


def test_provenance_without_a_source_is_caught(C, D, tmp_path):
    run_dir, records = build_run(D, tmp_path, verdicts=(True,))
    path = os.path.join(run_dir, records[0]["episode_path"])
    with h5py.File(path, "r+") as handle:
        handle["prov/target_pose"].attrs["source"] = ""
        handle.flush()
    report = C.run_check(run_dir)
    assert "prov_incomplete" in codes_of(report) or "prov_source_unnamed" in codes_of(report)


def test_truncated_episode_is_caught(C, D, tmp_path):
    run_dir, records = build_run(D, tmp_path, verdicts=(True,))
    path = os.path.join(run_dir, records[0]["episode_path"])
    with open(path, "r+b") as handle:
        handle.truncate(4096)
    report = C.run_check(run_dir)
    codes = codes_of(report)
    assert "size_mismatch" in codes
    assert "readback_failed" in codes


def test_edited_state_array_is_caught(C, D, tmp_path):
    run_dir, records = build_run(D, tmp_path, verdicts=(True,))
    path = os.path.join(run_dir, records[0]["episode_path"])
    with h5py.File(path, "r+") as handle:
        ds = handle["state/left_arm_joint_states"]
        # row 1 is constrained by A[0] == S[1], so this edit must break the alignment too
        ds[1, 0] = ds[1, 0] + 0.25
        handle.flush()
    report = C.run_check(run_dir)
    codes = codes_of(report)
    assert "content_sha256_mismatch" in codes
    assert "alignment_failed" in codes


def test_unledgered_episode_file_is_caught(C, D, tmp_path):
    run_dir, records = build_run(D, tmp_path, verdicts=(True,))
    src = os.path.join(run_dir, records[0]["episode_path"])
    shutil.copyfile(src, os.path.join(run_dir, "episodes", "r15gate_ep0000999.hdf5"))
    report = C.run_check(run_dir)
    assert report["verdict"] == C.FAIL
    assert "unledgered_episode" in codes_of(report)


def test_tmp_residue_is_caught(C, D, tmp_path):
    run_dir, _ = build_run(D, tmp_path, verdicts=(True,))
    open(os.path.join(run_dir, "episodes", "leftover.hdf5.tmp"), "wb").close()
    report = C.run_check(run_dir)
    assert "tmp_residue" in codes_of(report)


def test_edited_instruction_is_caught(C, D, tmp_path):
    run_dir, _ = build_run(D, tmp_path, verdicts=(True,))
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    manifest["instruction"] = "Move the red cube to the tray."
    json.dump(manifest, open(manifest_path, "w", encoding="utf-8"), indent=2)
    report = C.run_check(run_dir)
    codes = codes_of(report)
    assert "instruction_mismatch" in codes
    assert "instruction_not_official" in codes


def test_expert_status_must_be_implemented(C, D, tmp_path):
    run_dir, _ = build_run(D, tmp_path, verdicts=(True,))
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    manifest["expert"]["status"] = "BLOCKED_DATA"
    json.dump(manifest, open(manifest_path, "w", encoding="utf-8"), indent=2)
    report = C.run_check(run_dir)
    assert "expert_status_invalid" in codes_of(report)


# --------------------------------------------------------------------------- caps

def test_cap_accepted_per_task_is_caught(C, D, tmp_path):
    run_dir, records = build_run(D, tmp_path, verdicts=(True, False))
    ledger = read_ledger(run_dir)
    template = ledger[0]
    for i in range(99):
        clone = dict(template)
        clone["attempt_id"] = "dup_%03d" % i
        ledger.append(clone)
    assert len([r for r in ledger if r["status"] == "accepted"]) == 101
    write_ledger(run_dir, ledger)
    report = C.run_check(run_dir)
    assert report["verdict"] == C.FAIL
    assert "cap_accepted_per_task" in codes_of(report)
    # and the duplicated-but-consistent episodes are not the reason for the failure
    assert set(codes_of(report)) == {"cap_accepted_per_task"}


def test_cap_attempts_total_is_caught(C, D, tmp_path):
    run_dir, _ = build_run(D, tmp_path, verdicts=(True,))
    ledger = read_ledger(run_dir)
    for i in range(3000):
        ledger.append({"schema": ledger[0]["schema"], "attempt_id": "cap_%04d" % i,
                       "task": "other_task", "status": "refused",
                       "code": "expert_no_action"})
    write_ledger(run_dir, ledger)
    report = C.run_check(run_dir)
    assert "cap_attempts_total" in codes_of(report)
    assert "cap_attempts_per_task" in codes_of(report)


def test_no_accepted_episodes_fails_the_gate(C, D, tmp_path):
    run_dir = str(tmp_path / "run")
    cfg = make_cfg(D)
    record = D.collect(cfg.task, FakeEnv(), FakeExpert(), 5, run_dir, cfg=cfg,
                       privileged_fn=privileged_fn, max_steps=3)
    assert record["status"] == "refused"
    report = C.run_check(run_dir)
    assert report["verdict"] == C.FAIL
    assert "no_accepted_episodes" in codes_of(report)
    # the refusal itself must carry a code, and must not look like an episode
    assert report["counts"]["total"]["refused"] == 1


def test_refusal_without_a_code_is_caught(C, D, tmp_path):
    run_dir, records = build_run(D, tmp_path, verdicts=(True,))
    ledger = read_ledger(run_dir)
    ledger.append({"schema": ledger[0]["schema"], "attempt_id": "bad_refusal",
                   "task": "push_T", "status": "refused", "code": None})
    write_ledger(run_dir, ledger)
    report = C.run_check(run_dir)
    assert "refusal_without_code" in codes_of(report)


def test_missing_run_directory_is_an_error(tmp_path):
    run_dir = str(tmp_path / "does_not_exist")
    os.makedirs(run_dir)
    proc = subprocess.run([sys.executable, str(TOOL_PATH), run_dir],
                          capture_output=True, text=True)
    assert proc.returncode == 3
    assert "run_unreadable" in proc.stdout
