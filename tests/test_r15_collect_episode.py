#!/usr/bin/env python3
"""R15 collection driver tests -- pure CPU, fake environment double, no isaacsim.

The simulator is never started and no GPU is touched.  The env and the expert are injected
duck-typed doubles, exactly as the driver is designed to be driven, so every fail-closed
branch can be exercised for real:

  * happy path: accepted, verified on read-back, provenance recorded, official success;
  * a "successful-looking" episode whose OFFICIAL flag is False is recorded as a failure,
    and an expert that declares success is never consulted;
  * expert returns None / raises / returns a bad action -> refused, nothing written;
  * NaN observation -> refused;
  * A[t] != S[t+1] -> refused;
  * missing/undeclared provenance -> refused;
  * the official verdict being unreadable -> refused;
  * mission caps enforced from the run's own ledger (per task, per run, bytes);
  * a real round trip through training/r11_collector_writer.py with read-back, and the
    in-run dedupe index (the writer's opt-in gap) switched on by default.
"""
import importlib.util
import json
import os
import pathlib
import struct

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "training" / "r15_collect_episode.py"
REGISTRY_PATH = ROOT / "training" / "r15_expert_registry.py"
WRITER_PATH = ROOT / "training" / "r11_collector_writer.py"

h5py = pytest.importorskip("h5py")

BASE_DIM = 14
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
JOINT_GROUPS = (("left_arm", 6), ("left_ee", 1), ("right_arm", 6), ("right_ee", 1))
INSTRUCTION = "Push the T-shaped block to align it precisely with the gray T-shaped pad."
WRITER_TASK = "Push the T-shaped block"
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
    return _load(DRIVER_PATH, "r15_collect_episode_under_test")


@pytest.fixture(scope="module")
def W():
    return _load(WRITER_PATH, "r11_collector_writer_under_test")


@pytest.fixture(scope="module")
def R():
    return _load(REGISTRY_PATH, "r15_expert_registry_under_test")


# --------------------------------------------------------------------------- doubles

def jpeg(w=96, h=96, filler=1024, tag=0):
    """Structurally valid JPEG stream: SOI, SOF0 with real dimensions, SOS, data, EOI."""
    sof = (b"\xff\xc0" + struct.pack(">H", 17) + bytes([8]) + struct.pack(">HH", h, w)
           + bytes([3]) + bytes([1, 0x22, 0]) + bytes([2, 0x11, 1]) + bytes([3, 0x11, 1]))
    sos = b"\xff\xda" + struct.pack(">H", 12) + bytes([3, 1, 0, 2, 0x11, 3, 0x11, 0, 0x3f, 0])
    payload = bytes([(tag + i) % 251 for i in range(filler)])
    return b"\xff\xd8" + sof + sos + payload + b"\xff\xd9"


class FakeEnv(object):
    """Duck-typed double of RoboDojo's EvalEnv, including both official verdict branches.

    reset(seed=[s]) / get_obs() / take_action(dict) / is_episode_end() mirror
    RoboDojo/src/eval_client/eval_env.py:242, :300, :388, :882.  take_action() advances the
    sim and then runs the "official" check inside the step, exactly like
    take_action_batch() does at eval_env.py:506.
    """

    def __init__(self, steps_to_end=130, official_success=True, drift=0.0, nan_at=None,
                 check_inside_step=True, cameras=CAMERAS):
        self.steps_to_end = int(steps_to_end)
        self.official_success = bool(official_success)
        self.drift = float(drift)
        self.nan_at = nan_at
        self.check_inside_step = bool(check_inside_step)
        self.cameras = tuple(cameras)
        self.sent = []
        self.reset(seed=[0])

    # -- lifecycle
    def reset(self, seed=None):
        assert isinstance(seed, list), "the official reset() takes a list of seeds"
        self.seed = [int(s) for s in seed]
        self.count = 0
        self.end_flag = [False]
        self.success = [True]
        # the seed really changes the scene, so two seeds cannot collide by accident
        self.offset = float(self.seed[0] % 7) * 0.005
        self.sim = np.full(BASE_DIM, 0.05 + self.offset, dtype=np.float64)
        self.sent = []
        return None

    # -- observations
    def _obs(self):
        state = {}
        col = 0
        for group, dim in JOINT_GROUPS:
            state[group + "_joint_state"] = self.sim[col:col + dim].copy()
            col += dim
        state["left_ee_pose"] = np.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0])
        state["right_ee_pose"] = np.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0])
        if self.nan_at is not None and self.count >= self.nan_at:
            state["left_arm_joint_state"][1] = np.nan
        vision = {cam: {"color": jpeg(tag=7 * i + self.count + 13), "shape": (96, 96, 3)}
                  for i, cam in enumerate(self.cameras)}
        return {"state": state, "vision": vision, "action": dict(state),
                "instruction": INSTRUCTION, "data_format_version": "v1.0",
                "additional_info": {"frequency": 25}}

    def get_obs(self):
        return self._obs()

    # -- control
    def take_action(self, action):
        expected = {"left_arm_joint_state", "left_ee_joint_state",
                    "right_arm_joint_state", "right_ee_joint_state"}
        assert set(action) == expected, "unexpected action keys %r" % (sorted(action),)
        vec = np.concatenate([np.asarray(action[g + "_joint_state"], dtype=np.float64)
                              for g, _ in JOINT_GROUPS])
        assert vec.shape == (BASE_DIM,)
        self.sent.append(vec.copy())
        self.count += 1
        self.sim = vec + self.drift
        if self.check_inside_step:
            self.is_episode_end()

    # -- the official verdict
    def is_episode_end(self):
        if self.end_flag[0]:
            return True
        if self.count >= self.steps_to_end:
            self.end_flag[0] = True
            self.success[0] = self.official_success
            return True
        return False


class NoVerdictEnv(object):
    """An env that exposes no official success signal at all."""

    def reset(self, seed=None):
        self.steps = 0

    def get_obs(self):
        state = {}
        col = 0
        for group, dim in JOINT_GROUPS:
            state[group + "_joint_state"] = np.full(dim, 0.02)
            col += dim
        return {"state": state,
                "vision": {cam: {"color": jpeg(tag=i)} for i, cam in enumerate(CAMERAS)}}

    def take_action(self, action):
        self.steps += 1


class FakeExpert(object):
    """Configurable expert double.  It also *claims* success, which must be ignored."""

    def __init__(self, mode="linear"):
        self.mode = mode
        self.t = 0

    def act(self, obs, privileged):
        if self.mode == "none":
            return None
        if self.mode == "raise":
            raise RuntimeError("the planner cannot produce a target")
        if self.mode == "bad_shape":
            return np.zeros(13)
        if self.mode == "nan":
            return np.full(BASE_DIM, np.nan)
        self.t += 1
        return 0.01 + np.linspace(0.0, 0.5, BASE_DIM) * (0.001 * self.t)

    # deliberately present: the driver must never consult these
    declares_success = True

    def success(self):
        return True

    def is_success(self, *args, **kwargs):
        return True

    def info(self):
        return {"mode": self.mode, "t": self.t, "declares_success": True}


def privileged_fn(env, obs):
    return {"block_pose": np.array([0.10, 0.10, 0.00, 1.0, 0.0, 0.0, 0.0]),
            "target_pose": np.array([0.20, 0.10, 0.00, 1.0, 0.0, 0.0, 0.0]),
            "ee_pose": np.array([0.05, 0.10, 0.02, 1.0, 0.0, 0.0, 0.0])}


def make_cfg(D, **overrides):
    params = dict(run_id="r15test", task="push_T", instruction=INSTRUCTION,
                  step_lim=600, writer_task=WRITER_TASK, expert_name="fake_expert",
                  expert_version_sha="a" * 40, expert_status="IMPLEMENTED",
                  provenance=PROV, privileged_keys=("block_pose", "target_pose", "ee_pose"),
                  predicate_ref="RoboDojo/task/RoboDojo/tasks/push_T.py:19-29")
    params.update(overrides)
    return D.CollectConfig(**params)


def run_episode(D, tmp_path, env=None, expert=None, cfg=None, seed=7, **kwargs):
    cfg = cfg or make_cfg(D)
    env = env if env is not None else FakeEnv()
    expert = expert if expert is not None else FakeExpert()
    return D.collect(cfg.task, env, expert, seed, str(tmp_path / "run"), cfg=cfg,
                     privileged_fn=privileged_fn, **kwargs)


def seed_ledger(root, records, schema):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "ledger.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for rec in records:
            payload = {"schema": schema}
            payload.update(rec)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- happy path

def test_happy_path_writes_a_verified_episode_with_provenance(D, W, tmp_path):
    env = FakeEnv(steps_to_end=130)
    record = run_episode(D, tmp_path, env=env, expert=FakeExpert())

    assert record["status"] == "accepted", record
    assert record["success"] is True
    assert record["official_success"] is True
    assert record["termination"] == "official_success"
    assert record["steps"] == 130
    assert record["code"] is None
    path = str(tmp_path / "run" / record["episode_path"])
    assert os.path.isfile(path)

    # the real writer's read-back accepts the file
    report = W.verify_episode(path)
    assert report["verified"] is True
    assert report["frames"] == 130
    assert report["frequency"] == 25
    assert report["content_sha256"] == record["content_sha256"]
    assert report["state_action_sha256"] == record["state_action_sha256"]

    # the official verdict is only the official verdict
    assert "eval_env.py:882-901" in record["official_source"]["method"]
    assert "reward_manager.py:527-587" in record["official_source"]["reward"]

    # ledger is machine-readable and matches
    ledger = [json.loads(line) for line in
              open(str(tmp_path / "run" / "ledger.jsonl"), encoding="utf-8")]
    assert len(ledger) == 1
    assert ledger[0]["status"] == "accepted"
    assert ledger[0]["official_success"] is True
    assert ledger[0]["attempt_id"] == record["attempt_id"]

    with h5py.File(path, "r") as handle:
        assert int(handle["audit"].attrs["official_success"]) == 1
        assert "eval_env.py:882-901" in str(handle["audit"].attrs["official_source"])
        assert handle["prov"].attrs["declared_json"]
        for key in ("block_pose", "target_pose", "ee_pose"):
            ds = handle["prov/" + key]
            assert ds.shape[0] == 130, "prov/%s has %r steps" % (key, ds.shape)
            assert str(ds.attrs["source"]).strip()
            assert str(ds.attrs["file"]).strip()
            assert int(ds.attrs["line"]) > 0
        meta = json.loads(handle.attrs["meta_json"])
        assert meta["official"]["success"] is True
        assert meta["alignment"]["max_abs"] <= meta["alignment"]["tol"]
        assert str(handle.attrs["task"]) == WRITER_TASK
        assert handle["state/left_ee_poses"].shape if "state/left_ee_poses" in handle else True

    # A[t] == S[t+1] really holds, and the camera-inclusive digest covers the pixels
    with h5py.File(path, "r") as handle:
        streams = {cam: [bytes(handle["vision/%s/colors" % cam][t]) for t in range(130)]
                   for cam in CAMERAS}
        state = np.concatenate([handle["state/%s_joint_states" % g][()]
                                for g, _ in JOINT_GROUPS], axis=1)
        action = np.concatenate([handle["action/%s_joint_states" % g][()]
                                 for g, _ in JOINT_GROUPS], axis=1)
    recomputed = D.camera_inclusive_sha256(state, action, INSTRUCTION, streams)
    assert recomputed == record["camera_inclusive_sha256"]
    assert streams["cam_head"][0] == jpeg(tag=13)


def test_success_is_never_taken_from_the_expert(D, tmp_path):
    """A successful-looking episode whose OFFICIAL flag is False is recorded as failure."""
    env = FakeEnv(steps_to_end=120, official_success=False)
    expert = FakeExpert()
    assert expert.success() is True, "the double really does claim success"
    assert expert.is_success() is True
    record = run_episode(D, tmp_path, env=env, expert=expert)

    assert record["status"] == "accepted", record
    assert record["success"] is False, "the expert's claim leaked into the verdict"
    assert record["official_success"] is False
    assert record["termination"] == "official_failure"
    path = str(tmp_path / "run" / record["episode_path"])
    with h5py.File(path, "r") as handle:
        assert int(handle["audit"].attrs["official_success"]) == 0
        assert json.loads(handle.attrs["meta_json"])["official"]["success"] is False
    ledger = [json.loads(line) for line in
              open(str(tmp_path / "run" / "ledger.jsonl"), encoding="utf-8")]
    assert ledger[0]["official_success"] is False


# --------------------------------------------------------------------------- refusals

def test_expert_returning_no_action_is_refused(D, tmp_path):
    record = run_episode(D, tmp_path, env=FakeEnv(steps_to_end=130),
                         expert=FakeExpert(mode="none"))
    assert record["status"] == "refused"
    assert record["code"] == "expert_no_action"
    assert not os.path.isdir(str(tmp_path / "run" / "episodes")) or not [
        n for n in os.listdir(str(tmp_path / "run" / "episodes")) if n.endswith(".hdf5")]
    assert os.path.isfile(str(tmp_path / "run" / record["refusal_report"]))
    ledger = [json.loads(line) for line in
              open(str(tmp_path / "run" / "ledger.jsonl"), encoding="utf-8")]
    assert ledger[0]["status"] == "refused"
    assert ledger[0]["code"] == "expert_no_action"
    assert ledger[0]["official_success"] is None


def test_expert_exception_is_refused(D, tmp_path):
    record = run_episode(D, tmp_path, expert=FakeExpert(mode="raise"))
    assert record["status"] == "refused"
    assert record["code"] == "expert_error"


@pytest.mark.parametrize("mode", ["bad_shape", "nan"])
def test_malformed_expert_action_is_refused(D, tmp_path, mode):
    record = run_episode(D, tmp_path, expert=FakeExpert(mode=mode))
    assert record["status"] == "refused"
    assert record["code"] == "expert_bad_action"


def test_nan_observation_is_refused(D, tmp_path):
    record = run_episode(D, tmp_path, env=FakeEnv(steps_to_end=130, nan_at=3))
    assert record["status"] == "refused"
    assert record["code"] == "nonfinite_observation"
    assert not os.path.exists(str(tmp_path / "run" / "episodes") + "/r15test_ep0000000.hdf5")


def test_action_not_equal_to_the_next_state_is_refused(D, tmp_path):
    record = run_episode(D, tmp_path, env=FakeEnv(steps_to_end=130, drift=1e-3))
    assert record["status"] == "refused"
    assert record["code"] == "misaligned_state_action"
    assert "max|A[" in record["message"]
    assert record["episode_path"] if "episode_path" in record else True
    assert not [n for n in os.listdir(str(tmp_path / "run" / "episodes"))
                if n.endswith(".hdf5")]


def test_unreadable_official_verdict_is_refused(D, tmp_path):
    record = run_episode(D, tmp_path, env=NoVerdictEnv())
    assert record["status"] == "refused"
    assert record["code"] == "official_verdict_unavailable"


def test_privileged_value_without_a_declared_source_is_refused(D, tmp_path):
    cfg = make_cfg(D, provenance={"block_pose": {"source": "x"},
                                  "target_pose": PROV["target_pose"],
                                  "ee_pose": PROV["ee_pose"]})
    record = run_episode(D, tmp_path, cfg=cfg)
    assert record["status"] == "refused"
    assert record["code"] == "missing_provenance"
    assert "file" in record["message"] or "line" in record["message"]


def test_a_privileged_value_the_manifest_does_not_declare_is_refused(D, tmp_path):
    cfg = make_cfg(D, privileged_keys=("block_pose", "target_pose"),
                   provenance={"block_pose": PROV["block_pose"],
                               "target_pose": PROV["target_pose"]})
    record = run_episode(D, tmp_path, cfg=cfg)
    assert record["status"] == "refused"
    assert record["code"] == "undeclared_privileged"
    assert "ee_pose" in record["message"]


def test_declaring_privileged_use_without_provenance_fails_at_config_time(D):
    with pytest.raises(D.CollectRefused) as exc:
        make_cfg(D, privileged_keys=("block_pose",), provenance={})
    assert exc.value.code == "missing_provenance"


def test_an_instruction_that_does_not_name_the_task_is_refused_at_config_time(D):
    with pytest.raises(D.CollectRefused) as exc:
        make_cfg(D, instruction="Move the red cube to the tray.", writer_task="push_T")
    assert exc.value.code == "bad_config"


# --------------------------------------------------------------------------- caps

def test_cap_accepted_per_task_is_enforced_from_the_ledger(D, tmp_path):
    root = str(tmp_path / "run")
    seed_ledger(root, [{"attempt_id": "seed_%05d" % i, "task": "push_T",
                        "status": "accepted", "official_success": True, "bytes": 1000,
                        "episode_path": "episodes/x.hdf5"} for i in range(100)],
                D.SCHEMA)
    record = run_episode(D, tmp_path)
    assert record["status"] == "refused"
    assert record["code"] == "cap_accepted_per_task"
    assert not [n for n in os.listdir(os.path.join(root, "episodes"))
                if n.endswith(".hdf5")]


def test_cap_attempts_per_task_is_enforced(D, tmp_path):
    root = str(tmp_path / "run")
    seed_ledger(root, [{"attempt_id": "seed_%05d" % i, "task": "push_T",
                        "status": "refused", "code": "expert_no_action"}
                       for i in range(250)], D.SCHEMA)
    record = run_episode(D, tmp_path)
    assert record["status"] == "refused"
    assert record["code"] == "cap_attempts_per_task"


def test_cap_attempts_total_is_enforced(D, tmp_path):
    root = str(tmp_path / "run")
    tasks = ["t%02d" % i for i in range(12)]
    records = []
    for i in range(2988):
        records.append({"attempt_id": "seed_%05d" % i, "task": tasks[i % 12],
                        "status": "refused", "code": "expert_no_action"})
    records.append({"attempt_id": "seed_push", "task": "push_T", "status": "refused",
                    "code": "expert_no_action"})
    for i in range(11):
        records.append({"attempt_id": "seed_x%02d" % i, "task": "x", "status": "refused",
                        "code": "expert_no_action"})
    assert len(records) == 3000
    seed_ledger(root, records, D.SCHEMA)
    record = run_episode(D, tmp_path)
    assert record["status"] == "refused"
    assert record["code"] == "cap_attempts_total"


def test_cap_new_bytes_is_enforced_before_writing(D, tmp_path):
    root = str(tmp_path / "run")
    seed_ledger(root, [{"attempt_id": "seed", "task": "push_T", "status": "accepted",
                        "official_success": True,
                        "bytes": 250 * 1024 ** 3, "episode_path": "episodes/x.hdf5"}],
                D.SCHEMA)
    record = run_episode(D, tmp_path)
    assert record["status"] == "refused"
    assert record["code"] == "cap_new_bytes"
    assert not [n for n in os.listdir(os.path.join(root, "episodes"))
                if n.endswith(".hdf5")]


# --------------------------------------------------------------------------- writer + index

def test_round_trip_through_the_real_writer_and_read_back(D, W, tmp_path):
    record = run_episode(D, tmp_path)
    path = str(tmp_path / "run" / record["episode_path"])
    report = W.verify_episode(path)
    assert report["verified"] and report["frames"] == record["frames"]
    assert report["bytes"] == record["bytes"]
    assert report["content_sha256"] == record["content_sha256"]
    assert os.path.getsize(path) == record["bytes"]
    with open(path, "rb") as handle:
        raw = handle.read()
    import hashlib
    assert hashlib.sha256(raw).hexdigest() == record["file_sha256"]
    # nothing left behind
    assert not [n for n in os.listdir(str(tmp_path / "run" / "episodes")) if n.endswith(".tmp")]


def test_duplicate_content_is_refused_by_the_always_on_index(D, tmp_path):
    first = run_episode(D, tmp_path, seed=7)
    assert first["status"] == "accepted"
    second = run_episode(D, tmp_path, seed=7)          # identical trajectory and pixels
    assert second["status"] == "refused"
    assert second["code"] == "hash_collision"
    assert len([n for n in os.listdir(str(tmp_path / "run" / "episodes"))
                if n.endswith(".hdf5")]) == 1


def test_a_second_episode_with_a_new_seed_is_accepted(D, tmp_path):
    first = run_episode(D, tmp_path, seed=7)
    second = run_episode(D, tmp_path, seed=8)
    assert (first["status"], second["status"]) == ("accepted", "accepted")
    assert first["episode_path"] != second["episode_path"]
    assert first["content_sha256"] != second["content_sha256"]
    files = [n for n in os.listdir(str(tmp_path / "run" / "episodes")) if n.endswith(".hdf5")]
    assert len(files) == 2


def test_the_manifest_pins_the_run(D, tmp_path):
    run_episode(D, tmp_path)
    manifest = json.load(open(str(tmp_path / "run" / "run_manifest.json"), encoding="utf-8"))
    assert manifest["schema"] == "r15-collect-manifest-1"
    assert manifest["instruction"] == INSTRUCTION
    assert manifest["step_lim"] == 600
    assert "eval_env.py:882-901" in manifest["official_verdict"]["method"]
    assert set(manifest["privileged"]) == set(PROV)
    with pytest.raises(D.CollectRefused) as exc:
        run_episode(D, tmp_path, cfg=make_cfg(D, instruction=INSTRUCTION + " Go."))
    assert exc.value.code == "manifest_mismatch"


# --------------------------------------------------------------------------- registry wiring

def test_registry_covers_the_twelve_official_tasks(R):
    assert len(R.MISSION_TASKS) == 12
    expected = {"push_T": 600, "stack_bowls": 800, "sweep_blocks": 1000,
                "pack_objects_into_box": 1300, "hang_mugs": 800, "make_toast": 1400,
                "arrange_largest_number": 1050, "sort_nesting_dolls_by_size": 1050,
                "store_laptop_and_headphones": 800, "stack_blocks": 550,
                "fold_clothes": 500, "pour_liquid_into_cup": 400}
    assert set(R.MISSION_TASKS) == set(expected)
    for task_id, step_lim in expected.items():
        spec = R.get_task(task_id)
        assert spec.step_lim == step_lim, task_id
        assert spec.writer_task.lower() in spec.instruction.lower(), task_id
        assert spec.predicate_ref, task_id
        assert spec.status in (R.IMPLEMENTED, R.BLOCKED_DATA)
    implemented = [t for t in R.MISSION_TASKS if R.is_implemented(t)]
    assert implemented == ["push_T"]
    for task_id in R.MISSION_TASKS:
        spec = R.get_task(task_id)
        if spec.status == R.BLOCKED_DATA:
            assert spec.status_reason(), task_id
            assert "no expert implementation" in spec.status_reason(), task_id
            assert spec.unblocked_by, task_id
        else:
            assert spec.expert_version_sha() and spec.expert_version_sha() != "0" * 64


def test_registry_refuses_to_hand_out_an_expert_it_cannot_justify(R):
    with pytest.raises(R.RegistryBlocked) as exc:
        R.make_expert("stack_bowls")
    assert "BLOCKED_DATA" in str(exc.value)
    with pytest.raises(R.RegistryBlocked):
        R.make_expert("push_T")            # no IK, no home_joints
    with pytest.raises(R.RegistryError):
        R.get_task("no_such_task")


def test_push_t_privileged_reads_the_audited_getters(R):
    class FakeLayoutManager(object):
        def __init__(self):
            self.poses = {"t": ([0.10, 0.20, 0.00], [1.0, 0.0, 0.0, 0.0]),
                          "target_t": ([0.30, 0.20, 0.01], [1.0, 0.0, 0.0, 0.0])}

        def get_instance_name(self, label, env_idx=0):
            return label if label in self.poses else None

        def get_instance_pose(self, inst_name, env_idx=0):
            return self.poses[inst_name]

    class FakeEnvWithLayout(object):
        def __init__(self):
            func_parser = type("FP", (), {})()
            func_parser.layout_manager = FakeLayoutManager()
            reward_manager = type("RM", (), {})()
            reward_manager.func_parser = func_parser
            self.reward_manager = reward_manager

    obs = {"state": {"left_ee_pose": np.array([0.0, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0])}}
    values = R.push_t_privileged(FakeEnvWithLayout(), obs)
    assert np.allclose(values["block_pose"], [0.10, 0.20, 0.00, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(values["target_pose"], [0.30, 0.20, 0.01, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(values["ee_pose"], [0.0, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0])
    for key in ("block_pose", "target_pose", "ee_pose"):
        assert R.PUSH_T_PROVENANCE[key]["file"].startswith("RoboDojo/")
        assert int(R.PUSH_T_PROVENANCE[key]["line"]) > 0


def test_driver_config_from_the_registry_matches_the_official_metadata(D, R):
    cfg = D.config_from_task("push_T")
    assert cfg.instruction == R.PUSH_T_INSTRUCTION
    assert cfg.step_lim == 600
    assert cfg.writer_task.lower() in cfg.instruction.lower()
    assert set(cfg.privileged_keys) == {"block_pose", "target_pose", "ee_pose"}
    assert len(cfg.expert_version_sha) == 64
    assert cfg.expert_status == "IMPLEMENTED"
    with pytest.raises(D.CollectRefused) as exc:
        D.config_from_task("fold_clothes")
    assert exc.value.code == "expert_blocked"


def test_driver_module_is_importable_without_isaacsim(D):
    assert D.BASE_DIM == 14 and D.FREQUENCY_HZ == 25
    assert "eval_env.py:882-901" in D.OFFICIAL_VERDICT["method"]
    assert D.CAPS["accepted_per_task"] == 100
    assert D.CAPS["attempts_per_task"] == 250
    assert D.CAPS["attempts_total"] == 3000
    assert D.CAPS["new_bytes"] == 250 * 1024 ** 3
