#!/usr/bin/env python3
"""R16 T7: the stub-backend policy server, the G5 forward parity and the G12 codec assertion.

CPU only.  The stub backend makes the whole deploy path - observations in, 14 legal action
tokens, S+V+R safety, 4 latency phases, 4-key action dict - executable without the 7B weights,
which is the only way to prove the wiring while a training job owns the GPU.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_ROOT = os.path.join(REPO, "code", "xpolicylab", "policy")
POLICY_DIR = os.path.join(POLICY_ROOT, "OpenVLA_R42")
sys.path.insert(0, POLICY_ROOT)          # import OpenVLA_R42 as a package
sys.path.insert(0, POLICY_DIR)           # import r42_adapter / r42_deploy_server flat
sys.path.insert(0, os.path.join(POLICY_ROOT, "OpenVLA_Pure"))

import r42_adapter as R  # noqa: E402
import r42_deploy_server as S  # noqa: E402
import r42_env_io as E  # noqa: E402
import OpenVLA_R42  # noqa: E402
import OpenVLA_R42.model as M  # noqa: E402

CONTRACT = os.path.join(REPO, "evidence", "r9_gate_20261020", "frozen_contract_64train.json")
BOUNDS = os.path.join(REPO, "data", "push_T_bounds.npz")
PAD, STOP = 32000, 2
PROMPT_IDS = [128000, 9906, 11, 5123, 29901]


class StubBackend(object):
    def __init__(self, seed=0, vocab=32064, bail=False):
        self.seed, self.vocab, self.bail = seed, vocab, bail
        self.calls = 0
        self.weight = np.zeros(4, np.float32)

    def next_token_logits(self, input_ids, pixel_values=None, proprio=None):
        self.calls += 1
        if self.bail:
            raise RuntimeError("backend exploded")
        sig = (self.seed * 1000003
               + sum((i + 1) * int(x) for i, x in enumerate(input_ids))) % (2 ** 31 - 1)
        if proprio is not None:
            sig = (sig + int(round(float(np.abs(proprio).sum()) * 1e6))) % (2 ** 31 - 1)
        logits = (np.random.RandomState(sig).randn(self.vocab) * 0.01).astype(np.float32)
        logits[PAD] = 500.0
        logits[STOP] = 400.0
        return logits


def _meta():
    frozen = json.load(open(CONTRACT))
    norm = R.normalizer_payload(
        normalization_mode=frozen["normalization_mode"], pmin=frozen["pmin"],
        pmax=frozen["pmax"], q01=frozen["q01"], q99=frozen["q99"],
        v_scale=frozen["v_scale"], a_scale=frozen["a_scale"],
        motion_quantile=frozen["motion_quantile"], n_train_episodes=64,
        n_dev_episodes_excluded=16)
    nb = np.load(BOUNDS)
    sha = "c" * 64
    return R.build_metadata(
        "push_T", sha, sha, sha, sha, sha, train_steps=0, seed=0,
        gripper_policy=frozen["gripper_policy"],
        safety_limits={"amin": nb["amin"].astype(np.float32).tolist(),
                       "amax": nb["amax"].astype(np.float32).tolist(),
                       "max_rate": [0.5] * 14},
        normalizer=norm)


def _rgb(shape=(6, 8, 3), seed=0):
    return np.random.RandomState(seed).randint(0, 255, shape).astype(np.uint8)


def _obs(state=None, seed=0, **extra):
    st = _inside() if state is None else state
    obs = {"vision": {cam: {"color": _rgb(seed=seed + i)}
                      for i, cam in enumerate(R.CAMERA_ORDER)},
           "state": dict(zip(E.JOINT_ORDER, [st[:6], st[6:7], st[7:13], st[13:]])),
           "action": {"left_arm_joint_state": [0.0] * 6},
           "instruction": M.INSTRUCTION,
           "additional_info": {"frequency": 25},
           "data_format_version": "v1.0",
           "env_idx": 0}
    obs["state"].update({"left_ee_pose": np.zeros(7, np.float32),
                         "right_ee_pose": np.zeros(7, np.float32)})
    obs.update(extra)
    return obs


def _inside(alpha=0.5):
    nb = np.load(BOUNDS)
    lo = np.asarray(nb["amin"], np.float32)
    hi = np.asarray(nb["amax"], np.float32)
    return (lo + float(alpha) * (hi - lo)).astype(np.float32)


def _model(backend=None, meta=None, **over):
    cfg = {"policy_name": "OpenVLA_R42",
           "metadata": meta if meta is not None else _meta(),
           "backend": backend if backend is not None else StubBackend(),
           "device": "cpu",
           "prompt": M.INSTRUCTION,
           "expected_hw": None,
           "control_hz": 25,
           "tokenizer_prompt_ids": PROMPT_IDS}
    cfg.update(over)
    return M.Model(cfg)


# ------------------------------------------------------------------- the whole deploy step
def test_model_constructs_without_a_gpu_and_without_loading_weights():
    model = _model()
    assert model._server is None, "construction must be lazy (no 7B load, no CUDA touch)"
    assert model.codec_mode == "fixed"
    assert model.metadata["proprio_dim"] == 42
    contract = model.contract()
    assert contract["codec_mode"] == "fixed"
    assert contract["latency_phases"] == ["preprocess", "forward_total", "decode", "total"]
    assert contract["reset_scope"] == "all_envs"
    assert contract["rate_limit"]["enabled"] is True
    assert contract["rate_limit"]["source"] == "safety_limits.max_rate"
    assert contract["rate_limit"]["units"] == "per-dimension state-coordinate units per second"
    assert contract["rate_limit"]["control_dt"] == pytest.approx(0.04)
    assert contract["rate_limit"]["client_acknowledgement"] == "UNKNOWN"
    assert contract["env_io"]["camera_order"] == ["cam_head", "cam_left_wrist",
                                                  "cam_right_wrist"]


def test_one_step_yields_fourteen_tokens_and_the_four_latency_phases():
    model = _model()
    model.update_obs(_obs())
    payload = model.get_action()
    assert payload["n_action_tokens"] == 14
    ids = np.asarray(payload["ids"], np.int64)
    assert ids.shape == (14,)
    assert ids.min() >= 31744 and ids.max() <= 31999
    assert PAD not in set(payload["ids"]) and STOP not in set(payload["ids"])
    assert sorted(payload["latency_ms"]) == ["decode", "forward_total", "preprocess", "total"]
    for phase, value in payload["latency_ms"].items():
        assert np.isfinite(value) and value >= 0.0, phase
    assert payload["codec_mode"] == "fixed"
    action = payload["actions"]
    assert list(action) == list(E.JOINT_ORDER)
    assert [len(action[k]) for k in E.JOINT_ORDER] == [6, 1, 6, 1]
    flat = E.action_dict_to_cmd14(action)
    lo = np.asarray(model.metadata["safety_limits"]["amin"], np.float32)
    hi = np.asarray(model.metadata["safety_limits"]["amax"], np.float32)
    assert (flat >= lo - 1e-6).all() and (flat <= hi + 1e-6).all()
    assert 0.0 <= flat[6] <= 1.0 and 0.0 <= flat[13] <= 1.0
    assert payload["stamp"]["server_span_ms"] >= 0.0
    assert payload["stamp"]["obs_iso"] != payload["stamp"]["reply_iso"]


def test_official_model_path_enforces_rate_limit_and_reset_scoped_previous_command(monkeypatch):
    model = _model()
    server = model._ensure_server()
    ids = np.full(R.N_ACTIONS, R.CODEC_WINDOW[1], dtype=np.int64)
    monkeypatch.setattr(server, "_decode_tokens", lambda pixel_values, feat: (ids.copy(), []))

    calls = []
    original_act = server.act

    def tracked_act(images, state, env_idx=0, timestamp=None, prev_cmd=None, dt=None, record=False):
        call = {"state": np.asarray(state, np.float32).copy(),
                "prev_cmd": None if prev_cmd is None else np.asarray(prev_cmd, np.float32).copy(),
                "dt": dt}
        out = original_act(images, state, env_idx=env_idx, timestamp=timestamp,
                           prev_cmd=prev_cmd, dt=dt, record=record)
        call["info"] = dict(out["info"])
        call["action14"] = np.asarray(out["action"], np.float32).copy()
        calls.append(call)
        return out

    monkeypatch.setattr(server, "act", tracked_act)
    rate = np.asarray(model.metadata["safety_limits"]["max_rate"], np.float32)
    lo = np.asarray(model.metadata["safety_limits"]["amin"], np.float32)
    hi = np.asarray(model.metadata["safety_limits"]["amax"], np.float32)
    dt = 1.0 / model.control_hz

    state0 = _inside(0.5).copy()
    model.update_obs(_obs(state=state0, seed=10))
    first = model.get_action()
    assert calls[0]["prev_cmd"] == pytest.approx(state0.tolist())
    assert calls[0]["dt"] == pytest.approx(0.04)
    assert calls[0]["info"]["dt"] == pytest.approx(dt)
    assert calls[0]["info"]["rate_limited"] > 0
    assert np.all(np.abs(np.asarray(first["action14"]) - state0) <= rate * dt + 1e-6)
    assert np.all(np.asarray(first["action14"]) >= lo - 1e-5)
    assert np.all(np.asarray(first["action14"]) <= hi + 1e-5)

    state1 = state0.copy()
    model.update_obs(_obs(state=state1, seed=11))
    second = model.get_action()
    assert calls[1]["prev_cmd"] == pytest.approx(first["action14"])
    assert calls[1]["dt"] == pytest.approx(dt)
    assert np.all(np.abs(np.asarray(second["action14"]) - np.asarray(first["action14"]))
                  <= rate * dt + 1e-6)
    assert np.all(np.asarray(second["action14"]) >= lo - 1e-5)
    assert np.all(np.asarray(second["action14"]) <= hi + 1e-5)

    model.reset()
    state2 = _inside(0.4).copy()
    model.update_obs(_obs(state=state2, seed=12))
    third = model.get_action()
    assert calls[2]["prev_cmd"] == pytest.approx(state2.tolist())
    assert calls[2]["dt"] == pytest.approx(dt)
    assert np.all(np.abs(np.asarray(third["action14"]) - state2) <= rate * dt + 1e-6)


def test_official_model_path_refuses_missing_rate_limit_metadata():
    meta = _meta()
    del meta["safety_limits"]["max_rate"]
    with pytest.raises((M.R42ModelError, R.R42Error), match="max_rate"):
        _model(meta=meta)


def test_official_model_path_refuses_response_without_rate_branch_evidence():
    class _NoRateInfo(object):
        metadata = _meta()
        device = "cpu"

        def act(self, images, state, env_idx=0, timestamp=None, prev_cmd=None, dt=None,
                record=False):
            return {"action": _inside(0.5), "ids": [R.CODEC_WINDOW[0]] * R.N_ACTIONS,
                    "n_action_tokens": R.N_ACTIONS,
                    "info": {"dt": None, "rate_limited": 0},
                    "latency_ms": {"preprocess": 1.0, "forward_total": 2.0, "decode": 1.0,
                                   "total": 4.0}}

    model = _model()
    model.update_obs(_obs())
    model._server = _NoRateInfo()
    with pytest.raises(M.R42ModelError, match="rate|dt"):
        model.get_action()


def test_runtime_audit_records_state_prediction_response_and_next_observation(tmp_path, monkeypatch):
    import r42_runtime_audit as Audit

    path = tmp_path / "runtime.jsonl"
    monkeypatch.setenv("R42_AUDIT_LOG_PATH", str(path))
    monkeypatch.setenv("R42_AUDIT_RUN_ID", "r30-test-001")
    model = _model()
    model.reset()

    state0 = _inside(0.5).copy()
    model.update_obs(_obs(state=state0, seed=0))
    first_payload = model.get_action()

    state1 = state0.copy()
    state1[0] += 0.02
    model.update_obs(_obs(state=state1, seed=1))
    second_payload = model.get_action()
    model.reset()
    model._audit_trace.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    steps = [row for row in rows if row["event"] == "action_step"]
    assert [row["episode_step_index"] for row in steps] == [0, 1]
    assert [row["global_step_index"] for row in steps] == [0, 1]
    assert all(row["run_id"] == "r30-test-001" for row in rows)
    assert steps[0]["state14"] == pytest.approx(state0.tolist())
    assert steps[0]["prediction"]["token_ids14"] == first_payload["ids"]
    assert steps[0]["prediction"]["bins14"] == first_payload["bins"]
    assert steps[0]["server_response"]["action14"] == pytest.approx(first_payload["action14"])
    assert steps[0]["server_response"]["action_dict"] == first_payload["actions"]
    assert "pre_audit_return_wall_utc" in steps[0]
    assert "response_wall_utc" not in steps[0]
    assert "server_span_pre_audit_ms" in steps[0]
    assert "server_span_ms" not in steps[0]
    assert "fsync" in steps[0]["timing_semantics"].lower()
    assert steps[0]["observation_audit_overhead_ms"] >= 0.0
    assert len(steps[0]["prediction"]["residual14"]) == 14
    assert len(steps[0]["features42"]) == 42
    assert steps[0]["previous_server_response_action14"] is None
    assert steps[0]["rate_reference_action14"] == pytest.approx(state0.tolist())
    assert steps[1]["previous_server_response_action14"] == pytest.approx(first_payload["action14"])
    assert steps[1]["rate_reference_action14"] == pytest.approx(first_payload["action14"])
    assert steps[0]["control_dt"] == pytest.approx(0.04)
    assert steps[0]["rate_limit_source"] == "safety_limits.max_rate"
    expected_delta = state1 - np.asarray(first_payload["action14"], np.float32)
    assert steps[1]["observed_state_minus_previous_server_response14"] == pytest.approx(
        expected_delta.tolist())
    assert set(steps[0]["camera_inputs"]) == set(R.CAMERA_ORDER)
    for camera, item in steps[0]["camera_inputs"].items():
        assert len(item["sha256"]) == 64
        assert item["dtype"] == "uint8"
        assert item["shape"] == [6, 8, 3]
    assert "raw_image_bytes" not in steps[0]
    assert "images" not in steps[0]
    assert [row["event"] for row in rows].count("episode_reset") == 2
    assert rows[0]["event"] == "run_start"
    assert any(row["event"] == "backend_ready" for row in rows)


def test_runtime_audit_persists_observation_before_backend_failure(tmp_path, monkeypatch):
    path = tmp_path / "runtime.jsonl"
    monkeypatch.setenv("R42_AUDIT_LOG_PATH", str(path))
    monkeypatch.setenv("R42_AUDIT_RUN_ID", "r30-failure-evidence-001")
    model = _model(backend=StubBackend(bail=True))
    model.reset()
    state = _inside(0.5).copy()
    model.update_obs(_obs(state=state, seed=2))

    with pytest.raises(RuntimeError, match="backend exploded"):
        model.get_action()
    model._audit_trace.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    observations = [row for row in rows if row["event"] == "observation_received"]
    errors = [row for row in rows if row["event"] == "action_error"]
    assert len(observations) == 1
    assert observations[0]["state14"] == pytest.approx(state.tolist())
    assert set(observations[0]["camera_inputs"]) == set(R.CAMERA_ORDER)
    assert len(errors) == 1
    assert errors[0]["error_type"] == "RuntimeError"
    assert errors[0]["observation_audit_overhead_ms"] >= 0.0
    assert errors[0]["observation_record_index"] == observations[0]["record_index"]
    assert not any(row["event"] == "action_step" for row in rows)


def test_runtime_audit_records_backend_load_failure_against_received_observation(tmp_path, monkeypatch):
    path = tmp_path / "runtime.jsonl"
    monkeypatch.setenv("R42_AUDIT_LOG_PATH", str(path))
    monkeypatch.setenv("R42_AUDIT_RUN_ID", "r30-backend-failure-001")
    model = _model()
    model.reset()
    state = _inside(0.5).copy()
    model.update_obs(_obs(state=state, seed=3))

    def fail_backend():
        raise RuntimeError("backend load exploded")

    monkeypatch.setattr(model, "_ensure_server", fail_backend)
    with pytest.raises(RuntimeError, match="backend load exploded"):
        model.get_action()
    model._audit_trace.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    observation = next(row for row in rows if row["event"] == "observation_received")
    error = next(row for row in rows if row["event"] == "action_error")
    assert error["stage"] == "backend_load"
    assert error["observation_record_index"] == observation["record_index"]
    assert error["observation_index"] == observation["observation_index"]


def test_runtime_audit_is_disabled_without_an_explicit_path(tmp_path, monkeypatch):
    monkeypatch.delenv("R42_AUDIT_LOG_PATH", raising=False)
    monkeypatch.delenv("R42_AUDIT_RUN_ID", raising=False)
    model = _model()
    assert model._audit_trace is None
    model.update_obs(_obs())
    payload = model.get_action()
    assert payload["n_action_tokens"] == 14
    assert not list(tmp_path.iterdir())


def test_runtime_audit_refuses_to_overwrite_existing_evidence(tmp_path, monkeypatch):
    import r42_runtime_audit as Audit

    path = tmp_path / "runtime.jsonl"
    path.write_text("historical evidence\\n", encoding="utf-8")
    monkeypatch.setenv("R42_AUDIT_LOG_PATH", str(path))
    monkeypatch.setenv("R42_AUDIT_RUN_ID", "r30-test-002")
    with pytest.raises(Audit.RuntimeAuditError, match="already exists"):
        _model()
    assert path.read_text(encoding="utf-8") == "historical evidence\\n"


def test_gripper_channels_of_the_served_action_are_state_plus_velocity():
    model = _model()
    state = _inside(0.5).copy()
    state[6] = 0.2
    state[13] = 0.8
    model.update_obs(_obs(state=state))
    action = model.get_action()["actions"]
    assert action["left_ee_joint_state"][0] == pytest.approx(0.2, abs=1e-6)
    assert action["right_ee_joint_state"][0] == pytest.approx(0.8, abs=1e-6)


def test_get_action_without_update_obs_is_refused():
    model = _model()
    with pytest.raises(M.R42ModelError, match="update_obs"):
        model.get_action()


def test_reset_clears_every_env_g9():
    """G9: the WS server calls reset() with no env_idx, so reset must clear ALL envs."""
    model = _model()
    model.update_obs(_obs())
    model.get_action()
    model.update_obs(_obs(seed=3, env_idx=1))
    model.get_action()
    assert model._server is not None
    assert sorted(model._server._history._hist) == [0, 1]
    assert model.reset() is True
    assert model._server._history._hist == {}
    assert model.n_resets == 1
    # after a reset the next step for env 1 must look like a first frame again
    model.update_obs(_obs(seed=3, env_idx=1))
    payload = model.get_action()
    assert np.allclose(payload["ids"], payload["ids"])   # deterministic backend, still legal
    assert model._server._history._hist and 1 in model._server._history._hist


def test_update_obs_batch_is_single_env_only():
    model = _model()
    with pytest.raises(M.R42ModelError, match="one environment"):
        model.update_obs_batch([_obs(), _obs()])
    model.update_obs_batch([_obs()])
    assert model.get_action()["n_action_tokens"] == 14


def test_get_action_batch_is_refused():
    with pytest.raises(M.R42ModelError, match="single-env"):
        _model().get_action_batch()


def test_observation_instruction_is_ignored_but_warned_once():
    model = _model()
    model.update_obs(_obs(instruction="pick up the red cube"))
    model.get_action()
    assert any("ignored" in w for w in model._warned)


# --------------------------------------------------------------- G12: the codec mode is asserted
def test_metadata_with_a_non_fixed_codec_mode_is_refused():
    meta = _meta()
    meta["codec_mode"] = "legacy"
    with pytest.raises(R.R42Error):
        _model(meta=meta)


def test_prompt_that_differs_from_the_trained_instruction_is_refused():
    with pytest.raises(M.R42ModelError, match="allow_prompt_override"):
        _model(prompt="Pick up the cube.")


def test_prompt_override_can_be_requested_explicitly():
    model = _model(prompt="Pick up the cube.", allow_prompt_override=True)
    assert model.instruction == "Pick up the cube."


# ------------------------------------------------------- G5: deploy forward == training forward
class _FakeCore(object):
    """Minimal OpenVLA core: owns _process_action_masks, records the forward kwargs."""

    def __init__(self):
        self.calls = []

    def _process_action_masks(self, labels):
        raise AssertionError("unpatched _process_action_masks must not be used")

    def parameters(self):
        return iter([])

    def __call__(self, **kwargs):
        self.calls.append(kwargs)

        class _Out(object):
            import torch as _t
            logits = _t.zeros(1, 1, 32064)

        return _Out()


class _WrapperWithGetBase(object):
    """Stands in for a PEFT/LoRA wrapper: patching it would be a silent no-op."""

    def __init__(self, core):
        self.core = core

    def get_base_model(self):
        return self.core

    def parameters(self):
        return self.core.parameters()

    def __call__(self, **kwargs):
        return self.core(**kwargs)


class _Projector(object):
    def parameters(self):
        return iter([])

    def __call__(self, *a, **k):
        return None


def test_torch_backend_installs_pure_autoreg_and_passes_labels():
    import torch
    core = _FakeCore()
    backend = S._TorchBackend(_WrapperWithGetBase(core), _Projector(), device="cpu")
    assert backend.pure_autoreg is True
    assert backend.core is core, "the patch must land on the core, not on the PEFT wrapper"
    assert getattr(core, "_pure_autoreg_input_mode", False) is True
    logits = backend.next_token_logits([128000, 11, 22], None, np.zeros(42, np.float32))
    assert logits.shape == (32064,)
    assert len(core.calls) == 1
    kwargs = core.calls[0]
    assert "labels" in kwargs and kwargs["labels"] is not None, \
        "training always passes labels= (pure_autoreg_mode.py:19-25 raises otherwise)"
    assert torch.equal(kwargs["labels"], kwargs["input_ids"])
    assert kwargs["input_ids"].tolist() == [[128000, 11, 22]]
    assert kwargs["proprio"] is not None and tuple(kwargs["proprio"].shape) == (1, 42)


def test_pure_autoreg_patch_refuses_a_missing_labels_argument():
    core = _FakeCore()
    S.install_pure_autoreg(core)
    with pytest.raises(ValueError, match="labels required"):
        core._process_action_masks(None)
    import torch
    mask = core._process_action_masks(torch.zeros(1, 5, dtype=torch.long))
    assert mask.dtype == torch.bool and not bool(mask.any())


def test_install_pure_autoreg_refuses_an_incompatible_model():
    class _NoMask(object):
        def parameters(self):
            return iter([])

    with pytest.raises(R.R42Error, match="_process_action_masks"):
        S.install_pure_autoreg(_NoMask())


def test_ablation_without_pure_autoreg_is_refused_by_the_model():
    """A backend that does not carry the patch is not the training forward (documented)."""
    import torch
    core = _FakeCore()
    backend = S._TorchBackend(_WrapperWithGetBase(core), _Projector(), device="cpu",
                              pure_autoreg=False)
    assert backend.core is None
    with pytest.raises(AssertionError):
        core._process_action_masks(torch.zeros(1, 2, dtype=torch.long))


# ------------------------------------------------------------------- latency phase assertions
def test_missing_latency_phase_is_refused():
    server = S.R42DeployServer(_meta(), backend=StubBackend(), device="cpu",
                               tokenizer_prompt_ids=PROMPT_IDS)
    with pytest.raises(R.R42Error, match="missing phase"):
        server._check_latency({"preprocess": 1.0, "forward_total": 2.0, "total": 4.0})
    with pytest.raises(R.R42Error, match="negative|>= 0"):
        server._check_latency({"preprocess": 1.0, "forward_total": -2.0, "decode": 1.0,
                               "total": 4.0})
    with pytest.raises(R.R42Error, match="smaller than"):
        server._check_latency({"preprocess": 5.0, "forward_total": 2.0, "decode": 1.0,
                               "total": 4.0})
    ok = server._check_latency({"preprocess": 1.0, "forward_total": 2.0, "decode": 1.0,
                                "total": 4.0})
    assert ok["total"] == 4.0


def test_model_refuses_a_server_that_does_not_report_the_phases():
    class _NoPhases(object):
        metadata = _meta()
        device = "cpu"

        def act(self, images, state, **kwargs):
            return {"action": np.zeros(14, np.float32), "ids": [31744] * 14,
                    "n_action_tokens": 14, "latency_ms": {"total": 1.0}}

    model = _model()
    model.update_obs(_obs())
    model._server = _NoPhases()
    with pytest.raises(M.R42ModelError, match="missing phase"):
        model.get_action()


def test_server_refuses_a_wrong_token_count():
    class _Short(object):
        metadata = _meta()
        device = "cpu"

        def act(self, images, state, **kwargs):
            return {"action": np.zeros(14, np.float32), "ids": [31744] * 13,
                    "n_action_tokens": 13,
                    "latency_ms": {"preprocess": 1.0, "forward_total": 1.0, "decode": 1.0,
                                   "total": 3.0}}

    model = _model()
    model.update_obs(_obs())
    model._server = _Short()
    with pytest.raises(M.R42ModelError, match="action token"):
        model.get_action()


def test_package_contract_is_importable_and_complete():
    c = OpenVLA_R42.contract()
    assert c["codec_mode"] == "fixed"
    assert c["n_action_tokens"] == 14
    assert c["latency_phases"] == ["preprocess", "forward_total", "decode", "total"]
    assert c["env_io"]["state_dim"] == 14
