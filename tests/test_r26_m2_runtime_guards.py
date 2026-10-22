"""R26 negative safety probes for M2 task-bound CPU-only deployment path.

Real two-task normalizer, synthetic sealed checkpoint, no CUDA/HDF5/servo.
Intentional RED first.  This file must NOT normalize CPU stub activity into
an authorized real robot action stream.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
for rel in ("training", "code/xpolicylab/policy", "code/xpolicylab/policy/OpenVLA_R42", "code/xpolicylab/policy/OpenVLA_Pure"):
    path = str(ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import pytest

import m2_deploy_contract as C
import m2_deploy_server as S
import r42_adapter as R
from OpenVLA_R42.model_m2 import Model, M2ModelError
from test_r25_m2_deploy_contract import case as case
from test_r25_m2_deploy_server import FakeBackend, images


def _meta(case, task="stack_bowls"):
    args = dict(case)
    args["task"] = task
    return C.build_session_metadata(**args)[0]


def test_unauthorised_cpu_backend_fails_before_action(case):
    """An injected fake with no explicit test intent cannot use the wire model."""
    meta = _meta(case)
    with pytest.raises(M2ModelError, match="stub|test|authori|inject"):
        Model({"task_id": "stack_bowls", "metadata": meta,
               "backend": FakeBackend(), "device": "cpu",
               "tokenizer_prompt_ids": [5, 10], "expected_hw": None})


def test_cpu_fake_not_unlocked_by_real_7b_smoke_flag(case):
    """GPU smoke flag is NEVER an alias for allowing fake actions."""
    meta = _meta(case)
    with pytest.raises(M2ModelError, match="stub|test|authori|inject"):
        Model({"task_id": "stack_bowls", "metadata": meta,
               "backend": FakeBackend(), "device": "cpu",
               "allow_real_7b_smoke": True,
               "tokenizer_prompt_ids": [5, 10], "expected_hw": None})


def test_explicit_cpu_contract_test_can_act(case):
    """An explicit CPU-only contract-test fixture retains 14D functionality."""
    meta = _meta(case)
    model = Model({"task_id": "stack_bowls", "metadata": meta,
                   "backend": FakeBackend(), "device": "cpu",
                   "allow_stub_action_for_contract_test": True,
                   "tokenizer_prompt_ids": [5, 10], "expected_hw": None})
    assert model.contract()["strict_task_binding"] is True


def test_cv_environment_requires_optin(monkeypatch, case):
    """Environment inheritance must not silently turn a VLA run into CV."""
    meta = _meta(case, "push_T")
    srv = S.M2TaskDeployServer(
        meta, task="push_T", backend=FakeBackend(),
        device="cpu", tokenizer_prompt_ids=[8, 10])
    monkeypatch.setenv("R42_POLICY", "cv")
    with pytest.raises((ValueError, RuntimeError), match="CV|cv|diagnostic|override"):
        srv.act(images(), np.zeros(14, np.float32), timestamp=0.04)


def test_explicit_cv_diagnostic_preserves_task_codec(monkeypatch, case):
    meta = _meta(case, "stack_bowls")
    srv = S.M2TaskDeployServer(
        meta, task="stack_bowls", backend=FakeBackend(),
        device="cpu", tokenizer_prompt_ids=[8, 10],
        allow_cv_diagnostic=True)
    monkeypatch.setenv("R42_POLICY", "cv")
    result = srv.act(images(), np.zeros(14, np.float32), timestamp=0.04)
    want = R.token_ids_for_residual(
        np.zeros(14, np.float32),
        srv.session.normalizer["q01"],
        srv.session.normalizer["q99"])[1]
    assert result["policy_mode"] == "cv_diagnostic"
    assert np.array_equal(result["ids"], want)


def test_normal_m2_inference_reports_vla_policy_mode(monkeypatch, case):
    monkeypatch.delenv("R42_POLICY", raising=False)
    meta = _meta(case)
    srv = S.M2TaskDeployServer(meta, task="stack_bowls",
                                backend=FakeBackend(), device="cpu",
                                tokenizer_prompt_ids=[9, 11])
    out = srv.act(images(), np.zeros(14, np.float32), timestamp=1.0)
    assert out["policy_mode"] == "vla"


def test_replay_without_rgb_refused_for_real_processor(case):
    """Recorded fake output is NOT multimodal-7B replay without saved RGB."""
    meta = _meta(case)
    srv = S.M2TaskDeployServer(meta, task="stack_bowls",
                               backend=FakeBackend(), device="cpu",
                               tokenizer_prompt_ids=[9, 11])
    record = srv.act(images(), np.zeros(14, np.float32),
                     timestamp=1.0, record=True)["log"]
    # A real server has a processor.  Its replay must never use pixel_values=None
    # while advertising faithful visual model reproduction.
    srv.processor = object()
    with pytest.raises((ValueError, RuntimeError), match="image|RGB|replay|vision"):
        srv.replay(record)


def test_legacy_standard_launcher_remains_r15_not_m2():
    """A static negative contract: default launcher has NOT deployed M2."""
    root = Path(__file__).resolve().parents[1]
    path = root / "code/xpolicylab/policy/OpenVLA_R42"
    server = (path / "r42_serve.py").read_text()
    deploy = (path / "deploy.yml").read_text()
    assert "from OpenVLA_R42.model import Model" in server
    assert "from OpenVLA_R42.model_m2 import Model" not in server
    assert "task_name: push_T" in deploy


def test_stage_must_contain_actual_trained_prompt_token_hashes(case):
    """Text/keyword equality is NOT evidence of tokenizer-sequence parity."""
    import json
    p = Path(case["stage_evidence_path"])
    original = json.loads(p.read_text())
    original.pop("prompt_token_ids_sha256", None)
    p.write_text(json.dumps(original))
    with pytest.raises((ValueError, RuntimeError), match="prompt.*token|sha256|hash|evidence"):
        C.build_session_metadata(**case)


def test_session_meta_binds_trained_task_token_ids(case):
    """The staging report must preserve the processor's exact token-id hash."""
    import json
    stage = json.loads(Path(case["stage_evidence_path"]).read_text())
    meta = _meta(case, "stack_bowls")
    assert meta["trained_prompt_token_ids_sha256"] == (
        stage["prompt_token_ids_sha256"]["stack_bowls"])


def test_wrong_training_prompt_ids_fail_closed_before_model_decode(monkeypatch, case):
    meta = _meta(case)
    srv = S.M2TaskDeployServer(meta, task="stack_bowls", backend=FakeBackend(),
                               device="cpu")
    # The real implementation creates these through PurePromptBuilder, but the
    # CPU test isolates the hash guard with a deliberately WRONG tokenizer.
    monkeypatch.setattr(S.MT, "_make_prompt",
                        lambda proc, task, include_task_token: [123, 999])
    with pytest.raises((ValueError, RuntimeError), match="prompt.*token|sha256|hash|mismatch"):
        srv.build_prompt(object())


def test_matching_training_prompt_ids_admitted(monkeypatch, case):
    meta = _meta(case, "stack_bowls")
    srv = S.M2TaskDeployServer(meta, task="stack_bowls", backend=FakeBackend(),
                               device="cpu")
    trained = R.normalize_prompt_ids([100, 222, 301])
    monkeypatch.setattr(S.MT, "_make_prompt",
                        lambda proc, task, include_task_token: trained)
    assert srv.build_prompt(object()) == trained


def test_model_payload_marks_normal_vla_mode(monkeypatch, case):
    """The WEBSOCKET-visible payload must identify the policy actually used."""
    import m2_multitask_data as MD
    from test_openvla_r42_server import _obs
    monkeypatch.delenv("R42_POLICY", raising=False)
    meta = _meta(case, "stack_bowls")
    model = Model({"task_id": "stack_bowls", "metadata": meta,
                   "backend": FakeBackend(), "device": "cpu",
                   "allow_stub_action_for_contract_test": True,
                   "tokenizer_prompt_ids": [9, 20], "expected_hw": None})
    model.update_obs(_obs(instruction=MD.TASK_PROMPTS["stack_bowls"]))
    payload = model.get_action()
    assert payload["policy_mode"] == "vla"
    assert payload["serving_task"] == "stack_bowls"


def test_model_requires_config_optin_for_inherited_cv(monkeypatch, case):
    import m2_multitask_data as MD
    from test_openvla_r42_server import _obs
    monkeypatch.setenv("R42_POLICY", "cv")
    meta = _meta(case, "push_T")
    def make(allow):
        return Model({"task_id": "push_T", "metadata": meta,
                      "backend": FakeBackend(), "device": "cpu",
                      "allow_stub_action_for_contract_test": True,
                      "allow_cv_diagnostic": allow,
                      "tokenizer_prompt_ids": [9, 20], "expected_hw": None})
    denied = make(False)
    denied.update_obs(_obs(instruction=MD.TASK_PROMPTS["push_T"]))
    with pytest.raises((ValueError, RuntimeError), match="CV|cv|diagnostic"):
        denied.get_action()
    enabled = make(True)
    enabled.update_obs(_obs(instruction=MD.TASK_PROMPTS["push_T"]))
    res = enabled.get_action()
    assert res["policy_mode"] == "cv_diagnostic"
    assert res["serving_task"] == "push_T"
