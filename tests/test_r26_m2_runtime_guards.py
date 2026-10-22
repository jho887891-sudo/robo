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
