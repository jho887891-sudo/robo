"""RED: RoboDojo ModelTemplate must bind the explicit M2 task, not legacy R15."""
import json
import os

import numpy as np
import pytest

import m2_deploy_contract as C
import m2_multitask_data as MD
from test_r25_m2_deploy_contract import case as case
from test_r25_m2_deploy_server import FakeBackend
from test_openvla_r42_server import _obs

from OpenVLA_R42.model_m2 import Model, M2ModelError


def model(case, task="stack_bowls", **config):
    args = dict(case)
    args["task"] = task
    meta, report = C.build_session_metadata(**args)
    opts = {"task_id": task, "metadata": meta,
            "backend": FakeBackend(), "device": "cpu",
            "tokenizer_prompt_ids": [100, 222, 333],
            "control_hz": 25, "expected_hw": None}
    opts.update(config)
    return Model(opts), report


def test_model_template_cpu_stub_one_step_stack_bowls(case):
    m, report = model(case)
    assert m._server is None
    m.update_obs(_obs(instruction=MD.TASK_PROMPTS["stack_bowls"]))
    out = m.get_action()
    assert out["n_action_tokens"] == 14
    assert len(out["ids"]) == 14
    assert len(out["action14"]) == 14
    assert list(out["actions"]) == [
        "left_arm_joint_state", "left_ee_joint_state",
        "right_arm_joint_state", "right_ee_joint_state"]
    assert m.contract()["task_id"] == "stack_bowls"
    assert m._server.task == "stack_bowls"
    assert m.reset() is True


def test_wrong_instruction_is_rejected_not_warned_and_ignored(case):
    m, _ = model(case)
    with pytest.raises((ValueError, RuntimeError), match="instruction|task"):
        m.update_obs(_obs(instruction=MD.TASK_PROMPTS["push_T"]))
    assert m._server is None


def test_wrong_case_task_is_rejected(case):
    m, _ = model(case)
    with pytest.raises((ValueError, RuntimeError), match="task"):
        m.prepare_case({"task": "push_T"})


def test_missing_metadata_cannot_fallback_to_r15_bridge():
    with pytest.raises((ValueError, RuntimeError), match="metadata"):
        Model({"task_id": "stack_bowls", "device": "cpu",
               "backend": FakeBackend(), "tokenizer_prompt_ids": [10, 20]})


def test_missing_explicit_task_is_rejected_even_with_metadata(case):
    m, _ = model(case)
    with pytest.raises((ValueError, RuntimeError), match="task"):
        Model({"metadata": m.metadata, "device": "cpu",
               "backend": FakeBackend(), "tokenizer_prompt_ids": [10, 20]})


def test_bundle_metadata_path_is_supported(case, tmp_path):
    m, _ = model(case)
    bundle = tmp_path / "m2_session.json"
    bundle.write_text(json.dumps({"metadata": m.metadata,
                                  "attestation": {"no_optimizer_steps": True}}))
    inst = Model({"task_id": "stack_bowls", "metadata_path": str(bundle),
                  "backend": FakeBackend(), "device": "cpu",
                  "tokenizer_prompt_ids": [10, 20], "expected_hw": None})
    inst.update_obs(_obs(instruction=MD.TASK_PROMPTS["stack_bowls"]))
    assert inst.get_action()["n_action_tokens"] == 14


def test_real_gpu_path_is_explicitly_blocked_without_smoke_authorisation(case):
    m, _ = model(case)
    cfg = {"task_id": "stack_bowls", "metadata": m.metadata,
           "device": "cuda", "expected_hw": None}
    inst = Model(cfg)
    with pytest.raises((ValueError, RuntimeError), match="smoke|authoris|GPU|backend"):
        inst.warmup()
