"""TDD RED: M2 per-task stub inference must not leak R15 fixed prompt/normalizer.

No CUDA, HDF5 or model weights; real 2-task normalizer from the repo.
"""
import numpy as np
import pytest

import r42_adapter as R
from test_r25_m2_deploy_contract import case as case
import m2_deploy_contract as C
import m2_deploy_server as S
import m2_multitask_data as MD


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.kv_cache_enabled = False
        self.vision_cache_enabled = False

    def next_token_logits(self, input_ids, pixel_values, proprio):
        f = np.asarray(proprio, np.float32)
        assert f.shape == (42,)
        self.calls.append((list(input_ids), f.copy()))
        token = 31744 + (sum(input_ids) + int(f[0] * 100)) % 256
        logits = np.zeros(32064, np.float32)
        logits[2], logits[32000] = 1000., 1000.  # not legal action tokens
        logits[token] = 20.
        return logits


def server(case, task, ids, allow_cv_diagnostic=False):
    case["task"] = task
    meta, att = C.build_session_metadata(**case)
    assert att["server_status"].startswith("BLOCKED")
    backend = FakeBackend()
    service = S.M2TaskDeployServer(meta, task=task, backend=backend,
                                   device="cpu", tokenizer_prompt_ids=ids,
                                   allow_cv_diagnostic=allow_cv_diagnostic)
    return service, backend


def images():
    return {cam: np.zeros((8, 8, 3), dtype=np.uint8) for cam in R.CAMERA_ORDER}


def test_two_tasks_use_distinct_prompt_ids_and_history(case):
    a, ab = server(case.copy(), "push_T", [5, 100])
    b, bb = server(case.copy(), "stack_bowls", [5, 101])
    assert a.prompt_ids() != b.prompt_ids()
    s = np.zeros(14, np.float32)
    out_a = a.act(images(), s, env_idx=0, timestamp=0.04)
    out_b = b.act(images(), s, env_idx=0, timestamp=0.04)
    assert out_a["action"].shape == out_b["action"].shape == (14,)
    assert out_a["features"].shape == out_b["features"].shape == (42,)
    assert len(out_a["ids"]) == len(out_b["ids"]) == 14
    assert np.array_equal(ab.calls[0][1], out_a["features"])
    assert np.array_equal(bb.calls[0][1], out_b["features"])
    assert a.task == "push_T" and b.task == "stack_bowls"
    assert out_a["served_task"] == "push_T"
    assert out_b["served_task"] == "stack_bowls"


def test_m2_decoder_uses_task_specific_quantiles(case):
    x, _ = server(case.copy(), "stack_bowls", [88, 99])
    state = np.zeros(14, np.float32)
    res = x.act(images(), state, timestamp=0.04)
    manual = C.M2SessionContract(x.metadata, "stack_bowls")
    expected = manual.decode(res["bins"], state, np.zeros(14, np.float32))
    assert np.array_equal(res["action"], expected[0])
    assert np.array_equal(res["residual"], expected[1])
    assert list(res["latency_ms"]) == ["preprocess", "forward_total", "decode", "total"]


def test_model_must_reject_wrong_raw_observation_instruction(case):
    x, _ = server(case.copy(), "stack_bowls", [44, 55])
    x.assert_instruction(MD.TASK_PROMPTS["stack_bowls"])
    with pytest.raises((ValueError, RuntimeError), match="instruction|task"):
        x.assert_instruction(MD.TASK_PROMPTS["push_T"])


def test_incorrect_task_binding_fails_before_backend(case):
    a, _ = server(case.copy(), "push_T", [5, 100])
    with pytest.raises((ValueError, RuntimeError), match="task"):
        S.M2TaskDeployServer(a.metadata, task="stack_bowls",
                             backend=FakeBackend(), device="cpu",
                             tokenizer_prompt_ids=[5, 101])


def test_cv_policy_uses_selected_task_codec(monkeypatch, case):
    x, backend = server(case.copy(), "stack_bowls", [45, 90],
                        allow_cv_diagnostic=True)
    monkeypatch.setenv("R42_POLICY", "cv")
    state = np.zeros(14, np.float32)
    out = x.act(images(), state, timestamp=0.04)
    expected_ids = R.token_ids_for_residual(
        np.zeros(14, np.float32),
        x.session.normalizer["q01"],
        x.session.normalizer["q99"])[1]
    assert np.array_equal(out["ids"], expected_ids)
    assert not backend.calls
    assert all(x["policy"] == "cv_baseline" for x in out["step_logs"])


def test_reset_and_replay_do_not_leak_between_tasks(case):
    x, _ = server(case.copy(), "push_T", [51, 52])
    state = np.zeros(14, np.float32)
    out = x.act(images(), state, timestamp=1.0, record=True)
    assert out["log"]["task"] == "push_T"
    again = x.replay(out["log"])
    assert np.array_equal(out["ids"], again["ids"])
    assert np.array_equal(out["action"], again["action"])
    x.reset()
    assert x.session.history._hist == {}
    with pytest.raises((ValueError, RuntimeError), match="task"):
        x.replay(dict(out["log"], task="stack_bowls"))
