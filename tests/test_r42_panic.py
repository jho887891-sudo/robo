#!/usr/bin/env python3
"""R16 T10: panic paths.  Every one of these must fail closed and loudly, on CPU, with no model.

The rule under test is the deployment rule: a policy server that guesses is worse than one that
refuses, because a silently wrong 42D command looks exactly like a correct one until the robot
misses the block.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy"))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_R42"))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))

import r42_adapter as R  # noqa: E402
import r42_deploy_server as S  # noqa: E402
import r42_env_io as E  # noqa: E402
import r42_metadata_bridge as B  # noqa: E402
import OpenVLA_R42.model as M  # noqa: E402

CONTRACT = os.path.join(REPO, "evidence", "r9_gate_20261020", "frozen_contract_64train.json")
BOUNDS = os.path.join(REPO, "data", "push_T_bounds.npz")
PROMPT_IDS = [128000, 9906, 11, 5123, 29901]
PAD, STOP = 32000, 2


class StubBackend(object):
    def __init__(self, vocab=32064, bail=False):
        self.vocab, self.bail = vocab, bail
        self.calls = 0

    def next_token_logits(self, input_ids, pixel_values=None, proprio=None):
        self.calls += 1
        if self.bail:
            raise RuntimeError("backend exploded")
        logits = np.zeros(self.vocab, np.float32)
        if self.vocab > PAD:
            logits[PAD] = 500.0      # bait: the argmax outside the window is PAD
        if self.vocab > STOP:
            logits[STOP] = 400.0     # bait: and STOP
        if self.vocab > 31744 + 256:
            idx = (sum(int(x) for x in input_ids) + self.calls) % 256
            logits[31744 + idx] = 1.0    # a legal token inside 31744..31999
        return logits


def _meta(**over):
    frozen = json.load(open(CONTRACT))
    norm = R.normalizer_payload(
        normalization_mode=frozen["normalization_mode"], pmin=frozen["pmin"],
        pmax=frozen["pmax"], q01=frozen["q01"], q99=frozen["q99"],
        v_scale=frozen["v_scale"], a_scale=frozen["a_scale"],
        motion_quantile=frozen["motion_quantile"], n_train_episodes=64,
        n_dev_episodes_excluded=16)
    nb = np.load(BOUNDS)
    sha = "c" * 64
    meta = R.build_metadata(
        "push_T", sha, sha, sha, sha, sha, train_steps=0, seed=0,
        gripper_policy=frozen["gripper_policy"],
        safety_limits={"amin": nb["amin"].astype(np.float32).tolist(),
                       "amax": nb["amax"].astype(np.float32).tolist(),
                       "max_rate": [0.5] * 14},
        normalizer=norm)
    meta.update(over)
    return meta


def _inside(alpha=0.5):
    nb = np.load(BOUNDS)
    lo = np.asarray(nb["amin"], np.float32)
    hi = np.asarray(nb["amax"], np.float32)
    return (lo + float(alpha) * (hi - lo)).astype(np.float32)


def _obs(state=None, seed=0, **extra):
    st = _inside() if state is None else state
    obs = {"vision": {cam: {"color": np.random.RandomState(seed + i).randint(
        0, 255, (6, 8, 3)).astype(np.uint8)} for i, cam in enumerate(R.CAMERA_ORDER)},
        "state": {k: st[a:b] for k, a, b in
                  zip(E.JOINT_ORDER, (0, 6, 7, 13), (6, 7, 13, 14))},
        "instruction": M.INSTRUCTION,
        "additional_info": {"frequency": 25},
        "env_idx": 0}
    obs.update(extra)
    return obs


def _model(backend=None, meta=None, **over):
    cfg = {"policy_name": "OpenVLA_R42", "metadata": meta if meta is not None else _meta(),
           "backend": backend if backend is not None else StubBackend(), "device": "cpu",
           "prompt": M.INSTRUCTION, "expected_hw": None, "control_hz": 25,
           "tokenizer_prompt_ids": PROMPT_IDS}
    cfg.update(over)
    return M.Model(cfg)


# ---------------------------------------------------------------------- metadata panics
def test_a_7d_or_14d_metadata_is_refused():
    for dim in (7, 8, 14, 28):
        meta = _meta(proprio_dim=dim)
        with pytest.raises(R.R42Error):
            _model(meta=meta)


def test_metadata_with_a_broken_normalizer_sha_is_refused():
    meta = _meta(normalizer_sha256="e" * 64)
    with pytest.raises(R.R42Error, match="normalizer_sha256"):
        _model(meta=meta)


def test_metadata_with_a_legacy_alignment_version_is_refused():
    meta = _meta(alignment_version="r42-v1")
    with pytest.raises(R.R42Error, match="alignment_version"):
        _model(meta=meta)


def test_metadata_with_swapped_camera_order_is_refused():
    meta = _meta(camera_order=["cam_left_wrist", "cam_head", "cam_right_wrist"])
    with pytest.raises(R.R42Error, match="camera_order"):
        _model(meta=meta)


def test_missing_metadata_source_is_refused():
    with pytest.raises(M.R42ModelError, match="metadata"):
        M.Model({"backend": StubBackend(), "device": "cpu", "prompt": M.INSTRUCTION})


def test_nonexistent_metadata_path_is_refused():
    with pytest.raises(M.R42ModelError, match="does not exist"):
        M.Model({"metadata_path": "/nonexistent/r42_metadata.json", "device": "cpu",
                 "prompt": M.INSTRUCTION})


def test_deploy_cfg_must_be_a_dict():
    with pytest.raises(M.R42ModelError, match="dict"):
        M.Model(["not", "a", "dict"])


# ------------------------------------------------------------------------- input panics
def test_nan_state_never_reaches_the_policy():
    model = _model()
    state = _inside().copy()
    state[2] = np.nan
    with pytest.raises(E.R42EnvIOError, match="non-finite"):
        model.update_obs(_obs(state=state))
    assert model._inputs is None


def test_privileged_ground_truth_never_reaches_the_policy():
    model = _model()
    with pytest.raises(R.R42Error):
        model.update_obs(_obs(target_t=np.zeros(3, np.float32)))
    assert model._inputs is None


def test_swapped_camera_names_never_reach_the_policy():
    model = _model()
    obs = _obs()
    swapped = {"cam_head": obs["vision"]["cam_left_wrist"],
               "cam_left_wrist": obs["vision"]["cam_head"],
               "cam_right_wrist": obs["vision"]["cam_right_wrist"]}
    model.update_obs(dict(obs, vision=swapped))   # names are right, content is swapped
    # ... which the contract cannot detect by name; it detects dtype/shape problems instead:
    obs_bad = _obs()
    obs_bad["vision"]["cam_head"] = obs_bad["vision"]["cam_head"]["color"].astype(np.float32)
    with pytest.raises(E.R42EnvIOError, match="uint8"):
        model.update_obs(obs_bad)


def test_wrong_resolution_is_refused_on_the_deployment_contract():
    model = _model(expected_hw=(480, 640, 3))
    with pytest.raises(E.R42EnvIOError, match="480"):
        model.update_obs(_obs())


# ------------------------------------------------------------------------- backend panics
def test_backend_exception_becomes_a_refusal_not_a_wrong_action():
    model = _model(backend=StubBackend(bail=True))
    model.update_obs(_obs())
    with pytest.raises(RuntimeError, match="exploded"):
        model.get_action()


def test_argmax_bait_outside_the_window_is_ignored():
    model = _model()
    payloads = []
    for step in range(4):
        model.update_obs(_obs(seed=step))
        payloads.append(model.get_action())
    for payload in payloads:
        assert PAD not in payload["ids"] and STOP not in payload["ids"]
        assert min(payload["ids"]) >= 31744 and max(payload["ids"]) <= 31999


def test_backend_with_a_truncated_vocabulary_is_refused():
    model = _model(backend=StubBackend(vocab=1000))
    model.update_obs(_obs())
    with pytest.raises(R.R42Error):
        model.get_action()


def test_server_built_on_invalid_metadata_refuses_to_exist():
    with pytest.raises(R.R42Error, match="invalid metadata"):
        S.R42DeployServer({"proprio_dim": 14}, backend=StubBackend(), device="cpu")


def test_server_requires_a_backend_with_next_token_logits():
    class _Wrong(object):
        pass

    with pytest.raises(R.R42Error, match="next_token_logits"):
        S.R42DeployServer(_meta(), backend=_Wrong(), device="cpu",
                          tokenizer_prompt_ids=PROMPT_IDS)


def test_act_without_prompt_ids_is_refused():
    server = S.R42DeployServer(_meta(), backend=StubBackend(), device="cpu")
    with pytest.raises(R.R42Error, match="prompt ids"):
        server.act({cam: np.zeros((6, 8, 3), np.uint8) for cam in R.CAMERA_ORDER},
                   _inside())


def test_gripper_outside_the_unit_box_is_refused_before_the_sim_clips_it():
    cmd = np.zeros(14, np.float32)
    cmd[6] = 1.2
    with pytest.raises(E.R42EnvIOError, match="gripper"):
        E.cmd14_to_action_dict(cmd)


def test_serving_the_pristine_7b_without_an_adapter_is_refused(tmp_path):
    """A checkpoint_dir without lora_adapter must never be served as the R42 policy."""
    ck = os.path.join(str(tmp_path), "step300")
    os.makedirs(ck)
    with open(os.path.join(ck, "proprio_projector_42.pt"), "wb") as fh:
        fh.write(b"proj")
    model = M.Model({"metadata": _meta(), "checkpoint_dir": ck, "device": "cpu",
                     "prompt": M.INSTRUCTION,
                     "base_model_path": "/nonexistent/openvla-7b"})
    with pytest.raises(M.R42ModelError, match="no LoRA adapter"):
        model.warmup()


def test_serving_without_the_42d_projector_is_refused(tmp_path):
    ck = os.path.join(str(tmp_path), "step300")
    os.makedirs(os.path.join(ck, "lora_adapter"))
    model = M.Model({"metadata": _meta(), "checkpoint_dir": ck, "device": "cpu",
                     "prompt": M.INSTRUCTION,
                     "base_model_path": "/nonexistent/openvla-7b"})
    with pytest.raises(M.R42ModelError, match="42D proprio projector"):
        model.warmup()


def test_model_refuses_a_device_it_cannot_honour():
    cfg = {"metadata": _meta(), "device": "cuda", "prompt": M.INSTRUCTION,
           "base_model_path": "/nonexistent/openvla"}
    model = M.Model(cfg)
    assert model._server is None
    if not __import__("torch").cuda.is_available():
        with pytest.raises(M.R42ModelError, match="CUDA|torch"):
            model.warmup()


def test_bridge_refuses_a_checkpoint_dir_that_does_not_exist():
    with pytest.raises(B.R42BridgeError, match="does not exist"):
        B.build_stage_metadata("/nonexistent/step1")


def test_bridge_refuses_an_invalid_override_sha(tmp_path):
    ck = os.path.join(str(tmp_path), "step1")
    os.makedirs(os.path.join(ck, "lora_adapter"))
    with open(os.path.join(ck, "lora_adapter", "adapter_model.safetensors"), "wb") as fh:
        fh.write(b"lora")
    with open(os.path.join(ck, "proprio_projector_42.pt"), "wb") as fh:
        fh.write(b"proj")
    with pytest.raises(B.R42BridgeError, match="base_model_sha256"):
        B.build_stage_metadata(ck, base_model_sha256="too-short")
