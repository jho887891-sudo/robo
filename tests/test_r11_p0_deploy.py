#!/usr/bin/env python3
"""R11 P0-4 (RED first): real 7B deployment server skeleton contract.

No 7B weights are loaded here (CPU only): a deterministic stub backend is injected so that the
observable contract of the server can be tested without pretending to be a GPU smoke test:
  * metadata (with frozen normalizer + SHA) is mandatory,
  * three RGB cameras in the frozen order, RGBA/BGR refused,
  * 14 autoregressive action tokens, argmax restricted to 31744..31999 (never PAD/STOP),
  * physical bounds + rate rule enforced through the shared decode path,
  * per-env history/reset isolation,
  * replay performs no update (all weights frozen / requires_grad False).
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_R42"))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))

import r42_adapter as R  # noqa: E402
import r42_deploy_server as S  # noqa: E402

CONTRACT = os.path.join(REPO, "evidence", "r9_gate_20261020", "frozen_contract_64train.json")
PAD = 32000
STOP = 2


class StubBackend:
    """Deterministic stand-in with the same call signature as the real 7B backend."""

    def __init__(self, seed=0, bait=True):
        self.seed = seed
        self.vocab = 32064
        self.bait = bait
        self.calls = 0
        self.weight = np.zeros(8, np.float32)     # any in-place update is detected

    def next_token_logits(self, input_ids, pixel_values=None, proprio=None):
        """Deterministic pure function of (seed, input_ids, proprio): replay must reproduce it."""
        self.calls += 1
        sig = (self.seed * 1000003
               + sum((i + 1) * int(x) for i, x in enumerate(input_ids))) % (2 ** 31 - 1)
        if proprio is not None:
            sig = (sig + int(round(float(np.abs(proprio).sum()) * 1e6))) % (2 ** 31 - 1)
        logits = (np.random.RandomState(sig).randn(self.vocab) * 0.01).astype(np.float32)
        if self.bait:
            logits[PAD] = 500.0                   # would win if the window were not enforced
            logits[STOP] = 400.0
        return logits


def _meta():
    frozen = json.load(open(CONTRACT))
    norm = R.normalizer_payload(
        normalization_mode=frozen["normalization_mode"], pmin=frozen["pmin"], pmax=frozen["pmax"],
        q01=frozen["q01"], q99=frozen["q99"], v_scale=frozen["v_scale"],
        a_scale=frozen["a_scale"], motion_quantile=frozen["motion_quantile"],
        n_train_episodes=64, n_dev_episodes_excluded=16)
    nb = np.load(os.path.join(REPO, "data", "push_T_bounds.npz"))
    sha = "c" * 64
    return R.build_metadata("push_T", sha, sha, sha, sha, sha, train_steps=0, seed=0,
                            gripper_policy=frozen["gripper_policy"],
                            safety_limits={"amin": nb["amin"].tolist(),
                                           "amax": nb["amax"].tolist(),
                                           "max_rate": [0.5] * 14},
                            normalizer=norm)


def _images(shape=(64, 80, 3), dtype=np.uint8):
    rs = np.random.RandomState(1)
    return {cam: rs.randint(0, 255, shape).astype(dtype) for cam in R.CAMERA_ORDER}


@pytest.fixture()
def server():
    be = StubBackend()
    srv = S.R42DeployServer(_meta(), backend=be, device="cpu",
                            tokenizer_prompt_ids=[128000, 9906, 11, 5123, 29901])
    srv.reset(0)
    return srv


def test_metadata_is_mandatory_and_frozen():
    bad = _meta()
    bad.pop("normalizer")
    with pytest.raises(R.R42Error):
        S.R42DeployServer(bad, backend=StubBackend(), device="cpu")
    srv = S.R42DeployServer(_meta(), backend=StubBackend(), device="cpu")
    assert srv.assert_frozen() is True
    assert srv.n_action_tokens == 14
    assert srv.action_token_window == (31744, 31999)


def test_rejects_rgba_and_bgr(server):
    rgba = {cam: np.zeros((8, 8, 4), np.uint8) for cam in R.CAMERA_ORDER}
    with pytest.raises(R.R42Error):
        server.act(rgba, np.zeros(14, np.float32), env_idx=0)
    bgr = dict(_images())
    bgr["channel_order"] = "BGR"
    with pytest.raises(R.R42Error):
        server.act(bgr, np.zeros(14, np.float32), env_idx=0)
    wrong_order = dict(reversed(list(_images().items())))
    with pytest.raises(R.R42Error):
        server.act(wrong_order, np.zeros(14, np.float32), env_idx=0)


def test_fourteen_tokens_within_window(server):
    out = server.act(_images(), np.zeros(14, np.float32), env_idx=0)
    assert out["n_action_tokens"] == 14
    assert out["bins"].shape == (14,)
    assert out["bins"].min() >= 0 and out["bins"].max() <= 255
    assert out["action"].shape == (14,)
    assert np.isfinite(out["action"]).all()
    assert out["ids"].min() >= 31744 and out["ids"].max() <= 31999
    assert PAD not in set(int(x) for x in out["ids"])
    assert STOP not in set(int(x) for x in out["ids"])


def test_prompt_has_the_space_suffix_and_15_supervision_slots(server):
    assert server.prompt_ids()[-1] == 29871
    assert server.n_supervised == 15


def test_gripper_dims_use_copy_velocity(server):
    st0 = np.zeros(14, np.float32)
    server.act(_images(), st0, env_idx=0)
    st1 = st0 + 0.01
    out = server.act(_images(), st1, env_idx=0)
    v = out["velocity"]
    for d in R.GRIPPER_DIMS:
        assert abs(float(out["action"][d]) - float(st1[d] + v[d])) <= 1e-6


def test_physical_bounds_are_enforced(server):
    out = server.act(_images(), np.full(14, 1e6, np.float32), env_idx=0)
    lim = server.metadata["safety_limits"]
    assert (out["action"] >= np.asarray(lim["amin"], np.float32) - 1e-6).all()
    assert (out["action"] <= np.asarray(lim["amax"], np.float32) + 1e-6).all()


def test_env_isolation_and_reset(server):
    a = server.act(_images(), np.zeros(14, np.float32), env_idx=0)
    b = server.act(_images(), np.zeros(14, np.float32) + 0.01, env_idx=1)
    assert a["features"].shape == (42,) and b["features"].shape == (42,)
    assert not np.array_equal(a["features"], b["features"])
    server.reset(0)
    with pytest.raises(R.R42Error):
        server.history_features(0)
    assert np.isfinite(server.history_features(1)).all()


def test_replay_is_deterministic_and_updates_nothing(server):
    be = server.backend
    before = be.weight.copy()
    imgs = _images()
    st = np.zeros(14, np.float32)
    rec = server.act(imgs, st, env_idx=0, record=True)
    server.reset(0)
    rep = server.replay(rec["log"])
    assert np.array_equal(rec["ids"], rep["ids"])
    assert np.array_equal(rec["action"], rep["action"])
    assert np.array_equal(be.weight, before), "replay must not touch any weight"
    assert server.assert_frozen() is True


def test_latency_is_recorded(server):
    out = server.act(_images(), np.zeros(14, np.float32), env_idx=0)
    lat = out["latency_ms"]
    for key in ("preprocess", "forward_total", "decode", "total"):
        assert key in lat and lat[key] >= 0.0
    assert lat["forward_total"] >= 0.0
