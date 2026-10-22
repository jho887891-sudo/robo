#!/usr/bin/env python3
"""R16 T8/T9: full local WebSocket end to end, and the real latency percentiles.

T8 starts the actual XPolicyLab PolicyServer (RoboDojo/XPolicyLab/client_server/ws/model_server.py)
in a background event loop, binds it to an ephemeral port, and drives it with the actual
WsModelClient the RoboDojo env uses.  That is the only test in this lane that proves the wire:
msgpack frames, update_obs/get_action, reset, the 4-key action dict arriving at the client, and
the four server latency phases surviving the round trip.

T9 collects 200 steps (after a discarded warm-up) through the same socket and reports the
p50/p95/p99 of every phase plus the client-side RTT that the shipped stack never measured.

Needs: websockets + msgpack + msgpack_numpy + pydantic (the venv_oft runtime) and the RoboDojo
tree for client_server.  No GPU, no model, no simulator.
"""
import json
import os
import sys
import threading

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_ROOT = os.path.join(REPO, "code", "xpolicylab", "policy")
POLICY_DIR = os.path.join(POLICY_ROOT, "OpenVLA_R42")
sys.path.insert(0, POLICY_ROOT)          # import OpenVLA_R42 as a package
sys.path.insert(0, POLICY_DIR)           # import r42_adapter / r42_env_io flat
sys.path.insert(0, os.path.join(POLICY_ROOT, "OpenVLA_Pure"))

ROBODOJO = os.environ.get("ROBODOJO_ROOT", "/home/T7/ojh/robot_actual/RoboDojo")
for _p in (ROBODOJO, os.path.join(ROBODOJO, "XPolicyLab")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

pytest.importorskip("websockets")
pytest.importorskip("msgpack")
pytest.importorskip("pydantic")

import r42_adapter as R  # noqa: E402
import r42_env_io as E  # noqa: E402
import OpenVLA_R42.model as M  # noqa: E402

try:
    import asyncio

    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from client_server.ws.model_client import WsModelClient
    from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
    from client_server.ws.protocol.exceptions import WsError
    from client_server.ws.protocol.messages import MessageType
    HAVE_ROBODOJO = True
except Exception as exc:                       # pragma: no cover - environment dependent
    HAVE_ROBODOJO = False
    IMPORT_ERROR = repr(exc)

pytestmark = pytest.mark.skipif(
    not HAVE_ROBODOJO,
    reason="RoboDojo client_server not importable (set ROBODOJO_ROOT); %s"
           % ("" if HAVE_ROBODOJO else IMPORT_ERROR))

CONTRACT = os.path.join(REPO, "evidence", "r9_gate_20261020", "frozen_contract_64train.json")
BOUNDS = os.path.join(REPO, "data", "push_T_bounds.npz")
PAD, STOP = 32000, 2
PROMPT_IDS = [128000, 9906, 11, 5123, 29901]
WARMUP_STEPS = 10
MEASURED_STEPS = 200


class StubBackend(object):
    def __init__(self, seed=0, vocab=32064):
        self.seed, self.vocab = seed, vocab
        self.calls = 0

    def next_token_logits(self, input_ids, pixel_values=None, proprio=None):
        self.calls += 1
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


def _inside(alpha=0.5):
    nb = np.load(BOUNDS)
    lo = np.asarray(nb["amin"], np.float32)
    hi = np.asarray(nb["amax"], np.float32)
    return (lo + float(alpha) * (hi - lo)).astype(np.float32)


def _obs(seed=0, env_idx=0, alpha=0.5):
    state = _inside(alpha)
    rs = np.random.RandomState(seed)
    return {"vision": {cam: {"color": rs.randint(0, 255, (24, 32, 3)).astype(np.uint8),
                             "shape": (24, 32, 3)}
                       for cam in R.CAMERA_ORDER},
            "state": {k: state[a:b] for k, a, b in
                      zip(E.JOINT_ORDER, (0, 6, 7, 13), (6, 7, 13, 14))},
            "action": {"left_arm_joint_state": [0.0] * 6},
            "instruction": M.INSTRUCTION,
            "additional_info": {"frequency": 25},
            "data_format_version": "v1.0",
            "env_idx": env_idx}


def _model():
    return M.Model({"policy_name": "OpenVLA_R42", "metadata": _meta(),
                    "backend": StubBackend(), "device": "cpu", "prompt": M.INSTRUCTION,
                    "expected_hw": None, "control_hz": 25,
                    "tokenizer_prompt_ids": PROMPT_IDS})


class _ServerThread(object):
    def __init__(self, model):
        self.loop = asyncio.new_event_loop()
        self.server = PolicyServer(model, PolicyServerConfig(
            host="127.0.0.1", port=0, ws_ping_interval_s=None, ws_ping_timeout_s=None))
        self.loop.run_until_complete(self.server.start())
        self.url = self.server.url
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def stop(self):
        try:
            asyncio.run_coroutine_threadsafe(self.server.stop(), self.loop).result(timeout=10)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)


@pytest.fixture()
def served():
    srv = _ServerThread(_model())
    yield srv
    srv.stop()


# ------------------------------------------------------------------------------- T8: the wire
def test_full_ws_end_to_end_produces_the_four_key_action_dict(served):
    client = WsModelClient(url=served.url, evaluation_id="r42-e2e", trial_id="trial-0",
                          ws_ping_interval_s=None, ws_ping_timeout_s=None)
    try:
        assert client.call(func_name="reset") is True
        obs = _obs()
        client.call(func_name="update_obs", obs=obs)
        action = client.call(func_name="get_action")
    finally:
        client.close()
    assert isinstance(action, dict)
    assert list(action) == list(E.JOINT_ORDER)
    assert [len(action[k]) for k in E.JOINT_ORDER] == [6, 1, 6, 1]
    flat = E.action_dict_to_cmd14(action)
    assert np.isfinite(flat).all()
    assert 0.0 <= flat[6] <= 1.0 and 0.0 <= flat[13] <= 1.0


def test_ws_reset_between_episodes_keeps_the_wire_alive(served):
    client = WsModelClient(url=served.url, evaluation_id="r42-e2e", trial_id="trial-1",
                          ws_ping_interval_s=None, ws_ping_timeout_s=None)
    try:
        for episode in range(3):
            assert client.call(func_name="reset") is True
            client.call(func_name="update_obs", obs=_obs(seed=episode))
            action = client.call(func_name="get_action")
            assert len(action["left_arm_joint_state"]) == 6
    finally:
        client.close()


def test_the_four_server_phases_and_the_client_rtt_survive_the_round_trip(served):
    import r42_ws_timing as T
    client = T.R42WsTimingClient(url=served.url, evaluation_id="r42-e2e", trial_id="trial-2")
    try:
        client.reset()
        sample = client.infer(_obs())
    finally:
        client.close()
    assert sorted(sample["phases"]) == sorted(T.PHASES)
    for phase, value in sample["phases"].items():
        assert np.isfinite(value) and value >= 0.0, phase
    assert sample["client_rtt_ms"] > 0.0
    assert sample["server_span_ms"] is not None and sample["server_span_ms"] >= 0.0
    assert sample["net_ms"] is not None
    assert np.array_equal(sample["action14"], E.action_dict_to_cmd14(sample["action_dict"]))
    assert sample["n_action_tokens"] == 14
    # both are wall-clock measurements of different windows; neither may be negative
    assert sample["phases"]["total"] >= 0.0 and sample["client_rtt_ms"] >= 0.0
    assert abs(sample["clock_offset_ms"]) < 60_000.0     # same box: skew must be small


def test_ws_survives_a_malformed_infer_frame(served):
    """T10 over the wire: a frame without an observation is an error reply, not a dead server."""
    client = PolicyEvalClient(PolicyEvalClientConfig(
        url=served.url, evaluation_id="r42-e2e", ws_ping_interval_s=None,
        ws_ping_timeout_s=None))
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(client.connect(handshake=True))
        with pytest.raises(WsError):
            loop.run_until_complete(client.request(MessageType.INFER, {}))
        # the connection is still usable afterwards
        frame = loop.run_until_complete(client.infer(_obs()))
        assert frame.payload["n_action_tokens"] == 14
    finally:
        loop.run_until_complete(client.close())
        loop.close()


def test_ws_rejects_a_bad_observation_without_killing_the_server(served):
    """WsModelClient only caches update_obs locally; the server is what validates it."""
    client = WsModelClient(url=served.url, evaluation_id="r42-e2e", trial_id="trial-3",
                          ws_ping_interval_s=None, ws_ping_timeout_s=None)
    try:
        bad = _obs()
        bad["vision"]["cam_side"] = bad["vision"]["cam_head"]
        assert client.call(func_name="update_obs", obs=bad) is None
        with pytest.raises(Exception) as err:
            client.call(func_name="get_action")
        assert "cam_side" in str(err.value) or "unexpected camera" in str(err.value)
    finally:
        client.close()
    client = WsModelClient(url=served.url, evaluation_id="r42-e2e", trial_id="trial-4",
                          ws_ping_interval_s=None, ws_ping_timeout_s=None)
    try:
        client.call(func_name="update_obs", obs=_obs())
        assert len(client.call(func_name="get_action")["left_arm_joint_state"]) == 6
    finally:
        client.close()


# ------------------------------------------------------------------------ T9: latency numbers
def test_latency_percentiles_over_200_steps_with_warmup_excluded(served):
    import r42_ws_timing as T
    client = T.R42WsTimingClient(url=served.url, evaluation_id="r42-percentiles",
                                 trial_id="trial-p")
    try:
        client.reset()
        for _ in range(WARMUP_STEPS):               # excluded from the statistics
            client.infer(_obs())
        client.recorder.reset()
        client.samples = []
        for step in range(MEASURED_STEPS):
            client.infer(_obs(seed=step % 7), step=step)
        summary = client.summary()
    finally:
        client.close()

    assert summary["n_samples"] == MEASURED_STEPS
    assert summary["control_hz"] == pytest.approx(25.0)
    assert summary["budget_ms"] == pytest.approx(40.0)
    for phase in T.PHASES:
        stats = summary["phases"][phase]
        assert stats["n"] == MEASURED_STEPS
        assert stats["min"] <= stats["p50"] <= stats["p95"] <= stats["p99"] <= stats["max"]
        assert np.isfinite(stats["p50"]) and stats["p50"] >= 0.0
    assert summary["server_span_ms"]["n"] == MEASURED_STEPS
    assert summary["server_span_ms"]["p50"] <= summary["server_span_ms"]["p99"]
    assert summary["net_ms"]["n"] >= MEASURED_STEPS - 1
    assert summary["client_steps"] == MEASURED_STEPS
    assert summary["client_rtt_ms"]["n"] == MEASURED_STEPS
    assert summary["client_rtt_ms"]["p50"] <= summary["client_rtt_ms"]["p95"]         <= summary["client_rtt_ms"]["p99"]
    print("[T9] budget=%.1fms n=%d | server total p50=%.3f p95=%.3f p99=%.3f ms | "
          "client rtt p50=%.3f p95=%.3f p99=%.3f ms | net p50=%.3f ms"
          % (summary["budget_ms"], summary["n_samples"],
             summary["phases"]["total"]["p50"], summary["phases"]["total"]["p95"],
             summary["phases"]["total"]["p99"],
             summary["client_rtt_ms"]["p50"], summary["client_rtt_ms"]["p95"],
             summary["client_rtt_ms"]["p99"], summary["net_ms"]["p50"]))
