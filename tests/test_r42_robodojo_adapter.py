#!/usr/bin/env python3
"""R16 (RED first): RoboDojo client <-> 42D policy-server wiring and real latency accounting.

Everything here is CPU only.  No 7B weights, no CUDA, no container, no Isaac Sim: the deploy
server runs with the deterministic stub backend already used by tests/test_r11_p0_deploy.py and
this file only asserts the *wiring contract* between what the RoboDojo client sends
(TASK_ENV.get_obs() -> model_client.call(func_name="get_action")) and what the 42D policy server
consumes:

  * three RGB cameras, resolved to the frozen order (cam_head, cam_left_wrist, cam_right_wrist)
    with no silent positional trust and no channel swap,
  * 14D finite proprio state,
  * privileged ground-truth fields (target_t / privileged_*) are refused before the policy sees
    them; they may only be read through the explicit diagnostics path,
  * 14 action tokens per control step and a 14D joint command,
  * LatencyRecorder p50/p95/p99 + control-period (25 Hz = 40 ms) violations, exact numbers,
  * per-env history/prev-command isolation and reset,
  * replay performs no weight update.

The suite is deliberately self-contained (repo convention: no cross-test imports).
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
import r42_robodojo_adapter as A  # noqa: E402

CONTRACT = os.path.join(REPO, "evidence", "r9_gate_20261020", "frozen_contract_64train.json")
BOUNDS = os.path.join(REPO, "data", "push_T_bounds.npz")
PAD = 32000
STOP = 2
BUDGET_MS = 40.0          # 25 Hz control period


# --------------------------------------------------------------------------- stub backend
class StubBackend:
    """Deterministic stand-in with the real backend call signature (see r42_deploy_server)."""

    def __init__(self, seed=0, bait=True):
        self.seed = seed
        self.vocab = 32064
        self.bait = bait
        self.calls = 0
        self.weight = np.zeros(8, np.float32)     # any in-place update is detectable

    def next_token_logits(self, input_ids, pixel_values=None, proprio=None):
        self.calls += 1
        sig = (self.seed * 1000003
               + sum((i + 1) * int(x) for i, x in enumerate(input_ids))) % (2 ** 31 - 1)
        if proprio is not None:
            sig = (sig + int(round(float(np.abs(proprio).sum()) * 1e6))) % (2 ** 31 - 1)
        logits = (np.random.RandomState(sig).randn(self.vocab) * 0.01).astype(np.float32)
        if self.bait:
            logits[PAD] = 500.0
            logits[STOP] = 400.0
        return logits


class SpyServer:
    """Wraps the deploy server and records exactly which kwargs reached the policy."""

    def __init__(self, inner):
        self.inner = inner
        self.act_calls = []

    @property
    def metadata(self):
        return self.inner.metadata

    @property
    def backend(self):
        return self.inner.backend

    def act(self, images, state, **kwargs):
        self.act_calls.append({"images": dict(images), "kwargs": dict(kwargs),
                               "state": np.asarray(state, np.float32).copy()})
        return self.inner.act(images, state, **kwargs)

    def reset(self, env_idx=0):
        return self.inner.reset(env_idx)

    def replay(self, log):
        return self.inner.replay(log)

    def history_features(self, env_idx=0):
        return self.inner.history_features(env_idx)


def _meta(max_rate=(0.5,) * 14):
    frozen = json.load(open(CONTRACT))
    norm = R.normalizer_payload(
        normalization_mode=frozen["normalization_mode"], pmin=frozen["pmin"], pmax=frozen["pmax"],
        q01=frozen["q01"], q99=frozen["q99"], v_scale=frozen["v_scale"],
        a_scale=frozen["a_scale"], motion_quantile=frozen["motion_quantile"],
        n_train_episodes=64, n_dev_episodes_excluded=16)
    nb = np.load(BOUNDS)
    sha = "c" * 64
    limits = {"amin": nb["amin"].tolist(), "amax": nb["amax"].tolist()}
    if max_rate is not None:
        limits["max_rate"] = list(max_rate)
    return R.build_metadata("push_T", sha, sha, sha, sha, sha, train_steps=0, seed=0,
                            gripper_policy=frozen["gripper_policy"], safety_limits=limits,
                            normalizer=norm)


def _server(meta=None, backend=None):
    return S.R42DeployServer(meta or _meta(), backend=backend or StubBackend(), device="cpu",
                             tokenizer_prompt_ids=[128000, 9906, 11, 5123, 29901])


def _images(shape=(64, 80, 3), dtype=np.uint8):
    rs = np.random.RandomState(1)
    return {cam: rs.randint(0, 255, shape).astype(dtype) for cam in R.CAMERA_ORDER}


def _obs(state=None, images=None, **extra):
    obs = {"images": _images() if images is None else images,
           "state": np.zeros(14, np.float32) if state is None else state}
    obs.update(extra)
    return obs


def _lat(total, pre=0.5, dec=0.5):
    """Deterministic latency record whose phases sum exactly to the total."""
    return {"preprocess": pre, "forward_total": float(total) - pre - dec,
            "decode": dec, "total": float(total)}


def _inside(alpha=0.5):
    """A 14D state strictly inside the real push_T joint box (the box is asymmetric)."""
    nb = np.load(BOUNDS)
    lo = np.asarray(nb["amin"], np.float32)
    hi = np.asarray(nb["amax"], np.float32)
    return (lo + float(alpha) * (hi - lo)).astype(np.float32)


@pytest.fixture()
def adapter():
    ad = A.R42RoboDojoAdapter(SpyServer(_server()), control_dt=0.04)
    ad.reset(0)
    return ad


# --------------------------------------------------------------------- normal path (14 tokens)
def test_three_rgb_observation_yields_fourteen_token_action(adapter):
    out = adapter.step(_obs())
    assert out["n_action_tokens"] == 14
    act = np.asarray(out["action14"], np.float32)
    assert act.shape == (14,), "action14 must be the 14D joint command"
    assert np.isfinite(act).all()
    assert out["env_idx"] == 0
    for key in ("preprocess", "forward_total", "decode", "total"):
        assert key in out["latency"] and out["latency"][key] >= 0.0
    assert out["latency"]["budget_ms"] == pytest.approx(BUDGET_MS)
    assert adapter.recorder.n == 1
    lim = adapter.server.metadata["safety_limits"]
    assert (act >= np.asarray(lim["amin"], np.float32) - 1e-6).all()
    assert (act <= np.asarray(lim["amax"], np.float32) + 1e-6).all()


def test_action_tokens_and_bins_are_inside_the_legal_window(adapter):
    out = adapter.step(_obs())
    ids = np.asarray(out["ids"], np.int64)
    assert ids.shape == (14,)
    assert ids.min() >= 31744 and ids.max() <= 31999
    assert PAD not in set(int(x) for x in ids)
    assert STOP not in set(int(x) for x in ids)


def test_cam_high_alias_resolves_to_the_frozen_cam_head_slot(adapter):
    alias = {("cam_high" if k == "cam_head" else k): v for k, v in _images().items()}
    out_alias = adapter.step(_obs(images=alias))
    adapter.reset(0)
    out_canon = adapter.step(_obs(images=_images()))
    assert np.array_equal(out_alias["action14"], out_canon["action14"])


def test_unknown_camera_key_is_refused(adapter):
    bad = dict(_images())
    bad["cam_side"] = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(R.R42Error):
        adapter.step(_obs(images=bad))


def test_two_aliases_for_the_same_camera_are_ambiguous(adapter):
    bad = dict(_images())
    bad["cam_high"] = bad["cam_head"].copy()
    with pytest.raises(R.R42Error):
        adapter.step(_obs(images=bad))


def test_missing_camera_is_refused(adapter):
    bad = dict(_images())
    bad.pop("cam_right_wrist")
    with pytest.raises(R.R42Error):
        adapter.step(_obs(images=bad))


# ----------------------------------------------------------------------------- image guards
def test_rgba_camera_is_refused(adapter):
    rgba = {cam: np.zeros((8, 8, 4), np.uint8) for cam in R.CAMERA_ORDER}
    with pytest.raises(R.R42Error):
        adapter.step(_obs(images=rgba))


def test_bgr_channel_order_is_refused(adapter):
    bgr = dict(_images())
    bgr["channel_order"] = "BGR"
    with pytest.raises(R.R42Error):
        adapter.step(_obs(images=bgr))


def test_float_images_are_refused(adapter):
    flt = {cam: np.zeros((8, 8, 3), np.float32) for cam in R.CAMERA_ORDER}
    with pytest.raises(R.R42Error):
        adapter.step(_obs(images=flt))


# ------------------------------------------------------------------ positional camera order
def test_positional_images_need_a_declared_frozen_order(adapter):
    seq = [_images()[cam] for cam in R.CAMERA_ORDER]
    with pytest.raises(R.R42Error):
        adapter.step(_obs(images=seq))


def test_positional_images_with_wrong_declared_order_are_refused(adapter):
    names = ["cam_head", "cam_right_wrist", "cam_left_wrist"]
    seq = [_images()[cam] for cam in names]
    with pytest.raises(R.R42Error):
        adapter.step(_obs(images=seq, camera_names=names))


def test_positional_images_with_correct_order_match_the_named_path(adapter):
    cameras = list(R.CAMERA_ORDER)
    seq = [_images()[cam] for cam in cameras]
    out_seq = adapter.step(_obs(images=seq, camera_names=cameras))
    adapter.reset(0)
    out_named = adapter.step(_obs(images=_images()))
    assert np.array_equal(out_seq["action14"], out_named["action14"])


# ------------------------------------------------------------------------------ state guard
@pytest.mark.parametrize("state", [np.zeros(7, np.float32), np.zeros(28, np.float32),
                                   np.zeros(13, np.float32)])
def test_state_must_be_14d(adapter, state):
    with pytest.raises(R.R42Error):
        adapter.step(_obs(state=state))


def test_state_must_be_finite(adapter):
    bad = np.zeros(14, np.float32)
    bad[3] = np.nan
    with pytest.raises(R.R42Error):
        adapter.step(_obs(state=bad))
    bad[3] = np.inf
    with pytest.raises(R.R42Error):
        adapter.step(_obs(state=bad))


# ------------------------------------------------------------------- privileged ground truth
def test_target_t_is_refused_before_the_policy_sees_it(adapter):
    with pytest.raises(R.R42Error) as exc:
        adapter.step(_obs(target_t=np.zeros(3, np.float32)))
    assert "target_t" in str(exc.value)
    assert adapter.server.act_calls == [], "the policy must never be called with privileged obs"


def test_privileged_prefix_and_nested_fields_are_refused(adapter):
    with pytest.raises(R.R42Error):
        adapter.step(_obs(privileged_block_pose=np.zeros(3, np.float32)))
    with pytest.raises(R.R42Error):
        adapter.step(_obs(diag={"target_t": np.zeros(3, np.float32)}))
    assert adapter.server.act_calls == []


def test_diagnostics_are_readable_but_never_sent_to_the_policy(adapter):
    obs = _obs(target_t=np.array([1.0, 2.0, 0.25], np.float32))
    diag = adapter.diagnostics(obs)
    assert "target_t" in diag
    assert np.allclose(diag["target_t"], [1.0, 2.0, 0.25])
    with pytest.raises(R.R42Error):
        adapter.step(obs)
    adapter.step(_obs())
    assert len(adapter.server.act_calls) == 1
    sent = adapter.server.act_calls[0]["images"]
    assert tuple(sent.keys()) == R.CAMERA_ORDER
    assert not any("target" in k or "privileged" in k for k in sent)


# ----------------------------------------------------------------------- latency percentiles
def test_latency_percentiles_are_exact_on_a_known_sequence():
    rec = A.LatencyRecorder(control_dt=0.04)
    for total in range(1, 101):
        rec.record(_lat(total))
    s = rec.summary()
    assert s["n_samples"] == 100
    assert s["budget_ms"] == pytest.approx(40.0)
    assert s["control_hz"] == pytest.approx(25.0)
    tot = s["phases"]["total"]
    assert tot["p50"] == pytest.approx(50.5)
    assert tot["p95"] == pytest.approx(95.05)
    assert tot["p99"] == pytest.approx(99.01)
    assert tot["max"] == pytest.approx(100.0)
    assert tot["mean"] == pytest.approx(50.5)
    for phase in ("preprocess", "forward_total", "decode", "total"):
        assert s["phases"][phase]["n"] == 100


def test_latency_percentiles_on_a_short_sequence():
    rec = A.LatencyRecorder(control_dt=0.04)
    for total in (10.0, 20.0, 30.0, 40.0):
        rec.record(_lat(total))
    tot = rec.summary()["phases"]["total"]
    assert tot["p50"] == pytest.approx(25.0)
    assert tot["p95"] == pytest.approx(38.5)
    assert tot["p99"] == pytest.approx(39.7)


def test_percentile_helper_matches_the_documented_definition():
    assert A.percentile([5.0], 50) == 5.0
    assert A.percentile([5.0, 5.0, 5.0], 99) == 5.0
    assert A.percentile([0.0, 10.0], 50) == pytest.approx(5.0)


def test_control_period_violations_are_counted_strictly_above_40ms():
    rec = A.LatencyRecorder(control_dt=0.04)
    for total in (39.9, 40.0, 40.1, 55.0, 12.0):
        rec.record(_lat(total))
    s = rec.summary()
    assert s["violations"] == 2, "only totals strictly greater than 40 ms violate the period"
    assert s["violation_rate"] == pytest.approx(2.0 / 5.0)
    assert s["worst_phase"] in ("forward_total", "preprocess", "decode", "total")


def test_budget_follows_the_configured_control_dt():
    rec = A.LatencyRecorder(control_dt=0.02)
    for total in (19.0, 21.0):
        rec.record(_lat(total))
    s = rec.summary()
    assert s["budget_ms"] == pytest.approx(20.0)
    assert s["control_hz"] == pytest.approx(50.0)
    assert s["violations"] == 1


def test_recorder_refuses_malformed_and_negative_latency():
    rec = A.LatencyRecorder(control_dt=0.04)
    with pytest.raises(R.R42Error):
        rec.record({"total": 5.0})
    with pytest.raises(R.R42Error):
        rec.record({"preprocess": -1.0, "forward_total": 2.0, "decode": 0.5, "total": 5.0})


def test_summary_of_an_empty_recorder_is_safe():
    s = A.LatencyRecorder(control_dt=0.04).summary()
    assert s["n_samples"] == 0
    assert s["violations"] == 0
    assert all(v["p50"] == 0.0 for v in s["phases"].values())


def test_to_json_writes_the_summary(tmp_path):
    rec = A.LatencyRecorder(control_dt=0.04)
    for total in (10.0, 60.0):
        rec.record(_lat(total))
    path = tmp_path / "latency.json"
    rec.to_json(str(path))
    data = json.load(open(str(path)))
    assert data["summary"]["n_samples"] == 2
    assert data["summary"]["violations"] == 1
    assert data["summary"]["phases"]["total"]["p99"] == pytest.approx(59.5)
    assert len(data["samples"]) == 2


def test_adapter_feeds_the_recorder_across_steps(adapter):
    for _ in range(3):
        adapter.step(_obs())
    s = adapter.recorder.summary()
    assert s["n_samples"] == 3
    assert s["phases"]["total"]["max"] >= 0.0


# ----------------------------------------------------------------------- rate / env handling
def test_checkpoint_max_rate_holds_the_command_when_configured_to_zero():
    ad = A.R42RoboDojoAdapter(_server(_meta(max_rate=(0.0,) * 14)), control_dt=0.04)
    s0 = _inside(0.30)
    s1 = _inside(0.90)
    out0 = ad.step(_obs(state=s0))
    out1 = ad.step(_obs(state=s1))
    assert np.allclose(out0["action14"], s0, atol=1e-6)
    assert np.allclose(out1["action14"], out0["action14"], atol=1e-6)
    assert out1["rate"]["rate_limited"] > 0


def test_max_rate_step_scales_with_control_dt():
    meta = _meta(max_rate=(1.0,) * 14)
    fast = A.R42RoboDojoAdapter(_server(meta), control_dt=0.04)
    slow = A.R42RoboDojoAdapter(_server(meta), control_dt=0.01)
    st = np.zeros(14, np.float32)
    o_fast = fast.step(_obs(state=st))
    o_slow = slow.step(_obs(state=st))
    assert np.allclose(o_fast["rate"]["max_step"], 0.04)
    assert np.allclose(o_slow["rate"]["max_step"], 0.01)


def test_env_isolation_and_reset_clears_per_env_history(adapter):
    adapter.step(_obs(state=np.zeros(14, np.float32)), env_idx=0)
    adapter.step(_obs(state=_inside(0.6)), env_idx=1)
    assert adapter.server.history_features(0).shape == (42,)
    assert adapter.server.history_features(1).shape == (42,)
    adapter.reset(0)
    with pytest.raises(R.R42Error):
        adapter.server.history_features(0)
    assert np.isfinite(adapter.server.history_features(1)).all()
    adapter.step(_obs(state=np.zeros(14, np.float32)), env_idx=0)
    assert adapter.server.history_features(0).shape == (42,)


def test_reset_drops_the_held_previous_command():
    ad = A.R42RoboDojoAdapter(_server(_meta(max_rate=(0.0,) * 14)), control_dt=0.04)
    ad.step(_obs(state=_inside(0.3)))
    ad.reset(0)
    held = _inside(0.7)
    out = ad.step(_obs(state=held))
    assert np.allclose(out["action14"], held, atol=1e-6), (
        "after reset the previous command must be re-seeded from the state")


def test_replay_reproduces_the_step_and_updates_no_weight(adapter):
    be = adapter.server.backend
    before = be.weight.copy()
    rec = adapter.step(_obs(), env_idx=0, record=True)
    assert "log" in rec
    adapter.reset(0)
    rep = adapter.replay(rec["log"])
    assert np.array_equal(np.asarray(rec["ids"]), np.asarray(rep["ids"]))
    assert np.asarray(rep["action"]).shape == (14,)
    assert np.array_equal(be.weight, before), "replay must not touch any weight"


def test_adapter_refuses_a_server_that_returns_the_wrong_token_count():
    class ShortServer(SpyServer):
        def act(self, images, state, **kwargs):
            out = self.inner.act(images, state, **kwargs)
            out["n_action_tokens"] = 8
            return out
    bad = A.R42RoboDojoAdapter(ShortServer(_server()), control_dt=0.04)
    with pytest.raises(R.R42Error):
        bad.step(_obs())


def test_adapter_requires_a_server_with_metadata_and_act():
    with pytest.raises(R.R42Error):
        A.R42RoboDojoAdapter(object())
    with pytest.raises(R.R42Error):
        A.R42RoboDojoAdapter(_server(), control_dt=0.0)
    with pytest.raises(R.R42Error):
        A.R42RoboDojoAdapter(_server(), control_dt=float("nan"))


def test_contract_declares_the_frozen_wiring():
    ad = A.R42RoboDojoAdapter(_server(), control_dt=0.04)
    c = ad.contract()
    assert tuple(c["camera_order"]) == tuple(R.CAMERA_ORDER)
    assert tuple(R.CAMERA_ORDER) == ("cam_head", "cam_left_wrist", "cam_right_wrist")
    assert c["state_dim"] == 14
    assert c["n_action_tokens"] == 14
    assert c["proprio_dim"] == 42
    assert c["control_hz"] == pytest.approx(25.0)
    assert c["control_dt"] == pytest.approx(0.04)
    assert c["privileged_policy"] == "refused_before_policy"

