#!/usr/bin/env python3
"""The CV baseline served through the real R42 policy (workline C instrument).

R17 needs one closed-loop answer the offline audit can only suggest: does "keep your current velocity"
actually move the object? The CV baseline therefore runs through the SAME metadata, SAFE clip and protocol
as the VLA, selected by R42_POLICY=cv, and never touches the model. These tests pin that it is exactly
A = clip(S + V) and that the VLA path is untouched when the flag is off.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _s in ("training", "code/xpolicylab/policy/OpenVLA_Pure",
           "code/xpolicylab/policy/OpenVLA_R42"):
    sys.path.insert(0, os.path.join(REPO, _s))
import r42_adapter as R            # noqa: E402
import r42_deploy_server as S      # noqa: E402

NORM = os.path.join(REPO, "data", "m2_task_normalizer_2task.json")


def _metadata():
    payload = json.load(open(NORM))
    real = payload["tasks"]["push_T"]
    # The deployed metadata carries the R15 single-task contract (64 TRAIN / 16 DEV); the M2 payload has
    # 80/20 episode lists, so the counts are restated here and everything else is the real fitted block.
    blk = {k: real[k] for k in ("normalization_mode", "q01", "q99", "v_scale", "a_scale",
                                "pmin", "pmax")}
    blk["fit_source"] = "unit test of the CV branch (quantiles from the real 2-task payload)"
    blk["fit_episodes"] = ["train_%03d" % i for i in range(64)]
    blk["dev_episodes_excluded"] = ["dev_%03d" % i for i in range(16)]
    lo = np.asarray(blk["pmin"], np.float64)
    hi = np.asarray(blk["pmax"], np.float64)
    return R.build_metadata("push_T", "a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64,
                            train_steps=2000, seed=0,
                            gripper_policy={"gripper_dims": list(R.GRIPPER_DIMS)},
                            safety_limits={"amin": lo.tolist(), "amax": hi.tolist()},
                            # build_metadata expects ONE normalizer payload (the R15 form): the
                            # multi-task wrapper is resolved per task before this point
                            normalizer=dict(blk))


def test_cv_branch_returns_clipped_state_plus_velocity(monkeypatch):
    meta = _metadata()
    srv = S.R42DeployServer(meta, backend=None)
    monkeypatch.setenv("R42_POLICY", "cv")
    s0 = np.zeros(14, np.float32)
    s1 = np.full(14, 0.001, np.float32)
    imgs = {c: np.zeros((8, 8, 3), np.uint8) for c in R.CAMERA_ORDER}
    out0 = srv.act(imgs, s0)
    assert out0["info"]["policy"] == "cv_baseline"
    assert np.allclose(out0["action"], s0)          # no history -> v = 0
    out1 = srv.act(imgs, s1)
    lo = np.asarray(meta["safety_limits"]["amin"], np.float32)
    hi = np.asarray(meta["safety_limits"]["amax"], np.float32)
    expect = np.clip(s1 + (s1 - s0), lo, hi)
    assert np.allclose(out1["action"], expect), "cv must command clip(S + V)"
    assert np.allclose(out1["velocity"], s1 - s0)


def test_cv_branch_is_off_by_default(monkeypatch):
    monkeypatch.delenv("R42_POLICY", raising=False)
    meta = _metadata()
    srv = S.R42DeployServer(meta, backend=None)
    imgs = {c: np.zeros((8, 8, 3), np.uint8) for c in R.CAMERA_ORDER}
    with pytest.raises(Exception):
        srv.act(imgs, np.zeros(14, np.float32))    # no backend -> the VLA path must fail, not silently CV
