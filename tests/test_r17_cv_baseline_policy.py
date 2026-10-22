#!/usr/bin/env python3
"""The CV baseline served through the real R42 policy (workline C instrument).

R17 needs one closed-loop answer the offline audit can only suggest: does "keep your current velocity"
actually move the object? The baseline is expressed as "always predict a ZERO residual", which is exactly
A_hat = S + V, and it flows through the SAME token contract, decode_with_metadata and SAFE clip as the VLA
(selected by R42_POLICY=cv, no model, no GPU).

An earlier version returned a hand-built action dict and was rejected by the official client with
"server produced 0 action token(s) / 14 id(s)" - hence these tests check the TOKEN contract too.
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
    blk = {k: real[k] for k in ("normalization_mode", "q01", "q99", "v_scale", "a_scale",
                                "pmin", "pmax")}
    blk["fit_source"] = "unit test of the CV branch (quantiles from the real 2-task payload)"
    blk["fit_episodes"] = ["train_%03d" % i for i in range(64)]
    blk["dev_episodes_excluded"] = ["dev_%03d" % i for i in range(16)]
    return R.build_metadata("push_T", "a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64,
                            train_steps=2000, seed=0,
                            gripper_policy={"gripper_dims": list(R.GRIPPER_DIMS)},
                            safety_limits={"amin": list(map(float, blk["pmin"])),
                                           "amax": list(map(float, blk["pmax"]))},
                            normalizer=blk)


def _imgs():
    return {c: np.zeros((8, 8, 3), np.uint8) for c in R.CAMERA_ORDER}


def test_cv_branch_emits_the_full_token_contract(monkeypatch):
    monkeypatch.setenv("R42_POLICY", "cv")
    # the server reports prompt_len in every response, so a prompt id list must exist even though the
    # CV baseline itself never reads it
    srv = S.R42DeployServer(_metadata(), backend=None,
                            tokenizer_prompt_ids=list(range(5, 25)))
    out = srv.act(_imgs(), np.zeros(14, np.float32))
    ids = [int(x) for x in out["ids"]]
    assert len(ids) == R.N_ACTIONS, "the client refuses anything but 14 ids"
    assert out["n_action_tokens"] == R.N_ACTIONS
    assert all(R.TOKEN_BEGIN_IDX + 1 <= i <= R.TOKEN_BEGIN_IDX + 256 for i in ids)


def test_cv_branch_commands_state_plus_velocity(monkeypatch):
    monkeypatch.setenv("R42_POLICY", "cv")
    meta = _metadata()
    srv = S.R42DeployServer(meta, backend=None, tokenizer_prompt_ids=list(range(5, 25)))
    s0 = np.zeros(14, np.float32)
    s1 = np.full(14, 0.001, np.float32)
    out0 = srv.act(_imgs(), s0)
    # with no history the velocity is zero, so the command must be the current state itself
    assert np.allclose(out0["action"], np.clip(s0, meta["safety_limits"]["amin"],
                                               meta["safety_limits"]["amax"]), atol=2e-4)
    out1 = srv.act(_imgs(), s1)
    expect = np.clip(s1 + (s1 - s0), meta["safety_limits"]["amin"], meta["safety_limits"]["amax"])
    # the zero residual is coded to the nearest bin, so allow one codec bin of slack
    bin_w = (np.asarray(meta["normalizer"]["q99"], float)
             - np.asarray(meta["normalizer"]["q01"], float)) / 256.0
    assert np.allclose(out1["action"], expect, atol=float(bin_w.max())), \
        "cv must command clip(S + V) within one codec bin"
    assert np.allclose(out1["velocity"], s1 - s0)


def test_cv_branch_is_off_by_default(monkeypatch):
    monkeypatch.delenv("R42_POLICY", raising=False)
    srv = S.R42DeployServer(_metadata(), backend=None, tokenizer_prompt_ids=list(range(5, 25)))
    with pytest.raises(Exception):
        srv.act(_imgs(), np.zeros(14, np.float32))    # no backend -> the VLA path must fail loudly
