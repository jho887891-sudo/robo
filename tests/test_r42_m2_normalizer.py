#!/usr/bin/env python3
"""M2 multi-task normalizer dispatch in r42_adapter (deployability of a shared checkpoint).

Until this dispatch existed, an M2 checkpoint could be trained but never served: normalizer_payload()
hard-required a single 64/16 payload with no task dimension, while the three tasks need residual
scales that differ by 2.4x. These tests pin the additive contract without touching the R15 one.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _s in ("code/xpolicylab/policy/OpenVLA_R42", "code/xpolicylab/policy/OpenVLA_Pure", "training"):
    sys.path.insert(0, os.path.join(REPO, _s))
import r42_adapter as R  # noqa: E402

TASKS = ["push_T", "stack_bowls"]


def block(scale=0.02, n_train=80, n_dev=20):
    q01 = np.full(14, -float(scale), np.float32)
    q99 = np.full(14, float(scale), np.float32)
    return {"normalization_mode": "signed_motion",
            "pmin": np.full(14, -1.0, np.float32).tolist(),
            "pmax": np.full(14, 1.0, np.float32).tolist(),
            "v_scale": np.full(14, 0.01, np.float32).tolist(),
            "a_scale": np.full(14, 0.01, np.float32).tolist(),
            "motion_quantile": 0.99, "q01": q01.tolist(), "q99": q99.tolist(),
            "fit_source": {"quantile_source": "%d TRAIN episodes only" % n_train,
                           "n_train_episodes_for_quantiles": n_train,
                           "n_dev_episodes_excluded": n_dev}}


def payload(scales=(0.021, 0.052)):
    return {"schema": "m2-task-normalizer-v1",
            "tasks": {t: block(s) for t, s in zip(TASKS, scales)},
            "task_id": {t: i for i, t in enumerate(TASKS)},
            "fitted_on": {t: {"train_episodes": 80, "n_train_episodes": 80} for t in TASKS}}


def test_schema_detection_and_task_listing():
    p = payload()
    assert R.is_task_normalizer(p) is True
    assert R.normalizer_task_list(p) == TASKS
    assert R.is_task_normalizer({"normalization_mode": "signed_motion"}) is False
    assert R.normalizer_task_list({"q01": []}) == []


def test_each_task_resolves_to_its_own_block():
    p = payload()
    a = R.normalizer_for_task(p, "push_T")
    b = R.normalizer_for_task(p, "stack_bowls")
    assert abs(a["q99"][0] - 0.021) < 1e-6
    assert abs(b["q99"][0] - 0.052) < 1e-6
    assert a["q99"] != b["q99"]          # the 2.4x spread is actually kept apart


def test_missing_task_and_missing_block_fail_closed():
    p = payload()
    with pytest.raises(R.R42Error):
        R.normalizer_for_task(p, None)
    with pytest.raises(R.R42Error):
        R.normalizer_for_task(p, "sweep_blocks")
    bad = payload()
    bad["task_id"] = {"push_T": 0}       # stack_bowls has no frozen id
    with pytest.raises(R.R42Error):
        R.normalizer_for_task(bad, "stack_bowls")
    nofit = payload()
    nofit["fitted_on"] = {"push_T": {"train_episodes": 80}}
    with pytest.raises(R.R42Error):
        R.normalizer_for_task(nofit, "stack_bowls")


def test_a_block_with_no_train_or_dev_episodes_is_refused():
    p = payload()
    p["tasks"]["push_T"] = block(n_train=0)
    with pytest.raises(R.R42Error):
        R.normalizer_for_task(p, "push_T")
    p = payload()
    p["tasks"]["push_T"] = block(n_dev=0)
    with pytest.raises(R.R42Error):
        R.normalizer_for_task(p, "push_T")


def test_validate_normalizer_refuses_a_multitask_payload_without_a_task():
    with pytest.raises(R.R42Error):
        R.validate_normalizer(payload())


def test_whole_payload_sha_pins_every_task():
    p1, p2 = payload(), payload(scales=(0.021, 0.099))
    s1, s2 = R.normalizer_sha256_for(p1), R.normalizer_sha256_for(p2)
    assert len(s1) == 64 and s1 != s2      # changing only the second task changes the pin
    # and it equals the hash of the whole structure, not of one block
    assert s1 == R.normalizer_sha256(p1)
    # a single-task payload still hashes its validated block (R15 behaviour unchanged)
    single = block(0.02, 64, 16)
    assert R.normalizer_sha256_for(single) == R.normalizer_sha256(R.validate_normalizer(single))


def test_metadata_round_trip_names_the_task_set_and_refuses_a_foreign_task():
    p = payload()
    meta = R.build_metadata("multi:push_T+stack_bowls", "a" * 64, "b" * 64, "c" * 64,
                            "d" * 64, "e" * 64, train_steps=2000, seed=0,
                            gripper_policy={"gripper_dims": [6, 13]},
                            safety_limits={"amin": [-1.0] * 14, "amax": [1.0] * 14},
                            normalizer=p)
    assert meta["tasks"] == TASKS
    assert R.validate_metadata(meta) is True
    assert R.validate_metadata(meta, expect_task="push_T") is True
    with pytest.raises(R.R42Error):
        R.validate_metadata(meta, expect_task="sweep_blocks")
    # a metadata whose task list disagrees with the payload is refused
    broken = dict(meta)
    broken["tasks"] = ["push_T"]
    with pytest.raises(R.R42Error):
        R.validate_metadata(broken)


def test_single_task_metadata_is_unchanged():
    meta = R.build_metadata("push_T", "a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64,
                            train_steps=300, seed=0,
                            gripper_policy={"gripper_dims": [6, 13]},
                            safety_limits={"amin": [-1.0] * 14, "amax": [1.0] * 14},
                            normalizer=block(0.02, 64, 16))
    # a single-task checkpoint must NOT carry the key at all: every metadata file this repo already
    # produced has to stay byte-identical (pinned by the byte-parity test in test_r42_metadata_bridge)
    assert "tasks" not in meta
    assert meta["normalizer_sha256"] == R.normalizer_sha256(R.validate_normalizer(block(0.02, 64, 16)))
    assert R.validate_metadata(meta, expect_task="push_T") is True
    with pytest.raises(R.R42Error):
        R.validate_metadata(meta, expect_task="stack_bowls")
