#!/usr/bin/env python3
"""Deploy consistency for the THREE-task normalizer payload (issue #26, 12-task line).

The 2-task payload is covered by tests/test_r42_m2_normalizer.py. This pins the third task
(sweep_blocks) and the properties that matter when the shared model grows from 2 to 4 tasks:

  * the payload is complete and each task resolves to ITS OWN fitted block (not a shared one);
  * decoding the same residual under two tasks gives different bins, i.e. dispatch is real;
  * the metadata round-trip names all three tasks and refuses a task it was not fitted for;
  * the whole-payload sha256 pins every per-task block, so a single mangled block is detectable.

It runs on CPU from the committed artefacts only.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _s in ("training", "code/xpolicylab/policy/OpenVLA_Pure",
           "code/xpolicylab/policy/OpenVLA_R42"):
    sys.path.insert(0, os.path.join(REPO, _s))
import r42_adapter as R          # noqa: E402
import m2_task_normalizer as TN  # noqa: E402

PAYLOAD = os.path.join(REPO, "data", "m2_task_normalizer_3task.json")
TASKS = ["push_T", "stack_bowls", "sweep_blocks"]


@pytest.fixture(scope="module")
def payload():
    with open(PAYLOAD) as fh:
        return json.load(fh)


def test_payload_covers_all_three_tasks(payload):
    assert payload["schema"] == "m2-task-normalizer-v1"
    assert sorted(payload["tasks"]) == sorted(TASKS)
    assert R.is_task_normalizer(payload) is True
    assert R.normalizer_task_list(payload) == TASKS
    for t in TASKS:
        blk = payload["tasks"][t]
        assert len(blk["q01"]) == 14 and len(blk["q99"]) == 14
        assert all(float(b) > float(a) for a, b in zip(blk["q01"], blk["q99"])), \
            "%s: q01 must be below q99 on every dimension" % t


def test_each_task_resolves_to_its_own_fitted_block(payload):
    for t in TASKS:
        got = R.normalizer_for_task(payload, t)
        assert np.allclose(got["q01"], payload["tasks"][t]["q01"])
        assert np.allclose(got["q99"], payload["tasks"][t]["q99"])
    # the three blocks are genuinely different (a shared block would be a silent single-task model)
    q99 = {t: np.asarray(payload["tasks"][t]["q99"], float) for t in TASKS}
    assert not np.allclose(q99["push_T"], q99["sweep_blocks"])
    assert not np.allclose(q99["stack_bowls"], q99["sweep_blocks"])


def test_dispatch_changes_the_decoded_bins(payload):
    """Same residual, different task -> different legal bins. This is what makes routing real."""
    residual = np.array([0.01, -0.02, 0.03, -0.04, 0.005, -0.006, 0.007, -0.008,
                         0.02, -0.01, 0.004, -0.003, 0.002, -0.001], np.float32)
    ctx = {t: TN.decode_context(payload, t) for t in TASKS}
    ids = {}
    for t in TASKS:
        _idx, tok = R.token_ids_for_residual(residual, ctx[t]["q01"], ctx[t]["q99"])
        ids[t] = np.asarray(tok, np.int64) - R.TOKEN_BEGIN_IDX - 1
        assert ids[t].shape == (14,)
        assert ids[t].min() >= 0 and ids[t].max() <= 255
    assert not np.array_equal(ids["push_T"], ids["sweep_blocks"])
    assert not np.array_equal(ids["stack_bowls"], ids["sweep_blocks"])


def test_unknown_task_fails_closed(payload):
    with pytest.raises(Exception):
        R.normalizer_for_task(payload, "make_toast")
    with pytest.raises(Exception):
        R.normalizer_for_task(payload, None)


def test_metadata_round_trip_lists_three_tasks(payload):
    meta = R.build_metadata("multi:" + "+".join(TASKS), "a" * 64, "b" * 64, "c" * 64,
                            "d" * 64, "e" * 64, train_steps=2000, seed=0,
                            gripper_policy={"gripper_dims": [6, 13]},
                            safety_limits={"amin": [-1.0] * 14, "amax": [1.0] * 14},
                            normalizer=payload)
    assert meta["tasks"] == TASKS
    assert R.validate_metadata(meta) is True
    for t in TASKS:
        assert R.validate_metadata(meta, expect_task=t) is True
    with pytest.raises(R.R42Error):
        R.validate_metadata(meta, expect_task="fold_clothes")


def test_payload_sha_pins_every_block(payload):
    whole = TN.sha256_json(payload)
    assert whole == "930213f515773f7ac5e7975f60054065fa6236b2fb881d7fa245bbceb4065927", \
        "the committed 3-task payload changed; update the pin deliberately, never silently"
