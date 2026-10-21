#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only tests for the M2 task-aware normalizer contract (training/m2_task_normalizer.py).

Run:  /home/dgut/pusht_runtime/venv_oft/bin/python -m pytest tests/test_m2_task_normalizer.py -q

No GPU, no 7B model: everything here is numpy + h5py on synthetic episodes.  Two live-data tests
are skipped automatically when the ARX-X5 HDF5 root is not mounted.
"""
import copy
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import h5py                                                       # noqa: E402

from training import m2_multitask_data as M2                      # noqa: E402
from training import m2_task_normalizer as N                      # noqa: E402

REAL_ROOT = os.environ.get("M2_HDF5_ROOT", M2.DEFAULT_HDF5_ROOT)
HAVE_REAL = all(os.path.isdir(os.path.join(REAL_ROOT, t, "arx_x5", "data")) for t in M2.TASKS)
real_only = pytest.mark.skipif(not HAVE_REAL, reason="live HDF5 root not mounted: %s" % REAL_ROOT)
EVIDENCE_PAYLOAD = os.environ.get("M2_NORMALIZER_JSON",
                                  "/home/dgut/r15_lanes/M2T/m2_task_normalizer.json")

# two structurally identical tasks with DIFFERENT motion scales (like the real 2.4x spread)
SCALES = {"push_T": 1.0, "stack_bowls": 2.2}


# -------------------------------------------------------------------------------- fixtures
def _jpeg(n=64):
    return b"\xff\xd8" + b"\x00" * max(0, n - 4) + b"\xff\xd9"


def write_episode(d, name, T=120, scale=1.0, seed=0, instruction=None, freq=25):
    rng = np.random.RandomState(seed)
    S = np.cumsum(rng.randn(T, 14).astype(np.float64) * 0.05 * float(scale), axis=0)
    # smooth, [0,1] gripper channels (as in the real data): they must NOT dominate the residual
    phase = np.arange(T, dtype=np.float64) * 0.1
    S[:, 6] = np.clip(0.5 + 0.2 * np.sin(phase) + 0.002 * scale * rng.randn(T), 0.0, 1.0)
    S[:, 13] = np.clip(0.5 + 0.2 * np.cos(phase) + 0.002 * scale * rng.randn(T), 0.0, 1.0)
    A = np.empty_like(S)
    A[:-1] = S[1:]
    A[-1] = S[-1]
    p = os.path.join(d, name)
    with h5py.File(p, "w") as h:
        for grp, arr in (("state", S), ("action", A)):
            i = 0
            for key, w in zip(("left_arm_joint_states", "left_ee_joint_states",
                               "right_arm_joint_states", "right_ee_joint_states"),
                              (6, 1, 6, 1)):
                h.create_dataset("%s/%s" % (grp, key), data=arr[:, i:i + w].astype(np.float64))
                i += w
        for grp in ("state", "action"):
            h.create_dataset("%s/left_ee_poses" % grp, data=rng.randn(T, 7))
            h.create_dataset("%s/right_ee_poses" % grp, data=rng.randn(T, 7))
        h.create_dataset("state/left_delta_ee_poses", data=rng.randn(T, 7))
        h.create_dataset("state/right_delta_ee_poses", data=rng.randn(T, 7))
        h.create_dataset("additional_info/frequency", data=np.int64(freq))
        h.create_dataset("data_format_version", data=np.bytes_("v1.0"))
        h.create_dataset("instruction",
                         data=np.bytes_(instruction or M2.TASK_PROMPTS["push_T"]))
        for cam in M2.CAMERA_ORDER:
            g = h.create_group("vision/" + cam)
            g.create_dataset("colors", data=np.array([_jpeg()] * T, dtype="S64"))
            g.create_dataset("shape", data=np.array([480, 640, 3], dtype=np.int64))
            g.create_dataset("extrinsic_matrix", data=np.tile(np.eye(4), (T, 1, 1)))
            g.create_dataset("intrinsic_matrix", data=np.eye(3))
    return p


@pytest.fixture()
def ds(tmp_path):
    """A 2-task synthetic dataset root with splits, indexes and a complete multi-task payload."""
    root = tmp_path / "hdf5"
    for task, scale in SCALES.items():
        d = root / task / "arx_x5" / "data"
        d.mkdir(parents=True)
        for i in range(4):
            write_episode(str(d), "episode_%07d.hdf5" % i, T=120, scale=scale,
                          seed=100 * M2.task_id(task) + i,
                          instruction=M2.TASK_PROMPTS[task])
    indexes = {t: M2.load_task_index(t, str(root)) for t in SCALES}
    splits = {t: M2.split_task_index(indexes[t], dev_frac=0.5, seed=0) for t in SCALES}
    payload, provs = N.fit_multi_normalizer(list(SCALES), splits, indexes=indexes)
    dev_lists = {t: [k.split("/", 1)[-1] for k in splits[t]["dev"]] for t in SCALES}
    return {"root": str(root), "indexes": indexes, "splits": splits, "payload": payload,
            "provs": provs, "dev_lists": dev_lists}


def _mutate(payload, task, key, value):
    bad = copy.deepcopy(payload)
    bad["tasks"][task][key] = value
    return bad


# ------------------------------------------------------------------------- schema contract
def test_schema_and_required_keys_are_frozen():
    assert N.SCHEMA == "m2-task-normalizer-v1"
    assert N.REQUIRED_PAYLOAD_KEYS == ("schema", "tasks", "task_id", "fitted_on")
    for f in ("task", "task_id", "normalization_mode", "pmin", "pmax", "q01", "q99",
              "v_scale", "a_scale", "motion_quantile", "n_train_episodes",
              "n_dev_episodes_excluded"):
        assert f in N.REQUIRED_TASK_FIELDS, f
    assert N.BASE_DIM == 14
    assert N.TASKS == ("push_T", "stack_bowls", "sweep_blocks")


def test_payload_is_versioned_and_covers_every_task(ds):
    p = ds["payload"]
    assert p["schema"] == N.SCHEMA
    assert sorted(p["tasks"]) == ["push_T", "stack_bowls"]
    assert p["task_id"] == {"push_T": 0, "stack_bowls": 1}
    for t in p["tasks"]:
        fo = p["fitted_on"][t]
        assert fo["n_train_episodes"] == len(fo["train_episodes"]) == 2
        assert fo["train_episodes_sha256"] == N.sha256_json(list(fo["train_episodes"]))
        assert fo["payload_sha256"] == N.sha256_json(p["tasks"][t])
        assert fo["fit_source"] == "TRAIN episodes only"
    assert N.assert_payload_complete(p, tasks=["push_T", "stack_bowls"]) is True


# ----------------------------------------------------------------------------- provenance
def test_fit_is_train_only_and_records_provenance(ds):
    t = "push_T"
    train, dev = N.split_episodes_for_task(ds["splits"][t], t)
    blk, prov = N.fit_task_normalizer(t, ds["splits"][t], index=ds["indexes"][t])
    assert blk["fit_episodes"] == train
    assert blk["dev_episodes_excluded"] == dev
    assert blk["n_train_episodes"] == len(train) == 2
    assert blk["n_dev_episodes_excluded"] == len(dev) == 2
    assert not set(blk["fit_episodes"]) & set(blk["dev_episodes_excluded"])
    assert blk["task_id"] == 0 and blk["task"] == t
    assert blk["normalization_mode"] == "signed_motion"
    assert blk["proprio_dim"] == 42
    for key in ("pmin", "pmax", "q01", "q99", "v_scale", "a_scale"):
        assert len(blk[key]) == 14
    assert all(x > 0 for x in blk["v_scale"]) and all(x > 0 for x in blk["a_scale"])
    assert prov["train_episodes"] == train
    assert prov["train_episodes_sha256"] == N.sha256_json(train)
    assert prov["payload_sha256"] == N.sha256_json(blk)
    assert prov["episode_file_sha256"] is None      # only when hash_episodes=True
    blk2, prov2 = N.fit_task_normalizer(t, ds["splits"][t], index=ds["indexes"][t],
                                        hash_episodes=True)
    assert sorted(prov2["episode_file_sha256"]) == sorted(train)
    for ep, sha in prov2["episode_file_sha256"].items():
        assert len(sha) == 64
    assert N.sha256_json(blk) == N.sha256_json(blk2), "file hashing must not change the payload"


def test_fit_refuses_dev_as_train_and_foreign_task(ds):
    t, sp = "push_T", ds["splits"]["push_T"]
    train, dev = N.split_episodes_for_task(sp, t)
    swapped = {"train": dev, "dev": train, "task": t}
    blk, _ = N.fit_task_normalizer(t, swapped, index=ds["indexes"][t])
    assert blk["fit_episodes"] == dev, "the fit follows whatever TRAIN list it is given"
    other, _ = N.fit_task_normalizer("stack_bowls", ds["splits"]["stack_bowls"],
                                     index=ds["indexes"]["stack_bowls"])
    p = N.build_multi_payload([blk, other])
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(p, dev_lists=ds["dev_lists"])
    assert N.assert_payload_complete(p) is True, "without the authoritative split it is only a "
    "self-consistent payload; the split cross-check is what proves it was fitted on DEV"
    # the same episode may never sit on both sides of one split
    with pytest.raises(N.M2NormalizerError):
        N.split_episodes_for_task({"train": dev, "dev": dev, "task": t}, t)
    # a foreign task's qualified episode may not enter this task's split
    with pytest.raises(N.M2NormalizerError):
        N.split_episodes_for_task({"train": ["stack_bowls/" + train[0]], "dev": dev, "task": t},
                                  t)
    with pytest.raises(N.M2NormalizerError):
        N.split_episodes_for_task({"train": train, "dev": [], "task": t}, t)
    # an unknown task name never fits
    with pytest.raises(M2.M2DataError):
        N.fit_task_normalizer("stack_blocks", sp, index=ds["indexes"][t])


# --------------------------------------------------------------------------- payload_for
def test_payload_for_returns_the_task_block(ds):
    p = ds["payload"]
    for t in ("push_T", "stack_bowls"):
        blk = N.payload_for(p, t)
        assert blk["task"] == t and blk["task_id"] == M2.TASK_IDS[t]
        assert blk["q99"] == p["tasks"][t]["q99"]
    blk = N.payload_for(p, "push_T")
    blk["q99"][0] += 1.0
    assert blk["q99"] != p["tasks"]["push_T"]["q99"], "payload_for must return a copy"
    # the per-task scales really differ (a global normalizer would destroy this)
    ratio = (np.abs(p["tasks"]["stack_bowls"]["q99"]).max()
             / np.abs(p["tasks"]["push_T"]["q99"]).max())
    assert ratio > 1.5, ratio
    with pytest.raises(N.M2NormalizerError):
        N.payload_for(p, "sweep_blocks")
    with pytest.raises(N.M2NormalizerError):
        N.payload_for(p, "push_T_random")
    with pytest.raises(N.M2NormalizerError):
        N.payload_for({"schema": N.SCHEMA, "tasks": {}, "task_id": {}, "fitted_on": {}}, "push_T")


def test_contract_and_decode_context_match_the_block(ds):
    p = ds["payload"]
    for t in ("push_T", "stack_bowls"):
        ctx = N.decode_context(p, t)
        blk = p["tasks"][t]
        assert np.allclose(ctx["q01"], np.asarray(blk["q01"], np.float32))
        assert np.allclose(ctx["q99"], np.asarray(blk["q99"], np.float32))
        assert len(ctx["normalizer_sha256"]) == 64
        c = N.contract_for(p, t)
        f = c.transform(np.zeros(14, np.float32), np.zeros(14, np.float32), np.zeros(14, np.float32))
        assert f.shape == (42,)
        assert np.allclose(c.v_scale, np.asarray(blk["v_scale"], np.float32))


# -------------------------------------------------------------------------- fail closed
def test_missing_task_is_rejected(ds):
    # a payload whose three blocks disagree about which tasks exist is never accepted
    for key in ("tasks", "task_id", "fitted_on"):
        p = copy.deepcopy(ds["payload"])
        del p[key]["stack_bowls"]
        with pytest.raises(N.M2NormalizerError):
            N.assert_payload_complete(p)
    # a consistently dropped task is a 1-task payload: it is valid alone but never for this run
    p = copy.deepcopy(ds["payload"])
    for key in ("tasks", "task_id", "fitted_on"):
        del p[key]["stack_bowls"]
    assert N.assert_payload_complete(p) is True
    with pytest.raises(N.M2NormalizerError):
        N.payload_for(p, "stack_bowls")
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(p, tasks=["push_T", "stack_bowls"])
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_task_set(p, ["push_T", "stack_bowls"])
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(ds["payload"], tasks=["push_T", "stack_bowls", "sweep_blocks"])


def test_wrong_task_id_and_index_disagreement_are_rejected(ds):
    p = copy.deepcopy(ds["payload"])
    p["task_id"]["stack_bowls"] = 0
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(p)
    p = copy.deepcopy(ds["payload"])
    p["tasks"]["stack_bowls"]["task_id"] = 2
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(p)
    p = copy.deepcopy(ds["payload"])
    p["tasks"]["push_T"]["task"] = "stack_bowls"
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(p)
    p = copy.deepcopy(ds["payload"])
    p["tasks"]["push_T"]["task_id"] = 1
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(p)


def test_dev_fitted_payload_is_rejected(ds):
    # fit push_T on its DEV episodes (the classic "quantiles fitted on everything" mistake)
    t = "push_T"
    train, dev = N.split_episodes_for_task(ds["splits"][t], t)
    swapped = {"train": dev, "dev": train, "task": t}
    blk, _ = N.fit_task_normalizer(t, swapped, index=ds["indexes"][t])
    other, _ = N.fit_task_normalizer("stack_bowls", ds["splits"]["stack_bowls"],
                                     index=ds["indexes"]["stack_bowls"])
    p = N.build_multi_payload([blk, other])
    assert blk["fit_episodes"] == dev
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(p, dev_lists=ds["dev_lists"])
    # the fit list and the payload's own DEV list must not overlap either
    p2 = copy.deepcopy(ds["payload"])
    p2["tasks"]["push_T"]["fit_episodes"] = p2["tasks"]["push_T"]["dev_episodes_excluded"]
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(p2)
    # the payload's DEV list must equal the authoritative split's DEV list
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(ds["payload"], tasks=["push_T"],
                                  dev_lists={"push_T": train})


def test_shape_order_and_degenerate_fields_are_rejected(ds):
    p = ds["payload"]
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(_mutate(p, "push_T", "pmin", p["tasks"]["push_T"]["pmin"][:13]))
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(_mutate(p, "push_T", "q99", p["tasks"]["push_T"]["q01"]))
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(_mutate(p, "push_T", "pmax", p["tasks"]["push_T"]["pmin"]))
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(_mutate(p, "push_T", "v_scale", [0.0] * 14))
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(_mutate(p, "push_T", "a_scale", [-1.0] * 14))
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(_mutate(p, "push_T", "motion_quantile", 1.5))
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(_mutate(p, "push_T", "normalization_mode", "legacy_position"))
    q = copy.deepcopy(p)
    del q["tasks"]["push_T"]["q01"]
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(q)
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(_mutate(p, "push_T", "q01", [float("nan")] * 14))


def test_nothing_held_out_is_rejected(ds):
    p = copy.deepcopy(ds["payload"])
    p["tasks"]["push_T"]["n_dev_episodes_excluded"] = 0
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(p)
    p = copy.deepcopy(ds["payload"])
    p["tasks"]["push_T"]["dev_episodes_excluded"] = []
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(p)
    p = copy.deepcopy(ds["payload"])
    p["tasks"]["push_T"]["n_train_episodes"] = 0
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(p)


def test_edited_block_is_rejected_by_its_provenance_sha(ds):
    p = copy.deepcopy(ds["payload"])
    p["tasks"]["push_T"]["q99"][3] = float(p["tasks"]["push_T"]["q99"][3]) * 3.0
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(p)
    p = copy.deepcopy(ds["payload"])
    p["fitted_on"]["push_T"]["payload_sha256"] = "0" * 64
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(p)
    p = copy.deepcopy(ds["payload"])
    p["fitted_on"]["push_T"]["train_episodes"] = ["episode_0000009.hdf5"]
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(p)


def test_wrong_schema_is_rejected(ds):
    p = copy.deepcopy(ds["payload"])
    p["schema"] = "r42-normalizer-v1"
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(p)
    q = copy.deepcopy(ds["payload"])
    del q["schema"]
    with pytest.raises(N.M2NormalizerError):
        N.assert_payload_complete(q)


# ------------------------------------------------------------------------- payload sha
def test_normalizer_sha_covers_the_whole_payload(ds):
    p = ds["payload"]
    sha = N.normalizer_sha256(p)
    assert len(sha) == 64
    # key order must not matter
    reordered = {"fitted_on": p["fitted_on"], "task_id": p["task_id"], "tasks": p["tasks"],
                 "schema": p["schema"]}
    assert N.normalizer_sha256(reordered) == sha
    # changing ONE task's scale changes the sha (all tasks are pinned together)
    q = copy.deepcopy(p)
    q["tasks"]["stack_bowls"]["v_scale"][0] = float(q["tasks"]["stack_bowls"]["v_scale"][0]) * 1.01
    q["fitted_on"]["stack_bowls"]["payload_sha256"] = N.sha256_json(q["tasks"]["stack_bowls"])
    assert N.normalizer_sha256(q) != sha
    # ... as does the episode provenance
    r = copy.deepcopy(p)
    r["fitted_on"]["push_T"]["train_episodes"] = sorted(
        r["fitted_on"]["push_T"]["train_episodes"])
    assert N.normalizer_sha256(r) == sha
    r["fitted_on"]["push_T"]["split_sha256"] = "f" * 64
    assert N.normalizer_sha256(r) != sha
    # a partial payload can never produce a pin
    part = copy.deepcopy(p)
    del part["tasks"]["push_T"]
    with pytest.raises(N.M2NormalizerError):
        N.normalizer_sha256(part)


def test_payload_json_roundtrip_and_task_set_guard(ds, tmp_path):
    p = ds["payload"]
    path = str(tmp_path / "norm.json")
    sha = N.save_payload(path, p)
    back = N.load_payload(path, tasks=["push_T", "stack_bowls"])
    assert N.normalizer_sha256(back) == sha
    with pytest.raises(N.M2NormalizerError):
        N.load_payload(path, tasks=["push_T"])
    with pytest.raises(N.M2NormalizerError):
        N.load_payload(path, tasks=["push_T", "stack_bowls", "sweep_blocks"])
    with pytest.raises(N.M2NormalizerError):
        N.load_payload(str(tmp_path / "nope.json"))
    assert N.assert_payload_task_set(back, ["stack_bowls", "push_T"]) is True
    assert N.assert_same_normalizer(p, back) == sha
    q = copy.deepcopy(p)
    q["tasks"]["push_T"]["a_scale"][13] *= 2.0
    q["fitted_on"]["push_T"]["payload_sha256"] = N.sha256_json(q["tasks"]["push_T"])
    with pytest.raises(N.M2NormalizerError):
        N.assert_same_normalizer(p, q)
    d = N.describe(back)
    assert d["schema"] == N.SCHEMA
    assert d["residual_q99_spread_ratio"] > 1.5


# ------------------------------------------------------------------------- live dataset
@real_only
def test_real_live_fit_small_is_complete_and_cross_task_scaled(tmp_path):
    """Fit every real task on a few real TRAIN episodes: the payload must stay complete and the
    per-task residual scales must remain distinct (the whole point of this module)."""
    payloads = []
    for t in M2.TASKS:
        idx = M2.load_task_index(t, REAL_ROOT)
        sp = M2.split_task_index(idx, dev_frac=0.2, seed=M2.task_id(t))
        # NOTE: 12 TRAIN episodes, not 4.  Measured on the live data: with only 4 push_T TRAIN
        # episodes, dim 6 (left gripper) never moves, the 1%/99% state bounds collapse to a point
        # and TaskNormalizer correctly refuses the fit.  From 8 episodes on all 14 dims are
        # non-degenerate (see lanes/M2_trainer.md), so 12 keeps this test fast AND honest.
        train = sorted(k.split("/", 1)[-1] for k in sp["train"])[:12]
        dev = sorted(k.split("/", 1)[-1] for k in sp["dev"])[:4]
        blk, prov = N.fit_task_normalizer(t, {"train": train, "dev": dev, "task": t},
                                          index=idx, min_rows=50)
        assert blk["fit_episodes"] == sorted(train)
        payloads.append(blk)
    p = N.build_multi_payload(payloads)
    assert N.assert_payload_complete(p) is True
    q99 = {t: float(np.abs(np.asarray(p["tasks"][t]["q99"])).max()) for t in M2.TASKS}
    assert max(q99.values()) / min(q99.values()) > 1.5, q99
    assert len(N.normalizer_sha256(p)) == 64


@real_only
def test_real_evidence_payload_if_present():
    """Validate the SHIPPED payload if the evidence copy is mounted (no fit, no model)."""
    if not os.path.isfile(EVIDENCE_PAYLOAD):
        pytest.skip("evidence payload not mounted: %s" % EVIDENCE_PAYLOAD)
    p = N.load_payload(EVIDENCE_PAYLOAD)
    assert len(N.payload_tasks(p)) >= 2
    d = N.describe(p)
    assert d["residual_q99_spread_ratio"] > 1.5, d
    for t in N.payload_tasks(p):
        assert p["tasks"][t]["n_train_episodes"] >= 60
        assert p["tasks"][t]["n_dev_episodes_excluded"] >= 10
    assert N.assert_payload_complete(p) is True
