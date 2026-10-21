#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only tests for the M2 multi-task data contract (training/m2_multitask_data.py).

Run:  /home/dgut/pusht_runtime/venv_oft/bin/python -m pytest tests/test_m2_multitask_data.py -q

Real-data tests (the live ARX-X5 HDF5 root) are skipped automatically when the root is not
mounted; the synthetic-fixture tests always run, so the fail-closed behaviour, the split
algebra, the TRAIN-only normalizer and the prompt/task-token mapping are covered everywhere.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import h5py                                          # noqa: E402

from training import m2_multitask_data as M2         # noqa: E402

REAL_ROOT = os.environ.get("M2_HDF5_ROOT", M2.DEFAULT_HDF5_ROOT)
HAVE_REAL = all(os.path.isdir(os.path.join(REAL_ROOT, t, "arx_x5", "data")) for t in M2.TASKS)
real_only = pytest.mark.skipif(not HAVE_REAL, reason="live HDF5 root not mounted: %s" % REAL_ROOT)


# ------------------------------------------------------------------------------- fixtures
def _jpeg(n=64):
    return b"\xff\xd8" + b"\x00" * max(0, n - 4) + b"\xff\xd9"


def write_episode(d, name, T=120, widths=(6, 1, 6, 1), freq=25,
                  instruction=M2.TASK_PROMPTS["push_T"], seed=0, jpeg=_jpeg(64)):
    """Write one contract-conforming episode (A[t]==S[t+1]) or a deliberately broken one."""
    rng = np.random.RandomState(seed)
    S = np.cumsum(rng.randn(T, 14).astype(np.float64) * 0.05, axis=0)
    S[:, 6] = np.clip(np.abs(rng.randn(T)), 0.0, 1.0)
    S[:, 13] = np.clip(np.abs(rng.randn(T)), 0.0, 1.0)
    A = np.empty_like(S)
    A[:-1] = S[1:]
    A[-1] = S[-1]
    p = os.path.join(d, name)
    with h5py.File(p, "w") as h:
        for grp, arr in (("state", S), ("action", A)):
            i = 0
            for key, w in zip(("left_arm_joint_states", "left_ee_joint_states",
                               "right_arm_joint_states", "right_ee_joint_states"), widths):
                h.create_dataset("%s/%s" % (grp, key), data=arr[:, i:i + w].astype(np.float64))
                i += w
        h.create_dataset("state/left_ee_poses", data=rng.randn(T, 7))
        h.create_dataset("state/right_ee_poses", data=rng.randn(T, 7))
        h.create_dataset("action/left_ee_poses", data=rng.randn(T, 7))
        h.create_dataset("action/right_ee_poses", data=rng.randn(T, 7))
        h.create_dataset("state/left_delta_ee_poses", data=rng.randn(T, 7))
        h.create_dataset("state/right_delta_ee_poses", data=rng.randn(T, 7))
        h.create_dataset("additional_info/frequency", data=np.int64(freq))
        h.create_dataset("data_format_version", data=np.bytes_("v1.0"))
        h.create_dataset("instruction", data=np.bytes_(instruction))
        for cam in M2.CAMERA_ORDER:
            g = h.create_group("vision/" + cam)
            g.create_dataset("colors", data=np.array([jpeg] * T, dtype="S%d" % len(jpeg)))
            g.create_dataset("shape", data=np.array([480, 640, 3], dtype=np.int64))
            g.create_dataset("extrinsic_matrix", data=np.tile(np.eye(4), (T, 1, 1)))
            g.create_dataset("intrinsic_matrix", data=np.eye(3))
    return p


@pytest.fixture()
def task_dir(tmp_path):
    d = tmp_path / "push_T" / "arx_x5" / "data"
    d.mkdir(parents=True)
    for i in range(4):
        write_episode(str(d), "episode_%07d.hdf5" % i, seed=i)
    return str(d)


# ------------------------------------------------------------------------ task identity
def test_task_id_and_token_mapping_is_frozen():
    assert M2.TASKS == ("push_T", "stack_bowls", "sweep_blocks")
    assert M2.TASK_IDS == {"push_T": 0, "stack_bowls": 1, "sweep_blocks": 2}
    for t in M2.TASKS:
        assert M2.task_id(t) == M2.TASK_IDS[t]
        assert M2.task_token(t) == "<task:%s#%d>" % (t, M2.TASK_IDS[t])
    assert len({M2.task_id(t) for t in M2.TASKS}) == 3


def test_unknown_task_fails_closed():
    with pytest.raises(M2.M2DataError):
        M2.task_id("stack_blocks")
    with pytest.raises(M2.M2DataError):
        M2.task_prompt("push_T_random")
    with pytest.raises(M2.M2DataError):
        M2.task_token("")


# ----------------------------------------------------------------------------- prompts
def test_one_distinct_language_string_per_task():
    prompts = [M2.task_prompt(t) for t in M2.TASKS]
    assert len(set(prompts)) == len(M2.TASKS), "prompts must be task-distinguishing"
    for p in prompts:
        assert isinstance(p, str) and 5 < len(p) < 300
    assert "T-shaped block" in M2.task_prompt("push_T")
    assert "bowls" in M2.task_prompt("stack_bowls")
    assert "broom" in M2.task_prompt("sweep_blocks")


def test_prompt_builder_token_is_stable_and_idempotent():
    for t in M2.TASKS:
        plain = M2.build_prompt(t)
        assert plain["prompt"] == M2.task_prompt(t)
        assert plain["task_id"] == M2.TASK_IDS[t]
        tok = M2.build_prompt(t, include_token=True)
        assert tok["prompt"].startswith(M2.task_token(t) + " ")
        assert tok["prompt"].endswith(M2.task_prompt(t))
        again = M2.build_prompt(t, include_token=True)
        assert again == tok, "prompt builder must be deterministic"
        assert again["prompt"] == tok["prompt"]


def test_prompt_drift_fails_closed():
    good = M2.task_prompt("stack_bowls")
    assert M2.build_prompt("stack_bowls", instruction=good)["prompt_matches_dataset_instruction"]
    with pytest.raises(M2.M2DataError):
        M2.build_prompt("stack_bowls", instruction="Stack the bowls.")


# ------------------------------------------------------------------------------- splits
def test_split_is_deterministic_disjoint_and_complete():
    eps = ["episode_%07d.hdf5" % i for i in range(100)]
    a = M2.split_train_dev(eps, dev_frac=0.2, seed=0)
    b = M2.split_train_dev(list(reversed(eps)), dev_frac=0.2, seed=0)
    assert a["train"] == b["train"] and a["dev"] == b["dev"], "order-independent + seeded"
    assert not set(a["train"]) & set(a["dev"])
    assert sorted(a["train"] + a["dev"]) == sorted(eps)
    assert (a["n_train"], a["n_dev"]) == (80, 20)
    c = M2.split_train_dev(eps, dev_frac=0.2, seed=1)
    assert c["dev"] != a["dev"], "a different seed must move DEV"


def test_split_fails_closed():
    with pytest.raises(M2.M2DataError):
        M2.split_train_dev([])
    with pytest.raises(M2.M2DataError):
        M2.split_train_dev(["a.hdf5", "a.hdf5"])
    with pytest.raises(M2.M2DataError):
        M2.split_train_dev(["a.hdf5", "b.hdf5"], dev_frac=0.0)
    with pytest.raises(M2.M2DataError):
        M2.split_train_dev(["a.hdf5", "b.hdf5"], dev_frac=1.0)


def test_split_task_index_rejects_foreign_task_episode():
    idx = {"task": "push_T", "task_id": 0,
           "episodes": ["episode_0000000.hdf5"],
           "qualified_episodes": ["push_T/episode_0000000.hdf5",
                                  "stack_bowls/episode_0000000.hdf5"]}
    with pytest.raises(M2.M2DataError):
        M2.split_task_index(idx, dev_frac=0.5, seed=0)


def test_assert_splits_disjoint_catches_cross_task_claim():
    a = {"task": "push_T", "train": ["push_T/e0.hdf5"], "dev": ["push_T/e1.hdf5"]}
    b = {"task": "stack_bowls", "train": ["stack_bowls/e0.hdf5"],
         "dev": ["stack_bowls/e1.hdf5"]}
    assert M2.assert_splits_disjoint(a, b)
    b_bad = {"task": "stack_bowls", "train": ["push_T/e0.hdf5"], "dev": []}
    with pytest.raises(M2.M2DataError):
        M2.assert_splits_disjoint(a, b_bad)


# -------------------------------------------------------------------------------- index
def test_synthetic_index_and_qualified_names(task_dir):
    idx = M2.load_task_index("push_T", task_dir)
    assert idx["n_episodes"] == 4
    assert idx["episodes"] == sorted(idx["episodes"])
    assert idx["qualified_episodes"] == ["push_T/" + e for e in idx["episodes"]]
    assert idx["task_id"] == 0 and idx["task_token"] == "<task:push_T#0>"
    assert idx["cameras"] == list(M2.CAMERA_ORDER)
    assert idx["frequency"] == 25
    assert idx["frames_min"] == idx["frames_max"] == 120
    assert idx["total_frames"] == 480
    assert idx["prompt_matches_constant"] is True


def test_wrong_dim_episode_fails_closed(tmp_path):
    d = tmp_path / "bad" / "arx_x5" / "data"
    d.mkdir(parents=True)
    write_episode(str(d), "episode_0000000.hdf5", seed=0)
    idx = M2.load_task_index("push_T", str(d))
    assert idx["n_episodes"] == 1
    write_episode(str(d), "episode_0000001.hdf5", widths=(5, 1, 6, 1), seed=1)
    with pytest.raises(M2.M2DataError) as ei:
        M2.load_task_index("push_T", str(d))
    assert "left_arm_joint_states" in str(ei.value)


def test_bad_frequency_and_instruction_fail_closed(tmp_path):
    d1 = tmp_path / "f" / "arx_x5" / "data"
    d1.mkdir(parents=True)
    write_episode(str(d1), "episode_0000000.hdf5", freq=30)
    with pytest.raises(M2.M2DataError):
        M2.load_task_index("push_T", str(d1))
    d2 = tmp_path / "i" / "arx_x5" / "data"
    d2.mkdir(parents=True)
    write_episode(str(d2), "episode_0000000.hdf5", instruction=M2.TASK_PROMPTS["push_T"])
    write_episode(str(d2), "episode_0000001.hdf5", instruction="Do something else.")
    with pytest.raises(M2.M2DataError):
        M2.load_task_index("push_T", str(d2))


# ---------------------------------------------------------------------------- normalizer
def test_normalizer_is_fitted_on_train_episodes_only(task_dir):
    idx = M2.load_task_index("push_T", task_dir)
    sp = M2.split_task_index(idx, dev_frac=0.5, seed=0)
    train = [k.split("/", 1)[1] for k in sp["train"]]
    dev = [k.split("/", 1)[1] for k in sp["dev"]]
    n = M2.TaskNormalizer.fit_from_task_index(idx, train, dev, min_rows=10)
    assert n.fit_episodes == sorted(train), "fit provenance must be exactly TRAIN"
    assert not set(n.fit_episodes) & set(dev)
    assert n.task == "push_T" and n.n_rows >= 10
    meta = n.to_metadata()
    assert meta["motion_scale_fit_source"] == "TRAIN episodes only"
    assert meta["n_train_episodes"] == len(train)
    assert meta["n_dev_episodes_excluded"] == len(dev)
    assert set(meta["fit_episodes"]) == set(train)
    assert not set(meta["fit_episodes"]) & set(meta["dev_episodes_excluded"])
    # DEV must never be accepted as a fit source: any overlap with the DEV list is fatal
    with pytest.raises(M2.M2DataError):
        M2.TaskNormalizer.fit_from_task_index(idx, train + dev, dev, min_rows=10)
    with pytest.raises(M2.M2DataError):
        M2.TaskNormalizer.fit_from_task_index(idx, train + dev, train, min_rows=10)
    # episodes outside the index are fatal too
    with pytest.raises(M2.M2DataError):
        M2.TaskNormalizer.fit_from_task_index(idx, ["episode_9999999.hdf5"], dev, min_rows=10)
    # the TRAIN-only fit really is footed on fewer rows than an all-episodes fit
    n_all = M2.TaskNormalizer.fit_from_task_index(idx, idx["episodes"], (), min_rows=10)
    assert n.n_rows == 117 * len(train), n.n_rows
    assert n_all.n_rows == 117 * len(idx["episodes"]), n_all.n_rows
    assert n.n_rows < n_all.n_rows


def test_normalizer_contract_is_signed_motion_42d(task_dir):
    idx = M2.load_task_index("push_T", task_dir)
    n = M2.TaskNormalizer.fit_from_task_index(idx, idx["episodes"], (), min_rows=10)
    c = n.to_contract()
    f = c.transform(np.zeros(14), np.zeros(14), np.zeros(14))
    assert f.shape == (42,)
    assert n.to_metadata()["normalization_mode"] == "signed_motion"
    assert n.to_metadata()["proprio_dim"] == 42
    assert (n.v_scale > 0).all() and (n.a_scale > 0).all()
    assert len(n.action_q01) == 14 and len(n.r_q99) == 14


def test_normalizer_from_metadata_roundtrip(task_dir):
    idx = M2.load_task_index("push_T", task_dir)
    n = M2.TaskNormalizer.fit_from_task_index(idx, idx["episodes"], (), min_rows=10)
    m = n.to_metadata()
    n2 = M2.TaskNormalizer.from_metadata(json.loads(json.dumps(m)))
    for k in ("pmin", "pmax", "v_scale", "a_scale", "r_q01", "r_q99",
              "action_q01", "action_q99"):
        assert np.array_equal(getattr(n, k), getattr(n2, k))


# ------------------------------------------------------------------------------ rows
def test_build_row_shapes_task_and_no_future_leakage(task_dir):
    idx = M2.load_task_index("push_T", task_dir)
    n = M2.TaskNormalizer.fit_from_task_index(idx, idx["episodes"], (), min_rows=10)
    contract = n.to_contract()
    s, a = M2._episode_state_action(idx["hdf5_dir"], idx["episodes"][0])
    row = M2.build_row(s, a, 5, contract, "push_T", episode=idx["episodes"][0])
    assert row["features"].shape == (42,)
    assert row["residual"].shape == (14,)
    assert row["task"] == "push_T" and row["task_id"] == 0
    assert row["task_token"] == "<task:push_T#0>"
    assert row["prompt"] == M2.task_prompt("push_T")
    assert row["qualified_episode"] == "push_T/" + idx["episodes"][0]
    assert row["gripper_dims"] == (6, 13)
    assert np.array_equal(row["residual"], (a[5] - s[5]) - (s[5] - s[4]))
    assert M2.assert_row_no_future_leakage(s, a, 5, contract, "push_T") is not None


def test_build_row_rejects_t_without_history(task_dir):
    idx = M2.load_task_index("push_T", task_dir)
    n = M2.TaskNormalizer.fit_from_task_index(idx, idx["episodes"], (), min_rows=10)
    s, a = M2._episode_state_action(idx["hdf5_dir"], idx["episodes"][0])
    for bad_t in (-1, 0, 1):
        with pytest.raises(Exception):
            M2.build_row(s, a, bad_t, n.to_contract(), "push_T")


def test_build_row_rejects_wrong_state_dim(task_dir):
    idx = M2.load_task_index("push_T", task_dir)
    n = M2.TaskNormalizer.fit_from_task_index(idx, idx["episodes"], (), min_rows=10)
    s, a = M2._episode_state_action(idx["hdf5_dir"], idx["episodes"][0])
    with pytest.raises(Exception):
        M2.build_row(s[:, :13], a, 5, n.to_contract(), "push_T")
    with pytest.raises(M2.M2DataError):
        M2.build_row(s, a, 5, n.to_contract(), "stack_blocks")


# --------------------------------------------------------------- real dataset (live root)
@real_only
def test_real_index_per_task():
    seen = {}
    for t in M2.TASKS:
        idx = M2.load_task_index(t, REAL_ROOT)
        assert idx["n_episodes"] == 100
        assert 100 < idx["frames_min"] <= idx["frames_median"] <= idx["frames_max"]
        assert idx["cameras"] == list(M2.CAMERA_ORDER)
        assert idx["frequency"] == 25
        assert idx["gripper_dims"] == [6, 13]
        assert idx["prompt_matches_constant"] is True, idx["instruction"]
        assert idx["n_distinct_instructions"] == 1
        seen[t] = idx["instruction"]
    assert len(set(seen.values())) == 3, "the three tasks must carry different instructions"


@real_only
def test_real_cross_task_isolation_and_split():
    a = M2.load_task_index("push_T", REAL_ROOT)
    b = M2.load_task_index("stack_bowls", REAL_ROOT)
    c = M2.load_task_index("sweep_blocks", REAL_ROOT)
    ev = M2.assert_task_isolation(a, b)
    assert ev["qualified_intersection"] == 0
    assert ev["bare_name_intersection"] == 100, "bare names collide -> must key on task"
    M2.assert_task_isolation(a, c)
    M2.assert_task_isolation(b, c)
    with pytest.raises(M2.M2DataError):
        M2.assert_task_isolation(a, a)
    splits = [M2.split_task_index(i, dev_frac=0.2, seed=0) for i in (a, b, c)]
    assert M2.assert_splits_disjoint(*splits)
    for sp in splits:
        assert (sp["n_train"], sp["n_dev"]) == (80, 20)
        assert all(k.startswith(sp["task"] + "/") for k in sp["train"] + sp["dev"])
    # a stack_bowls episode can never appear in push_T's index or split
    push_keys = set(splits[0]["train"] + splits[0]["dev"])
    assert not any(k.startswith("stack_bowls/") for k in push_keys)
    assert push_keys == set(a["qualified_episodes"])


@real_only
def test_real_normalizer_train_only_and_row(tmp_path):
    idx = M2.load_task_index("push_T", REAL_ROOT)
    sp = M2.split_task_index(idx, dev_frac=0.2, seed=0)
    train = [k.split("/", 1)[1] for k in sp["train"]]
    dev = [k.split("/", 1)[1] for k in sp["dev"]]
    n = M2.TaskNormalizer.fit_from_task_index(idx, train, dev)
    assert n.fit_episodes == sorted(train)
    assert not set(n.fit_episodes) & set(dev)
    with pytest.raises(M2.M2DataError):
        M2.TaskNormalizer.fit_from_task_index(idx, train + dev, dev)
    s, a = M2._episode_state_action(idx["hdf5_dir"], train[0])
    row = M2.build_row(s, a, 10, n.to_contract(), "push_T", episode=train[0])
    assert row["features"].shape == (42,) and row["residual"].shape == (14,)
    assert np.isfinite(row["features"]).all() and np.isfinite(row["residual"]).all()


@real_only
def test_real_gripper_channels_are_normalised():
    idx = M2.load_task_index("push_T", REAL_ROOT)
    lo = np.full(14, np.inf)
    hi = np.full(14, -np.inf)
    for e in idx["episodes"][:10]:
        s, a = M2._episode_state_action(idx["hdf5_dir"], e)
        lo = np.minimum(lo, a.min(0))
        hi = np.maximum(hi, a.max(0))
    for d in M2.GRIPPER_DIMS:
        assert 0.0 <= lo[d] and hi[d] <= 1.0, "gripper dim %d outside [0,1]: %s..%s" % (d, lo[d], hi[d])
    assert hi[6] > 0.5 and hi[13] > 0.5, "gripper must actually open somewhere"
