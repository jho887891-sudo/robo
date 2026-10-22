#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2 multi-task data contract for ONE shared 7B VLA (push_T / stack_bowls / sweep_blocks).

ADDITIVE module. It does not modify, and is not imported by, any existing entrypoint.

Evidence this module is built on (see lanes/M2_data_audit.md):
  * all three tasks carry the SAME 29-dataset HDF5 schema, float64, T rows, no attributes;
  * 14D block order [left_arm(6), left_ee(1), right_arm(6), right_ee(1)] in BOTH state and
    action for all three tasks, gripper channels at dims (6, 13) in [0,1];
  * A[t] == S[t+1] bit-exactly (max|d| = 0.0) for all 300 episodes;
  * 3 RGB JPEG cameras, identical names AND order (cam_head, cam_left_wrist,
    cam_right_wrist), 480x640x3, FFD8/FFD9 buffers, 25 Hz;
  * exactly ONE task-distinguishing instruction string per task.

What this module deliberately does NOT do: it does not re-implement the R15 feature/target
math. build_row() delegates to training/r42_train_contract.py (TC.build_training_row), which
is the same code path the R15 trainer uses, so the 42D feature + 14D velocity-residual target
contract cannot drift between single-task and multi-task training.

Episode identity: episode FILE NAMES are identical across tasks (every task has
episode_0000000.hdf5 ... episode_0000099.hdf5). A bare file name is therefore AMBIGUOUS
across tasks. Every cross-task structure in this module is keyed by the qualified name
"<task>/<file>" and fail-closed helpers reject bare-name mixing.
"""
import json
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("training", "diagnostics", "code/xpolicylab/policy/OpenVLA_Pure",
             "code/xpolicylab/policy/OpenVLA_R42"):
    _p = os.path.join(REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r42_train_contract as TC                       # noqa: E402  (R15 row contract)
from r8_normalization import (                        # noqa: E402  (R8 normalizer)
    NormalizationContract, SIGNED_MOTION, fit_scales)

# ----------------------------------------------------------------------------- contract
BASE_DIM = 14
N42 = 42
GRIPPER_DIMS = (6, 13)
CAMERA_ORDER = ("cam_head", "cam_left_wrist", "cam_right_wrist")
ALIGNMENT_VERSION = "r42-v2"
TARGET_REPRESENTATION = "velocity_residual"
FREQUENCY_HZ = 25
MIN_HISTORY_FRAMES = 2
MOTION_QUANTILE = 0.99
EPS_SCALE = 1e-6
D_MODES = (SIGNED_MOTION,)

DEFAULT_HDF5_ROOT = os.environ.get("M2_HDF5_ROOT",
                                   "/home/T7/ojh/robot_actual/data/data/hdf5")

TASKS = ("push_T", "stack_bowls", "sweep_blocks")
TASK_IDS = {"push_T": 0, "stack_bowls": 1, "sweep_blocks": 2}

# One language string per task. These are BYTE-IDENTICAL to the /instruction dataset that is
# stored in all 100 episodes of the corresponding task (verified; tests re-verify on real data).
TASK_PROMPTS = {
    "push_T": "Push the T-shaped block to align it precisely with the gray T-shaped pad.",
    "stack_bowls": "Stack the three bowls together.",
    "sweep_blocks": "Pick up the broom, hand it over to the right hand, then use the dustpan "
                    "to sweep the blocks.",
}
PROMPT_SOURCE = "/instruction dataset of each task (25 Hz ARX-X5 RoboDojo HDF5)"

# Datasets that MUST exist in every episode, with their exact trailing shape and kind.
_FLOAT_BLOCKS = ("left_arm_joint_states", "left_ee_joint_states",
                 "right_arm_joint_states", "right_ee_joint_states")
_EXPECTED_TRAILING = {
    "state/left_arm_joint_states": (6,), "state/left_ee_joint_states": (1,),
    "state/right_arm_joint_states": (6,), "state/right_ee_joint_states": (1,),
    "action/left_arm_joint_states": (6,), "action/left_ee_joint_states": (1,),
    "action/right_arm_joint_states": (6,), "action/right_ee_joint_states": (1,),
    "state/left_ee_poses": (7,), "state/right_ee_poses": (7,),
    "action/left_ee_poses": (7,), "action/right_ee_poses": (7,),
    "state/left_delta_ee_poses": (7,), "state/right_delta_ee_poses": (7,),
    "additional_info/frequency": (),
    "data_format_version": (),
    "instruction": (),
}


class M2DataError(ValueError):
    """Raised on every contract violation; the M2 path never degrades silently."""


# ------------------------------------------------------------------------------- helpers
def task_id(task):
    if task not in TASK_IDS:
        raise M2DataError("unknown task %r; M2 knows %s" % (task, TASKS))
    return int(TASK_IDS[task])


def task_token(task):
    """Stable, human-readable task marker (never re-numbered, never re-ordered)."""
    return "<task:%s#%d>" % (task, task_id(task))


def task_prompt(task):
    task_id(task)
    return TASK_PROMPTS[task]


def qualified(task, episode):
    """'<task>/<episode file>' - the only cross-task-safe episode key."""
    task_id(task)
    if not isinstance(episode, str) or "/" in episode:
        raise M2DataError("episode must be a bare file name, got %r" % (episode,))
    if not episode.endswith(".hdf5"):
        raise M2DataError("episode %r must end with .hdf5" % (episode,))
    return "%s/%s" % (task, episode)


def split_qualified(key):
    if not isinstance(key, str) or key.count("/") != 1:
        raise M2DataError("expected '<task>/<episode>' key, got %r" % (key,))
    t, ep = key.split("/")
    task_id(t)
    qualified(t, ep)
    return t, ep


def task_dir(task, hdf5_root=None):
    """Resolve the episode directory for a task.

    Accepts either the DATASET ROOT ('.../data/data/hdf5', containing one sub-directory per
    task) or the TASK directory itself ('.../hdf5/<task>/arx_x5/data'), so callers cannot get
    the two confused - and both are still fully validated by load_task_index().
    """
    task_id(task)
    if hdf5_root is None:
        return os.path.join(DEFAULT_HDF5_ROOT, task, "arx_x5", "data")
    d = str(hdf5_root)
    nested = os.path.join(d, "arx_x5", "data")
    if os.path.isdir(nested):
        return nested
    if os.path.basename(d) == "data" and os.path.basename(os.path.dirname(d)) == "arx_x5":
        return d
    if os.path.isdir(os.path.join(d, "data")) and os.path.basename(d) == "arx_x5":
        return os.path.join(d, "data")
    # a dataset root: descend to <root>/<task>/arx_x5/data
    cand = os.path.join(d, task, "arx_x5", "data")
    if os.path.isdir(cand):
        return cand
    return d


# -------------------------------------------------------------------------- episode index
def _inspect_episode(path, task, verify_targets=False):
    """Fail-closed structural inspection of ONE episode. Metadata only unless verify_targets."""
    import h5py

    if not os.path.isfile(path):
        raise M2DataError("episode file missing: %s" % path)
    info = {"path": path, "episode": os.path.basename(path), "task": task}
    with h5py.File(path, "r") as h:
        # 1) required datasets, dtype kind and trailing shape
        for rel, trailing in _EXPECTED_TRAILING.items():
            if rel not in h:
                raise M2DataError("%s: missing dataset /%s" % (path, rel))
        for rel, trailing in _EXPECTED_TRAILING.items():
            ds = h[rel]
            got = tuple(int(x) for x in ds.shape[1:])
            if got != trailing:
                raise M2DataError("%s: /%s trailing shape %s != required %s"
                                  % (path, rel, got, trailing))
            if trailing and not np.issubdtype(ds.dtype, np.floating):
                raise M2DataError("%s: /%s dtype %s is not floating point"
                                  % (path, rel, ds.dtype))
        # 2) exact 14D layout: the four blocks must each be 2D [T, d] with the R15 widths
        widths = [h["state/%s" % k].shape[1] for k in _FLOAT_BLOCKS]
        if widths != [6, 1, 6, 1]:
            raise M2DataError("%s: 14D layout widths %s != [6, 1, 6, 1]" % (path, widths))
        T = int(h["state/left_arm_joint_states"].shape[0])
        if T < MIN_HISTORY_FRAMES + 2:
            raise M2DataError("%s: episode too short (T=%d)" % (path, T))
        # 3) every time-indexed dataset must agree on T
        for rel in list(_EXPECTED_TRAILING) + [
                "vision/%s/colors" % c for c in CAMERA_ORDER]:
            if rel.startswith("additional_info") or rel in ("data_format_version", "instruction"):
                continue
            if str(h[rel].shape[0]) != str(T):
                raise M2DataError("%s: /%s has T=%s but state has T=%d"
                                  % (path, rel, h[rel].shape[0], T))
        # 4) camera set and order
        if "vision" not in h:
            raise M2DataError("%s: missing /vision group" % path)
        cams = tuple(h["vision"].keys())
        if cams != CAMERA_ORDER:
            raise M2DataError("%s: camera set/order %s != %s" % (path, cams, CAMERA_ORDER))
        for c in CAMERA_ORDER:
            shp = tuple(int(x) for x in h["vision/%s/shape" % c][:])
            if shp != (480, 640, 3):
                raise M2DataError("%s: camera %s shape %s != (480, 640, 3)" % (path, c, shp))
            col = h["vision/%s/colors" % c]
            if col.ndim != 1 or col.dtype.kind != "S":
                raise M2DataError("%s: camera %s colors must be a 1-D byte-string buffer, got "
                                  "%s %s" % (path, c, col.shape, col.dtype))
            head = bytes(col[0][:2])
            tail = bytes(col[0][-2:])
            if head != b"\xff\xd8" or tail != b"\xff\xd9":
                raise M2DataError("%s: camera %s frame 0 is not a JPEG (SOI %r EOI %r)"
                                  % (path, c, head, tail))
        # 5) instruction + frequency
        instr = h["instruction"][()]
        if isinstance(instr, bytes):
            instr = instr.decode("utf-8", "replace")
        else:
            instr = str(instr)
        if not instr.strip():
            raise M2DataError("%s: empty /instruction" % path)
        freq = int(h["additional_info/frequency"][()])
        if freq != FREQUENCY_HZ:
            raise M2DataError("%s: frequency %d != %d Hz" % (path, freq, FREQUENCY_HZ))
        fmt = h["data_format_version"][()]
        fmt = fmt.decode("utf-8", "replace") if isinstance(fmt, bytes) else str(fmt)
        info.update({"T": T, "instruction": instr, "frequency": freq,
                     "data_format_version": fmt, "cameras": list(cams)})
        if verify_targets:
            state = TC.read14(h, "state")
            action = TC.read14(h, "action")
            worst = TC.assert_a_equals_next_state(state, action, atol=1e-6)
            info["a_eq_s_max_abs"] = float(worst)
    return info


def load_task_index(task, hdf5_dir=None, verify_targets=False,
                    require_single_instruction=True):
    """Index ONE task. Fail-closed: any structural deviation raises M2DataError.

    Returns a plain dict (JSON-serialisable apart from numpy scalars, which are cast).
    """
    task_id(task)
    d = task_dir(task, hdf5_dir)
    if not os.path.isdir(d):
        raise M2DataError("task %s: no such directory %s" % (task, d))
    names = sorted(f for f in os.listdir(d) if f.endswith(".hdf5"))
    if not names:
        raise M2DataError("task %s: no *.hdf5 episodes in %s" % (task, d))
    bad = [n for n in names if not n.startswith("episode_")]
    if bad:
        raise M2DataError("task %s: unexpected non-episode files %s" % (task, bad[:5]))
    if len({n for n in names}) != len(names):
        raise M2DataError("task %s: duplicate episode names" % task)

    per_ep = {}
    instrs = set()
    freqs = set()
    fmts = set()
    cams = set()
    for n in names:
        info = _inspect_episode(os.path.join(d, n), task, verify_targets=verify_targets)
        per_ep[n] = info
        instrs.add(info["instruction"])
        freqs.add(info["frequency"])
        fmts.add(info["data_format_version"])
        cams.add(tuple(info["cameras"]))
    if len(cams) != 1:
        raise M2DataError("task %s: inconsistent camera sets %s" % (task, cams))
    if len(freqs) != 1:
        raise M2DataError("task %s: inconsistent frequency %s" % (task, freqs))
    if require_single_instruction and len(instrs) != 1:
        raise M2DataError("task %s: %d distinct instructions; a shared per-task prompt needs "
                          "exactly one" % (task, len(instrs)))
    frames = {n: int(per_ep[n]["T"]) for n in names}
    fl = np.asarray(sorted(frames.values()), dtype=np.int64)
    return {
        "task": task,
        "task_id": task_id(task),
        "task_token": task_token(task),
        "hdf5_dir": d,
        "n_episodes": len(names),
        "episodes": names,
        "qualified_episodes": [qualified(task, n) for n in names],
        "frames": frames,
        "frames_min": int(fl.min()),
        "frames_median": float(np.median(fl)),
        "frames_max": int(fl.max()),
        "total_frames": int(fl.sum()),
        "instruction": sorted(instrs)[0] if instrs else "",
        "n_distinct_instructions": len(instrs),
        "prompt_matches_constant": bool(instrs == {TASK_PROMPTS[task]}),
        "frequency": sorted(freqs)[0],
        "data_format_version": sorted(fmts),
        "cameras": list(sorted(cams)[0]),
        "gripper_dims": list(GRIPPER_DIMS),
        "base_dim": BASE_DIM,
        "n42": N42,
        "alignment_version": ALIGNMENT_VERSION,
        "amplitude_verified_targets": bool(verify_targets),
    }


# ------------------------------------------------------------------------------- splits
def split_train_dev(episodes, dev_frac=0.2, seed=0):
    """Deterministic, episode-disjoint TRAIN/DEV split of a sequence of episode keys.

    * operates on whatever keys it is given (bare names for a single task, or qualified
      '<task>/<episode>' names - qualified is REQUIRED once more than one task is involved);
    * the shuffle is seeded, so the same (episodes, dev_frac, seed) always gives the same
      partition, independent of directory listing order;
    * fails closed on duplicates, on empty input, on a degenerate split.
    """
    eps = list(episodes)
    if not eps:
        raise M2DataError("split_train_dev: empty episode list")
    if len(set(eps)) != len(eps):
        dup = sorted({e for e in eps if eps.count(e) > 1})
        raise M2DataError("split_train_dev: duplicate episodes %s" % dup[:5])
    if not (0.0 < float(dev_frac) < 1.0):
        raise M2DataError("split_train_dev: dev_frac must be in (0,1), got %r" % (dev_frac,))
    n_dev = int(round(len(eps) * float(dev_frac)))
    n_dev = max(1, min(len(eps) - 1, n_dev))
    order = sorted(eps)
    random.Random(int(seed)).shuffle(order)
    dev = sorted(order[:n_dev])
    train = sorted(order[n_dev:])
    if set(train) & set(dev):
        raise M2DataError("split_train_dev: TRAIN/DEV overlap")
    if sorted(train + dev) != sorted(eps):
        raise M2DataError("split_train_dev: split does not cover the episode list")
    return {"train": train, "dev": dev, "dev_frac": float(dev_frac), "seed": int(seed),
            "n_train": len(train), "n_dev": len(dev), "task": None}


def split_task_index(index, dev_frac=0.2, seed=0):
    """Per-task 80/20 TRAIN/DEV over QUALIFIED names, tagged with the task id."""
    task = index["task"]
    out = split_train_dev(index["qualified_episodes"], dev_frac=dev_frac, seed=seed)
    out["task"] = task
    out["task_id"] = task_id(task)
    for k in ("train", "dev"):
        for key in out[k]:
            t, _ep = split_qualified(key)
            if t != task:
                raise M2DataError("split_task_index(%s): foreign episode %r in %s"
                                  % (task, key, k))
    return out


def assert_splits_disjoint(*splits):
    """Every split must be pairwise disjoint AND internally disjoint."""
    seen = {}
    for sp in splits:
        tag = sp.get("task") or "?"
        keys = list(sp["train"]) + list(sp["dev"])
        if set(sp["train"]) & set(sp["dev"]):
            raise M2DataError("split %s: TRAIN/DEV overlap" % tag)
        for k in keys:
            if k in seen and seen[k] != tag:
                raise M2DataError("episode %r claimed by both %s and %s" % (k, seen[k], tag))
            seen[k] = tag
    return True


def assert_task_isolation(index_a, index_b):
    """Two task indexes may not share a single episode, even if the names look identical."""
    qa = set(index_a["qualified_episodes"])
    qb = set(index_b["qualified_episodes"])
    if qa & qb:
        raise M2DataError("tasks %s and %s share episodes %s"
                          % (index_a["task"], index_b["task"], sorted(qa & qb)[:5]))
    if index_a["task"] == index_b["task"]:
        raise M2DataError("assert_task_isolation needs two DIFFERENT tasks")
    if index_a["task_id"] == index_b["task_id"]:
        raise M2DataError("task ids collide: %s" % index_a["task_id"])
    # a bare-name index would be ambiguous: prove the names really do collide
    return {"tasks": [index_a["task"], index_b["task"]],
            "bare_name_intersection": len(set(index_a["episodes"]) & set(index_b["episodes"])),
            "qualified_intersection": 0}


# ---------------------------------------------------------------------------- normalizer
def _episode_state_action(hdf5_dir, episode):
    import h5py
    p = os.path.join(hdf5_dir, episode)
    with h5py.File(p, "r") as h:
        s = TC.read14(h, "state")
        a = TC.read14(h, "action")
    if s.shape != a.shape:
        raise M2DataError("%s: state %s / action %s shape mismatch" % (p, s.shape, a.shape))
    return s, a


def _motion_and_residual(state, action):
    """V[t] = S[t]-S[t-1], Acc[t] = (S[t]-S[t-1]) - (S[t-1]-S[t-2]), R[t] = (A[t]-S[t]) - V[t].

    Only history-safe frames are produced: t >= 2 and t <= T-2 (a target needs S[t+1]).
    """
    s = np.asarray(state, np.float32)
    a = np.asarray(action, np.float32)
    T = s.shape[0]
    ts = np.arange(MIN_HISTORY_FRAMES, T - 1, dtype=np.int64)
    if ts.size == 0:
        raise M2DataError("episode too short for a history-safe frame pool (T=%d)" % T)
    vel = s[ts] - s[ts - 1]
    acc = vel - (s[ts - 1] - s[ts - 2])
    res = (a[ts] - s[ts]) - vel
    return vel, acc, res, ts


class TaskNormalizer:
    """Per-task quantile normalizer fitted on TRAIN episodes ONLY (fail-closed provenance)."""

    def __init__(self, task, pmin, pmax, action_q01, action_q99, v_scale, a_scale,
                 r_q01, r_q99, fit_episodes, dev_episodes, n_rows, quantile=MOTION_QUANTILE):
        self.task = task
        task_id(task)
        self.pmin = self._vec14(pmin, "pmin")
        self.pmax = self._vec14(pmax, "pmax")
        if not (self.pmax > self.pmin).all():
            raise M2DataError("%s: pmax must exceed pmin on every dim" % task)
        self.action_q01 = self._vec14(action_q01, "action_q01")
        self.action_q99 = self._vec14(action_q99, "action_q99")
        if not (self.action_q99 > self.action_q01).all():
            raise M2DataError("%s: action q99 must exceed q01 on every dim" % task)
        self.v_scale = self._positive14(v_scale, "v_scale")
        self.a_scale = self._positive14(a_scale, "a_scale")
        self.r_q01 = self._vec14(r_q01, "r_q01")
        self.r_q99 = self._vec14(r_q99, "r_q99")
        if not (self.r_q99 > self.r_q01).all():
            raise M2DataError("%s: residual q99 must exceed q01 on every dim" % task)
        self.fit_episodes = [str(e) for e in fit_episodes]
        self.dev_episodes = [str(e) for e in dev_episodes]
        if set(self.fit_episodes) & set(self.dev_episodes):
            raise M2DataError("%s: normalizer was fitted on DEV episodes %s"
                              % (task, sorted(set(self.fit_episodes) & set(self.dev_episodes))[:5]))
        if not self.fit_episodes:
            raise M2DataError("%s: normalizer has no TRAIN episodes" % task)
        self.n_rows = int(n_rows)
        self.quantile = float(quantile)

    @staticmethod
    def _vec14(x, name):
        v = np.asarray(x, np.float32).reshape(-1)
        if v.shape != (BASE_DIM,):
            raise M2DataError("%s must be 14D, got %s" % (name, v.shape))
        if not np.isfinite(v).all():
            raise M2DataError("%s has non-finite values" % name)
        return v

    @classmethod
    def _positive14(cls, x, name):
        v = cls._vec14(x, name)
        if (v <= 0).any():
            raise M2DataError("%s must be strictly positive" % name)
        return v

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit_from_task_index(cls, index, train_episodes, dev_episodes=(), quantile=MOTION_QUANTILE,
                            bounds_quantile=0.99, min_rows=100, include_actions=False):
        """Fit from the TRAIN episodes of one task index. DEV names are only used to REFUSE."""
        task = index["task"]
        train = sorted(str(e) for e in train_episodes)
        dev = sorted(str(e) for e in dev_episodes)
        unknown = [e for e in train if e not in index["episodes"]]
        if unknown:
            raise M2DataError("%s: TRAIN episodes not in the index: %s" % (task, unknown[:5]))
        if set(train) & set(dev):
            raise M2DataError("%s: DEV episodes passed as TRAIN: %s"
                              % (task, sorted(set(train) & set(dev))[:5]))
        if len(train) < 2:
            raise M2DataError("%s: need >= 2 TRAIN episodes" % task)
        s_all, a_all, v_all, acc_all, r_all = [], [], [], [], []
        for ep in train:
            s, a = _episode_state_action(index["hdf5_dir"], ep)
            vel, acc, res, _ts = _motion_and_residual(s, a)
            s_all.append(s[A_SAFE_SLICE(s.shape[0])])
            a_all.append(a[A_SAFE_SLICE(a.shape[0])])
            v_all.append(vel)
            acc_all.append(acc)
            r_all.append(res)
        S = np.concatenate(s_all, 0).astype(np.float32)
        A = np.concatenate(a_all, 0).astype(np.float32)
        V = np.concatenate(v_all, 0).astype(np.float32)
        AC = np.concatenate(acc_all, 0).astype(np.float32)
        Rm = np.concatenate(r_all, 0).astype(np.float32)
        if len(V) < min_rows:
            raise M2DataError("%s: only %d TRAIN rows, need >= %d" % (task, len(V), min_rows))
        pmin = np.quantile(S, 1.0 - bounds_quantile, axis=0)
        pmax = np.quantile(S, bounds_quantile, axis=0)
        if not (pmax > pmin).all():
            raise M2DataError("%s: degenerate state bounds (pmin==pmax)" % task)
        aq01 = np.quantile(A, 0.01, axis=0)
        aq99 = np.quantile(A, 0.99, axis=0)
        if not (aq99 > aq01).all():
            raise M2DataError("%s: degenerate action quantiles" % task)
        v_scale, a_scale = fit_scales(V, AC, quantile)
        rq01 = np.quantile(Rm, 0.01, axis=0)
        rq99 = np.quantile(Rm, 0.99, axis=0)
        if not (rq99 > rq01).all():
            raise M2DataError("%s: degenerate residual quantiles" % task)
        return cls(task, pmin, pmax, aq01, aq99, v_scale, a_scale, rq01, rq99,
                   fit_episodes=train, dev_episodes=dev, n_rows=len(V), quantile=quantile)

    # ------------------------------------------------------------------ contract
    def to_contract(self):
        """The exact R8/R15 NormalizationContract the 42D feature needs."""
        return NormalizationContract(SIGNED_MOTION, self.pmin, self.pmax,
                                     self.v_scale, self.a_scale, self.quantile)

    def to_metadata(self):
        return {
            "task": self.task, "task_id": task_id(self.task),
            "proprio_mode": "svacc", "proprio_dim": N42,
            "normalization_mode": SIGNED_MOTION,
            "pmin": self.pmin.tolist(), "pmax": self.pmax.tolist(),
            "action_q01": self.action_q01.tolist(), "action_q99": self.action_q99.tolist(),
            "v_scale": self.v_scale.tolist(), "a_scale": self.a_scale.tolist(),
            "residual_q01": self.r_q01.tolist(), "residual_q99": self.r_q99.tolist(),
            "motion_quantile": self.quantile,
            "motion_scale_fit_source": "TRAIN episodes only",
            "n_train_episodes": len(self.fit_episodes),
            "n_dev_episodes_excluded": len(self.dev_episodes),
            "n_rows": self.n_rows,
            "fit_episodes": self.fit_episodes,
            "dev_episodes_excluded": self.dev_episodes,
            "alignment_version": ALIGNMENT_VERSION,
            "target_representation": TARGET_REPRESENTATION,
        }

    @classmethod
    def from_metadata(cls, meta):
        return cls(meta["task"], meta["pmin"], meta["pmax"], meta["action_q01"],
                   meta["action_q99"], meta["v_scale"], meta["a_scale"],
                   meta["residual_q01"], meta["residual_q99"],
                   meta["fit_episodes"], meta.get("dev_episodes_excluded", []),
                   meta["n_rows"], meta.get("motion_quantile", MOTION_QUANTILE))

    def describe(self):
        return {"task": self.task, "task_id": task_id(self.task),
                "n_train_episodes": len(self.fit_episodes), "n_rows": self.n_rows,
                "gripper_state_bounds": [float(self.pmin[6]), float(self.pmax[6]),
                                         float(self.pmin[13]), float(self.pmax[13])],
                "gripper_action_q01_q99": [float(self.action_q01[6]), float(self.action_q99[6]),
                                           float(self.action_q01[13]),
                                           float(self.action_q99[13])],
                "residual_q01_abs_max": float(np.abs(self.r_q01).max()),
                "residual_q99_abs_max": float(np.abs(self.r_q99).max())}


def A_SAFE_SLICE(T):
    """Frames usable as a state sample: all of them (state has no target requirement)."""
    return slice(0, int(T))


# ------------------------------------------------------------------------------- row build
def build_row(state, action, t, contract, task, episode=None):
    """One supervised row: 42D [S,V,Acc] feature + 14D velocity-residual target.

    Delegates the math to the R15 contract (TC.build_training_row) so the multi-task path and
    the single-task R15 trainer cannot drift. Only observed history t, t-1, t-2 is used; the
    only future quantity is the target A[t], which the contract asserts equals S[t+1].
    """
    tid = task_id(task)
    if contract is None:
        raise M2DataError("build_row needs a NormalizationContract")
    row = TC.build_training_row(state, action, t, contract)
    feats = np.asarray(row["features"])
    res = np.asarray(row["residual"])
    if feats.shape != (N42,):
        raise M2DataError("42D feature has shape %s" % (feats.shape,))
    if res.shape != (BASE_DIM,):
        raise M2DataError("14D residual has shape %s" % (res.shape,))
    if tuple(row["gripper_dims"]) != GRIPPER_DIMS:
        raise M2DataError("gripper dims %s != %s" % (row["gripper_dims"], GRIPPER_DIMS))
    out = dict(row)
    out["task"] = task
    out["task_id"] = tid                 # task id is preserved through the row build
    out["task_token"] = task_token(task)
    out["prompt"] = task_prompt(task)
    out["qualified_episode"] = qualified(task, episode) if episode else None
    if out["task_id"] != TASK_IDS[out["task"]]:
        raise M2DataError("task id drifted inside build_row")
    return out


def assert_row_no_future_leakage(state, action, t, contract, task=None):
    """Mutating every row >= t+1 must not change the row (R15 contract, re-used here)."""
    if task is not None:
        task_id(task)
    return TC.assert_no_future_leakage(state, action, t, contract)


def build_prompt(task, include_token=False, instruction=None):
    """Task-conditioned prompt. One language string per task, plus the stable task token."""
    tid = task_id(task)
    text = task_prompt(task)
    if instruction is not None and str(instruction) != text:
        raise M2DataError("task %s: supplied instruction %r drifted from the frozen prompt %r"
                          % (task, str(instruction), text))
    prompt = ("%s %s" % (task_token(task), text)) if include_token else text
    return {"task": task, "task_id": tid, "task_token": task_token(task),
            "prompt": prompt, "include_token": bool(include_token),
            "prompt_matches_dataset_instruction": (str(instruction) == text
                                                   if instruction is not None else None)}


# -------------------------------------------------------------------------- split artefacts
def build_m2_splits(hdf5_root=None, seed=0, push_t_split80=None, dev_frac=0.2):
    """Per-task TRAIN/DEV (80/20) splits over QUALIFIED names, plus per-task normalizer fits.

    push_T re-uses the frozen data/split80.json (train/val -> TRAIN/DEV) so push_T numbers stay
    comparable with R15; the other tasks get a deterministic seeded split with a decorrelated
    seed (seed + task_id).
    """
    indexes = {}
    splits = {}
    for t in TASKS:
        indexes[t] = load_task_index(t, hdf5_root)
    for t in TASKS:
        idx = indexes[t]
        if t == "push_T" and push_t_split80 and os.path.isfile(push_t_split80):
            raw = json.load(open(push_t_split80))
            train = sorted(str(x) for x in raw["train"])
            dev = sorted(str(x) for x in raw["val"])
            missing = [e for e in train + dev if e not in idx["episodes"]]
            if missing:
                raise M2DataError("split80.json names not present in push_T: %s" % missing[:5])
            if sorted(train + dev) != sorted(idx["episodes"]):
                raise M2DataError("split80.json does not cover exactly the push_T episodes")
            sp = {"train": [qualified(t, e) for e in train],
                  "dev": [qualified(t, e) for e in dev],
                  "dev_frac": float(dev_frac), "seed": int(raw.get("split_seed", seed)),
                  "n_train": len(train), "n_dev": len(dev), "task": t,
                  "task_id": task_id(t),
                  "provenance": "reused frozen data/split80.json (train/val)"}
        else:
            sp = split_task_index(idx, dev_frac=dev_frac, seed=int(seed) + task_id(t))
            sp["provenance"] = "seeded split_train_dev(seed=%d)" % (int(seed) + task_id(t))
        splits[t] = sp
    assert_splits_disjoint(*splits.values())
    out = {"seed": int(seed), "dev_frac": float(dev_frac),
           "alignment_version": ALIGNMENT_VERSION, "tasks": {}, "splits": splits}
    for t in TASKS:
        idx = indexes[t]
        sp = splits[t]
        n_rows = 0
        for e in [k.split("/", 1)[1] for k in sp["train"]]:
            n_rows += max(0, idx["frames"][e] - 3)
        out["tasks"][t] = {
            "task_id": task_id(t), "n_episodes": idx["n_episodes"],
            "frames_min": idx["frames_min"], "frames_median": idx["frames_median"],
            "frames_max": idx["frames_max"], "total_frames": idx["total_frames"],
            "instruction": idx["instruction"],
            "prompt_matches_constant": idx["prompt_matches_constant"],
            "cameras": idx["cameras"], "frequency": idx["frequency"],
            "train_episodes": sp["n_train"], "dev_episodes": sp["n_dev"],
            "train_frames": int(sum(idx["frames"][k.split("/", 1)[1]] for k in sp["train"])),
            "dev_frames": int(sum(idx["frames"][k.split("/", 1)[1]] for k in sp["dev"])),
            "train_rows_t2_to_Tm2": int(n_rows),
            "provenance": sp["provenance"],
        }
    return out, indexes, splits


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="M2 multi-task data audit + split writer")
    ap.add_argument("--hdf5-root", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-splits", default=None)
    ap.add_argument("--push-t-split80", default=os.path.join(REPO, "data", "split80.json"))
    ap.add_argument("--verify-targets", action="store_true")
    a = ap.parse_args(argv)

    for t in TASKS:
        idx = load_task_index(t, a.hdf5_root, verify_targets=a.verify_targets)
        print("[%s] task_id=%d eps=%d frames min/med/max=%d/%.0f/%d total=%d freq=%dHz cams=%s"
              % (t, idx["task_id"], idx["n_episodes"], idx["frames_min"],
                 idx["frames_median"], idx["frames_max"], idx["total_frames"],
                 idx["frequency"], ",".join(idx["cameras"])))
        print("      instruction=%r (distinct=%d, matches frozen=%s)"
              % (idx["instruction"], idx["n_distinct_instructions"],
                 idx["prompt_matches_constant"]))
    summary, indexes, splits = build_m2_splits(a.hdf5_root, seed=a.seed,
                                               push_t_split80=a.push_t_split80)
    total_train = total_dev = 0
    for t in TASKS:
        s = summary["tasks"][t]
        total_train += s["train_episodes"]
        total_dev += s["dev_episodes"]
        print("[%s] TRAIN %d eps / %d frames / %d rows  DEV %d eps / %d frames  (%s)"
              % (t, s["train_episodes"], s["train_frames"], s["train_rows_t2_to_Tm2"],
                 s["dev_episodes"], s["dev_frames"], s["provenance"]))
        norm = TaskNormalizer.fit_from_task_index(
            indexes[t], [k.split("/", 1)[1] for k in splits[t]["train"]],
            [k.split("/", 1)[1] for k in splits[t]["dev"]])
        print("      normalizer train-only: %s" % json.dumps(norm.describe(), sort_keys=True))
    print("TOTAL: 3-task TRAIN %d episodes, DEV %d episodes" % (total_train, total_dev))
    if a.out_splits:
        payload = {"schema": "m2_splits/v1", "summary": summary,
                   "splits": splits}
        with open(a.out_splits, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        print("wrote %s" % a.out_splits)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
