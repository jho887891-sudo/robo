#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2 task-aware normalizer contract for the ONE shared multi-task R42 trainer.

Why this module exists
----------------------
The three M2 tasks (push_T / stack_bowls / sweep_blocks) are structurally identical (same 14D
[six arm joints, ee, six arm joints, ee] block, same 3 cameras, same 25 Hz, A[t]==S[t+1]) but
their velocity-residual TARGET scales are NOT: the TRAIN-only residual q99 abs max is
0.0213 / 0.0520 / 0.0415 (push_T / stack_bowls / sweep_blocks), a 2.4x spread.  The frozen R9/R15
contract (training/r9_trainer_contract.load_frozen_contract) and r42_adapter.normalizer_payload()
accept exactly ONE 14D payload with no task id, so a shared model would have to clip or quantise
away most of stack_bowls' target range.  This module is the additive, versioned, fail-closed
REPLACEMENT for that single-payload assumption:

    {"schema": "m2-task-normalizer-v1",
     "tasks":  {task: <14D payload>},        # one per task, fitted on that task's TRAIN only
     "task_id": {task: int},                 # the frozen, never-renumbered task ids
     "fitted_on": {task: {...provenance...}}} # episodes + sha256 of exactly what was fitted

Nothing in the repo is modified.  The trainer (training/m2_train_shared.py) reads the payload,
selects the per-task block with payload_for() and feeds it to r42_adapter.build_labels() /
decode_command(); the gate (tools/m2_gate_check.py) recomputes the baselines with the same
payload.  normalizer_sha256() pins the WHOLE multi-task payload (every task, every episode list)
so a checkpoint cannot be resumed or deployed with one task's scales silently swapped.

Field-name contract (deliberate, documented because the repo has two naming schemes)
-----------------------------------------------------------------------------------
  * q01 / q99          = the VELOCITY-RESIDUAL quantiles, i.e. the values the 256-bin codec
                         normalises R=(A-S)-V with.  These are byte-compatible with the frozen
                         single-task contract's meta["q01"]/meta["q99"] (tools/
                         make_frozen_contract.py computes them from velocity_residual_quantiles).
  * action_q01/action_q99 = the raw 14D action quantiles of the same TRAIN episodes, carried
                         alongside for provenance / deployment diagnostics only (never used by
                         the codec on this path).
  * pmin / pmax        = positional bounds of the STATE slot (S of the 42D feature).
  * v_scale / a_scale  = signed-motion scales of the V and Acc slots.

Fail-closed rules (every one of them raises M2NormalizerError)
-------------------------------------------------------------
  * unknown task name, non-frozen task id, duplicate id;
  * a payload missing a task, carrying an extra task, or with tasks/task_id/fitted_on disagreeing;
  * a per-task block with a missing field, wrong shape (not 14D), q99<=q01, pmax<=pmin,
    non-positive motion scale, non-finite value, or a task/task_id that drifted;
  * a block that was fitted on DEV: its fit_episodes intersect its own dev_episodes_excluded,
    or intersect the authoritative DEV list passed by the caller, or its recorded DEV list
    differs from the authoritative one;
  * a block with n_train_episodes == 0 or n_dev_episodes_excluded == 0 (nothing was held out,
    so "fitted on TRAIN only" is unproven);
  * a provenance sha that does not match the block it claims to describe.
"""
import copy
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("training", "diagnostics", "code/xpolicylab/policy/OpenVLA_Pure",
             "code/xpolicylab/policy/OpenVLA_R42"):
    _p = os.path.join(REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# One module object for m2_multitask_data, whichever name it was first imported under: otherwise
# 'training.m2_multitask_data' and 'm2_multitask_data' would be two modules with two DIFFERENT
# M2DataError classes and a caller could neither catch nor test the right one.
def _resolve_m2():
    import importlib
    for name in ("training.m2_multitask_data", "m2_multitask_data"):
        if name in sys.modules:
            return sys.modules[name]
    try:
        return importlib.import_module("m2_multitask_data")
    except ImportError:
        return importlib.import_module("training.m2_multitask_data")


M2 = _resolve_m2()
sys.modules.setdefault("m2_multitask_data", M2)
sys.modules.setdefault("training.m2_multitask_data", M2)

SCHEMA = "m2-task-normalizer-v1"
BASE_DIM = M2.BASE_DIM
N42 = M2.N42
TASKS = M2.TASKS
TASK_IDS = M2.TASK_IDS
REQUIRED_TASK_FIELDS = ("task", "task_id", "normalization_mode", "pmin", "pmax", "q01", "q99",
                        "v_scale", "a_scale", "motion_quantile", "n_train_episodes",
                        "n_dev_episodes_excluded", "fit_episodes", "dev_episodes_excluded")
REQUIRED_PAYLOAD_KEYS = ("schema", "tasks", "task_id", "fitted_on")
QUANTILE_SEMANTICS = ("q01/q99 are the velocity-residual quantiles consumed by the 256-bin "
                      "codec; identical semantics to the frozen single-task contract q01/q99")


class M2NormalizerError(ValueError):
    """Raised on every multi-task normalizer contract violation.  Never degrades silently."""


# ------------------------------------------------------------------------------- utilities
def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(obj):
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _check_task(task):
    if not isinstance(task, str) or task not in TASK_IDS:
        raise M2NormalizerError("unknown task %r; the M2 multi-task normalizer knows %s"
                                % (task, list(TASKS)))
    return task


def _vec14(x, name, task="?"):
    v = np.asarray(x, np.float32).reshape(-1)
    if v.shape != (BASE_DIM,):
        raise M2NormalizerError("%s: %s must be %dD, got %s"
                                % (task, name, BASE_DIM, (np.asarray(x).shape,)))
    if not np.isfinite(v).all():
        raise M2NormalizerError("%s: %s has non-finite values" % (task, name))
    return v


def _bare(task, key):
    """Accept '<task>/<episode.hdf5>' or a bare '<episode.hdf5>' for THIS task only."""
    if not isinstance(key, str) or not key.strip():
        raise M2NormalizerError("%s: episode key must be a non-empty string, got %r" % (task, key))
    if "/" in key:
        parts = key.split("/")
        if len(parts) != 2:
            raise M2NormalizerError("%s: malformed qualified episode key %r" % (task, key))
        got_task, ep = parts
        if got_task != task:
            raise M2NormalizerError("%s: episode %r belongs to task %r (cross-task mixing is "
                                    "forbidden)" % (task, key, got_task))
    else:
        ep = key
    if not ep.endswith(".hdf5"):
        raise M2NormalizerError("%s: episode %r must end with .hdf5" % (task, ep))
    return ep


def split_episodes_for_task(split, task, require_dev=True):
    """TRAIN / DEV bare episode names of ONE task, from a per-task split dict (fail closed)."""
    _check_task(task)
    if not isinstance(split, dict):
        raise M2NormalizerError("%s: split must be a dict with 'train'/'dev', got %r"
                                % (task, type(split)))
    for key in ("train", "dev"):
        if key not in split:
            raise M2NormalizerError("%s: split is missing %r (have %s)"
                                    % (task, key, sorted(split)))
    declared = split.get("task")
    if declared is not None and declared != task:
        raise M2NormalizerError("%s: split belongs to task %r" % (task, declared))
    if declared is None and split.get("task_id") is not None \
            and int(split["task_id"]) != TASK_IDS[task]:
        raise M2NormalizerError("%s: split task_id %r does not match the frozen id %d"
                                % (task, split["task_id"], TASK_IDS[task]))
    train = sorted(_bare(task, k) for k in split["train"])
    dev = sorted(_bare(task, k) for k in split["dev"])
    if len(set(train)) != len(train) or len(set(dev)) != len(dev):
        raise M2NormalizerError("%s: split contains duplicate episodes" % task)
    if not train:
        raise M2NormalizerError("%s: split has no TRAIN episodes" % task)
    if require_dev and not dev:
        raise M2NormalizerError("%s: split has no DEV episodes, so 'fitted on TRAIN only' cannot "
                                "be proven" % task)
    if set(train) & set(dev):
        raise M2NormalizerError("%s: TRAIN/DEV overlap %s"
                                % (task, sorted(set(train) & set(dev))[:5]))
    return train, dev


# --------------------------------------------------------------------------------- fitting
def fit_task_normalizer(task, split, index=None, hdf5_root=None, quantile=M2.MOTION_QUANTILE,
                        bounds_quantile=0.99, min_rows=100, hash_episodes=False):
    """Fit ONE task's 14D payload on THAT TASK'S TRAIN EPISODES ONLY, with provenance.

    split    : per-task split dict ({'train': [...], 'dev': [...]}, qualified or bare names).
    index    : an m2_multitask_data.load_task_index() dict; loaded from hdf5_root if omitted.
    Returns  : the per-task payload dict (see the module docstring for the field contract).
    """
    tid = M2.task_id(task)
    train, dev = split_episodes_for_task(split, task)
    if index is None:
        index = M2.load_task_index(task, hdf5_root)
    if index.get("task") != task:
        raise M2NormalizerError("index is for task %r, not %r" % (index.get("task"), task))
    if int(index.get("task_id", -1)) != tid:
        raise M2NormalizerError("%s: index task_id %r != frozen %d"
                                % (task, index.get("task_id"), tid))

    norm = M2.TaskNormalizer.fit_from_task_index(index, train, dev, quantile=quantile,
                                                bounds_quantile=bounds_quantile,
                                                min_rows=min_rows)
    meta = norm.to_metadata()
    if list(meta["fit_episodes"]) != list(train):
        raise M2NormalizerError("%s: fitted episodes %s != the TRAIN list %s"
                                % (task, meta["fit_episodes"][:5], train[:5]))

    payload = {
        "task": task,
        "task_id": tid,
        "task_token": M2.task_token(task),
        "prompt": M2.task_prompt(task),
        "normalization_mode": meta["normalization_mode"],
        "proprio_mode": meta["proprio_mode"],
        "proprio_dim": int(meta["proprio_dim"]),
        "pmin": [float(x) for x in meta["pmin"]],
        "pmax": [float(x) for x in meta["pmax"]],
        "q01": [float(x) for x in meta["residual_q01"]],
        "q99": [float(x) for x in meta["residual_q99"]],
        "v_scale": [float(x) for x in meta["v_scale"]],
        "a_scale": [float(x) for x in meta["a_scale"]],
        "motion_quantile": float(meta["motion_quantile"]),
        "action_q01": [float(x) for x in meta["action_q01"]],
        "action_q99": [float(x) for x in meta["action_q99"]],
        "n_train_episodes": int(meta["n_train_episodes"]),
        "n_dev_episodes_excluded": int(meta["n_dev_episodes_excluded"]),
        "n_rows": int(meta["n_rows"]),
        "fit_episodes": list(meta["fit_episodes"]),
        "dev_episodes_excluded": list(meta["dev_episodes_excluded"]),
        "fit_source": "TRAIN episodes only",
        "motion_scale_fit_source": meta.get("motion_scale_fit_source"),
        "quantile_semantics": QUANTILE_SEMANTICS,
        "alignment_version": meta.get("alignment_version"),
        "target_representation": meta.get("target_representation"),
        "gripper_dims": list(M2.GRIPPER_DIMS),
    }
    # the fit procedure guarantees these; assert rather than assume
    if payload["fit_source"] != "TRAIN episodes only":
        raise M2NormalizerError("%s: fit_source drifted" % task)
    if set(payload["fit_episodes"]) & set(payload["dev_episodes_excluded"]):
        raise M2NormalizerError("%s: the normalizer was fitted on DEV episodes" % task)
    prov = _provenance(task, train, dev, index, hash_episodes=hash_episodes,
                       quantile=quantile, bounds_quantile=bounds_quantile,
                       n_rows=payload["n_rows"])
    # the per-task block is hashed BEFORE any provenance is attached to it, so its sha256 stays
    # recomputable from the block alone (the provenance lives in the payload's 'fitted_on').
    prov["payload_sha256"] = sha256_json(payload)
    prov["quantile_semantics"] = QUANTILE_SEMANTICS
    return payload, prov


def _provenance(task, train, dev, index, hash_episodes=False, quantile=None,
                bounds_quantile=None, n_rows=None):
    hdf5_dir = index.get("hdf5_dir")
    prov = {"task": task, "task_id": TASK_IDS[task], "hdf5_dir": hdf5_dir,
            "train_episodes": list(train), "n_train_episodes": len(train),
            "dev_episodes_excluded": list(dev), "n_dev_episodes_excluded": len(dev),
            "train_episodes_sha256": sha256_json(list(train)),
            "split_sha256": sha256_json({"task": task, "train": list(train), "dev": list(dev)}),
            "quantile": float(quantile if quantile is not None else M2.MOTION_QUANTILE),
            "bounds_quantile": float(bounds_quantile if bounds_quantile is not None else 0.99),
            "n_rows": None if n_rows is None else int(n_rows),
            "fit_source": "TRAIN episodes only",
            "episode_file_sha256": None}
    if hash_episodes:
        files = {}
        if not hdf5_dir or not os.path.isdir(hdf5_dir):
            raise M2NormalizerError("%s: cannot hash episodes, no hdf5_dir (%r)"
                                    % (task, hdf5_dir))
        for ep in train:
            p = os.path.join(hdf5_dir, ep)
            if not os.path.isfile(p):
                raise M2NormalizerError("%s: TRAIN episode missing: %s" % (task, p))
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            files[ep] = h.hexdigest()
        prov["episode_file_sha256"] = files
    return prov


def build_multi_payload(task_payloads):
    """Assemble the versioned multi-task payload from per-task blocks (fail closed)."""
    if not task_payloads:
        raise M2NormalizerError("need at least one per-task payload")
    tasks = {}
    task_id = {}
    fitted_on = {}
    for tp in task_payloads:
        if not isinstance(tp, dict):
            raise M2NormalizerError("per-task payload must be a dict, got %r" % (type(tp),))
        task = _check_task(tp.get("task"))
        if task in tasks:
            raise M2NormalizerError("duplicate per-task payload for %r" % (task,))
        block = copy.deepcopy(tp)
        block.pop("provenance", None)
        tasks[task] = block
        task_id[task] = TASK_IDS[task]
        if int(block.get("task_id", -1)) != TASK_IDS[task]:
            raise M2NormalizerError("%s: block task_id %r != frozen %d"
                                    % (task, block.get("task_id"), TASK_IDS[task]))
        fitted_on[task] = {
            "train_episodes": list(block["fit_episodes"]),
            "n_train_episodes": int(block["n_train_episodes"]),
            "dev_episodes_excluded": list(block["dev_episodes_excluded"]),
            "n_dev_episodes_excluded": int(block["n_dev_episodes_excluded"]),
            "train_episodes_sha256": sha256_json(list(block["fit_episodes"])),
            "split_sha256": sha256_json({"task": task,
                                         "train": list(block["fit_episodes"]),
                                         "dev": list(block["dev_episodes_excluded"])}),
            "payload_sha256": sha256_json(block),
            "fit_source": "TRAIN episodes only",
        }
    payload = {"schema": SCHEMA, "tasks": tasks, "task_id": task_id, "fitted_on": fitted_on}
    assert_payload_complete(payload)
    return payload


def fit_multi_normalizer(tasks, splits, hdf5_root=None, indexes=None, quantile=M2.MOTION_QUANTILE,
                         bounds_quantile=0.99, min_rows=100, hash_episodes=False):
    """Fit every requested task on its own TRAIN episodes and return the multi-task payload."""
    task_list = _normalize_task_list(tasks)
    split_map = _split_map(splits)
    missing = [t for t in task_list if t not in split_map]
    if missing:
        raise M2NormalizerError("no split for task(s) %s (have %s)"
                                % (missing, sorted(split_map)))
    blocks, provs = [], {}
    for t in task_list:
        idx = None if indexes is None else indexes.get(t)
        block, prov = fit_task_normalizer(t, split_map[t], index=idx, hdf5_root=hdf5_root,
                                          quantile=quantile, bounds_quantile=bounds_quantile,
                                          min_rows=min_rows, hash_episodes=hash_episodes)
        blocks.append(block)
        provs[t] = prov
    payload = build_multi_payload(blocks)
    for t in task_list:                       # the assembled payload must carry the fit provenance
        want = payload["fitted_on"][t]["payload_sha256"]
        if want != provs[t]["payload_sha256"]:
            raise M2NormalizerError("%s: assembled payload sha %s != fitted sha %s"
                                    % (t, want, provs[t]["payload_sha256"]))
    return payload, provs


# ----------------------------------------------------------------------------- validation
def _normalize_task_list(tasks):
    if isinstance(tasks, str):
        tasks = [x.strip() for x in tasks.split(",") if x.strip()]
    out = []
    for t in tasks:
        _check_task(t)
        if t not in out:
            out.append(t)
    if not out:
        raise M2NormalizerError("empty task list")
    return sorted(out, key=lambda x: TASK_IDS[x])


def _split_map(splits):
    if not isinstance(splits, dict):
        raise M2NormalizerError("splits must be a dict of {task: split} or an m2_splits artifact")
    if "splits" in splits and isinstance(splits["splits"], dict):
        splits = splits["splits"]
    if "train" in splits or "dev" in splits:
        raise M2NormalizerError("splits looks like a SINGLE split; a multi-task fit needs "
                                "{task: split}")
    return splits


def payload_tasks(payload):
    """Tasks present in a payload, in frozen task-id order (raises if malformed)."""
    if not isinstance(payload, dict):
        raise M2NormalizerError("payload must be a dict")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise M2NormalizerError("payload has no 'tasks' block")
    out = []
    for t in tasks:
        _check_task(t)
        out.append(t)
    return sorted(out, key=lambda x: TASK_IDS[x])


def assert_payload_complete(payload, tasks=None, dev_lists=None, forbid_dev_fit=True):
    """Full fail-closed validation of a multi-task normalizer payload.  Returns True."""
    if not isinstance(payload, dict):
        raise M2NormalizerError("payload must be a dict, got %r" % (type(payload),))
    for key in REQUIRED_PAYLOAD_KEYS:
        if key not in payload:
            raise M2NormalizerError("payload is missing %r (have %s)"
                                    % (key, sorted(payload)))
    if payload.get("schema") != SCHEMA:
        raise M2NormalizerError("payload schema %r != %r (a single-task R9/R15 payload is not a "
                                "multi-task payload)" % (payload.get("schema"), SCHEMA))
    present = payload_tasks(payload)
    for key in ("task_id", "fitted_on"):
        block = payload.get(key)
        if not isinstance(block, dict):
            raise M2NormalizerError("payload %r must be a dict" % key)
        if sorted(block) != sorted(present):
            raise M2NormalizerError("payload %r covers %s but 'tasks' covers %s"
                                    % (key, sorted(block), present))
    if tasks is not None:
        want = _normalize_task_list(tasks)
        missing = [t for t in want if t not in present]
        extra = [t for t in present if t not in want]
        if missing or extra:
            raise M2NormalizerError("payload task set %s != requested %s (missing %s, extra %s)"
                                    % (present, want, missing, extra))
    ids_seen = {}
    for t in present:
        if int(payload["task_id"][t]) != TASK_IDS[t]:
            raise M2NormalizerError("%s: payload task_id %r != frozen %d"
                                    % (t, payload["task_id"][t], TASK_IDS[t]))
        if int(payload["task_id"][t]) in ids_seen:
            raise M2NormalizerError("task ids collide: %s and %s both -> %r"
                                    % (ids_seen[int(payload["task_id"][t])], t,
                                       payload["task_id"][t]))
        ids_seen[int(payload["task_id"][t])] = t
    for t in present:
        _assert_block_complete(t, payload["tasks"][t])
        _assert_provenance(t, payload, dev_lists=dev_lists, forbid_dev_fit=forbid_dev_fit)
    return True


def _assert_block_complete(task, block):
    if not isinstance(block, dict):
        raise M2NormalizerError("%s: per-task payload must be a dict" % task)
    missing = [k for k in REQUIRED_TASK_FIELDS if k not in block]
    if missing:
        raise M2NormalizerError("%s: per-task payload missing fields %s" % (task, missing))
    if block.get("task") != task:
        raise M2NormalizerError("%s: per-task payload declares task %r" % (task, block.get("task")))
    if int(block.get("task_id", -1)) != TASK_IDS[task]:
        raise M2NormalizerError("%s: per-task payload declares task_id %r (frozen %d)"
                                % (task, block.get("task_id"), TASK_IDS[task]))
    from r8_normalization import SIGNED_MOTION
    if block.get("normalization_mode") != SIGNED_MOTION:
        raise M2NormalizerError("%s: normalization_mode %r != %r"
                                % (task, block.get("normalization_mode"), SIGNED_MOTION))
    if block.get("proprio_dim") is not None and int(block["proprio_dim"]) != N42:
        raise M2NormalizerError("%s: proprio_dim %r != %d"
                                % (task, block.get("proprio_dim"), N42))
    pmin = _vec14(block["pmin"], "pmin", task)
    pmax = _vec14(block["pmax"], "pmax", task)
    q01 = _vec14(block["q01"], "q01", task)
    q99 = _vec14(block["q99"], "q99", task)
    vs = _vec14(block["v_scale"], "v_scale", task)
    as_ = _vec14(block["a_scale"], "a_scale", task)
    if not (pmax > pmin).all():
        raise M2NormalizerError("%s: pmax must exceed pmin on every dim" % task)
    if not (q99 > q01).all():
        raise M2NormalizerError("%s: q99 must exceed q01 on every dim" % task)
    if not ((vs > 0).all() and (as_ > 0).all()):
        raise M2NormalizerError("%s: motion scales must be strictly positive" % task)
    mq = float(block["motion_quantile"])
    if not (0.0 < mq < 1.0):
        raise M2NormalizerError("%s: motion_quantile %r must be in (0,1)" % (task, mq))
    for key in ("fit_episodes", "dev_episodes_excluded"):
        ep = block[key]
        if not isinstance(ep, (list, tuple)) or not ep:
            raise M2NormalizerError("%s: %s must be a non-empty list" % (task, key))
        for e in ep:
            if not isinstance(e, str) or "/" in e or not e.endswith(".hdf5"):
                raise M2NormalizerError("%s: %s holds a non-bare episode key %r" % (task, key, e))
        if len(set(ep)) != len(ep):
            raise M2NormalizerError("%s: %s has duplicates" % (task, key))
    if int(block["n_train_episodes"]) != len(block["fit_episodes"]):
        raise M2NormalizerError("%s: n_train_episodes %r != %d episodes"
                                % (task, block["n_train_episodes"], len(block["fit_episodes"])))
    if int(block["n_train_episodes"]) < 1:
        raise M2NormalizerError("%s: normalizer was fitted on 0 episodes" % task)
    if int(block["n_dev_episodes_excluded"]) != len(block["dev_episodes_excluded"]):
        raise M2NormalizerError("%s: n_dev_episodes_excluded %r != %d episodes"
                                % (task, block["n_dev_episodes_excluded"],
                                   len(block["dev_episodes_excluded"])))
    if int(block["n_dev_episodes_excluded"]) < 1:
        raise M2NormalizerError("%s: n_dev_episodes_excluded == 0: nothing was held out, so "
                                "'fitted on TRAIN only' is unproven" % task)
    overlap = sorted(set(block["fit_episodes"]) & set(block["dev_episodes_excluded"]))
    if overlap:
        raise M2NormalizerError("%s: the normalizer was fitted on DEV episodes %s"
                                % (task, overlap[:5]))
    return True


def _assert_provenance(task, payload, dev_lists=None, forbid_dev_fit=True):
    block = payload["tasks"][task]
    fo = payload["fitted_on"][task]
    if not isinstance(fo, dict):
        raise M2NormalizerError("%s: fitted_on entry must be a dict" % task)
    for key in ("train_episodes", "n_train_episodes"):
        if key not in fo:
            raise M2NormalizerError("%s: fitted_on is missing %r" % (task, key))
    if list(fo["train_episodes"]) != list(block["fit_episodes"]):
        raise M2NormalizerError("%s: fitted_on.train_episodes disagrees with the block's "
                                "fit_episodes" % task)
    if int(fo["n_train_episodes"]) != int(block["n_train_episodes"]):
        raise M2NormalizerError("%s: fitted_on.n_train_episodes disagrees with the block" % task)
    want = fo.get("payload_sha256")
    if want is None:
        raise M2NormalizerError("%s: fitted_on carries no payload_sha256 (provenance unproven)"
                                % task)
    got = sha256_json(block)
    if got != want:
        raise M2NormalizerError("%s: per-task payload sha256 %s != recorded %s (the block was "
                                "edited after fitting)" % (task, got, want))
    if forbid_dev_fit:
        dev = set(block["dev_episodes_excluded"])
        bad = sorted(set(block["fit_episodes"]) & dev)
        if bad:
            raise M2NormalizerError("%s: the normalizer was fitted on DEV episodes %s"
                                    % (task, bad[:5]))
        if dev_lists is not None and task in dev_lists:
            auth = set(_bare(task, k) for k in dev_lists[task])
            if auth != dev:
                raise M2NormalizerError("%s: payload DEV list differs from the authoritative "
                                        "split (payload-only %s, split-only %s)"
                                        % (task, sorted(dev - auth)[:5], sorted(auth - dev)[:5]))
            bad = sorted(set(block["fit_episodes"]) & auth)
            if bad:
                raise M2NormalizerError("%s: the normalizer was fitted on DEV episodes %s "
                                        "(authoritative split)" % (task, bad[:5]))
    return True


def payload_for(payload, task):
    """The 14D payload of ONE task (validated against the whole multi-task payload)."""
    assert_payload_complete(payload)
    _check_task(task)
    tasks = payload_tasks(payload)
    if task not in tasks:
        raise M2NormalizerError("payload has no normalizer for task %r (has %s)"
                                % (task, tasks))
    if int(payload["task_id"][task]) != TASK_IDS[task]:
        raise M2NormalizerError("%s: task id %r != frozen %d"
                                % (task, payload["task_id"][task], TASK_IDS[task]))
    return copy.deepcopy(payload["tasks"][task])


def decode_context(payload, task):
    """Everything build_labels()/decode_command() need for ONE task (numpy, no model)."""
    block = payload_for(payload, task)
    return {"task": task, "task_id": TASK_IDS[task],
            "q01": np.asarray(block["q01"], np.float32),
            "q99": np.asarray(block["q99"], np.float32),
            "pmin": np.asarray(block["pmin"], np.float32),
            "pmax": np.asarray(block["pmax"], np.float32),
            "v_scale": np.asarray(block["v_scale"], np.float32),
            "a_scale": np.asarray(block["a_scale"], np.float32),
            "normalizer_sha256": normalizer_sha256(payload)}


def contract_for(payload, task):
    """The R8 NormalizationContract of ONE task (drives the 42D [S,V,Acc] feature)."""
    block = payload_for(payload, task)
    from r8_normalization import NormalizationContract, SIGNED_MOTION
    return NormalizationContract(SIGNED_MOTION, block["pmin"], block["pmax"],
                                 block["v_scale"], block["a_scale"], block["motion_quantile"])


def normalizer_sha256(payload):
    """sha256 over the WHOLE multi-task payload (so a checkpoint pins every task at once)."""
    assert_payload_complete(payload)
    return sha256_json(payload)


def assert_payload_task_set(payload, tasks):
    """Resume/deploy guard: the payload must describe EXACTLY this task set."""
    return assert_payload_complete(payload, tasks=tasks)


def assert_same_normalizer(payload_a, payload_b, what="normalizer"):
    a = normalizer_sha256(payload_a)
    b = normalizer_sha256(payload_b)
    if a != b:
        raise M2NormalizerError("%s mismatch: %s != %s (a checkpoint from a different "
                                "normalizer fit must not be resumed)" % (what, a, b))
    return a


# ----------------------------------------------------------------------------------- io
def save_payload(path, payload):
    assert_payload_complete(payload)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    return normalizer_sha256(payload)


def load_payload(path, tasks=None, dev_lists=None):
    if not os.path.isfile(path):
        raise M2NormalizerError("multi-task normalizer payload not found: %s" % path)
    with open(path) as fh:
        payload = json.load(fh)
    assert_payload_complete(payload, tasks=tasks, dev_lists=dev_lists)
    return payload


def describe(payload):
    assert_payload_complete(payload)
    out = {"schema": payload["schema"], "normalizer_sha256": normalizer_sha256(payload),
           "tasks": {}}
    for t in payload_tasks(payload):
        b = payload["tasks"][t]
        out["tasks"][t] = {
            "task_id": b["task_id"],
            "residual_q99_abs_max": float(np.abs(np.asarray(b["q99"], np.float32)).max()),
            "residual_q01_abs_max": float(np.abs(np.asarray(b["q01"], np.float32)).max()),
            "n_train_episodes": b["n_train_episodes"],
            "n_dev_episodes_excluded": b["n_dev_episodes_excluded"],
            "n_rows": b.get("n_rows"),
            "gripper_state_bounds": [float(b["pmin"][6]), float(b["pmax"][6]),
                                     float(b["pmin"][13]), float(b["pmax"][13])],
            "v_scale_abs_max": float(np.abs(np.asarray(b["v_scale"], np.float32)).max()),
        }
    out["residual_q99_spread_ratio"] = _spread_ratio(payload, "q99")
    return out


def _spread_ratio(payload, key):
    vals = [float(np.abs(np.asarray(payload["tasks"][t][key], np.float32)).max())
            for t in payload_tasks(payload)]
    if len(vals) < 2:
        return 1.0
    lo = min(vals)
    return float(max(vals) / lo) if lo > 0 else float("inf")


# ----------------------------------------------------------------------------------- CLI
def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Fit + validate the M2 task-aware normalizer payload")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--splits", default=os.path.join(REPO, "data", "m2_splits.json"))
    ap.add_argument("--hdf5-root", default=None)
    ap.add_argument("--out", required=True, help="where to write the versioned payload JSON")
    ap.add_argument("--hash-episodes", action="store_true",
                    help="also record the sha256 of every TRAIN episode file (slower)")
    a = ap.parse_args(argv)
    with open(a.splits) as fh:
        splits = json.load(fh)
    tasks = [x.strip() for x in a.tasks.split(",") if x.strip()]
    payload, provs = fit_multi_normalizer(tasks, splits, hdf5_root=a.hdf5_root,
                                          hash_episodes=a.hash_episodes)
    dev_lists = {t: [k.split("/", 1)[-1] for k in _split_map(splits)[t]["dev"]]
                 for t in tasks}
    sha = save_payload(a.out, payload)
    assert_payload_complete(payload, tasks=tasks, dev_lists=dev_lists)
    print(json.dumps(describe(payload), indent=1, sort_keys=True))
    print("normalizer_sha256 = %s" % sha)
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
