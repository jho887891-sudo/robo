#!/usr/bin/env python3
"""R44: joint_delta_v1 action-target contract (CPU only, numpy).

Why a new contract and not a new normalizer for the old one
-----------------------------------------------------------
The deployed R42 checkpoint is trained on the *velocity residual*

    R_t = (A_t - S_t) - V_t

with per-task q01/q99 fitted on R.  R44 trains on the plain joint displacement

    Delta_t = A_t - S_t

where A_t is the expert action at frame t and S_t the measured 14D state (the HDF5
contract is A_t == S_{t+1} bit-exactly, re-verified by diagnostics/r44_prepare_delta.py).

R and Delta are different random variables with different support, different quantiles and
different physical meaning.  Swapping the normalizer under a checkpoint trained on R would
silently re-scale every action token, so the target identity is carried explicitly:
CONTRACT_ID, a payload sha256 over the whole normalizer and a per-checkpoint metadata field.
refuse_legacy() fails closed on any payload that is not this contract.
"""
import hashlib
import json

import numpy as np

BASE_DIM = 14
N_DIM = 42
N_ACTIONS = 14
CONTRACT_ID = "joint_delta_v1"
TARGET_DESCRIPTION = "Delta_t = A_t - S_t (A_t == S_{t+1}, no velocity term)"
CODEC_MODE = "fixed"
N_BINS = 256
ACTION_TOKEN_BEGIN_IDX = 31743
LEGAL_TOKEN_MIN = ACTION_TOKEN_BEGIN_IDX + 1
LEGAL_TOKEN_MAX = ACTION_TOKEN_BEGIN_IDX + N_BINS
PAD_TOKEN_ID = 32000
GRIPPER_DIMS = (6, 13)
MAX_NORM_QUANT_ERROR = 2.0 / (N_BINS - 1)   # 0.007843 in normalised [-1,1] units


class R44ContractError(ValueError):
    """Any violation of the joint_delta_v1 target contract."""


def _a14(x, name):
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if a.shape != (BASE_DIM,):
        raise R44ContractError("%s must be 14D, got %s" % (name, a.shape))
    if not np.isfinite(a).all():
        raise R44ContractError("%s contains non-finite values" % name)
    return a


# ------------------------------------------------------------------ data-contract checks
def delta_target(state, action):
    """Delta_t = A_t - S_t. The only place the training target is defined."""
    s = _a14(state, "state")
    a = _a14(action, "action")
    return (a - s).astype(np.float32)


def verify_step_contract(state_t, action_t, state_next, atol=0.0):
    """A_t == S_{t+1} (bit-exact by default) and the delta is therefore the S->S' motion."""
    a = _a14(action_t, "action")
    s_next = _a14(state_next, "state_next")
    err = float(np.max(np.abs(a - s_next)))
    if not (err <= atol):
        raise R44ContractError("A_t != S_{t+1} (max abs diff %.3e > %.3e)" % (err, atol))
    return delta_target(state_t, a)


# ------------------------------------------------------------------------- normalizer
def _frozen_payload(target, codec, tasks, fit):
    return {"schema": "r44-delta-normalizer/v1", "target": target, "codec": codec,
            "n_bins": N_BINS, "token_window": [LEGAL_TOKEN_MIN, LEGAL_TOKEN_MAX],
            "fit": fit, "tasks": tasks}


def payload_sha256(payload):
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def fit_normalizer(deltas_by_task, quantiles=(0.01, 0.99), split_sha256=None,
                   min_rows=100):
    """TRAIN-only per-task quantiles of the displacement target."""
    q_lo, q_hi = float(quantiles[0]), float(quantiles[1])
    if not (0.0 < q_lo < q_hi < 1.0):
        raise R44ContractError("quantiles must satisfy 0<lo<hi<1, got %r" % (quantiles,))
    tasks = {}
    for task, rows in sorted(deltas_by_task.items()):
        d = np.asarray(rows, dtype=np.float32)
        if d.ndim != 2 or d.shape[1] != BASE_DIM:
            raise R44ContractError("%s deltas must be (N,14), got %s" % (task, d.shape))
        if len(d) < min_rows:
            raise R44ContractError("%s has only %d TRAIN rows (< %d)" % (task, len(d), min_rows))
        if not np.isfinite(d).all():
            raise R44ContractError("%s deltas contain non-finite values" % task)
        q01 = np.quantile(d, q_lo, axis=0).astype(np.float32)
        q99 = np.quantile(d, q_hi, axis=0).astype(np.float32)
        if (q99 <= q01).any():
            raise R44ContractError("%s has a degenerate displacement dimension" % task)
        tasks[task] = {"q01": q01.tolist(), "q99": q99.tolist(), "n_rows": int(len(d)),
                       "target_mean": d.mean(axis=0).astype(np.float32).tolist(),
                       "target_std": d.std(axis=0).astype(np.float32).tolist(),
                       "zero_fraction": float(np.mean(np.all(d == 0.0, axis=1)))}
    fit = {"source": "TRAIN only", "quantiles": [q_lo, q_hi],
           "split_sha256": split_sha256, "n_dims": BASE_DIM,
           "target": TARGET_DESCRIPTION}
    payload = _frozen_payload(CONTRACT_ID, CODEC_MODE, tasks, fit)
    payload["sha256"] = payload_sha256(payload)
    return payload


def validate_normalizer(payload, expect_contract=CONTRACT_ID):
    """Fail closed unless this payload matches the complete frozen displacement/codec contract."""
    if not isinstance(payload, dict):
        raise R44ContractError("normalizer payload must be a dict")
    if payload.get("schema") != "r44-delta-normalizer/v1":
        raise R44ContractError("normalizer schema is not r44-delta-normalizer/v1")
    got = payload.get("target")
    if got != expect_contract:
        raise R44ContractError(
            "normalizer target %r != %r: a residual (R = A-S-V) normalizer cannot be reused "
            "for the displacement target" % (got, expect_contract))
    if payload.get("codec") != CODEC_MODE:
        raise R44ContractError("codec %r != %r" % (payload.get("codec"), CODEC_MODE))
    if int(payload.get("n_bins", -1)) != N_BINS:
        raise R44ContractError("normalizer n_bins %r != frozen %d" % (payload.get("n_bins"), N_BINS))
    if payload.get("token_window") != [LEGAL_TOKEN_MIN, LEGAL_TOKEN_MAX]:
        raise R44ContractError("normalizer token window differs from the frozen legal action-token window")
    fit = payload.get("fit")
    if not isinstance(fit, dict):
        raise R44ContractError("normalizer fit metadata is missing")
    if fit.get("source") != "TRAIN only":
        raise R44ContractError("normalizer was not explicitly fitted on TRAIN only")
    if int(fit.get("n_dims", -1)) != BASE_DIM or fit.get("target") != TARGET_DESCRIPTION:
        raise R44ContractError("normalizer fit target/dimension metadata differs from joint_delta_v1")
    if payload.get("sha256") != payload_sha256(payload):
        raise R44ContractError("normalizer sha256 mismatch: payload was modified after fitting")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise R44ContractError("normalizer task table is empty or malformed")
    for task, block in tasks.items():
        if not isinstance(block, dict):
            raise R44ContractError("%s normalizer block is malformed" % task)
        q01 = _a14(block["q01"], "%s.q01" % task)
        q99 = _a14(block["q99"], "%s.q99" % task)
        if (q99 <= q01).any():
            raise R44ContractError("%s q99 <= q01" % task)
    return True


def normalizer_for_task(payload, task):
    validate_normalizer(payload)
    task = str(task)
    if task not in payload["tasks"]:
        raise R44ContractError("task %r is not in the frozen normalizer (has %s)"
                               % (task, sorted(payload["tasks"])))
    blk = payload["tasks"][task]
    return np.asarray(blk["q01"], np.float32), np.asarray(blk["q99"], np.float32)


def refuse_legacy(meta, field="target_representation"):
    """A checkpoint must declare this contract; an R42 residual checkpoint must not pass."""
    if not isinstance(meta, dict):
        raise R44ContractError("checkpoint metadata must be a dict")
    if meta.get(field) != CONTRACT_ID:
        raise R44ContractError("checkpoint %s=%r is not %r: refusing to treat an old residual "
                               "checkpoint as a delta checkpoint"
                               % (field, meta.get(field), CONTRACT_ID))
    return True


# ------------------------------------------------------------------------------ codec
def _codec():
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    pure = os.path.join(here, "..", "code", "xpolicylab", "policy", "OpenVLA_Pure")
    pure = os.path.abspath(pure)
    if pure not in sys.path:
        sys.path.insert(0, pure)
    import action_codec
    return action_codec


def normalize_delta(delta, q01, q99):
    ac = _codec()
    return ac.normalize(delta, q01, q99)


def encode_delta(delta, task, payload):
    """Delta (14D, physical) -> 14 bin indices (0..255) -> 14 legal token ids."""
    ac = _codec()
    q01, q99 = normalizer_for_task(payload, task)
    norm = ac.normalize(delta, q01, q99)
    idx = ac.encode_indices(norm, mode=ac.MODE_FIXED)
    if idx.shape != (N_ACTIONS,) or idx.min() < 0 or idx.max() > N_BINS - 1:
        raise R44ContractError("encoded indices out of range: %r" % (idx,))
    return idx.astype(np.int64), ac.token_ids(idx)


def decode_delta(idx, task, payload):
    ac = _codec()
    q01, q99 = normalizer_for_task(payload, task)
    norm = ac.decode_indices(np.asarray(idx, np.int64), mode=ac.MODE_FIXED)
    return ac.denormalize(norm, q01, q99).astype(np.float32)


def recover_action(state, idx, task, payload):
    """Offline decoder for the new contract: A_hat = S + Delta_hat (no velocity, no residual)."""
    s = _a14(state, "state")
    d = decode_delta(idx, task, payload)
    return (s + d).astype(np.float32)


def roundtrip_error(state, action, task, payload):
    """Encode the expert displacement and decode it back; returns the physical error vector."""
    delta = delta_target(state, action)
    idx, tokens = encode_delta(delta, task, payload)
    if int((tokens < LEGAL_TOKEN_MIN).sum() + (tokens > LEGAL_TOKEN_MAX).sum()) != 0:
        raise R44ContractError("token ids left the legal window")
    a_hat = recover_action(state, idx, task, payload)
    return (a_hat - _a14(action, "action")).astype(np.float32), idx, tokens


# -------------------------------------------------------------------- epoch sampler
class EpochSampler:
    """Epoch-complete, no-replacement ordering: every TRAIN sample exactly once per epoch."""

    def __init__(self, n_samples, base_seed, epoch):
        n = int(n_samples)
        if n <= 0:
            raise R44ContractError("n_samples must be positive")
        if int(epoch) < 0:
            raise R44ContractError("epoch must be >= 0")
        self.n = n
        self.base_seed = int(base_seed)
        self.epoch = int(epoch)
        rng = np.random.default_rng([self.base_seed, self.epoch])
        self.order = rng.permutation(n).astype(np.int64)

    def order_hash(self):
        return hashlib.sha256(self.order.tobytes()).hexdigest()

    def coverage(self):
        counts = np.bincount(self.order, minlength=self.n).astype(np.int64)
        return {"unique_seen": int((counts > 0).sum()),
                "total": int(self.n),
                "min_visits": int(counts.min()),
                "max_visits": int(counts.max()),
                "complete": bool(counts.min() == 1 and counts.max() == 1
                                 and (counts > 0).sum() == self.n)}
