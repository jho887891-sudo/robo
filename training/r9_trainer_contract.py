#!/usr/bin/env python3
"""R9 Gate B: 42D [S,V,Acc] + velocity-residual target (R) for the training entrypoint.

training/train_pushT_pure2.py is left BYTE-IDENTICAL: it still trains 14D absolute/residual
targets, uses a 14D ProprioProjector and 80-episode target quantiles. This module is the
explicit 42D/R path. It refuses (fail-closed) anything that would silently mix the two:
  * a frozen contract that was not fitted on exactly 64 TRAIN / 16 DEV episodes,
  * a projector width other than 42,
  * a frame index below 2 (V and Acc are undefined there),
  * a target mode other than velocity_residual when 42D features are used.
"""
import json
import os

import numpy as np

import action_codec as AC
import r8_vla_adapter as A
from r8_normalization import NormalizationContract, SIGNED_MOTION

BASE_DIM = 14
N_DIM = 42
GRIPPER_DIMS = (6, 13)
TARGET_VELOCITY_RESIDUAL = "velocity_residual"
MIN_HISTORY_FRAMES = 2
TOKEN_BEGIN = 31744
TOKEN_END = 31999


def load_frozen_contract(path, require_64_16=True):
    """Fail closed: only a 64-TRAIN / 16-DEV signed-motion contract may drive 42D training."""
    if not os.path.isfile(path):
        raise FileNotFoundError("frozen contract not found: %s" % path)
    meta = json.load(open(path))
    prov = meta.get("frozen_from", {})
    if require_64_16 and (prov.get("n_train_episodes_for_quantiles") != 64
                          or prov.get("n_dev_episodes_excluded") != 16):
        raise ValueError("frozen contract must come from 64 TRAIN / 16 DEV (got %r / %r)"
                         % (prov.get("n_train_episodes_for_quantiles"),
                            prov.get("n_dev_episodes_excluded")))
    if meta.get("normalization_mode") != SIGNED_MOTION:
        raise ValueError("frozen contract must use signed_motion normalisation, got %r"
                         % (meta.get("normalization_mode"),))
    if int(meta.get("proprio_dim", -1)) != N_DIM:
        raise ValueError("frozen contract proprio_dim must be %d" % N_DIM)
    if meta.get("target_representation") != TARGET_VELOCITY_RESIDUAL:
        raise ValueError("frozen contract target_representation must be %r"
                         % TARGET_VELOCITY_RESIDUAL)
    q01 = np.asarray(meta.get("q01", []), np.float32)
    q99 = np.asarray(meta.get("q99", []), np.float32)
    if q01.shape != (BASE_DIM,) or q99.shape != (BASE_DIM,) or not (q99 > q01).all():
        raise ValueError("frozen contract must carry finite 14D q01<q99")
    return NormalizationContract.from_metadata(meta), meta


def select_splits(split_path, n_dev=16):
    """64 TRAIN / 16 DEV / 20 old holdout, episode-disjoint, fail closed on overlap."""
    split = json.load(open(split_path))
    train = sorted(str(x) for x in split["train"])
    holdout = sorted(str(x) for x in split["val"])
    dev = sorted(set(train[::max(1, len(train) // n_dev)][:n_dev]))
    fit = [e for e in train if e not in set(dev)]
    if set(fit) & set(dev) or set(fit) & set(holdout) or set(dev) & set(holdout):
        raise ValueError("TRAIN/DEV/holdout overlap")
    return {"train": fit, "dev": dev, "holdout": holdout}


def frame_pool(n_frames, require_history=True):
    """History-safe frames: t >= 2 (and t <= n-2 when the row needs a target)."""
    if require_history:
        t = np.arange(MIN_HISTORY_FRAMES, n_frames - 1, dtype=np.int64)
    else:
        t = np.arange(0, max(1, n_frames - 1), dtype=np.int64)
    if t.size == 0:
        raise ValueError("episode too short for a history-safe frame pool: %d" % n_frames)
    return t


def check_frame_index(t):
    if int(t) < MIN_HISTORY_FRAMES:
        raise ValueError("frame %d has no [S,V,Acc] history (need t >= %d)"
                         % (t, MIN_HISTORY_FRAMES))
    return int(t)


def proprio42(state_now, state_prev, state_prev2, contract):
    out = A.build_proprio(np.asarray(state_now, np.float32),
                          np.asarray(state_prev, np.float32),
                          np.asarray(state_prev2, np.float32), contract)
    if out.shape != (N_DIM,) or not np.isfinite(out).all():
        raise ValueError("42D feature invalid: %s" % (out.shape,))
    return out


def target_residual(state, action, velocity):
    return A.training_target(np.asarray(state, np.float32), np.asarray(action, np.float32),
                             np.asarray(velocity, np.float32))


def encode_tokens(residual, q01, q99):
    """R -> [-1,1] -> 256-bin index -> legal OpenVLA action token (31744..31999)."""
    idx = AC.encode_indices(AC.normalize(residual, q01, q99), mode=AC.MODE_FIXED)
    ids = AC.token_ids(idx)
    AC.assert_legal_token_ids(ids)
    return np.asarray(idx, np.int64), np.asarray(ids, np.int64)


def decode_command(tokens, q01, q99, state, velocity, bounds_path):
    idx = np.asarray(tokens, np.int64).reshape(-1)
    if idx.size == BASE_DIM and idx.max() >= TOKEN_BEGIN:
        idx = idx - TOKEN_BEGIN
    nb = np.load(bounds_path)
    cmd, decoded, excluded = A.decode_tokens_to_command(
        idx, q01, q99, state, velocity, AC,
        cv_command=np.asarray(state, np.float32) + np.asarray(velocity, np.float32))
    return (A.clip_command(cmd, nb["amin"].astype(np.float32), nb["amax"].astype(np.float32)),
            decoded, excluded)


def assert_projector_dim(projector_dim):
    if int(projector_dim) != N_DIM:
        raise ValueError("42D features require a %d-wide projector, got %d"
                         % (N_DIM, projector_dim))
    return True


def assert_target_mode(mode):
    if mode != TARGET_VELOCITY_RESIDUAL:
        raise ValueError("42D path supervises %r, not %r" % (TARGET_VELOCITY_RESIDUAL, mode))
    return True


def standard_entrypoint_gaps():
    """What training/train_pushT_pure2.py does NOT yet provide (documented, not guessed)."""
    return [
        {"gap": "proprio_dim", "standard": "14 (current S_t only)", "r9_required": "42 [S,V,Acc]"},
        {"gap": "proprio_normalisation", "standard": "position bounds pmin/pmax",
         "r9_required": "signed_motion q99 for V/Acc, position bounds kept for S"},
        {"gap": "target", "standard": "AT.ABSOLUTE / AT.RESIDUAL (A or A-S)",
         "r9_required": "velocity_residual R=(A-S)-V"},
        {"gap": "quantiles", "standard": "action_stats(all 80 train episodes)",
         "r9_required": "frozen q01/q99 from 64 TRAIN only"},
        {"gap": "frame_sampling", "standard": "t = randrange(0, T-1)",
         "r9_required": "t >= 2 (history-safe); offline t in [2, n-2]"},
        {"gap": "checkpoint_meta", "standard": "variant/cams/proprio/q01/q99",
         "r9_required": "plus normalisation mode, codec, split hash, contract sha"},
    ]


def integration_contract(contract, meta, contract_path, bounds_path, split_path):
    import hashlib

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    return {
        "proprio_mode": A.PROPRIO_MODE, "proprio_dim": N_DIM,
        "proprio_layout": "[state, velocity, acceleration]",
        "normalization_mode": contract.mode, "motion_quantile": meta.get("motion_quantile"),
        "target_representation": TARGET_VELOCITY_RESIDUAL,
        "action_codec_mode": meta.get("action_codec_mode", "fixed"),
        "frame_pool": "t >= 2 (V,Acc defined); offline t in [2, n-2]",
        "gripper_policy": meta.get("gripper_policy"),
        "history_boundary_policy": meta.get("history_boundary_policy"),
        "frozen_contract_path": os.path.abspath(contract_path),
        "frozen_contract_sha256": sha(contract_path),
        "bounds_sha256": sha(bounds_path),
        "split_sha256": sha(split_path),
        "q01": np.asarray(meta["q01"], np.float32).tolist(),
        "q99": np.asarray(meta["q99"], np.float32).tolist(),
    }
