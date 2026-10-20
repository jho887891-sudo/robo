#!/usr/bin/env python3
"""OpenVLA_R42: 42D [S,V,Acc] + velocity-residual deployment adapter (CPU only, new module).

Isolated from the historical OpenVLA_Pure 14/28D deployment: nothing here imports or mutates
its checkpoints, trainer or deploy config. Every function is pure numpy/torch-free so it can be
unit-tested without a GPU; the real 7B call is NOT made here.
"""
import hashlib
import json

import numpy as np

BASE_DIM = 14
N42 = 42
CODEC_WINDOW = (31744, 31999)
N_ACTIONS = 14
STOP_ID = 2
IGNORE_INDEX = -100
CAMERA_ORDER = ("cam_head", "cam_left_wrist", "cam_right_wrist")
GRIPPER_DIMS = (6, 13)
TARGET = "velocity_residual"
ALIGNMENT_VERSION = "r42-v1"


class R42Error(ValueError):
    pass


def _a14(x, name):
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    if v.shape != (BASE_DIM,):
        raise R42Error("%s must be 14D, got %s" % (name, v.shape))
    if not np.isfinite(v).all():
        raise R42Error("%s has non-finite values" % name)
    return v


class EpisodeHistory:
    """Per-environment history of OBSERVED states only (never predictions, never other envs)."""

    def __init__(self, contract):
        self.contract = contract
        self._hist = {}
        self._ts = {}

    def reset(self, env_idx=0):
        self._hist.pop(env_idx, None)
        self._ts.pop(env_idx, None)

    def observe(self, env_idx, state, timestamp=None):
        now = _a14(state, "state")
        if timestamp is not None:
            last = self._ts.get(env_idx)
            if last is not None and float(timestamp) < last:
                raise R42Error("timestamp went backwards for env %s: %r < %r"
                               % (env_idx, timestamp, last))
            self._ts[env_idx] = float(timestamp)
        h = self._hist.setdefault(env_idx, [])
        h.append(now.copy())
        feat = self.features(env_idx)      # uses up to THREE observed states
        if len(h) > 2:
            del h[0]                        # keep only what the next step needs
        return feat

    def features(self, env_idx=0):
        h = self._hist.get(env_idx)
        if not h:
            raise R42Error("no observation for env %s yet" % (env_idx,))
        now = h[-1]
        prev = h[-2] if len(h) >= 2 else None
        prev2 = h[-3] if len(h) >= 3 else None
        v = np.zeros(BASE_DIM, np.float32) if prev is None else (now - prev)
        acc = np.zeros(BASE_DIM, np.float32) if (prev is None or prev2 is None) \
            else ((now - prev) - (prev - prev2))
        feat = self.contract.transform(now, v.astype(np.float32), acc.astype(np.float32))
        if feat.shape != (N42,) or not np.isfinite(feat).all():
            raise R42Error("42D feature invalid: %s" % (feat.shape,))
        return feat.astype(np.float32)


def validate_images(images):
    """Reject wrong camera order / dtype / layout before they reach the model."""
    if not isinstance(images, dict):
        raise R42Error("images must be a dict keyed by camera name")
    if tuple(images.keys()) != CAMERA_ORDER:
        raise R42Error("camera order must be %s, got %s" % (CAMERA_ORDER, tuple(images.keys())))
    out = {}
    for cam in CAMERA_ORDER:
        arr = np.asarray(images[cam])
        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            raise R42Error("%s must be HxWx3/4 uint8, got %s" % (cam, arr.shape))
        if arr.dtype != np.uint8:
            raise R42Error("%s must be uint8, got %s" % (cam, arr.dtype))
        out[cam] = arr
    return out


def token_ids_for_residual(residual, q01, q99):
    """R -> 14 bin indices -> 14 legal action token ids (window enforced)."""
    r = _a14(residual, "residual")
    q01 = np.asarray(q01, np.float32)
    q99 = np.asarray(q99, np.float32)
    if not (q99 > q01).all():
        raise R42Error("q99 must exceed q01 for every dim")
    norm = np.clip(2.0 * (r - q01) / (q99 - q01 + 1e-8) - 1.0, -1.0, 1.0)
    levels = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    idx = np.clip(np.digitize(norm, levels) - 1, 0, 255)
    ids = CODEC_WINDOW[0] + idx.astype(np.int64)
    if ids.min() < CODEC_WINDOW[0] or ids.max() > CODEC_WINDOW[1]:
        raise R42Error("action token ids outside %s: %s" % (CODEC_WINDOW, ids.tolist()))
    return idx.astype(np.int64), ids


def build_labels(prompt_ids, residual, q01, q99):
    """Causal label mask: prompt is IGNOREd, only the 14 action tokens + STOP are supervised."""
    pids = [int(x) for x in np.asarray(prompt_ids).reshape(-1)]
    if not pids:
        raise R42Error("prompt_ids is empty")
    _idx, ids = token_ids_for_residual(residual, q01, q99)
    supervised = ids.tolist() + [STOP_ID]
    input_ids = pids + supervised
    labels = [IGNORE_INDEX] * len(pids) + supervised
    if len(supervised) != N_ACTIONS + 1:
        raise R42Error("expected %d supervised positions, got %d" % (N_ACTIONS + 1, len(supervised)))
    return {"input_ids": input_ids, "labels": labels, "action_ids": ids.tolist(),
            "n_prompt": len(pids), "n_supervised": len(supervised),
            "window": list(CODEC_WINDOW)}


def decode_command(indices, q01, q99, state, velocity, limits=None, prev_cmd=None, dt=None,
                  gripper_dims=GRIPPER_DIMS):
    """A_hat = S + V + R_hat with the DEV gripper fallback and safety clipping."""
    idx = np.asarray(indices, np.int64).reshape(-1)
    if idx.shape != (BASE_DIM,):
        raise R42Error("expected 14 bin indices")
    if idx.min() < 0 or idx.max() > 255:
        raise R42Error("bin indices out of 0..255")
    q01 = np.asarray(q01, np.float32)
    q99 = np.asarray(q99, np.float32)
    levels = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    r_hat = np.clip((levels[idx] + 1.0) / 2.0 * (q99 - q01) + q01, q01, q99)
    s = _a14(state, "state")
    v = _a14(velocity, "velocity")
    cmd = s + v + r_hat
    cmd = cmd.copy()
    for d in gripper_dims:
        cmd[d] = s[d] + v[d]
    info = {"clipped_joint": 0, "rate_limited": 0, "gripper_fallback_dims": list(gripper_dims)}
    if limits:
        lo = np.asarray(limits["amin"], np.float32)
        hi = np.asarray(limits["amax"], np.float32)
        before = cmd.copy()
        cmd = np.clip(cmd, lo, hi)
        info["clipped_joint"] = int((before != cmd).sum())
        if prev_cmd is not None and dt is not None and limits.get("max_rate") is not None:
            prev = _a14(prev_cmd, "prev_cmd")
            rate = np.asarray(limits["max_rate"], np.float32) * float(dt)
            delta = cmd - prev
            limited = np.clip(delta, -rate, rate)
            info["rate_limited"] = int((np.abs(delta) > rate).sum())
            cmd = prev + limited
    return cmd.astype(np.float32), r_hat.astype(np.float32), info


REQUIRED_METADATA = ("task_id", "base_model_sha256", "lora_sha256", "projector_sha256",
                     "proprio_dim", "proprio_layout", "history_policy", "camera_order",
                     "prompt_source", "frozen_contract_sha256", "split_sha256", "codec_mode",
                     "action_token_window", "target_representation", "train_steps", "seed",
                     "gripper_policy", "safety_limits", "alignment_version")


def build_metadata(task_id, base_model_sha256, lora_sha256, projector_sha256,
                   frozen_contract_sha256, split_sha256, train_steps, seed,
                   gripper_policy, safety_limits, prompt_source="tokenizer_prompt",
                   codec_mode="fixed"):
    for name, val in (("base_model_sha256", base_model_sha256), ("lora_sha256", lora_sha256),
                      ("projector_sha256", projector_sha256),
                      ("frozen_contract_sha256", frozen_contract_sha256),
                      ("split_sha256", split_sha256)):
        if not (isinstance(val, str) and len(val) == 64):
            raise R42Error("%s must be a 64-char sha256, got %r" % (name, val))
    return {"task_id": task_id, "base_model_sha256": base_model_sha256,
            "lora_sha256": lora_sha256, "projector_sha256": projector_sha256,
            "proprio_dim": N42, "proprio_layout": "[state14, velocity14, acceleration14]",
            "history_policy": "V=0,Acc=0 at t0; Acc=0 at t1; per-env isolated, observed only",
            "camera_order": list(CAMERA_ORDER), "prompt_source": prompt_source,
            "frozen_contract_sha256": frozen_contract_sha256, "split_sha256": split_sha256,
            "codec_mode": codec_mode, "action_token_window": list(CODEC_WINDOW),
            "target_representation": TARGET, "train_steps": int(train_steps),
            "seed": int(seed), "gripper_policy": gripper_policy,
            "safety_limits": safety_limits, "alignment_version": ALIGNMENT_VERSION}


def validate_metadata(meta, expect_task=None):
    """Fail closed on any R42 checkpoint that could silently be a 14/28D or prismatic-7D model."""
    if meta.get("proprio_dim") != N42:
        raise R42Error("R42 checkpoint must declare proprio_dim=42, got %r" % (meta.get("proprio_dim"),))
    if int(meta.get("proprio_dim", 0)) in (7, 8, 14, 28):
        raise R42Error("legacy prismatic/14D/28D checkpoint is not an R42 checkpoint")
    if meta.get("action_token_window") != list(CODEC_WINDOW):
        raise R42Error("action_token_window must be %s" % (list(CODEC_WINDOW),))
    if meta.get("target_representation") != TARGET:
        raise R42Error("target_representation must be %r" % TARGET)
    if meta.get("alignment_version") != ALIGNMENT_VERSION:
        raise R42Error("alignment_version must be %r" % ALIGNMENT_VERSION)
    missing = [k for k in REQUIRED_METADATA if k not in meta]
    if missing:
        raise R42Error("checkpoint metadata missing fields: %s" % missing)
    if list(meta["camera_order"]) != list(CAMERA_ORDER):
        raise R42Error("camera_order mismatch: %s" % (meta["camera_order"],))
    if expect_task is not None and meta["task_id"] != expect_task:
        raise R42Error("task mismatch: checkpoint %r vs requested %r"
                       % (meta["task_id"], expect_task))
    for key in ("base_model_sha256", "lora_sha256", "projector_sha256",
                "frozen_contract_sha256", "split_sha256"):
        if len(str(meta[key])) != 64:
            raise R42Error("%s is not a sha256" % key)
    return True


def save_metadata(path, meta):
    validate_metadata(meta)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_metadata(path, expect_task=None):
    meta = json.load(open(path))
    validate_metadata(meta, expect_task=expect_task)
    return meta
