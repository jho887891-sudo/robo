#!/usr/bin/env python3
"""OpenVLA_R42: 42D [S,V,Acc] + velocity-residual deployment adapter (CPU only, new module).

Isolated from the historical OpenVLA_Pure 14/28D deployment: nothing here imports or mutates
its checkpoints, trainer or deploy config.  Everything is pure numpy at import time so the
contract can be unit-tested without a GPU; the real 7B call is NOT made here.

R11 P0 changes (all driven by a red regression first, see tests/test_r11_p0_*.py):
  P0-1  the 256-bin codec is no longer hand-written here; encode/decode delegate to the single
        source of truth `action_codec` so bin -> token -> residual is bit-exact with the codec
        that produced the frozen 64-TRAIN quantiles.  The prompt/label builder now applies the
        trailing <space> (29871) rule of the legacy entrypoint, so the HF causal shift lands on
        exactly the 14 action tokens + STOP.
  P0-2  the prismatic ACTION_DIM=7 / NUM_ACTIONS_CHUNK=8 / PROPRIO_DIM=8 defaults are never used
        on this path; `assert_runtime_contract()` fails closed unless the running prismatic
        really carries the 14D/31743/2 constants, and `assert_projector_width()` rejects a 14D
        projector.
  P0-3  checkpoint metadata carries the frozen normalizer payload (positional bounds for S,
        signed q99 motion scales for V/Acc, residual q01/q99) plus its canonical SHA and the
        64-TRAIN/16-DEV provenance; save/load round-trips reproduce the action bit-for-bit.
  P0-3b `decode_command` validates dt/rate/prev_cmd and refuses an action that cannot satisfy the
        physical limits and the rate rule simultaneously instead of silently clamping it.
"""
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PURE = os.path.abspath(os.path.join(_HERE, "..", "OpenVLA_Pure"))
if _PURE not in sys.path:
    sys.path.insert(0, _PURE)
import action_codec as AC  # noqa: E402  (single source of truth for the 256-bin codec)

BASE_DIM = 14
N42 = 42
N_ACTIONS = 14
TOKEN_BEGIN_IDX = AC.ACTION_TOKEN_BEGIN_IDX          # 31743
CODEC_WINDOW = (AC.LEGAL_TOKEN_MIN, AC.LEGAL_TOKEN_MAX)  # (31744, 31999)
PAD_TOKEN_ID = AC.PAD_TOKEN_ID                       # 32000
STOP_ID = AC.STOP_INDEX                              # 2
IGNORE_INDEX = -100
SPACE_TOKEN_ID = 29871                               # llama "▁" appended by the prompt builder
CAMERA_ORDER = ("cam_head", "cam_left_wrist", "cam_right_wrist")
GRIPPER_DIMS = (6, 13)
TARGET = "velocity_residual"
ALIGNMENT_VERSION = "r42-v2"
N_REQUIRED_METADATA = 21


class R42Error(ValueError):
    pass


def _a14(x, name):
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    if v.shape != (BASE_DIM,):
        raise R42Error("%s must be 14D, got %s" % (name, v.shape))
    if not np.isfinite(v).all():
        raise R42Error("%s has non-finite values" % name)
    return v


def _arr14(x, name):
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if a.shape != (BASE_DIM,):
        raise R42Error("%s must be a 14D vector" % name)
    if not np.isfinite(a).all():
        raise R42Error("%s contains non-finite values" % name)
    return a


# ----------------------------------------------------------------- prismatic poison guard
_PRISMATIC_KEYS = ("ACTION_" + "DIM", "NUM_ACTIONS_" + "CHUNK", "PROPRIO_" + "DIM")


def prismatic_constants():
    """Read the live prismatic dimension globals (they are chosen from sys.argv!)."""
    import prismatic.vla.constants as C
    return {
        "action_dim": int(getattr(C, _PRISMATIC_KEYS[0])),
        "num_actions_chunk_unused": int(getattr(C, _PRISMATIC_KEYS[1])),
        "proprio_dim_unused": int(getattr(C, _PRISMATIC_KEYS[2])),
        "action_token_begin_idx": int(C.ACTION_TOKEN_BEGIN_IDX),
        "stop_index": int(C.STOP_INDEX),
        "ignore_index": int(C.IGNORE_INDEX),
        "robot_platform": str(getattr(C, "ROBOT_PLATFORM", "unknown")),
    }


def assert_action_contract(action_dim=None, begin_idx=None, stop_idx=None, ignore_index=None):
    """Fail closed unless the runtime carries the 14D action codec this module encodes with."""
    if action_dim is not None and int(action_dim) != N_ACTIONS:
        raise R42Error(
            "prismatic %s=%d leaked into the 14D R42 path (expected %d); prismatic selects its "
            "constants from sys.argv, so make sure the arx platform marker is present before the "
            "model modules are imported" % (_PRISMATIC_KEYS[0], int(action_dim), N_ACTIONS))
    if begin_idx is not None and int(begin_idx) != TOKEN_BEGIN_IDX:
        raise R42Error("ACTION_TOKEN_BEGIN_IDX must be %d, got %r" % (TOKEN_BEGIN_IDX, begin_idx))
    if stop_idx is not None and int(stop_idx) != STOP_ID:
        raise R42Error("STOP_INDEX must be %d, got %r" % (STOP_ID, stop_idx))
    if ignore_index is not None and int(ignore_index) != IGNORE_INDEX:
        raise R42Error("IGNORE_INDEX must be %d, got %r" % (IGNORE_INDEX, ignore_index))
    return {"action_dim": N_ACTIONS, "action_token_begin_idx": TOKEN_BEGIN_IDX,
            "stop_index": STOP_ID, "ignore_index": IGNORE_INDEX}


def assert_runtime_contract():
    """Full fail-closed check of the live runtime; returns the observed evidence dict."""
    c = prismatic_constants()
    obs = assert_action_contract(action_dim=c["action_dim"],
                                 begin_idx=c["action_token_begin_idx"],
                                 stop_idx=c["stop_index"], ignore_index=c["ignore_index"])
    obs["num_actions_chunk_unused"] = c["num_actions_chunk_unused"]
    obs["proprio_dim_unused"] = c["proprio_dim_unused"]
    obs["robot_platform"] = c["robot_platform"]
    obs["r42_proprio_dim"] = N42
    obs["r42_action_tokens"] = N_ACTIONS
    return obs


def assert_projector_width(width_or_state, expected=N42):
    """Accept a width, a 2D tensor shape or a projector state_dict; reject every legacy width."""
    w = width_or_state
    if isinstance(w, dict):
        if "fc1.weight" not in w:
            raise R42Error("projector state dict lacks fc1.weight")
        w = w["fc1.weight"].shape
    shape = None
    if hasattr(w, "shape"):                      # torch.Tensor (torch.Size is a tuple subclass)
        shape = tuple(int(x) for x in w.shape)
    elif isinstance(w, (tuple, list)):           # torch.Size / a plain shape tuple
        shape = tuple(int(x) for x in w)
    if shape is not None:
        if len(shape) < 2:
            raise R42Error("projector weights must be at least 2D, got %s" % (shape,))
        w = shape[1]
    try:
        width = int(w)
    except (TypeError, ValueError):
        raise R42Error("cannot read a projector input width from %r" % (width_or_state,))
    if width != int(expected):
        raise R42Error("projector input width %d != %d (legacy 7/8/14/28D projector refused)"
                       % (width, int(expected)))
    return width


# ------------------------------------------------------------------------- 256-bin codec
def token_ids_for_residual(residual, q01, q99):
    """R -> 14 bin indices -> 14 legal action token ids (bit-exact with action_codec)."""
    r = _a14(residual, "residual")
    q01 = _arr14(q01, "q01")
    q99 = _arr14(q99, "q99")
    if not (q99 > q01).all():
        raise R42Error("q99 must exceed q01 for every dim")
    idx = np.asarray(AC.encode_indices(AC.normalize(r, q01, q99), mode=AC.MODE_FIXED), np.int64)
    ids = np.asarray(AC.token_ids(idx), np.int64)
    AC.assert_legal_token_ids(ids)
    return idx, ids


def residual_from_indices(indices, q01, q99):
    """14 bin indices -> residual, through the same codec (exact inverse of the encoder)."""
    i = np.asarray(indices, np.int64).reshape(-1)
    if i.shape != (BASE_DIM,):
        raise R42Error("expected 14 bin indices, got %s" % (i.shape,))
    if i.min() < 0 or i.max() > AC.N_BINS - 1:
        raise R42Error("bin indices must lie in 0..255, got [%d, %d]" % (i.min(), i.max()))
    q01 = _arr14(q01, "q01")
    q99 = _arr14(q99, "q99")
    if not (q99 > q01).all():
        raise R42Error("q99 must exceed q01 for every dim")
    levels = np.asarray(AC.decode_indices(i, mode=AC.MODE_FIXED), np.float32)
    return np.asarray(AC.denormalize(levels, q01, q99), np.float32)


def normalize_prompt_ids(prompt_ids, space_suffix=SPACE_TOKEN_ID):
    """Apply the legacy entrypoint rule: the prompt always ends with the <space> token."""
    pids = [int(x) for x in np.asarray(prompt_ids).reshape(-1)]
    if not pids:
        raise R42Error("prompt_ids is empty")
    if pids[-1] != int(space_suffix):
        pids = pids + [int(space_suffix)]
    return pids


def build_labels(prompt_ids, residual, q01, q99, space_suffix=SPACE_TOKEN_ID):
    """Causal label mask identical to train_pushT_pure2.make_ids() (prompt IGNOREd).

    input_ids = prompt + 14 action tokens + STOP(2)
    labels    = [-100] * len(prompt) + 14 action tokens + STOP(2)
    HF causal shift predicts label[p] from logits[p-1], so the 15 supervised predictions consume
    the logits at the last prompt position and the 14 following positions.
    """
    pids = normalize_prompt_ids(prompt_ids, space_suffix=space_suffix)
    _idx, ids = token_ids_for_residual(residual, q01, q99)
    supervised = ids.tolist() + [STOP_ID]
    input_ids = pids + supervised
    labels = [IGNORE_INDEX] * len(pids) + supervised
    if len(supervised) != N_ACTIONS + 1:
        raise R42Error("expected %d supervised positions, got %d"
                       % (N_ACTIONS + 1, len(supervised)))
    if len(labels) != len(input_ids):
        raise R42Error("labels and input_ids must have the same length")
    if STOP_ID in ids.tolist() or PAD_TOKEN_ID in ids.tolist():
        raise R42Error("STOP/PAD must never be used as an action target")
    return {"input_ids": input_ids, "labels": labels, "action_ids": ids.tolist(),
            "n_prompt": len(pids), "n_supervised": len(supervised),
            "window": list(CODEC_WINDOW)}


# ------------------------------------------------------------------------ episode history
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
            if last is not None and float(timestamp) <= last:
                raise R42Error("timestamp must strictly increase for env %s: %r <= %r"
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

    def velocity(self, env_idx=0):
        h = self._hist.get(env_idx)
        if not h:
            raise R42Error("no observation for env %s yet" % (env_idx,))
        if len(h) < 2:
            return np.zeros(BASE_DIM, np.float32)
        return (h[-1] - h[-2]).astype(np.float32)


# --------------------------------------------------------------------------- image guard
def validate_images(images, channel_order="RGB"):
    """Reject wrong camera order / dtype / layout / colour convention before the model sees them."""
    if not isinstance(images, dict):
        raise R42Error("images must be a dict keyed by camera name")
    images = dict(images)
    order = images.pop("channel_order") if "channel_order" in images else channel_order
    if str(order).upper() != "RGB":
        raise R42Error("channel order must be RGB, got %r (BGR is forbidden)" % (order,))
    missing = [c for c in CAMERA_ORDER if c not in images]
    if missing:
        raise R42Error("missing camera(s) %s" % missing)
    extra = [k for k in images if k not in CAMERA_ORDER]
    if extra:
        raise R42Error("unexpected camera keys %s (frozen order %s)" % (extra, CAMERA_ORDER))
    if tuple(images.keys()) != CAMERA_ORDER:
        raise R42Error("camera order must be %s, got %s" % (CAMERA_ORDER, tuple(images.keys())))
    out = {}
    for cam in CAMERA_ORDER:
        arr = np.asarray(images[cam])
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise R42Error("%s must be HxWx3 RGB uint8, got %s (RGBA/BGR rejected)"
                           % (cam, arr.shape))
        if arr.dtype != np.uint8:
            raise R42Error("%s must be uint8, got %s" % (cam, arr.dtype))
        out[cam] = arr
    return out


# ---------------------------------------------------------------------- action decoding
def decode_command(indices, q01, q99, state, velocity, limits=None, prev_cmd=None, dt=None,
                   gripper_dims=GRIPPER_DIMS, on_unsafe="raise", min_dt=1e-6):
    """A_hat = S + V + R_hat with the DEV gripper fallback and a double physical/rate guarantee.

    Safety proof for the rate-limited branch: `prev_cmd` is required to lie inside [amin, amax]
    and the clipped target does too, so `prev + clip(target - prev, -rate*dt, +rate*dt)` stays
    inside the box by convexity.  The final re-clip is therefore a no-op; if it ever changes a
    dimension the combination of constraints is inconsistent and the action is refused
    (`on_unsafe="raise"`) or replaced by a defined hold (`on_unsafe="hold"`).
    """
    r_hat = residual_from_indices(indices, q01, q99)
    s = _a14(state, "state")
    v = _a14(velocity, "velocity")
    gdims = tuple(int(d) for d in gripper_dims)
    if len(set(gdims)) != 2 or any(d < 0 or d >= BASE_DIM for d in gdims):
        raise R42Error("gripper_dims must be two distinct dims inside 0..13, got %s" % (gdims,))
    cmd = (s + v + r_hat).astype(np.float32)
    for d in gdims:
        cmd[d] = s[d] + v[d]
    info = {"clipped_joint": 0, "rate_limited": 0, "gripper_fallback_dims": list(gdims),
            "rate_conflict_dims": [], "unsafe": None, "dt": None if dt is None else float(dt)}
    if limits is not None:
        lo = _arr14(limits["amin"], "amin")
        hi = _arr14(limits["amax"], "amax")
        if (hi <= lo).any():
            raise R42Error("amax must exceed amin")
        before = cmd.copy()
        cmd = np.clip(cmd, lo, hi)
        info["clipped_joint"] = int((before != cmd).sum())
        rate_cfg = limits.get("max_rate")
        if prev_cmd is None and dt is None:
            pass          # no rate rule requested; physical bounds alone apply
        else:
            if prev_cmd is None or dt is None or rate_cfg is None:
                raise R42Error("rate limiting needs prev_cmd, dt and max_rate together")
            prev = _arr14(prev_cmd, "prev_cmd")
            if not np.all((prev >= lo - 1e-6) & (prev <= hi + 1e-6)):
                raise R42Error("prev_cmd is outside the physical limits; rate rule undefined")
            dtf = float(dt)
            if not np.isfinite(dtf) or dtf <= float(min_dt):
                raise R42Error("dt must be finite and > %g, got %r" % (min_dt, dt))
            rate = _arr14(rate_cfg, "max_rate")
            if (rate < 0).any():
                raise R42Error("max_rate must be non-negative")
            step = (rate * dtf).astype(np.float32)
            delta = cmd - prev
            info["rate_limited"] = int((np.abs(delta) > step).sum())
            cmd = (prev + np.clip(delta, -step, step)).astype(np.float32)
            rec = np.clip(cmd, lo, hi)
            bad = np.abs(rec - prev) > (step + 1e-6)
            if bad.any():
                info["rate_conflict_dims"] = [int(x) for x in np.nonzero(bad)[0]]
                if str(on_unsafe) == "hold":
                    rec = rec.copy()
                    rec[bad] = prev[bad]
                    info["unsafe"] = "rate_conflict_hold"
                else:
                    raise R42Error("action_unsafe: dims %s cannot satisfy the physical limits and "
                                   "the rate rule at the same time" % info["rate_conflict_dims"])
            cmd = rec
    if not np.isfinite(cmd).all():
        raise R42Error("decoded_command_not_finite")
    return cmd.astype(np.float32), r_hat.astype(np.float32), info


# ------------------------------------------------------------------- frozen normalizer
def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def normalizer_sha256(payload):
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def normalizer_payload(normalization_mode, pmin, pmax, q01, q99, v_scale=None, a_scale=None,
                       motion_quantile=None, n_train_episodes=64, n_dev_episodes_excluded=16,
                       require_legacy_fit=True):
    """The frozen normalizer that a checkpoint must carry (and that DEV must never be fitted on).

    require_legacy_fit=True is the single-task R15 contract (exactly 64 TRAIN / 16 DEV). The M2
    per-task blocks come from an 80/20 split, so they are validated with the flag off, which still
    refuses a payload with no TRAIN episodes or no excluded DEV episodes.
    """
    if require_legacy_fit and (int(n_train_episodes) != 64 or int(n_dev_episodes_excluded) != 16):
        raise R42Error("normalizer must be fitted on 64 TRAIN episodes with 16 DEV excluded "
                       "(got %r TRAIN / %r DEV excluded); the legacy 80-episode quantiles are "
                       "refused" % (n_train_episodes, n_dev_episodes_excluded))
    if (not require_legacy_fit) and (int(n_train_episodes) < 1
                                     or int(n_dev_episodes_excluded) < 1):
        raise R42Error("a per-task normalizer block must record at least one TRAIN episode and at "
                       "least one excluded DEV episode (got %r TRAIN / %r DEV excluded)"
                       % (n_train_episodes, n_dev_episodes_excluded))
    if normalization_mode != "signed_motion":
        raise R42Error("only signed_motion normalisation may drive the 42D path, got %r"
                       % (normalization_mode,))
    if v_scale is None or a_scale is None:
        raise R42Error("signed_motion requires fitted v_scale and a_scale")
    payload = {"normalization_mode": str(normalization_mode),
               "pmin": _arr14(pmin, "pmin").tolist(), "pmax": _arr14(pmax, "pmax").tolist(),
               "v_scale": _arr14(v_scale, "v_scale").tolist(),
               "a_scale": _arr14(a_scale, "a_scale").tolist(),
               "motion_quantile": None if motion_quantile is None else float(motion_quantile),
               "q01": _arr14(q01, "q01").tolist(), "q99": _arr14(q99, "q99").tolist(),
               "fit_source": {"quantile_source": "64 TRAIN episodes only",
                              "n_train_episodes_for_quantiles": int(n_train_episodes),
                              "n_dev_episodes_excluded": int(n_dev_episodes_excluded)}}
    if (np.asarray(payload["q99"], np.float32) <= np.asarray(payload["q01"], np.float32)).any():
        raise R42Error("q99 must exceed q01 for every dim")
    if (np.asarray(payload["pmax"], np.float32) <= np.asarray(payload["pmin"], np.float32)).any():
        raise R42Error("pmax must exceed pmin for every dim")
    if (np.asarray(payload["v_scale"], np.float32) <= 0).any() or \
            (np.asarray(payload["a_scale"], np.float32) <= 0).any():
        raise R42Error("motion scales must be strictly positive")
    return payload


def validate_normalizer(payload, require_legacy_fit=True):
    if not isinstance(payload, dict):
        raise R42Error("checkpoint must carry a normalizer payload dict")
    if is_task_normalizer(payload):
        raise R42Error("this is a multi-task payload; call normalizer_for_task(payload, task) so "
                       "the task is explicit")
    src = payload.get("fit_source") or {}
    return normalizer_payload(
        normalization_mode=payload.get("normalization_mode"),
        pmin=payload.get("pmin"), pmax=payload.get("pmax"), q01=payload.get("q01"),
        q99=payload.get("q99"), v_scale=payload.get("v_scale"), a_scale=payload.get("a_scale"),
        motion_quantile=payload.get("motion_quantile"),
        n_train_episodes=src.get("n_train_episodes_for_quantiles", -1),
        n_dev_episodes_excluded=src.get("n_dev_episodes_excluded", -1),
        require_legacy_fit=require_legacy_fit)


M2_TASK_SCHEMA = "m2-task-normalizer-v1"


def is_task_normalizer(payload):
    """True for the M2 per-task payload produced by training/m2_task_normalizer.py."""
    return isinstance(payload, dict) and payload.get("schema") == M2_TASK_SCHEMA


def task_normalizer_tasks(payload):
    if not is_task_normalizer(payload):
        return None
    return sorted((payload.get("tasks") or {}).keys())


def normalizer_for_task(payload, task=None):
    """The 14D normalizer that applies to task.

    A legacy single-task payload has no task dimension, so it is returned for any task (that is how
    every R15 checkpoint works). A multi-task payload requires an explicit task and fails closed on a
    missing block or a task id that disagrees with the frozen mapping.
    """
    if not is_task_normalizer(payload):
        return validate_normalizer(payload)
    tasks = payload.get("tasks") or {}
    if task is None:
        raise R42Error("a multi-task normalizer needs an explicit task; it carries %s"
                       % (sorted(tasks),))
    if task not in tasks:
        raise R42Error("task %r has no normalizer block (have %s)" % (task, sorted(tasks)))
    ids = payload.get("task_id") or {}
    if ids and ids.get(str(task)) is None:
        raise R42Error("task %r has no frozen task id in the payload" % (task,))
    block = validate_normalizer(tasks[task], require_legacy_fit=False)
    fitted = payload.get("fitted_on") or {}
    if fitted and str(task) not in fitted:
        raise R42Error("task %r has a normalizer block but no fit provenance" % (task,))
    return block


def normalizer_sha256_for(payload):
    """sha256 that pins the WHOLE normalizer: all tasks for M2, the single payload for R15."""
    if not is_task_normalizer(payload):
        return normalizer_sha256(validate_normalizer(payload))
    for task in task_normalizer_tasks(payload):
        normalizer_for_task(payload, task)
    return normalizer_sha256(payload)


def normalizer_task_list(payload):
    """The task list a checkpoint covers, so metadata can name it without guessing."""
    return task_normalizer_tasks(payload) or []


# ------------------------------------------------------------------------- metadata
REQUIRED_METADATA = ("task_id", "base_model_sha256", "lora_sha256", "projector_sha256",
                     "proprio_dim", "proprio_layout", "history_policy", "camera_order",
                     "prompt_source", "frozen_contract_sha256", "split_sha256", "codec_mode",
                     "action_token_window", "target_representation", "train_steps", "seed",
                     "gripper_policy", "safety_limits", "alignment_version",
                     "normalizer", "normalizer_sha256")


def build_metadata(task_id, base_model_sha256, lora_sha256, projector_sha256,
                   frozen_contract_sha256, split_sha256, train_steps, seed,
                   gripper_policy, safety_limits, normalizer=None, prompt_source="tokenizer_prompt",
                   codec_mode="fixed"):
    for name, val in (("base_model_sha256", base_model_sha256), ("lora_sha256", lora_sha256),
                      ("projector_sha256", projector_sha256),
                      ("frozen_contract_sha256", frozen_contract_sha256),
                      ("split_sha256", split_sha256)):
        if not (isinstance(val, str) and len(val) == 64):
            raise R42Error("%s must be a 64-char sha256, got %r" % (name, val))
    if normalizer is None:
        raise R42Error("an R42 checkpoint must carry the frozen normalizer payload")
    if is_task_normalizer(normalizer):
        for _task in task_normalizer_tasks(normalizer):
            normalizer_for_task(normalizer, _task)
        norm = normalizer
        tasks_covered = normalizer_task_list(normalizer)
    else:
        norm = validate_normalizer(normalizer)
        tasks_covered = []
    meta = {"task_id": task_id,
            "base_model_sha256": base_model_sha256,
            "lora_sha256": lora_sha256, "projector_sha256": projector_sha256,
            "proprio_dim": N42, "proprio_layout": "[state14, velocity14, acceleration14]",
            "history_policy": "V=0,Acc=0 at t0; Acc=0 at t1; per-env isolated, observed only",
            "camera_order": list(CAMERA_ORDER), "prompt_source": prompt_source,
            "frozen_contract_sha256": frozen_contract_sha256, "split_sha256": split_sha256,
            "codec_mode": codec_mode, "action_token_window": list(CODEC_WINDOW),
            "target_representation": TARGET, "train_steps": int(train_steps),
            "seed": int(seed), "gripper_policy": gripper_policy,
            "safety_limits": safety_limits, "alignment_version": ALIGNMENT_VERSION,
            "normalizer": norm, "normalizer_sha256": normalizer_sha256_for(normalizer)}
    # ONLY a multi-task checkpoint names its task set, so every single-task metadata file stays
    # byte-identical to the ones this repo already produced (pinned by the byte-parity test).
    if tasks_covered:
        meta["tasks"] = tasks_covered
    return meta


def validate_metadata(meta, expect_task=None):
    """Fail closed on any R42 checkpoint that could silently be a 14/28D or prismatic-7D model."""
    if not isinstance(meta, dict):
        raise R42Error("checkpoint metadata must be a dict")
    missing = [k for k in REQUIRED_METADATA if k not in meta]
    if missing:
        raise R42Error("checkpoint metadata missing fields: %s" % missing)
    if meta.get("proprio_dim") != N42:
        raise R42Error("R42 checkpoint must declare proprio_dim=42, got %r"
                       % (meta.get("proprio_dim"),))
    if int(meta.get("proprio_dim", 0)) in (7, 8, 14, 28):
        raise R42Error("legacy prismatic/14D/28D checkpoint is not an R42 checkpoint")
    if meta.get("action_token_window") != list(CODEC_WINDOW):
        raise R42Error("action_token_window must be %s" % (list(CODEC_WINDOW),))
    if meta.get("target_representation") != TARGET:
        raise R42Error("target_representation must be %r" % TARGET)
    if meta.get("alignment_version") != ALIGNMENT_VERSION:
        raise R42Error("alignment_version must be %r" % ALIGNMENT_VERSION)
    if list(meta["camera_order"]) != list(CAMERA_ORDER):
        raise R42Error("camera_order mismatch: %s" % (meta["camera_order"],))
    covered = list(meta.get("tasks") or [])
    if expect_task is not None:
        if covered:
            if expect_task not in covered:
                raise R42Error("task %r is not served by this checkpoint (covers %s)"
                               % (expect_task, covered))
        elif meta["task_id"] != expect_task:
            raise R42Error("task mismatch: checkpoint %r vs requested %r"
                           % (meta["task_id"], expect_task))
    for key in ("base_model_sha256", "lora_sha256", "projector_sha256",
                "frozen_contract_sha256", "split_sha256", "normalizer_sha256"):
        if len(str(meta[key])) != 64:
            raise R42Error("%s is not a sha256" % key)
    if is_task_normalizer(meta["normalizer"]):
        n_tasks = task_normalizer_tasks(meta["normalizer"])
        if not covered or sorted(covered) != n_tasks:
            raise R42Error("metadata tasks %s disagree with the normalizer payload %s"
                           % (covered, n_tasks))
    recomputed = normalizer_sha256_for(meta["normalizer"])
    if recomputed != str(meta["normalizer_sha256"]):
        raise R42Error("normalizer_sha256 mismatch: metadata says %s, payload hashes to %s"
                       % (meta["normalizer_sha256"], recomputed))
    limits = meta.get("safety_limits") or {}
    for key in ("amin", "amax"):
        if key not in limits:
            raise R42Error("safety_limits must carry %s" % key)
        _arr14(limits[key], "safety_limits.%s" % key)
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


def decode_with_metadata(meta, indices, state, velocity, prev_cmd=None, dt=None,
                         on_unsafe="raise", task=None):
    """Deployment entry: everything (quantiles, limits, gripper policy) comes from the checkpoint.

    task is required for a multi-task checkpoint and simply ignored for an R15 single-task payload,
    so callers do not have to branch on the schema.
    """
    validate_metadata(meta)
    norm = normalizer_for_task(meta["normalizer"], task)
    return decode_command(indices, norm["q01"], norm["q99"], state, velocity,
                          limits=meta["safety_limits"], prev_cmd=prev_cmd, dt=dt,
                          gripper_dims=tuple(meta["gripper_policy"]["gripper_dims"]),
                          on_unsafe=on_unsafe)
