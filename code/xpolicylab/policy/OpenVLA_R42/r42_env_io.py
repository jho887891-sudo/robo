#!/usr/bin/env python3
"""R16: RoboDojo environment dict <-> flat 14D joint space + frozen 3-camera order.

One job only: convert what the RoboDojo policy client actually sends
(env/observation_manager/obs_manager.py:76-118) into exactly what the 42D policy server
consumes, and convert the server's 14D command back into the 4-key dict that
eval_env.take_action() accepts (src/eval_client/eval_env.py:418-441).  Everything here fails
closed: a wrong camera name, a swapped joint, a float image, an RGBA image, a 13D state or an
action dict with an extra key raises instead of being reshaped, sliced or reordered.

Frozen layout (r42_adapter.GRIPPER_DIMS == (6, 13) is the evidence that this order is right):

    dim 0..5    left_arm_joint_state   (6,)
    dim 6       left_ee_joint_state    (1,)      <- gripper, clipped to [0, 1] by the sim
    dim 7..12   right_arm_joint_state  (6,)
    dim 13      right_ee_joint_state   (1,)      <- gripper, clipped to [0, 1] by the sim

Cameras (r42_adapter.CAMERA_ORDER, the training order, no BGR swap):

    cam_head, cam_left_wrist, cam_right_wrist    each HxWx3 uint8 RGB (480x640 on push_T at 25 Hz)

The observation also carries 'action' (the *recorded* dataset action) and 'instruction'.
'action' is future information with respect to the decision and is therefore DROPPED, never
forwarded; the caller can assert that through the returned 'dropped' tuple.  Ground-truth
fields (target_*, privileged_*, gt_*, oracle_*, ...) are refused outright by the same guard the
adapter uses.

CPU only, numpy only at import time.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PURE = os.path.abspath(os.path.join(_HERE, "..", "OpenVLA_Pure"))
for _p in (_HERE, _PURE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r42_adapter as R  # noqa: E402
import r42_robodojo_adapter as A  # noqa: E402  (privileged-field guard, one implementation)

# ------------------------------------------------------------------------- frozen contract
VISION_KEY = "vision"
STATE_KEY = "state"
COLOR_ANNOTATOR = "color"
CAMERA_ORDER = tuple(R.CAMERA_ORDER)
EXPECTED_RGB_SHAPE = (480, 640, 3)          # push_T cameras, HxWxC, audited from the client
EXPECTED_CONTROL_HZ = 25.0                  # additional_info.frequency
CONTROL_DT = 1.0 / EXPECTED_CONTROL_HZ
JOINT_ORDER = ("left_arm_joint_state", "left_ee_joint_state",
               "right_arm_joint_state", "right_ee_joint_state")
JOINT_DIMS = (6, 1, 6, 1)
STATE_DIM = sum(JOINT_DIMS)                 # 14
GRIPPER_DIMS = tuple(R.GRIPPER_DIMS)        # (6, 13)
GRIPPER_RANGE = (0.0, 1.0)                  # eval_env clips the gripper to [0, 1]
# Extra keys the audited client sends inside 'state' that the 42D space does not use.
STATE_IGNORED_KEYS = ("left_ee_pose", "right_ee_pose")
# Top-level keys the client sends that must never become policy input.
OBS_IGNORED_KEYS = ("action", "additional_info", "data_format_version", "instruction",
                    "env_idx", "timestamp", "record", "camera_names", "channel_order")
OBS_REQUIRED_KEYS = (VISION_KEY, STATE_KEY)
# Annotator sub-keys obs_manager may attach next to "color" (obs_manager.py:99-118).
VISION_ANNOTATORS = ("color", "shape", "frequency")


class R42EnvIOError(R.R42Error):
    """Any violation of the client <-> 42D input/output contract."""


# --------------------------------------------------------------------------------- helpers
def _as_ndarray(value, name):
    try:
        arr = np.asarray(value)
    except Exception as exc:                      # ragged list, object dtype, ...
        raise R42EnvIOError("%s is not array-like: %r" % (name, exc))
    return arr


def _check_image(cam, value, expected_hw=None):
    """One camera -> contiguous HxWx3 uint8.  No reshape, no channel drop, no BGR swap."""
    if isinstance(value, dict):
        extra = [k for k in value if k not in VISION_ANNOTATORS]
        if extra:
            raise R42EnvIOError("%s carries unexpected annotator key(s) %s (allowed %s)"
                                % (cam, extra, list(VISION_ANNOTATORS)))
        if COLOR_ANNOTATOR not in value:
            raise R42EnvIOError("%s has no %r annotator" % (cam, COLOR_ANNOTATOR))
        declared = value.get("shape")
        arr = _as_ndarray(value[COLOR_ANNOTATOR], "%s.color" % cam)
        if declared is not None and tuple(int(x) for x in np.asarray(declared).reshape(-1)) \
                != tuple(arr.shape):
            raise R42EnvIOError("%s declares shape %s but carries %s"
                                % (cam, tuple(declared), arr.shape))
    else:
        arr = _as_ndarray(value, cam)
    if arr.dtype != np.uint8:
        raise R42EnvIOError("%s must be uint8 (the client sends RGB uint8); got %s"
                            % (cam, arr.dtype))
    if arr.ndim != 3:
        raise R42EnvIOError("%s must be HxWx3; got ndim=%d shape=%s (CHW is not accepted, "
                            "and nothing is transposed for you)" % (cam, arr.ndim, arr.shape))
    if arr.shape[-1] != 3:
        raise R42EnvIOError("%s must have exactly 3 colour channels; got shape %s "
                            "(RGBA/1-channel is refused, not sliced)" % (cam, arr.shape))
    if expected_hw is not None:
        want = tuple(int(x) for x in expected_hw)
        if tuple(arr.shape) != want:
            raise R42EnvIOError("%s must be %s (HxWx3) on push_T; got %s"
                                % (cam, want, tuple(arr.shape)))
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise R42EnvIOError("%s is empty: %s" % (cam, arr.shape))
    return np.ascontiguousarray(arr)


def _flat_state_from_dict(state):
    missing = [k for k in JOINT_ORDER if k not in state]
    if missing:
        raise R42EnvIOError("state is missing joint key(s) %s; the 42D order is %s"
                            % (missing, list(JOINT_ORDER)))
    unknown = [k for k in state if k not in JOINT_ORDER and k not in STATE_IGNORED_KEYS]
    if unknown:
        raise R42EnvIOError("state carries unknown key(s) %s (joint keys %s, ignored %s)"
                            % (unknown, list(JOINT_ORDER), list(STATE_IGNORED_KEYS)))
    parts = []
    for key, dim in zip(JOINT_ORDER, JOINT_DIMS):
        arr = _as_ndarray(state[key], "state.%s" % key)
        if arr.ndim > 1:
            raise R42EnvIOError("state.%s must be a flat (%d,) vector; got shape %s"
                                % (key, dim, arr.shape))
        flat = arr.reshape(-1)
        if flat.shape != (dim,):
            raise R42EnvIOError("state.%s must be %dD; got shape %s"
                                % (key, dim, tuple(flat.shape)))
        if flat.dtype.kind not in "iuf":
            raise R42EnvIOError("state.%s must be numeric; got dtype %s" % (key, flat.dtype))
        parts.append(flat.astype(np.float32))
    out = np.concatenate(parts).astype(np.float32)
    if out.shape != (STATE_DIM,):
        raise R42EnvIOError("flat state must be %dD, got %s" % (STATE_DIM, out.shape))
    if not np.isfinite(out).all():
        raise R42EnvIOError("state has non-finite values")
    return out


def _flat_state(state):
    """dict of per-joint arrays, or an already-flat 14D vector, -> (14,) float32."""
    if isinstance(state, dict):
        return _flat_state_from_dict(state)
    arr = _as_ndarray(state, "state")
    if arr.ndim > 1:
        raise R42EnvIOError("state must be a flat (%d,) vector or a per-joint dict; got shape %s"
                            % (STATE_DIM, arr.shape))
    flat = arr.reshape(-1)
    if flat.shape != (STATE_DIM,):
        raise R42EnvIOError("flat state must be %dD in the order %s; got shape %s"
                            % (STATE_DIM, list(JOINT_ORDER), tuple(flat.shape)))
    if flat.dtype.kind not in "iuf":
        raise R42EnvIOError("state must be numeric; got dtype %s" % (flat.dtype,))
    out = flat.astype(np.float32)
    if not np.isfinite(out).all():
        raise R42EnvIOError("state has non-finite values")
    return out


# ------------------------------------------------------------------ observation -> inputs
def obs_dict_to_inputs(obs, expected_hw=EXPECTED_RGB_SHAPE, expected_hz=EXPECTED_CONTROL_HZ):
    """The client's observation dict -> the 42D policy inputs, fail closed.

    Parameters
    ----------
    obs          : the dict returned by TASK_ENV.get_obs(); must carry 'vision' and 'state'.
    expected_hw  : the HxW the deployment camera contract requires.  None disables the
                   resolution check (tests / other robots); the default is push_T's 480x640.
    expected_hz  : the control rate declared in additional_info.frequency; a different rate
                   invalidates the per-step rate rule, so it is refused.

    Returns
    -------
    dict with
        images     : ordered dict cam -> HxWx3 uint8 in r42_adapter.CAMERA_ORDER
        state      : (14,) float32 in JOINT_ORDER  [left_arm6, left_ee1, right_arm6, right_ee1]
        env_idx    : int
        instruction: str or None (advisory; the policy pins its own trained instruction)
        timestamp  : float or None
        frequency  : float or None
        dropped    : tuple of ignored top-level keys (contains "action" when present)
    """
    if not isinstance(obs, dict):
        raise R42EnvIOError("observation must be a dict, got %r" % (type(obs),))
    A.assert_no_privileged(obs)                    # ground truth never reaches the policy
    missing = [k for k in OBS_REQUIRED_KEYS if k not in obs]
    if missing:
        raise R42EnvIOError("observation is missing %s (keys %s)"
                            % (missing, sorted(obs.keys())))

    # ---- cameras: resolved by name, never by dict order, nothing repositioned silently
    vision = obs[VISION_KEY]
    if not isinstance(vision, dict):
        raise R42EnvIOError("observation[%r] must be a dict keyed by camera name, got %r"
                            % (VISION_KEY, type(vision)))
    declared_names = obs.get("camera_names")
    if declared_names is not None:
        names = list(declared_names)
        if [str(n) for n in names] != list(CAMERA_ORDER):
            raise R42EnvIOError("declared camera_names %s are not the frozen order %s"
                                % (names, list(CAMERA_ORDER)))
    unknown = [k for k in vision if k not in CAMERA_ORDER]
    if unknown:
        raise R42EnvIOError("unexpected camera key(s) %s; the frozen order is %s"
                            % (unknown, list(CAMERA_ORDER)))
    absent = [c for c in CAMERA_ORDER if c not in vision]
    if absent:
        raise R42EnvIOError("missing camera(s) %s; the frozen order is %s"
                            % (absent, list(CAMERA_ORDER)))
    images = {cam: _check_image(cam, vision[cam], expected_hw=expected_hw)
              for cam in CAMERA_ORDER}

    # ---- proprio
    state = _flat_state(obs[STATE_KEY])

    # ---- advisory metadata
    info = obs.get("additional_info")
    frequency = None
    if info is not None:
        if not isinstance(info, dict):
            raise R42EnvIOError("additional_info must be a dict, got %r" % (type(info),))
        unknown_info = [k for k in info if k != "frequency"]
        if unknown_info:
            raise R42EnvIOError("additional_info carries unknown key(s) %s" % unknown_info)
        if "frequency" in info:
            try:
                frequency = float(info["frequency"])
            except (TypeError, ValueError):
                raise R42EnvIOError("additional_info.frequency is not a number: %r"
                                    % (info["frequency"],))
            if not np.isfinite(frequency) or frequency <= 0.0:
                raise R42EnvIOError("additional_info.frequency must be finite and > 0")
            if expected_hz is not None and abs(frequency - float(expected_hz)) > 1e-6:
                raise R42EnvIOError(
                    "client control rate %.6g Hz != %.6g Hz: the checkpoint rate rule "
                    "(max_rate * dt) is defined at %.6g s/step; refusing a silent rate change"
                    % (frequency, float(expected_hz), CONTROL_DT))

    instruction = obs.get("instruction")
    if instruction is not None and not isinstance(instruction, str):
        raise R42EnvIOError("instruction must be a string, got %r" % (type(instruction),))
    env_idx = int(obs.get("env_idx", 0))
    if env_idx < 0:
        raise R42EnvIOError("env_idx must be >= 0, got %d" % env_idx)
    ts = obs.get("timestamp")
    if ts is not None:
        ts = float(ts)
        if not np.isfinite(ts):
            raise R42EnvIOError("timestamp must be finite")

    # ---- everything else is deliberately dropped (and reported, so a test can prove it)
    dropped = tuple(sorted(k for k in obs if k not in OBS_REQUIRED_KEYS))
    allowed_dropped = set(OBS_IGNORED_KEYS)
    for key in dropped:
        if key not in allowed_dropped:
            # An unrecognised top-level key could be privilege smuggling or a client change:
            # refuse instead of guessing that it is harmless.
            raise R42EnvIOError("unrecognised observation key %r; the audited client sends %s"
                                % (key, sorted(allowed_dropped)))
    return {"images": images, "state": state, "env_idx": env_idx,
            "instruction": instruction, "timestamp": ts, "frequency": frequency,
            "dropped": dropped}


def assert_no_action_leak(inputs):
    """Fail closed if anything action-shaped survived the conversion."""
    if "action" in inputs:
        raise R42EnvIOError("converted inputs still carry an 'action' entry")
    for key in inputs:
        if key in ("images", "state", "env_idx", "instruction", "timestamp", "frequency",
                   "dropped"):
            continue
        raise R42EnvIOError("unexpected converted-input key %r" % (key,))
    return True


# ------------------------------------------------------------------ 14D command <-> action dict
def _cmd14(cmd, name="cmd"):
    arr = _as_ndarray(cmd, name)
    if arr.ndim > 1:
        raise R42EnvIOError("%s must be flat 14D, got shape %s" % (name, arr.shape))
    flat = arr.reshape(-1)
    if flat.shape != (STATE_DIM,):
        raise R42EnvIOError("%s must be %dD in the order %s, got shape %s"
                            % (name, STATE_DIM, list(JOINT_ORDER), tuple(flat.shape)))
    if flat.dtype.kind not in "iuf":
        raise R42EnvIOError("%s must be numeric, got dtype %s" % (name, flat.dtype))
    out = flat.astype(np.float32)
    if not np.isfinite(out).all():
        raise R42EnvIOError("%s has non-finite values" % name)
    return out


def cmd14_to_action_dict(cmd, gripper_range=GRIPPER_RANGE, tol=1e-6):
    """(14,) joint command -> the 4-key dict eval_env.take_action() expects.

    The sim clips the two gripper channels to [0, 1] (eval_env.py:428-430) and clamps nothing
    else, so a gripper value outside [0, 1] would be silently changed by the simulator: that is
    refused here instead (the command box already guarantees it, so this only fires on a bug).
    """
    flat = _cmd14(cmd)
    lo, hi = float(gripper_range[0]), float(gripper_range[1])
    for dim in GRIPPER_DIMS:
        value = float(flat[dim])
        if value < lo - tol or value > hi + tol:
            raise R42EnvIOError(
                "gripper dim %d = %.6f is outside [%g, %g]; the simulator would clip it "
                "silently, so the command is refused" % (dim, value, lo, hi))
    out, start = {}, 0
    for key, dim in zip(JOINT_ORDER, JOINT_DIMS):
        block = flat[start:start + dim]
        out[key] = [float(x) for x in block]
        start += dim
    if start != STATE_DIM or list(out) != list(JOINT_ORDER):
        raise R42EnvIOError("action dict assembly is inconsistent")
    assert_action_dict(out)
    return out


def assert_action_dict(action):
    """Validate a client-facing action dict without touching it."""
    if not isinstance(action, dict):
        raise R42EnvIOError("action must be a dict, got %r" % (type(action),))
    missing = [k for k in JOINT_ORDER if k not in action]
    if missing:
        raise R42EnvIOError("action is missing key(s) %s" % missing)
    extra = [k for k in action if k not in JOINT_ORDER]
    if extra:
        raise R42EnvIOError("action carries key(s) %s that eval_env cannot consume (allowed %s)"
                            % (extra, list(JOINT_ORDER)))
    for key, dim in zip(JOINT_ORDER, JOINT_DIMS):
        arr = _as_ndarray(action[key], "action.%s" % key)
        if arr.reshape(-1).shape != (dim,):
            raise R42EnvIOError("action.%s must be %dD, got shape %s"
                                % (key, dim, tuple(arr.shape)))
        if arr.dtype.kind not in "iuf":
            raise R42EnvIOError("action.%s must be numeric, got dtype %s"
                                % (key, arr.dtype))
        if not np.isfinite(arr.astype(np.float64)).all():
            raise R42EnvIOError("action.%s has non-finite values" % key)
    return True


def action_dict_to_cmd14(action):
    """Exact inverse of cmd14_to_action_dict (same fail-closed checks)."""
    assert_action_dict(action)
    parts = [np.asarray(action[key], np.float32).reshape(-1)
             for key, _dim in zip(JOINT_ORDER, JOINT_DIMS)]
    out = np.concatenate(parts).astype(np.float32)
    if out.shape != (STATE_DIM,):
        raise R42EnvIOError("reassembled command is %s, expected (%d,)"
                            % (out.shape, STATE_DIM))
    return out


def contract():
    """The conversion contract as observed, for evidence files and the R16 report."""
    return {"camera_order": list(CAMERA_ORDER),
            "camera_annotator": COLOR_ANNOTATOR,
            "expected_image_shape": list(EXPECTED_RGB_SHAPE),
            "image_dtype": "uint8",
            "joint_order": list(JOINT_ORDER),
            "joint_dims": list(JOINT_DIMS),
            "state_dim": STATE_DIM,
            "gripper_dims": list(GRIPPER_DIMS),
            "gripper_range": list(GRIPPER_RANGE),
            "control_hz": EXPECTED_CONTROL_HZ,
            "control_dt": CONTROL_DT,
            "state_ignored_keys": list(STATE_IGNORED_KEYS),
            "obs_dropped_keys": list(OBS_IGNORED_KEYS),
            "action_keys": list(JOINT_ORDER)}
