"""History-proprio contract: P_t = [S_t, S_(t-1)] (28D) with a fair no-history control.

Why this module exists
----------------------
R2-R5 fed the policy only the CURRENT state S_t, while the constant-velocity
baseline that beat it uses two states. The pilot therefore needs an input-interface
change that is (a) minimal, (b) identical in shape and capacity for both arms of the
A/B, and (c) impossible to accidentally leak the future.

Modes (one input vector layout each, all normalized by the SAME train-derived bounds):

  none      -> 14D  P_t = S_t                    legacy R2-R5 behaviour, unchanged
  duplicate -> 28D  P_t = [S_t, S_t]             control arm: same capacity, NO history
  true      -> 28D  P_t = [S_t, S_(t-1)]         treatment arm: real past state

Rules enforced here (and tested):
  * slot 0 is ALWAYS the current state, slot 1 is ALWAYS the past state;
  * both slots are normalized with the same pmin/pmax because they carry the same
    physical quantity -- the previous slot never uses an action, an action delta or
    any future frame;
  * the first frame of an episode has no S_(t-1); slot 1 is filled with S_0, i.e.
    the implied velocity is zero. Training and deployment use THIS SAME rule;
  * deployment keeps a per-env buffer so S_(t-1) is the previously OBSERVED state
    of the same environment (never a prediction), and reset() drops it.

numpy only: no torch import, so the CPU test suite can cover the whole contract.
"""
import numpy as np

BASE_DIM = 14
NONE = "none"
DUPLICATE = "duplicate"
TRUE = "true"
MODES = (NONE, DUPLICATE, TRUE)
PROPRIO_DIM = {NONE: BASE_DIM, DUPLICATE: 2 * BASE_DIM, TRUE: 2 * BASE_DIM}

MODE_DESCRIPTION = {
    NONE: "legacy 14D current state only (R2-R5)",
    DUPLICATE: "28D control: current state in both slots (no history information)",
    TRUE: "28D treatment: current state + previous state",
}


def proprio_dim(mode):
    if mode not in PROPRIO_DIM:
        raise ValueError("unknown proprio history mode %r (expected one of %s)" % (mode, MODES))
    return PROPRIO_DIM[mode]


def normalize_block(state, pmin, pmax):
    """Identical formula to training/train_pushT_pure2.py::norm_pr."""
    s = np.asarray(state, dtype=np.float32)
    lo = np.asarray(pmin, dtype=np.float32)
    hi = np.asarray(pmax, dtype=np.float32)
    if s.shape[-1] != BASE_DIM or lo.shape != (BASE_DIM,) or hi.shape != (BASE_DIM,):
        raise ValueError("state/pmin/pmax must be 14D: %s %s %s" % (s.shape, lo.shape, hi.shape))
    if not np.isfinite(s).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("non-finite state or proprio bounds")
    if (hi <= lo).any():
        raise ValueError("invalid proprio bounds (pmax <= pmin)")
    return np.clip(2.0 * (s - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0).astype(np.float32)


def build(state_now, state_prev, pmin, pmax, mode=TRUE):
    """Return the proprio vector for one observation as a (PROPRIO_DIM[mode],) array."""
    dim = proprio_dim(mode)
    now = np.asarray(state_now, dtype=np.float32).reshape(-1)
    prev = now if state_prev is None else np.asarray(state_prev, dtype=np.float32).reshape(-1)
    if now.shape != (BASE_DIM,) or prev.shape != (BASE_DIM,):
        raise ValueError("state_now/state_prev must be 14D")
    if not np.isfinite(now).all() or not np.isfinite(prev).all():
        raise ValueError("non-finite state_now/state_prev")
    cur_n = normalize_block(now, pmin, pmax)
    if mode == NONE:
        return cur_n
    # The control arm must NOT be able to read the past state: overwrite the slot.
    past_n = cur_n if mode == DUPLICATE else normalize_block(prev, pmin, pmax)
    out = np.concatenate([cur_n, past_n]).astype(np.float32)
    if out.shape != (dim,):
        raise ValueError("unexpected proprio width %s (expected %d)" % (out.shape, dim))
    return out


def first_frame_policy():
    """Machine-readable statement of the t == 0 rule (same in training and deployment)."""
    return {"rule": "previous slot = current state when no earlier frame exists",
            "implied_velocity": 0.0,
            "applies_to": ["episode frame 0", "first observation after reset()"]}


class HistoryBuffer:
    """Deployment-side cache of the previously OBSERVED state, per environment."""

    def __init__(self, mode=TRUE):
        proprio_dim(mode)
        self.mode = mode
        self._last = {}

    def reset(self, env_idx=None):
        if env_idx is None:
            self._last.clear()
        else:
            self._last.pop(env_idx, None)

    def push(self, state, env_idx=0):
        """Record the observation and return (state_now, state_prev) for this step."""
        now = np.asarray(state, dtype=np.float32).reshape(-1)
        if now.shape != (BASE_DIM,) or not np.isfinite(now).all():
            raise ValueError("state must be a finite 14D vector")
        prev = self._last.get(env_idx)
        self._last[env_idx] = now.copy()
        return now, (now.copy() if prev is None else prev)

    def observe(self, state, pmin, pmax, env_idx=0):
        now, prev = self.push(state, env_idx)
        return build(now, prev, pmin, pmax, self.mode)


def checkpoint_fields(mode):
    """Metadata a 28D checkpoint MUST carry; legacy checkpoints carry neither key."""
    return {"proprio_dim": proprio_dim(mode), "proprio_history_mode": mode,
            "proprio_layout": None if mode == NONE else "[current_state, previous_state]",
            "first_frame_policy": None if mode == NONE else first_frame_policy()["rule"]}


def resolve_mode(meta, requested=None):
    """Fail closed: a 28D checkpoint without explicit metadata must not be loaded."""
    declared_dim = meta.get("proprio_dim", BASE_DIM)
    declared_mode = meta.get("proprio_history_mode", NONE)
    if declared_mode not in MODES:
        raise ValueError("checkpoint has unknown proprio_history_mode=%r" % (declared_mode,))
    if declared_dim != PROPRIO_DIM[declared_mode]:
        raise ValueError("checkpoint proprio_dim=%r inconsistent with mode=%r"
                         % (declared_dim, declared_mode))
    if requested is not None and requested != declared_mode:
        raise ValueError("proprio history mismatch: checkpoint=%s deployment=%s"
                         % (declared_mode, requested))
    return declared_mode


def verify_projector_width(meta, projector_proprio_dim, mode=None):
    """The projector's first layer must match the checkpoint's declared width."""
    declared_dim = meta.get("proprio_dim", BASE_DIM)
    if int(projector_proprio_dim) != int(declared_dim):
        raise ValueError("proprio projector expects %s inputs but checkpoint declares proprio_dim=%s"
                         % (projector_proprio_dim, declared_dim))
    return resolve_mode(meta, mode)


def time_alignment_report(state_now, state_prev, frame_idx):
    """Refuse history that is not strictly the previous frame of the same episode."""
    if int(frame_idx) < 0:
        raise ValueError("negative frame index")
    if int(frame_idx) == 0:
        if state_prev is not None and not np.allclose(np.asarray(state_prev), np.asarray(state_now)):
            raise ValueError("frame 0 must use the first-frame policy (previous == current)")
        return {"frame": 0, "policy": "previous_state := current_state"}
    if state_prev is None:
        raise ValueError("frame %d requires an explicit previous state" % int(frame_idx))
    return {"frame": int(frame_idx), "policy": "previous_state := observed state at t-1"}
