#!/usr/bin/env python3
"""R8 contract: 42-dim history proprio plus the velocity-residual target.

This module is a *contract*, not a model.  It defines exactly how the inputs,
the supervision and the action recovery must be built so that training and
deployment cannot disagree, and so that the CPU suite can prove the contract on
real HDF5 data before anything is put on a GPU.

Definitions (identical to the R7 state-dynamics work)
    V_t   = S_t - S_{t-1}
    Acc_t = S_t - 2 S_{t-1} + S_{t-2}
    X_t   = [S_t, V_t, Acc_t]                       (14 + 14 + 14 = 42)
    R_t   = (A_t - S_t) - V_t                       (supervision)
    A_hat = S_t + V_t + R_hat                       (recovery)

Boundary rule (must be the SAME in training and deployment)
    The episode is assumed to start at rest, so every state index before the
    first observed frame is clamped to S_0:

        t = 0 :  S_{-1} := S_0,  S_{-2} := S_0   ->  V_0 = 0,  Acc_0 = 0
        t = 1 :  S_{-1} := S_0 (observed), S_{-2} := S_0
                 -> V_1 = S_1 - S_0,  Acc_1 = V_1

    Deployment therefore only needs the last two OBSERVED states of the same
    environment; no prediction is ever fed back into the history.

Nothing in this module imports torch, so every rule here is CPU-testable.
"""

from __future__ import annotations

import os
import sys

import numpy as np

BASE_DIM = 14
PROPRIO42_DIM = 3 * BASE_DIM


def _as_states(states):
    arr = np.asarray(states, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != BASE_DIM:
        raise ValueError("states must be (N,14), got %s" % (arr.shape,))
    if not np.isfinite(arr).all():
        raise ValueError("states contain non-finite values")
    return arr


def velocity(states):
    """V_t = S_t - S_{t-1} with the rest-before-start boundary rule."""
    states = _as_states(states)
    prev = np.vstack([states[:1], states[:-1]])
    return states - prev


def acceleration(states):
    """Acc_t = S_t - 2 S_{t-1} + S_{t-2} with states clamped to S_0 before start."""
    states = _as_states(states)
    prev = np.vstack([states[:1], states[:-1]])
    prev2 = np.vstack([states[:1], states[:1], states[:-2]]) if len(states) >= 2 else states
    return states - 2.0 * prev + prev2


def build_proprio42(states):
    """X_t = [S_t, V_t, Acc_t], shape (N, 42), past-only by construction."""
    states = _as_states(states)
    V = velocity(states)
    Acc = acceleration(states)
    X = np.concatenate([states, V, Acc], axis=1)
    if X.shape[1] != PROPRIO42_DIM:
        raise AssertionError("proprio42 width must be 42, got %d" % X.shape[1])
    return X


def velocity_residual(states, actions):
    """R_t = (A_t - S_t) - V_t."""
    states = _as_states(states)
    actions = _as_states(actions)
    if actions.shape != states.shape:
        raise ValueError("actions/states shape mismatch")
    return (actions - states) - velocity(states)


def recover_action(states, R_hat):
    """A_hat = S_t + V_t + R_hat (never A = S_t + R_hat, never a second -V)."""
    states = _as_states(states)
    R_hat = _as_states(R_hat)
    if R_hat.shape != states.shape:
        raise ValueError("R_hat/states shape mismatch")
    return states + velocity(states) + R_hat


def fit_bounds_on_train(train_residuals, q_low=1.0, q_high=99.0, min_halfwidth=1.0e-6):
    """Per-dimension (q01, q99) of R, computed on TRAIN frames only.

    A channel that barely moves can have q99 == q01 on the training split (its
    residual window collapses to a point), which would make the 256-bin encoder
    divide by zero.  Such a channel carries no usable signal anyway, so the
    window is widened to a declared half-width instead of failing silently or
    aborting: the widening is explicit and is reported by window_coverage().
    """
    R = _as_states(train_residuals)
    lo = np.percentile(R, q_low, axis=0)
    hi = np.percentile(R, q_high, axis=0)
    # <= 0 catches a point window; < min_halfwidth also catches one that is only
    # a float32 ulp wide, which the codec's 1e-8 stabiliser would dominate.
    degenerate = (hi - lo) < min_halfwidth
    if degenerate.any():
        centre = 0.5 * (lo + hi)
        lo = np.where(degenerate, centre - min_halfwidth, lo)
        hi = np.where(degenerate, centre + min_halfwidth, hi)
    return lo.astype(np.float32), hi.astype(np.float32)


def window_coverage(residuals, q01, q99):
    """How much of each channel the q01..q99 window actually covers.

    A 256-bin tokenizer can only represent values inside the window, so a channel
    whose residual routinely leaves it is effectively truncated no matter how
    many bins are used.  This is reported per channel rather than assumed away.
    """
    R = _as_states(residuals)
    lo = np.asarray(q01, dtype=np.float64)
    hi = np.asarray(q99, dtype=np.float64)
    inside = (R >= lo[None, :]) & (R <= hi[None, :])
    width = hi - lo
    return {
        "coverage_fraction": inside.mean(axis=0).tolist(),
        "clipped_fraction": (1.0 - inside.mean(axis=0)).tolist(),
        "window_width": width.tolist(),
        "abs_max": np.abs(R).max(axis=0).tolist(),
        "truncation_ratio": (np.abs(R).max(axis=0) / np.where(width > 0, width, np.inf)).tolist(),
    }


def token_roundtrip(residuals, q01, q99, codec):
    """Encode R with the shared 256-bin codec and decode it back physically."""
    R = _as_states(residuals)
    tokens = codec.token_ids_for_actions(R.astype(np.float32), q01, q99)
    codec.assert_legal_token_ids(np.asarray(tokens).reshape(-1))
    back = codec.actions_from_token_ids(np.asarray(tokens), q01, q99)
    return np.asarray(back, dtype=np.float64), np.asarray(tokens)


FLOAT32_EPS = float(np.finfo(np.float32).eps)


#: The shared codec adds this to every normalisation denominator.
CODEC_STABILISER = 1.0e-8


def quantization_error_bound(q01, q99, ulp_slack=4.0):
    """Worst-case per-dimension decode error of the shared 256-bin codec.

    Measured property of the codec (action_codec.encode_indices): it applies
    digitize() and then subtracts one, i.e. it FLOOR-quantises towards -inf
    instead of rounding to the nearest level.  The error budget is therefore one
    full bin width, (q99-q01+1e-8)/255, not half of it.

    Two further terms are added rather than ignored:
      * the codec's 1e-8 stabiliser also appears in denormalise, which biases the
        top of the window by up to 1e-8;
      * float32 rounding of the normalise/denormalise pipeline.
    """
    lo = np.asarray(q01, dtype=np.float64)
    hi = np.asarray(q99, dtype=np.float64)
    scale = hi - lo + CODEC_STABILISER
    one_bin = scale / 255.0
    magnitude = np.maximum(np.maximum(np.abs(lo), np.abs(hi)), scale)
    return one_bin + CODEC_STABILISER + ulp_slack * FLOAT32_EPS * magnitude


def clip_to_window(residuals, q01, q99):
    """What the tokenizer can represent: R clipped into the train window."""
    R = _as_states(residuals)
    lo = np.asarray(q01, dtype=np.float64)[None, :]
    hi = np.asarray(q99, dtype=np.float64)[None, :]
    return np.clip(R, lo, hi)


def deploy_step(history, state_now, R_hat, q01, q99, codec):
    """One deployment step: consume an observed state, return X and the physical action."""
    history.push(state_now)
    window = history.window()
    X = build_proprio42(window[-3:] if len(window) >= 3 else window)[-1:]
    R = np.asarray(R_hat, dtype=np.float64).reshape(1, BASE_DIM)
    return X, recover_action(np.asarray(state_now, dtype=np.float64).reshape(1, BASE_DIM), R)


class ObservedHistory:
    """Per-environment buffer of the last two OBSERVED states (never predictions)."""

    def __init__(self, dim=BASE_DIM, pad=3):
        self.dim = dim
        self.pad = pad
        self.items = []

    def push(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(self.dim)
        if not np.isfinite(state).all():
            raise ValueError("non-finite observation")
        self.items.append(state)
        if len(self.items) > self.pad + 2:
            self.items = self.items[-(self.pad + 2):]

    def window(self):
        """Oldest-to-newest observed states, padded with S_0 at the start."""
        if not self.items:
            raise ValueError("empty history")
        window = list(self.items)
        while len(window) < 3:
            window = [window[0]] + window
        return window

    def reset(self):
        self.items = []


def load_codec():
    """Import the single shared action codec shipped with the OpenVLA policy."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "code", "xpolicylab", "policy", "OpenVLA_Pure")
    if path not in sys.path:
        sys.path.insert(0, path)
    import action_codec  # noqa: WPS433 (imported lazily so plain numpy stays enough)

    return action_codec
