"""R8 contract: 42-dim three-frame proprio + velocity-residual action target.

PROPRIO (past-only, three observed frames)
------------------------------------------
    P_t = [S_t, V_t, Acc_t]                       (14 + 14 + 14 = 42)
    V_t   = S_t - S_(t-1)                          t >= 1   (0 at t = 0)
    Acc_t = S_t - 2*S_(t-1) + S_(t-2)              t >= 2   (0 at t < 2)

Boundary policy (identical in training, evaluation and deployment):
    t = 0 : V = 0, Acc = 0      (only one observation exists)
    t = 1 : V = S1 - S0, Acc = 0 (two observations exist, no third)
Deployment replays the same rule from a per-environment buffer of the last TWO
OBSERVED states; predictions are never fed back into the history.

TARGET (velocity residual, distinct from R5's position residual)
---------------------------------------------------------------
    training supervision : R_t = (A_t - S_t) - V_t
    inference command    : A_hat = S_t + V_t + R_hat
R5/R6 used Delta = A - S; using Delta here would silently change the objective, so
this module keeps the two representations in separate, explicitly named functions.

Quantiles: q01/q99 MUST be computed on the R representation from TRAIN episodes only
(see velocity_residual_quantiles). Decoding uses the frozen R2 256-bin codec unchanged.
"""
import numpy as np

BASE_DIM = 14
N_DIM = 3 * BASE_DIM
PROPRIO_SVACC = "svacc"
PROPRIO_DIM = {PROPRIO_SVACC: N_DIM}

# Target representations that exist in this project. Only VELOCITY_RESIDUAL is R8.
ABS_POSITION = "absolute"
POSITION_RESIDUAL = "residual"          # R5/R6: A - S
VELOCITY_RESIDUAL = "velocity_residual"  # R8: (A - S) - V
TARGETS = (ABS_POSITION, POSITION_RESIDUAL, VELOCITY_RESIDUAL)


# --------------------------------------------------------------------- validation
def _as14(x, name):
    a = np.asarray(x, dtype=np.float32)
    if a.shape[-1] != BASE_DIM:
        raise ValueError("%s must have a last dimension of %d, got %s" % (name, BASE_DIM, a.shape))
    if not np.isfinite(a).all():
        raise ValueError("%s contains non-finite values" % name)
    return a


def normalize_block(state, pmin, pmax):
    s = _as14(state, "state")
    lo = np.asarray(pmin, np.float32)
    hi = np.asarray(pmax, np.float32)
    if lo.shape != (BASE_DIM,) or hi.shape != (BASE_DIM,):
        raise ValueError("pmin/pmax must be 14D")
    if (hi <= lo).any():
        raise ValueError("pmax must exceed pmin for every dimension")
    return np.clip(2.0 * (s - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------- proprio
def velocity(state_now, state_prev):
    return _as14(state_now, "state_now") - _as14(state_prev, "state_prev")


def acceleration(state_now, state_prev, state_prev2):
    s0 = _as14(state_now, "state_now")
    s1 = _as14(state_prev, "state_prev")
    s2 = _as14(state_prev2, "state_prev2")
    return (s0 - s1) - (s1 - s2)


def build_proprio(state_now, state_prev=None, state_prev2=None, pmin=None, pmax=None,
                  normalized=True):
    """Return the 42D P_t = [S_t, V_t, Acc_t] with the documented boundary rules."""
    now = _as14(state_now, "state_now")
    if pmin is None or pmax is None:
        raise ValueError("pmin/pmax are required")
    if state_prev is None:
        raise ValueError("state_prev is required; use first_frames() for the t=0 boundary")
    prev = _as14(state_prev, "state_prev")
    v = now - prev
    acc = np.zeros(BASE_DIM, np.float32) if state_prev2 is None else acceleration(now, prev, state_prev2)
    if normalized:
        now_n, v_n, acc_n = (normalize_block(now, pmin, pmax),
                             normalize_block(v, pmin, pmax),
                             normalize_block(acc, pmin, pmax))
        out = np.concatenate([now_n, v_n, acc_n])
    else:
        out = np.concatenate([now, v, acc])
    if out.shape != (N_DIM,) or not np.isfinite(out).all():
        raise ValueError("bad 42D proprio vector: %s" % (out.shape,))
    return out.astype(np.float32)


def boundary_flags(t):
    """Which history terms are defined at frame t (machine-readable policy)."""
    t = int(t)
    if t < 0:
        raise ValueError("negative frame index")
    return {"frame": t, "velocity_defined": t >= 1, "acceleration_defined": t >= 2,
            "policy": ("t=0: V=0,Acc=0" if t == 0 else
                       "t=1: Acc=0" if t == 1 else "t>=2: full three-frame history")}


class HistoryBuffer:
    """Deployment-side cache of the last TWO observed states, per environment.

    push() records an observation and returns (S_t, S_(t-1), S_(t-2) or None), so the
    same build_proprio() boundary rules apply online. reset() drops the history, and a
    fresh environment therefore reproduces the t=0 / t=1 behaviour exactly.
    """

    def __init__(self):
        self._hist = {}

    def reset(self, env_idx=None):
        if env_idx is None:
            self._hist.clear()
        else:
            self._hist.pop(env_idx, None)

    def push(self, state, env_idx=0):
        now = _as14(np.asarray(state, np.float32).reshape(-1), "state")
        h = self._hist.setdefault(env_idx, [])
        prev = h[-1] if len(h) >= 1 else None
        prev2 = h[-2] if len(h) >= 2 else None
        h.append(now.copy())
        if len(h) > 2:
            del h[0]
        return now, prev, prev2

    def observe(self, state, pmin, pmax, env_idx=0):
        now, prev, prev2 = self.push(state, env_idx)
        if prev is None:
            # t = 0: no history at all -> V = 0, Acc = 0 by duplicating the observation
            zero = np.zeros(BASE_DIM, np.float32)
            cur_n = normalize_block(now, pmin, pmax)
            return np.concatenate([cur_n, normalize_block(zero, pmin, pmax),
                                   normalize_block(zero, pmin, pmax)]).astype(np.float32)
        return build_proprio(now, prev, prev2, pmin, pmax)


def first_frames():
    return {"frame_0": boundary_flags(0), "frame_1": boundary_flags(1),
            "frame_2_plus": boundary_flags(2)}


# ------------------------------------------------------- velocity-residual target
def validate_target_mode(mode):
    if mode not in TARGETS:
        raise ValueError("unknown target representation %r (expected one of %s)" % (mode, TARGETS))
    return mode


def training_target(state, action, velocity, mode=VELOCITY_RESIDUAL):
    """Physical quantity to discretise. R8 uses VELOCITY_RESIDUAL only."""
    validate_target_mode(mode)
    s = _as14(state, "state")
    a = _as14(action, "action")
    v = _as14(velocity, "velocity")
    if mode == ABS_POSITION:
        return a.copy()
    if mode == POSITION_RESIDUAL:
        return (a - s).astype(np.float32)
    return ((a - s) - v).astype(np.float32)


def command(state, velocity, decoded_target, mode=VELOCITY_RESIDUAL):
    """Inverse transform into an ABSOLUTE joint-position command."""
    validate_target_mode(mode)
    s = _as14(state, "state")
    v = _as14(velocity, "velocity")
    t = _as14(decoded_target, "decoded_target")
    if mode == ABS_POSITION:
        return t.copy()
    if mode == POSITION_RESIDUAL:
        return (s + t).astype(np.float32)
    return (s + v + t).astype(np.float32)


def velocity_residual_quantiles(states, actions, velocities):
    """q01/q99 of R, computed on the TRAIN rows the caller passes in."""
    s = _as14(states, "states")
    a = _as14(actions, "actions")
    v = _as14(velocities, "velocities")
    if s.shape != a.shape or s.shape != v.shape:
        raise ValueError("states/actions/velocities must share a shape")
    if len(s) < 100:
        raise ValueError("refusing to compute quantiles from fewer than 100 rows")
    r = (a - s) - v
    lo = np.quantile(r, 0.01, axis=0).astype(np.float32)
    hi = np.quantile(r, 0.99, axis=0).astype(np.float32)
    if (hi <= lo).any():
        raise ValueError("degenerate quantiles: q99 <= q01 for some dimension")
    return lo, hi


# -------------------------------------------------------------------- checkpoint
def checkpoint_fields(mode=VELOCITY_RESIDUAL, q01=None, q99=None, target_mode=None):
    tmode = target_mode or mode
    validate_target_mode(tmode)
    out = {"proprio_mode": PROPRIO_SVACC, "proprio_dim": PROPRIO_DIM[PROPRIO_SVACC],
           "proprio_layout": "[state, velocity, acceleration]",
           "target_representation": tmode,
           "boundary_policy": "V=0,Acc=0 at t=0; Acc=0 at t=1",
           "quantile_source": "TRAIN episodes only, computed on the target representation"}
    if q01 is not None and q99 is not None:
        out["q01"] = np.asarray(q01, np.float32).tolist()
        out["q99"] = np.asarray(q99, np.float32).tolist()
    return out


def resolve_checkpoint(meta, projector_proprio_dim=None):
    """Fail closed: an R8 checkpoint must declare itself; legacy checkpoints stay legacy."""
    declared = meta.get("proprio_mode")
    if declared is None:
        # Legacy R2-R6 checkpoint: 14D or 28D, no R8 semantics implied.
        legacy_dim = int(meta.get("proprio_dim", BASE_DIM))
        if legacy_dim not in (BASE_DIM, 2 * BASE_DIM):
            raise ValueError("legacy checkpoint declares unsupported proprio_dim=%r" % legacy_dim)
        return {"kind": "legacy", "proprio_dim": legacy_dim,
                "target_representation": meta.get("action_target_mode", ABS_POSITION)}
    if declared != PROPRIO_SVACC:
        raise ValueError("unknown proprio_mode %r" % (declared,))
    if int(meta.get("proprio_dim", -1)) != N_DIM:
        raise ValueError("R8 checkpoint must declare proprio_dim=%d" % N_DIM)
    tmode = meta.get("target_representation")
    if tmode != VELOCITY_RESIDUAL:
        raise ValueError("R8 checkpoint must declare target_representation=%r" % VELOCITY_RESIDUAL)
    for key in ("q01", "q99"):
        if key not in meta or len(meta[key]) != BASE_DIM:
            raise ValueError("R8 checkpoint must carry a 14D %s for the R target" % key)
    if projector_proprio_dim is not None and int(projector_proprio_dim) != N_DIM:
        raise ValueError("projector expects %s inputs, R8 requires %d"
                         % (projector_proprio_dim, N_DIM))
    return {"kind": "r8", "proprio_dim": N_DIM, "target_representation": VELOCITY_RESIDUAL}


def integration_checklist():
    """The audit items required before any 7B run, with this round's status."""
    return [
        {"item": "data loader emits (t, t-1, t-2) with the documented boundary rules",
         "status": "implemented + tested", "where": "r8_contract.build_proprio / HistoryBuffer"},
        {"item": "42D ProprioProjector", "status": "width verified in contract; weights NOT trained",
         "where": "resolve_checkpoint(projector_proprio_dim=42)"},
        {"item": "q01/q99 computed on the R target from TRAIN only",
         "status": "implemented + tested", "where": "velocity_residual_quantiles"},
        {"item": "256-bin token encoding unchanged (R2 fixed codec)",
         "status": "reused unchanged", "where": "action_codec"},
        {"item": "decoded target -> physical command S + V + R_hat",
         "status": "implemented + tested", "where": "command()"},
        {"item": "KV-cache autoregressive inference",
         "status": "interface unchanged (proprio width only) - needs GPU smoke test",
         "where": "OpenVLA_Pure/model.py"},
        {"item": "action reconstruction / clipping to joint bounds",
         "status": "implemented, bounds clipping remains the caller's responsibility",
         "where": "command()"},
        {"item": "checkpoint metadata (proprio mode, dim, target representation, quantiles)",
         "status": "implemented + fail-closed tests", "where": "checkpoint_fields/resolve_checkpoint"},
        {"item": "first/second frame history initialisation",
         "status": "implemented + tested (identical training/deployment)",
         "where": "boundary_flags / HistoryBuffer"},
        {"item": "train/val/deploy time alignment", "status": "implemented + tested",
         "where": "tests/test_r8_contract.py"},
    ]
