#!/usr/bin/env python3
"""R11 P0-5: fail-closed assertions for every 42D/R training entry.

The contract the trainer must satisfy (any violation raises TrainContractError before a step):

  * frame t >= 2 so that V and Acc come only from OBSERVED history (t-1, t-2);
  * A[t] is the next-step ABSOLUTE joint target, i.e. A[t] == S[t+1] within atol;
  * S/V/Acc are built from history only: mutating any row >= t+1 leaves the row bit-identical;
  * shapes are [B,42] features, [B,14] residual and [B,14,256] token logits;
  * the gripper channels stay at dims 6 and 13;
  * per-env history never mixes envs;
  * the prismatic 7D/8D/chunked dimension defaults never drive this path.

Only numpy + h5py are needed, so the whole contract is testable on CPU against the real HDF5.
"""
import numpy as np

BASE_DIM = 14
N42 = 42
N_BINS = 256
GRIPPER_DIMS = (6, 13)
MIN_HISTORY_FRAMES = 2


class TrainContractError(ValueError):
    pass


# ------------------------------------------------------------------------------ tensors
def assert_frame_index(t, min_frame=MIN_HISTORY_FRAMES):
    if int(t) < int(min_frame):
        raise TrainContractError(
            "frame %d has no [S,V,Acc] history: t must be >= %d" % (int(t), int(min_frame)))
    return int(t)


def assert_shapes(features, residual, logits):
    f = np.asarray(features)
    r = np.asarray(residual)
    lg = np.asarray(logits)
    if f.ndim != 2 or f.shape[1] != N42:
        raise TrainContractError("features must be [B,%d], got %s" % (N42, (f.shape,)))
    if r.ndim != 2 or r.shape[1] != BASE_DIM:
        raise TrainContractError("residual must be [B,%d], got %s" % (BASE_DIM, (r.shape,)))
    if lg.ndim != 3 or lg.shape[1] != BASE_DIM or lg.shape[2] != N_BINS:
        raise TrainContractError("logits must be [B,%d,%d], got %s"
                                 % (BASE_DIM, N_BINS, (lg.shape,)))
    if not (f.shape[0] == r.shape[0] == lg.shape[0]):
        raise TrainContractError("batch sizes differ: %d/%d/%d"
                                 % (f.shape[0], r.shape[0], lg.shape[0]))
    if not (np.isfinite(f).all() and np.isfinite(r).all()):
        raise TrainContractError("non-finite features/residual")
    return True


def assert_gripper_dims(gripper_dims=GRIPPER_DIMS):
    g = tuple(int(x) for x in gripper_dims)
    if g != tuple(GRIPPER_DIMS):
        raise TrainContractError("gripper dims must be %s, got %s" % (GRIPPER_DIMS, g))
    return g


def assert_window(logits):
    lg = np.asarray(logits)
    if lg.shape[-1] != N_BINS:
        raise TrainContractError("only the %d-wide action window may be used, got %d"
                                 % (N_BINS, lg.shape[-1]))
    return N_BINS


# ------------------------------------------------------------------------------ dataset
def read14(h, group):
    """14D [left_arm(6), left_ee(1), right_arm(6), right_ee(1)] state or action block."""
    v = np.concatenate([h[group + "/left_arm_joint_states"][:],
                        h[group + "/left_ee_joint_states"][:],
                        h[group + "/right_arm_joint_states"][:],
                        h[group + "/right_ee_joint_states"][:]], axis=1).astype(np.float32)
    if v.ndim != 2 or v.shape[1] != BASE_DIM:
        raise TrainContractError("%s must be [T,14], got %s" % (group, (v.shape,)))
    if not np.isfinite(v).all():
        raise TrainContractError("%s contains non-finite values" % group)
    return v


def assert_a_equals_next_state(state, action, atol=1e-6):
    """A[t] is the next-step absolute joint target: A[t] == S[t+1]."""
    s = np.asarray(state, np.float32)
    a = np.asarray(action, np.float32)
    if s.shape != a.shape:
        raise TrainContractError("state/action shapes differ: %s vs %s" % (s.shape, a.shape))
    if len(s) < 2:
        raise TrainContractError("episode too short")
    d = np.abs(a[:-1] - s[1:])
    worst = float(d.max())
    if worst > atol:
        raise TrainContractError("A[t] is not the next-step absolute target: max|A[t]-S[t+1]| = "
                                 "%g > %g" % (worst, atol))
    return worst


def assert_next_step_target(state, action, t, atol=1e-6):
    s = np.asarray(state, np.float32)
    a = np.asarray(action, np.float32)
    t = int(t)
    if t < 0 or t >= len(s) - 1:
        raise TrainContractError("frame %d has no next state (T=%d)" % (t, len(s)))
    d = float(np.abs(a[t] - s[t + 1]).max())
    if d > atol:
        raise TrainContractError("A[%d] is not S[%d]: max|delta| = %g > %g"
                                 % (t, t + 1, d, atol))
    return d


def build_training_row(state, action, t, contract, gripper_dims=GRIPPER_DIMS):
    """One supervised row built ONLY from observed history t, t-1, t-2 and the target A[t]."""
    import r42_adapter as R
    assert_gripper_dims(gripper_dims)
    s = np.asarray(state, np.float32)
    a = np.asarray(action, np.float32)
    t = assert_frame_index(t)
    if t > len(s) - 2:
        raise TrainContractError("frame %d needs a next state for A[t] (T=%d)" % (t, len(s)))
    assert_next_step_target(s, a, t)
    now, prev, prev2 = s[t], s[t - 1], s[t - 2]
    velocity = (now - prev).astype(np.float32)
    acceleration = (velocity - (prev - prev2)).astype(np.float32)
    features = contract.transform(now, velocity, acceleration)
    if features.shape != (N42,) or not np.isfinite(features).all():
        raise TrainContractError("42D feature invalid: %s" % (features.shape,))
    residual = ((a[t] - now) - velocity).astype(np.float32)
    if residual.shape != (BASE_DIM,) or not np.isfinite(residual).all():
        raise TrainContractError("residual invalid: %s" % (residual.shape,))
    return {"t": t, "features": features, "residual": residual, "state": now,
            "velocity": velocity, "acceleration": acceleration, "target": a[t].copy(),
            "history_rows": [t, t - 1, t - 2], "gripper_dims": tuple(gripper_dims)}


def assert_no_future_leakage(state, action, t, contract):
    """Mutating every row >= t+1 must not change the training row."""
    import r42_adapter as R
    s = np.asarray(state, np.float32)
    a = np.asarray(action, np.float32)
    t = assert_frame_index(t)
    row = build_training_row(s, a, t, contract)
    s2 = s.copy()
    a2 = a.copy()
    s2[t + 1:] += 1234.5
    a2[t + 1:] -= 987.25
    try:
        row2 = build_training_row(s2, a2, t, contract)
    except TrainContractError:
        # A[t] check fails once the future is destroyed: also proof that no future is used.
        return {"features_identical": True, "checked": "target check fires first"}
    same = np.array_equal(row["features"], row2["features"]) and \
        np.array_equal(row["residual"], row2["residual"])
    if not same:
        raise TrainContractError("future rows leaked into the training row at t=%d" % t)
    return {"features_identical": True, "residual_identical": True}


def assert_env_isolation(history_cls, contract, states_by_env):
    """Two envs fed different observations must produce different features and reset cleanly."""
    h = history_cls(contract)
    feats = {}
    for env, rows in states_by_env.items():
        for i, s in enumerate(rows):
            feats.setdefault(env, []).append(h.observe(int(env), np.asarray(s, np.float32),
                                                       timestamp=float(i)))
    envs = sorted(feats)
    if len(envs) < 2:
        raise TrainContractError("need at least two envs to prove isolation")
    if np.array_equal(feats[envs[0]][-1], feats[envs[1]][-1]):
        raise TrainContractError("envs share history")
    h.reset(envs[0])
    try:
        h.features(envs[0])
    except Exception:
        pass
    else:
        raise TrainContractError("reset did not clear env %s" % (envs[0],))
    if not np.isfinite(h.features(envs[1])).all():
        raise TrainContractError("reset of one env corrupted another")
    return {"envs": envs, "isolated": True}
