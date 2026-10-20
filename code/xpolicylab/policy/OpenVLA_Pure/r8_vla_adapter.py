"""R8 VLA adapter: 42-dim proprio projector + velocity-residual command path.

This module is the single place where the R8 *contract* meets the OpenVLA *runtime*:

  P_t   = [S_t, V_t, Acc_t]        (42D, past-only, boundary rules from r8_contract)
  R_t   = (A_t - S_t) - V_t        (training supervision)
  A_hat = S_t + V_t + R_hat        (deployment command)

Feature layout and normalization come from `r8_normalization.NormalizationContract`
(the DEV-selected scheme is `signed_motion`), so training and inference cannot diverge.

Only numpy is required at import time; torch/prismatic are imported inside the helpers
that need them, so CPU contract tests run without a GPU or a 7B checkpoint.
"""
import json
import numpy as np

BASE_DIM = 14
N_DIM = 42
PROPRIO_MODE = "svacc"
TARGET_MODE = "velocity_residual"

# DEV-selected from the R8 normalization audit: the 12 arm dimensions use the model,
# the two gripper dimensions (6 = left gripper, 13 = right gripper) use copy-V.
DEV_MODEL_DIMS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPER_DIMS = (6, 13)


def _a14(x, name):
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if a.shape != (BASE_DIM,):
        raise ValueError("%s must be 14D, got %s" % (name, a.shape))
    if not np.isfinite(a).all():
        raise ValueError("%s contains non-finite values" % name)
    return a


# ------------------------------------------------------------------- proprio path
def history_terms(state_now, state_prev, state_prev2):
    """V and Acc from three observed frames, with the documented boundary rules."""
    now = _a14(state_now, "state_now")
    if state_prev is None:
        raise ValueError("state_prev is required (use ProprioHistory42 for frame 0)")
    prev = _a14(state_prev, "state_prev")
    v = now - prev
    acc = np.zeros(BASE_DIM, np.float32) if state_prev2 is None else (v - (prev - _a14(state_prev2, "state_prev2")))
    return v.astype(np.float32), acc.astype(np.float32)


def build_proprio(state_now, state_prev, state_prev2, contract):
    """42D feature vector through the frozen normalization contract."""
    v, acc = history_terms(state_now, state_prev, state_prev2)
    out = contract.transform(_a14(state_now, "state_now"), v, acc)
    if out.shape != (N_DIM,):
        raise ValueError("expected %d features, got %s" % (N_DIM, out.shape))
    return out.astype(np.float32)


class ProprioHistory42:
    """Per-environment cache of the last TWO observed states (never predictions)."""

    def __init__(self, contract):
        self.contract = contract
        self._hist = {}

    def reset(self, env_idx=None):
        if env_idx is None:
            self._hist.clear()
        else:
            self._hist.pop(env_idx, None)

    def push(self, state, env_idx=0):
        now = _a14(np.asarray(state, np.float32), "state")
        h = self._hist.setdefault(env_idx, [])
        prev = h[-1] if len(h) >= 1 else None
        prev2 = h[-2] if len(h) >= 2 else None
        h.append(now.copy())
        if len(h) > 2:
            del h[0]
        return now, prev, prev2

    def observe(self, state, env_idx=0):
        """Returns (features42, info) where info records which history terms exist."""
        now, prev, prev2 = self.push(state, env_idx)
        info = {"velocity_defined": prev is not None,
                "acceleration_defined": prev2 is not None}
        if prev is None:
            # frame 0: no history -> V = 0, Acc = 0 through the same contract
            zero = np.zeros(BASE_DIM, np.float32)
            feats = self.contract.transform(now, zero, zero)
            return feats.astype(np.float32), info
        feats = build_proprio(now, prev, prev2, self.contract)
        return feats, info


# ---------------------------------------------------------------- target / command
def training_target(state, action, velocity):
    """R = (A - S) - V, the R8 supervision."""
    s = _a14(state, "state")
    a = _a14(action, "action")
    v = _a14(velocity, "velocity")
    return ((a - s) - v).astype(np.float32)


def command_from_residual(state, velocity, residual):
    """A_hat = S + V + R_hat."""
    return (_a14(state, "state") + _a14(velocity, "velocity")
            + _a14(residual, "residual")).astype(np.float32)


def decode_tokens_to_command(tokens, q01, q99, state, velocity, codec,
                             model_dims=DEV_MODEL_DIMS, cv_command=None):
    """Decode 256-bin indices -> residual -> command, with the DEV gripper policy applied.

    `cv_command` (S + V) is required when any dimension is excluded from the model
    (the two gripper channels) so the policy is explicit rather than implicit.
    """
    idx = np.asarray(tokens, dtype=np.int64).reshape(-1)
    if idx.shape != (BASE_DIM,):
        raise ValueError("expected 14 token indices")
    decoded = codec.denormalize(codec.decode_indices(idx, mode=codec.MODE_FIXED),
                                np.asarray(q01, np.float32), np.asarray(q99, np.float32))
    cmd = command_from_residual(state, velocity, decoded)
    excluded = [d for d in range(BASE_DIM) if d not in set(model_dims)]
    if excluded:
        if cv_command is None:
            raise ValueError("cv_command (S + V) required for the excluded dimensions %s" % excluded)
        cv = _a14(cv_command, "cv_command")
        cmd = cmd.copy()
        cmd[excluded] = cv[excluded]
    return cmd.astype(np.float32), decoded.astype(np.float32), excluded


def clip_command(command, amin, amax):
    c = _a14(command, "command")
    lo = _a14(amin, "amin")
    hi = _a14(amax, "amax")
    if (hi <= lo).any():
        raise ValueError("amax must exceed amin")
    return np.clip(c, lo, hi).astype(np.float32)


# -------------------------------------------------------------------- torch pieces
def build_projector(llm_dim, device="cuda"):
    """Real 42D ProprioProjector (same class OpenVLA uses)."""
    import torch
    from prismatic.models.projectors import ProprioProjector
    proj = ProprioProjector(proprio_dim=N_DIM, llm_dim=int(llm_dim))
    return proj.to(device)


def load_projector_state(projector, state_dict):
    """Fail-closed load that checks the input width before touching the weights."""
    import torch
    if "fc1.weight" not in state_dict:
        raise ValueError("state dict lacks fc1.weight")
    width = int(state_dict["fc1.weight"].shape[1])
    if width != N_DIM:
        raise ValueError("projector expects %d inputs but R8 requires %d" % (width, N_DIM))
    projector.load_state_dict(state_dict)
    return projector


def projector_param_count(llm_dim):
    return N_DIM * int(llm_dim) + int(llm_dim) + int(llm_dim) * int(llm_dim) + int(llm_dim)


# --------------------------------------------------------------------- metadata
def metadata(contract, q01, q99, codec_mode="fixed"):
    """Everything a checkpoint must carry so training == inference."""
    meta = contract.to_metadata()
    meta.update({"proprio_mode": PROPRIO_MODE, "proprio_dim": N_DIM,
                 "target_representation": TARGET_MODE,
                 "action_codec_mode": codec_mode,
                 "gripper_policy": {"model_dims": list(DEV_MODEL_DIMS),
                                    "gripper_dims": list(GRIPPER_DIMS),
                                    "rule": "gripper channels use S + V (DEV-selected)"},
                 "q01": np.asarray(q01, np.float32).tolist(),
                 "q99": np.asarray(q99, np.float32).tolist(),
                 "history_boundary_policy": "V=0,Acc=0 at t=0; Acc=0 at t=1"})
    return meta


def resolve_metadata(meta, projector_dim=None):
    """Fail-closed resolution used by the smoke test and by deployment."""
    from r8_normalization import NormalizationContract
    if meta.get("proprio_mode") != PROPRIO_MODE:
        raise ValueError("checkpoint is not an R8 svacc checkpoint: proprio_mode=%r"
                         % (meta.get("proprio_mode"),))
    if int(meta.get("proprio_dim", -1)) != N_DIM:
        raise ValueError("R8 checkpoint must declare proprio_dim=%d" % N_DIM)
    if meta.get("target_representation") != TARGET_MODE:
        raise ValueError("R8 checkpoint must declare target_representation=%r" % TARGET_MODE)
    for key in ("q01", "q99"):
        if key not in meta or len(meta[key]) != BASE_DIM:
            raise ValueError("R8 checkpoint must carry a 14D %s" % key)
    if projector_dim is not None and int(projector_dim) != N_DIM:
        raise ValueError("projector width %s != %d" % (projector_dim, N_DIM))
    contract = NormalizationContract.from_metadata(meta)
    return {"contract": contract, "q01": np.asarray(meta["q01"], np.float32),
            "q99": np.asarray(meta["q99"], np.float32),
            "gripper_policy": meta.get("gripper_policy")}


def save_metadata(path, meta):
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=1)
    return path


def load_metadata(path):
    with open(path) as fh:
        return json.load(fh)
