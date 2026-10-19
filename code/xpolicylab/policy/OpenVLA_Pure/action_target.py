"""Shared absolute-vs-residual action target contract for pure OpenVLA.

No weights or model-specific imports. Training targets and deployment predictions
MUST use the same target representation and train-split q01/q99 statistics.

absolute: target = next joint position A
residual: target = next joint position A - current observed joint state S
The physical command sent to RoboDojo is ALWAYS an absolute joint position.
"""
import numpy as np

ABSOLUTE = "absolute"
RESIDUAL = "residual"
TARGET_MODES = (ABSOLUTE, RESIDUAL)
ACTION_DIM = 14


def validate_mode(mode):
    if mode not in TARGET_MODES:
        raise ValueError("unknown action target mode %r (expected %s)" % (mode, TARGET_MODES))
    return mode


def _pair(state, action):
    s = np.asarray(state, dtype=np.float32)
    a = np.asarray(action, dtype=np.float32)
    if s.shape != a.shape or s.ndim == 0 or s.shape[-1] != ACTION_DIM:
        raise ValueError("state/action shape mismatch or last dimension is not 14: %s / %s" % (s.shape, a.shape))
    if not np.isfinite(s).all() or not np.isfinite(a).all():
        raise ValueError("non-finite state/action")
    return s, a


def training_target(state, action, mode=ABSOLUTE):
    """Obtain the physical 14D quantity to discretize for training."""
    validate_mode(mode)
    s, a = _pair(state, action)
    return a.copy() if mode == ABSOLUTE else (a - s).astype(np.float32)


def joint_command(state, decoded_target, mode=ABSOLUTE):
    """Inverse transform into an *absolute* joint-position command."""
    validate_mode(mode)
    s, a = _pair(state, decoded_target)
    return a.copy() if mode == ABSOLUTE else (s + a).astype(np.float32)


def validate_stats(q01, q99):
    lo = np.asarray(q01, dtype=np.float32)
    hi = np.asarray(q99, dtype=np.float32)
    if lo.shape != (ACTION_DIM,) or hi.shape != (ACTION_DIM,):
        raise ValueError("expected q01 and q99 vectors with 14 entries")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or (hi < lo).any():
        raise ValueError("invalid/nonfinite action target quantiles")
    return lo, hi


def resolve_checkpoint_target(meta, configured_mode=ABSOLUTE):
    """Fail closed for residual checkpoints lacking explicit target metadata.

    Older absolute-position checkpoints did not store target mode. They may be
    loaded as absolute ONLY. A residual checkpoint MUST store a residual mode.
    """
    requested = validate_mode(configured_mode)
    declared = meta.get("action_target_mode")
    if declared is None:
        if requested != ABSOLUTE:
            raise ValueError("residual deployment requires action_target_mode=residual in checkpoint metadata")
        return ABSOLUTE
    validate_mode(declared)
    if declared != requested:
        raise ValueError("action target mismatch: checkpoint=%s deployment=%s" % (declared, requested))
    return requested


def file_sha256(path):
    """Hash checkpoint files so a residual target cannot be paired with old weights."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record_checkpoint_hashes(meta, adapter_dir, projector_path):
    """Add binding hashes to metadata after checkpoint files are written."""
    import os
    out = dict(meta)
    out["adapter_sha256"] = file_sha256(os.path.join(adapter_dir, "adapter_model.safetensors"))
    out["proprio_projector_sha256"] = file_sha256(projector_path)
    return out


def verify_checkpoint_hashes(meta, adapter_dir, projector_path, mode):
    """Residual policy requires exact file identities; legacy absolute may lack hashes."""
    import os
    validate_mode(mode)
    expected_adapter = meta.get("adapter_sha256")
    expected_projector = meta.get("proprio_projector_sha256")
    if mode == RESIDUAL and (not expected_adapter or not expected_projector):
        raise ValueError("residual checkpoint metadata must bind adapter and projector SHA-256")
    for name, path, expected in (
        ("adapter", os.path.join(adapter_dir, "adapter_model.safetensors"), expected_adapter),
        ("proprio_projector", projector_path, expected_projector),
    ):
        if expected and file_sha256(path) != expected:
            raise ValueError("%s checksum mismatch: checkpoint artifacts and statistics do not belong together" % name)
    return True


def verify_runtime_modeling(meta, core_model, mode):
    """Check the loaded HF dynamic module, not merely an unused source file."""
    import inspect
    validate_mode(mode)
    expected = meta.get("modeling_sha256")
    if not expected and mode == RESIDUAL:
        raise ValueError("residual checkpoint lacks runtime modeling_sha256 identity")
    if expected:
        actual = file_sha256(inspect.getfile(type(core_model)))
        if actual != expected:
            raise ValueError("active modeling_prismatic.py differs from checkpoint training module")
    return True
