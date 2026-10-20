"""Shared HDF5 access for R6 history-proprio diagnostics (numpy/h5py only, no torch).

The 14D action/state vector keeps the EXACT concatenation order used by
training/train_pushT_pure2.py::c14 (left arm 6, left gripper 1, right arm 6,
right gripper 1), so any number produced here is comparable to R2-R5 results.
"""
import json
import os

import numpy as np

ACTION_DIM = 14
ARM_DIM = 6
LEFT = slice(0, 7)      # left arm joints 0..5 + left gripper 6
RIGHT = slice(7, 14)    # right arm joints 7..12 + right gripper 13
DIM_NAMES = (["left_arm_%d" % i for i in range(6)] + ["left_gripper"]
             + ["right_arm_%d" % i for i in range(6)] + ["right_gripper"])

HDF5_DIR = os.environ.get("PUSHT_HDF5", "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
SPLIT_PATH = os.environ.get("PUSHT_SPLIT", "/home/dgut/pusht_runtime/split80.json")


def load_split(path=None):
    """Train/val episode lists; fail closed on overlap or duplicates."""
    with open(path or SPLIT_PATH) as fh:
        split = json.load(fh)
    train, val = list(split["train"]), list(split["val"])
    if not train or not val:
        raise ValueError("empty train or val episode list")
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError("duplicate episodes inside a split")
    if set(train) & set(val):
        raise ValueError("train/val episode overlap")
    return {"train": train, "val": val}


def c14(h, group):
    """Identical concatenation order to the training script's c14."""
    return np.concatenate([
        h[group + "/left_arm_joint_states"][:], h[group + "/left_ee_joint_states"][:],
        h[group + "/right_arm_joint_states"][:], h[group + "/right_ee_joint_states"][:],
    ], axis=1).astype(np.float64)


def validate_arrays(state, action, name=""):
    s = np.asarray(state, dtype=np.float64)
    a = np.asarray(action, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != ACTION_DIM:
        raise ValueError("%s state must be (T,%d), got %s" % (name, ACTION_DIM, s.shape))
    if a.shape != s.shape:
        raise ValueError("%s action/state shape mismatch %s vs %s" % (name, a.shape, s.shape))
    if not np.isfinite(s).all() or not np.isfinite(a).all():
        raise ValueError("%s non-finite state or action" % name)
    return s, a


def load_episode(name, hdf5_dir=None):
    """Return {'state','action','frequency','instruction','frames'} for one episode."""
    import h5py
    path = os.path.join(hdf5_dir or HDF5_DIR, name)
    with h5py.File(path, "r") as h:
        state = c14(h, "state")
        action = c14(h, "action")
        freq = int(h["additional_info/frequency"][()]) if "additional_info/frequency" in h else None
        instr = h["instruction"][()]
        if isinstance(instr, bytes):
            instr = instr.decode("utf-8", "replace")
    state, action = validate_arrays(state, action, name)
    return {"episode": name, "state": state, "action": action,
            "frames": int(state.shape[0]), "frequency": freq, "instruction": instr}


def load_episodes(names, hdf5_dir=None):
    return [load_episode(n, hdf5_dir) for n in names]


def valid_frames(n_frames):
    """Frames t where BOTH t-1 and t+1 exist (so E1/E3 are well defined)."""
    if n_frames < 4:
        raise ValueError("episode too short for temporal alignment analysis: %d" % n_frames)
    return np.arange(1, n_frames - 1, dtype=np.int64)


def arm_of(dim):
    return "left" if dim < 7 else "right"
