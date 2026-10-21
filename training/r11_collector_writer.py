#!/usr/bin/env python3
"""Atomic HDF5 episode writer for the Push-T collector (pure CPU, no simulator).

R11/R15 Lane D.  The previous round could not write a single episode, so this writer exists to
make every future episode auditable in the same terms the existing freeze pipeline uses:

  * layout identical to the downloaded 100 episodes in
    data/data/hdf5/push_T/arx_x5/data: state/action split into the four joint groups
    (6, 1, 6, 1 = 14D in the order left_arm, left_ee, right_arm, right_ee), three JPEG byte
    streams under vision/<cam>/colors, instruction, additional_info/frequency = 25;
  * hash definitions byte-identical to tools/r8_eval_manifest.py (content_sha256 and
    state_action_sha256), so the manifest/freeze/disjoint tooling accepts the file as-is;
  * the size floor/ceiling of tools/r10_freeze_guard.py (>= 100 kB, plausible cameras);
  * an atomic write protocol: <name>.tmp -> flush -> os.fsync -> close -> O_EXCL claim ->
    os.rename.  A pre-existing file is never overwritten, and an interrupted write leaves no
    .tmp behind;
  * fail-closed refusal of fabricated data: zero or constant trajectories, too few frames,
    missing cameras, non-JPEG frames, non-finite arrays, A[t] != S[t+1], an empty or wrong
    instruction, a duplicated name or content hash against the reference set.

Nothing here starts a simulator, imports torch or touches a GPU: it is a pure CPU file writer
that the collector calls once per accepted episode.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone

import numpy as np

__all__ = [
    "write_episode",
    "verify_episode",
    "episode_hashes",
    "build_existing_index",
    "jpeg_size",
    "EpisodeRejected",
    "BASE_DIM",
    "JOINT_GROUPS",
    "CAMERAS",
    "FREQUENCY_HZ",
    "MIN_FRAMES",
    "MIN_BYTES",
    "MAX_BYTES",
]

BASE_DIM = 14
JOINT_GROUPS = (("left_arm", 6), ("left_ee", 1), ("right_arm", 6), ("right_ee", 1))
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
FREQUENCY_HZ = 25
MIN_FRAMES = 100
MIN_BYTES = 100_000                     # freeze guard: anything smaller is a placeholder
MAX_BYTES = 256 * 1024 * 1024           # a real 360-frame episode is ~90 MB
ALIGN_TOL = 1e-6
MIN_FRAME_SIDE = 64
DATA_FORMAT_VERSION = b"r11-collector-1"
GENERATOR = "training/r11_collector_writer.py"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SOF_MARKERS = (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF)


class EpisodeRejected(ValueError):
    """An episode was refused.  The code attribute is stable and machine-checkable."""

    def __init__(self, code, message):
        super().__init__("[%s] %s" % (code, message))
        self.code = str(code)
        self.message = str(message)


# --------------------------------------------------------------------------- helpers

def _reject(code, message):
    raise EpisodeRejected(code, message)


def _as_matrix(values, name):
    if values is None:
        _reject("bad_shape", "%s is None" % name)
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        _reject("bad_shape", "%s is not numeric: %r" % (name, exc))
    if arr.ndim != 2:
        _reject("bad_shape", "%s must be 2D [T, %d], got shape %r"
                % (name, BASE_DIM, tuple(arr.shape)))
    if arr.shape[1] != BASE_DIM:
        _reject("bad_shape", "%s must be [T, %d], got %r"
                % (name, BASE_DIM, tuple(arr.shape)))
    return arr


def jpeg_size(buf):
    """Return (height, width) parsed from a JPEG SOF marker, or None if it is not a JPEG."""
    try:
        b = bytes(buf)
    except (TypeError, ValueError):
        return None
    n = len(b)
    if n < 4 or b[0:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 1 < n:
        if b[i] != 0xFF:
            i += 1
            continue
        while i < n and b[i] == 0xFF:
            i += 1
        if i >= n:
            return None
        marker = b[i]
        i += 1
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xD9:
            return None
        if i + 1 >= n:
            return None
        seg_len = (b[i] << 8) | b[i + 1]
        if seg_len < 2:
            return None
        if marker in _SOF_MARKERS:
            if i + 7 >= n:
                return None
            height = (b[i + 3] << 8) | b[i + 4]
            width = (b[i + 5] << 8) | b[i + 6]
            return (int(height), int(width))
        if marker == 0xDA:          # entropy-coded data before any SOF: give up
            return None
        i += seg_len
    return None


def _concat_groups(handle, group):
    parts = [handle["%s/%s_joint_states" % (group, name)][()] for name, _ in JOINT_GROUPS]
    return np.concatenate(parts, axis=1).astype(np.float64)


def episode_hashes(state, action, instruction):
    """The freeze pipeline's hash definitions (tools/r8_eval_manifest.py), reproduced exactly."""
    state = np.ascontiguousarray(state, dtype="<f8")
    action = np.ascontiguousarray(action, dtype="<f8")
    content = hashlib.sha256()
    content.update(state.tobytes())
    content.update(action.tobytes())
    content.update(str(instruction).encode("utf-8"))
    content.update(np.int64(len(state)).tobytes())
    sa = hashlib.sha256(state.tobytes() + action.tobytes()).hexdigest()
    return {"content_sha256": content.hexdigest(), "state_action_sha256": sa}


def build_existing_index(root):
    """Map filename -> {content_sha256, state_action_sha256} for a reference directory.

    Only the state/action arrays and the instruction are read, so indexing the 100 reference
    episodes costs a few megabytes of I/O (the camera streams are never touched).
    """
    import h5py

    root = os.fspath(root)
    if not os.path.isdir(root):
        _reject("bad_reference_dir", "reference directory does not exist: %r" % (root,))
    index = {}
    for name in sorted(os.listdir(root)):
        if not name.endswith(".hdf5"):
            continue
        path = os.path.join(root, name)
        try:
            with h5py.File(path, "r") as handle:
                state = _concat_groups(handle, "state")
                action = _concat_groups(handle, "action")
                ins = handle["instruction"][()]
        except Exception as exc:                       # noqa: BLE001
            _reject("bad_reference_episode", "%s could not be read: %r" % (name, exc))
        if isinstance(ins, bytes):
            ins = ins.decode("utf-8", "replace")
        index[name] = episode_hashes(state, action, str(ins))
    return index


def _validate_frames(rgb_frames, frames, cameras, min_frame_side, require_jpeg):
    if not isinstance(rgb_frames, dict):
        _reject("missing_camera", "rgb_frames must be a dict of camera -> [bytes], got %r"
                % (type(rgb_frames).__name__,))
    missing = [cam for cam in cameras if cam not in rgb_frames]
    if missing:
        _reject("missing_camera", "missing camera stream(s): %s" % ", ".join(missing))
    out = {}
    payload = 0
    for cam in cameras:
        stream = rgb_frames[cam]
        try:
            n = len(stream)
        except TypeError:
            _reject("bad_frame_count", "camera %s is not a sequence" % cam)
        if n != frames:
            _reject("bad_frame_count", "camera %s has %d frames but the episode has %d"
                    % (cam, n, frames))
        bufs = []
        for t, raw in enumerate(stream):
            if not isinstance(raw, (bytes, bytearray, memoryview, np.bytes_)):
                _reject("bad_frame_bytes", "camera %s frame %d is %r, not JPEG bytes"
                        % (cam, t, type(raw).__name__))
            buf = bytes(raw)
            if not buf:
                _reject("bad_frame_bytes", "camera %s frame %d is empty" % (cam, t))
            if require_jpeg:
                size = jpeg_size(buf)
                if size is None:
                    _reject("bad_frame_bytes",
                            "camera %s frame %d is not a JPEG stream (no SOI/SOF header)"
                            % (cam, t))
                height, width = size
                if height < min_frame_side or width < min_frame_side:
                    _reject("frame_too_small",
                            "camera %s frame %d is %dx%d, below the %dpx floor"
                            % (cam, t, height, width, min_frame_side))
            bufs.append(buf)
            payload += len(buf)
        out[cam] = bufs
    return out, payload


def _write_hdf5(tmp, task, instruction, seed, expert_version_sha, run_id, episode_id,
                state, action, frames, frequency_hz, meta_json, created_utc,
                state_action_sha256, ee_poses):
    import h5py

    vlen_bytes = h5py.special_dtype(vlen=bytes)
    vlen_uint8 = h5py.special_dtype(vlen=np.uint8)
    n_frames = int(state.shape[0])
    with h5py.File(tmp, "w") as handle:
        for group, arrays in (("state", state), ("action", action)):
            grp = handle.create_group(group)
            col = 0
            for name, dim in JOINT_GROUPS:
                grp.create_dataset(name + "_joint_states",
                                   data=np.ascontiguousarray(arrays[:, col:col + dim],
                                                             dtype="<f8"))
                col += dim
        if ee_poses:
            for side in ("left", "right"):
                data = np.ascontiguousarray(ee_poses[side], dtype="<f8")
                handle.create_dataset("state/%s_ee_poses" % side, data=data)
                handle.create_dataset("action/%s_ee_poses" % side, data=data)
        for cam in CAMERAS:
            # JPEG payloads are arbitrary binary, so they are stored as a variable-length
            # uint8 sequence (vlen=bytes would be a VLEN string and rejects embedded NULs).
            # Readers recover bytes with bytes(ds[t]); r11_train_pushT_r42.py already does
            # np.frombuffer(bytes(buf), np.uint8) so both this and the reference layout work.
            ds = handle.create_dataset("vision/%s/colors" % cam, (n_frames,),
                                       dtype=vlen_uint8)
            for t, buf in enumerate(frames[cam]):
                ds[t] = np.frombuffer(buf, dtype=np.uint8)
            ds.attrs["encoding"] = "jpeg"
            ds.attrs["payload"] = "raw-jpeg-bytes"
        handle.create_dataset("instruction", data=instruction.encode("utf-8"),
                              dtype=vlen_bytes)
        handle.create_dataset("data_format_version", data=DATA_FORMAT_VERSION,
                              dtype=vlen_bytes)
        info = handle.create_group("additional_info")
        info.create_dataset("frequency", data=np.int64(frequency_hz))
        attrs = handle.attrs
        attrs["run_id"] = run_id
        attrs["episode_id"] = int(episode_id)
        attrs["task"] = task
        attrs["seed"] = int(seed)
        attrs["expert_version_sha"] = expert_version_sha
        attrs["created_utc"] = created_utc
        attrs["generator"] = GENERATOR
        attrs["frequency_hz"] = int(frequency_hz)
        attrs["n_frames"] = n_frames
        attrs["meta_json"] = meta_json
        attrs["state_action_sha256"] = state_action_sha256
        attrs["content_sha256_note"] = ("content_sha256 is reported by write_episode and "
                                        "cannot be stored here without self-reference")
        handle.flush()
        try:
            os.fsync(handle.id.get_vfd_handle())
        except Exception:                              # noqa: BLE001  (driver dependent)
            pass


def _fsync_path(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- verify

def verify_episode(path, min_frames=MIN_FRAMES, min_bytes=MIN_BYTES, max_bytes=MAX_BYTES,
                   frequency_hz=FREQUENCY_HZ, align_tol=ALIGN_TOL, cameras=CAMERAS,
                   require_jpeg=True, min_frame_side=MIN_FRAME_SIDE):
    """Read an episode back and re-derive every property the freeze pipeline needs."""
    import h5py

    path = os.fspath(path)
    if not os.path.isfile(path):
        _reject("verify_missing", "no such episode: %r" % (path,))
    nbytes = os.path.getsize(path)
    if nbytes < min_bytes:
        _reject("verify_size", "%s is only %d bytes (< %d): a placeholder, not an episode"
                % (os.path.basename(path), nbytes, min_bytes))
    if nbytes > max_bytes:
        _reject("verify_size", "%s is %d bytes (> the %d byte cap)"
                % (os.path.basename(path), nbytes, max_bytes))
    try:
        handle = h5py.File(path, "r")
    except Exception as exc:                           # noqa: BLE001
        _reject("verify_schema", "%s is not a readable HDF5: %r"
                % (os.path.basename(path), exc))
    with handle:
        try:
            state = _concat_groups(handle, "state")
            action = _concat_groups(handle, "action")
        except Exception as exc:                       # noqa: BLE001
            _reject("verify_schema", "state/action groups are incomplete: %r" % (exc,))
        if state.shape != action.shape or state.shape[1] != BASE_DIM:
            _reject("verify_schema", "state/action shapes %r / %r are not [T, %d]"
                    % (tuple(state.shape), tuple(action.shape), BASE_DIM))
        frames = int(state.shape[0])
        if frames < min_frames:
            _reject("verify_frames", "only %d frames (< %d)" % (frames, min_frames))
        if not (np.isfinite(state).all() and np.isfinite(action).all()):
            _reject("verify_nonfinite", "state/action contain non-finite values")
        misalign = float(np.abs(action[:-1] - state[1:]).max()) if frames > 1 else 0.0
        if misalign > align_tol:
            _reject("verify_alignment",
                    "max|A[t]-S[t+1]| = %.6g exceeds %.6g: action is not the next state"
                    % (misalign, align_tol))
        if "instruction" not in handle:
            _reject("verify_instruction", "missing instruction")
        raw = handle["instruction"][()]
        instruction = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        if len(instruction.strip()) < 8:
            _reject("verify_instruction", "instruction is empty: %r" % (instruction,))
        freq = None
        if "additional_info/frequency" in handle:
            freq = int(handle["additional_info/frequency"][()])
        if freq != int(frequency_hz):
            _reject("verify_frequency", "frequency %r != %d Hz" % (freq, frequency_hz))
        cam_stats = {}
        for cam in cameras:
            key = "vision/%s/colors" % cam
            if key not in handle:
                _reject("verify_camera", "missing camera stream %s" % cam)
            ds = handle[key]
            if int(ds.shape[0]) != frames:
                _reject("verify_camera", "camera %s has %d frames, expected %d"
                        % (cam, int(ds.shape[0]), frames))
            total = 0
            for t in range(frames):
                buf = bytes(ds[t])
                if not buf:
                    _reject("verify_camera", "camera %s frame %d is empty" % (cam, t))
                if require_jpeg and jpeg_size(buf) is None:
                    _reject("verify_camera", "camera %s frame %d is not JPEG" % (cam, t))
                total += len(buf)
            cam_stats[cam] = {"frames": frames, "bytes": total}
        if "data_format_version" in handle:
            version = handle["data_format_version"][()]
            version = (version.decode("utf-8", "replace") if isinstance(version, bytes)
                       else str(version))
        else:
            version = None
    hashes = episode_hashes(state, action, instruction)
    return {"path": path, "file": os.path.basename(path), "bytes": int(nbytes),
            "frames": frames, "frequency": freq, "instruction": instruction,
            "cameras": cam_stats, "data_format_version": version,
            "content_sha256": hashes["content_sha256"],
            "state_action_sha256": hashes["state_action_sha256"],
            "max_abs_action_minus_next_state": misalign, "verified": True}


# --------------------------------------------------------------------------- write

def write_episode(new_root, run_id, episode_id, task, instruction, state, action,
                  rgb_frames, seed, expert_version_sha, meta,
                  frequency_hz=FREQUENCY_HZ, min_frames=MIN_FRAMES,
                  min_bytes=MIN_BYTES, max_bytes=MAX_BYTES, align_tol=ALIGN_TOL,
                  existing_dirs=(), existing_index=None, allow_existing_root=False,
                  ee_poses=None, cameras=CAMERAS, require_jpeg=True,
                  min_frame_side=MIN_FRAME_SIDE):
    """Write one audited episode atomically and return its verification report.

    new_root            output directory; it must not exist unless allow_existing_root=True
    run_id, episode_id  produce the file name <run_id>_ep<episode_id:07d>.hdf5
    state, action       [T, 14] arrays in the order left_arm, left_ee, right_arm, right_ee
    rgb_frames          {"cam_head": [jpeg bytes]*T, "cam_left_wrist": ..., "cam_right_wrist": ...}
    existing_dirs       reference dirs whose file names may not be reused
    existing_index      build_existing_index() of the reference set; blocks duplicate content
    """
    root = os.fspath(new_root)
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id or ""):
        _reject("bad_run_id", "run_id must match %s, got %r" % (_RUN_ID_RE.pattern, run_id))
    if isinstance(episode_id, bool) or not isinstance(episode_id, (int, np.integer)):
        _reject("bad_episode_id", "episode_id must be a non-negative integer, got %r"
                % (episode_id,))
    if int(episode_id) < 0:
        _reject("bad_episode_id", "episode_id must be >= 0, got %r" % (episode_id,))
    if not isinstance(task, str) or not task.strip():
        _reject("bad_task", "task must be a non-empty string, got %r" % (task,))
    if not isinstance(instruction, str) or len(instruction.strip()) < 8:
        _reject("bad_instruction", "instruction must be a non-empty sentence, got %r"
                % (instruction,))
    if task.strip().lower() not in instruction.lower():
        _reject("bad_instruction",
                "instruction %r does not name the task %r; the freeze guard refuses it"
                % (instruction[:60], task))
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        _reject("bad_seed", "seed must be an integer, got %r" % (seed,))
    if (not isinstance(expert_version_sha, str) or len(expert_version_sha) < 8
            or any(c not in "0123456789abcdef" for c in expert_version_sha.lower())):
        _reject("bad_expert_version",
                "expert_version_sha must be >= 8 hex characters, got %r"
                % (expert_version_sha,))
    try:
        meta_json = json.dumps(dict(meta or {}), sort_keys=True)
    except (TypeError, ValueError) as exc:
        _reject("bad_meta", "meta must be JSON serialisable: %r" % (exc,))

    state = _as_matrix(state, "state")
    action = _as_matrix(action, "action")
    if state.shape != action.shape:
        _reject("bad_shape", "state %r and action %r are not the same shape"
                % (tuple(state.shape), tuple(action.shape)))
    frames = int(state.shape[0])
    if frames < int(min_frames):
        _reject("too_few_frames", "%d frames < the %d frame floor" % (frames, min_frames))
    if not (np.isfinite(state).all() and np.isfinite(action).all()):
        _reject("nonfinite", "state/action contain non-finite values")
    if float(np.abs(action).max()) == 0.0:
        _reject("degenerate_zero_action",
                "the action trace is all zeros: a zero trajectory is not an episode")
    if bool(np.all(action.max(axis=0) - action.min(axis=0) < 1e-9)):
        _reject("degenerate_constant_action",
                "every action row is identical: a constant trajectory is not an episode")
    misalign = float(np.abs(action[:-1] - state[1:]).max()) if frames > 1 else 0.0
    if misalign > align_tol:
        _reject("misaligned_state_action",
                "max|A[t]-S[t+1]| = %.6g exceeds %.6g" % (misalign, align_tol))
    frames_by_cam, payload = _validate_frames(rgb_frames, frames, tuple(cameras),
                                              min_frame_side, require_jpeg)
    ee_valid = None
    if ee_poses:
        if not isinstance(ee_poses, dict):
            _reject("bad_ee_poses", "ee_poses must be a dict of side -> [T, 7]")
        ee_valid = {}
        for side in ("left", "right"):
            if side not in ee_poses:
                _reject("bad_ee_poses", "ee_poses is missing the %r arm" % (side,))
            arr = np.asarray(ee_poses[side], dtype=np.float64)
            if arr.shape != (frames, 7):
                _reject("bad_ee_poses", "ee_poses[%r] must be [%d, 7], got %r"
                        % (side, frames, tuple(arr.shape)))
            if not np.isfinite(arr).all():
                _reject("bad_ee_poses", "ee_poses[%r] contains non-finite values" % (side,))
            ee_valid[side] = arr

    hashes = episode_hashes(state, action, instruction)
    final_name = "%s_ep%07d.hdf5" % (run_id, int(episode_id))
    final_path = os.path.join(root, final_name)
    tmp_path = final_path + ".tmp"

    estimate = frames * BASE_DIM * 8 * 2 + payload + 65536
    if estimate > int(max_bytes):
        _reject("size_cap", "estimated %d bytes > the %d byte cap" % (estimate, max_bytes))

    if os.path.exists(root) and not os.path.isdir(root):
        _reject("root_exists", "%r exists and is not a directory" % (root,))
    if os.path.isdir(root) and not allow_existing_root:
        _reject("root_exists",
                "output directory %r already exists; refusing to write into it "
                "(pass allow_existing_root=True only when the collector owns it)" % (root,))
    if os.path.exists(final_path):
        _reject("target_exists", "%r already exists; this writer never overwrites"
                % (final_name,))
    if os.path.exists(tmp_path):
        _reject("target_exists", "%r is a leftover temporary file; refusing to clobber it"
                % (os.path.basename(tmp_path),))
    for ref in existing_dirs or ():
        ref = os.fspath(ref)
        if os.path.isdir(ref) and final_name in os.listdir(ref):
            _reject("name_collision",
                    "%r already exists in the reference set %r" % (final_name, ref))
    if existing_index:
        for name, rec in dict(existing_index).items():
            if not isinstance(rec, dict):
                continue
            if rec.get("content_sha256") == hashes["content_sha256"]:
                _reject("hash_collision",
                        "content_sha256 collides with reference episode %r" % (name,))
            if rec.get("state_action_sha256") == hashes["state_action_sha256"]:
                _reject("hash_collision",
                        "state_action_sha256 collides with reference episode %r" % (name,))

    created_root = False
    stage = "none"
    created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        if not os.path.isdir(root):
            os.makedirs(root, exist_ok=False)
            created_root = True
        stage = "tmp"
        _write_hdf5(tmp_path, task, instruction, seed, expert_version_sha, run_id,
                    int(episode_id), state, action, frames_by_cam, int(frequency_hz),
                    meta_json, created_utc, hashes["state_action_sha256"], ee_valid)
        _fsync_path(tmp_path)
        stage = "claim"
        handle = os.open(final_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(handle)
        os.rename(tmp_path, final_path)
        stage = "renamed"
        _fsync_dir(root)
        size = int(os.path.getsize(final_path))
        if size > int(max_bytes):
            _reject("size_cap", "written episode is %d bytes > the %d byte cap"
                    % (size, max_bytes))
        report = verify_episode(final_path, min_frames=min_frames, min_bytes=min_bytes,
                                max_bytes=max_bytes, frequency_hz=frequency_hz,
                                align_tol=align_tol, cameras=tuple(cameras),
                                require_jpeg=require_jpeg, min_frame_side=min_frame_side)
        stage = "verified"
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        if stage in ("claim", "renamed") and os.path.exists(final_path):
            try:
                os.unlink(final_path)
            except OSError:
                pass
        if created_root:
            try:
                if not os.listdir(root):
                    os.rmdir(root)
            except OSError:
                pass
        raise

    report.update({"run_id": run_id, "episode_id": int(episode_id), "task": task,
                   "seed": int(seed), "expert_version_sha": expert_version_sha,
                   "created_utc": created_utc, "meta": dict(meta or {}),
                   "content_sha256": hashes["content_sha256"],
                   "state_action_sha256": hashes["state_action_sha256"],
                   "estimated_bytes": int(estimate),
                   "camera_streams": list(cameras),
                   "tmp_left_behind": False})
    return report
