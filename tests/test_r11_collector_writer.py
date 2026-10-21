#!/usr/bin/env python3
"""Lane D RED tests -- atomic HDF5 episode writer for the Push-T collector (pure CPU).

No simulator is started and no GPU is touched: the writer is exercised on synthetic but
structurally real episodes (genuine JPEG byte streams, 14D state/action with A[t] == S[t+1]).

The contract pinned here is the one the existing freeze pipeline depends on
(tools/r8_eval_manifest.py hash definitions, tools/r10_freeze_guard.py schema rules):
atomic .tmp -> fsync -> rename, never overwrite, no .tmp residue, fail-closed on fabricated
(zero / constant / short / camera-less) trajectories, and a read-back verification whose
hashes are byte-identical to the official manifest tool's.
"""
import importlib.util
import json
import os
import pathlib
import struct

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE = ROOT / "training" / "r11_collector_writer.py"
MANIFEST_TOOL = ROOT / "tools" / "r8_eval_manifest.py"

BASE_DIM = 14
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
INSTRUCTION = "Push the T-shaped block to align it precisely with the gray T-shaped pad."


def load():
    assert MODULE.is_file(), "missing writer module: %s" % MODULE
    spec = importlib.util.spec_from_file_location("r11_collector_writer", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_manifest_tool():
    import h5py  # noqa: F401  (the tool needs it; skip cleanly when absent)
    spec = importlib.util.spec_from_file_location("r8_eval_manifest", MANIFEST_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jpeg(w=96, h=96, filler=1024, tag=0):
    """Structurally valid JPEG stream: SOI, SOF0 with real dimensions, SOS, entropy, EOI."""
    sof = (b"\xff\xc0" + struct.pack(">H", 17) + bytes([8]) + struct.pack(">HH", h, w)
           + bytes([3]) + bytes([1, 0x22, 0]) + bytes([2, 0x11, 1]) + bytes([3, 0x11, 1]))
    sos = b"\xff\xda" + struct.pack(">H", 12) + bytes([3, 1, 0, 2, 0x11, 3, 0x11, 0, 0x3f, 0])
    payload = bytes([(tag + i) % 251 for i in range(filler)])
    return b"\xff\xd8" + sof + sos + payload + b"\xff\xd9"


def episode(frames=120, base=0.0, filler=1024, cameras=CAMERAS):
    T = frames
    direction = np.linspace(0.0, 1.0, BASE_DIM)
    state = np.stack([base + 0.001 * t * direction for t in range(T)]).astype(np.float64)
    action = np.empty_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1] + 0.0005 * direction
    rgb = {cam: [jpeg(filler=filler, tag=i + 7 * k) for i in range(T)]
           for k, cam in enumerate(cameras)}
    return state, action, rgb


def write(tmp_path, name="run", episode_id=0, root=None, **kw):
    W = load()
    state, action, rgb = episode()
    payload = dict(new_root=str(root or (tmp_path / name)), run_id="r11laneD",
                   episode_id=episode_id, task="Push the T-shaped block",
                   instruction=INSTRUCTION, state=state, action=action,
                   rgb_frames=rgb, seed=1234, expert_version_sha="a" * 40,
                   meta={"expert": "geometric_push_t", "phase_log": ["approach", "push"]})
    payload.update(kw.pop("payload", {}))
    return W, payload, kw


# ---------------------------------------------------------------- happy path

def test_writes_an_episode_and_passes_read_back_verification(tmp_path):
    W, payload, kw = write(tmp_path)
    rep = W.write_episode(**payload, **kw)
    path = rep["path"]
    assert os.path.isfile(path)
    assert path.endswith(".hdf5")
    assert os.path.basename(path) == "r11laneD_ep0000000.hdf5"
    assert not any(n.endswith(".tmp") for n in os.listdir(os.path.dirname(path)))
    assert rep["verified"] is True
    assert rep["frames"] == 120
    assert rep["frequency"] == 25
    assert len(rep["content_sha256"]) == 64 and len(rep["state_action_sha256"]) == 64

    import h5py
    with h5py.File(path, "r") as h:
        for group in ("state", "action"):
            parts = [h[group + "/left_arm_joint_states"][()], h[group + "/left_ee_joint_states"][()],
                     h[group + "/right_arm_joint_states"][()], h[group + "/right_ee_joint_states"][()]]
            assert np.concatenate(parts, axis=1).shape == (120, 14)
        assert int(h["additional_info/frequency"][()]) == 25
        assert h["instruction"][()].decode("utf-8") == INSTRUCTION
        for cam in CAMERAS:
            ds = h["vision/" + cam + "/colors"]
            assert ds.shape[0] == 120
            frame = bytes(ds[0])
            assert frame == payload["rgb_frames"][cam][0], "the JPEG payload changed on disk"
            assert frame[:2] == b"\xff\xd8" and frame[-2:] == b"\xff\xd9"
        assert h.attrs["seed"] == 1234
        assert h.attrs["expert_version_sha"] == "a" * 40
        assert json.loads(h.attrs["meta_json"])["expert"] == "geometric_push_t"
        assert h.attrs["state_action_sha256"] == rep["state_action_sha256"]


def test_reported_hashes_match_the_official_manifest_tool(tmp_path):
    W, payload, kw = write(tmp_path)
    rep = W.write_episode(**payload, **kw)
    M = load_manifest_tool()
    rec = M.episode_record(rep["path"])
    assert rec["content_sha256"] == rep["content_sha256"]
    assert rec["state_action_sha256"] == rep["state_action_sha256"]
    assert rec["length"] == 120
    assert rec["frequency"] == 25
    assert rec["instruction"] == INSTRUCTION


def test_standalone_hash_helper_is_reproducible(tmp_path):
    W = load()
    state, action, _ = episode(frames=8, filler=8)
    a = W.episode_hashes(state, action, INSTRUCTION)
    b = W.episode_hashes(state.copy(), action.copy(), INSTRUCTION)
    assert a == b
    assert a["content_sha256"] != a["state_action_sha256"]
    c = W.episode_hashes(state, action, INSTRUCTION + " ")
    assert c["content_sha256"] != a["content_sha256"]


def test_verify_episode_detects_a_truncated_file(tmp_path):
    W, payload, kw = write(tmp_path)
    rep = W.write_episode(**payload, **kw)
    with open(rep["path"], "r+b") as fh:
        fh.truncate(4096)
    with pytest.raises(Exception):
        W.verify_episode(rep["path"])


# ---------------------------------------------------------------- never overwrite

def test_existing_target_directory_is_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    os.makedirs(payload["new_root"])
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**payload, **kw)
    assert exc.value.code == "root_exists"
    assert os.listdir(payload["new_root"]) == []


def test_existing_root_is_allowed_only_when_asked_explicitly(tmp_path):
    W, payload, kw = write(tmp_path)
    os.makedirs(payload["new_root"])
    rep = W.write_episode(**payload, allow_existing_root=True, **kw)
    assert rep["verified"] is True


def test_same_name_is_never_overwritten(tmp_path):
    W, payload, kw = write(tmp_path)
    rep = W.write_episode(**payload, allow_existing_root=True, **kw)
    before = pathlib.Path(rep["path"]).read_bytes()
    state2, action2, rgb2 = episode(base=5.0)
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, state=state2, action=action2, rgb_frames=rgb2, **kw),
                        allow_existing_root=True)
    assert exc.value.code == "target_exists"
    assert pathlib.Path(rep["path"]).read_bytes() == before, "the first episode was modified"
    assert not any(n.endswith(".tmp") for n in os.listdir(os.path.dirname(rep["path"])))


def test_collision_with_a_reference_episode_name_is_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    used = tmp_path / "used"
    os.makedirs(used)
    open(os.path.join(str(used), "r11laneD_ep0000000.hdf5"), "wb").close()
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**payload, existing_dirs=(str(used),), **kw)
    assert exc.value.code == "name_collision"


def test_collision_with_a_reference_content_hash_is_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    hashes = W.episode_hashes(payload["state"], payload["action"], payload["instruction"])
    index = {"episode_0000000.hdf5": hashes}
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**payload, existing_index=index, **kw)
    assert exc.value.code == "hash_collision"


# ---------------------------------------------------------------- fabricated data

def test_zero_action_episode_is_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    action = np.zeros((120, 14))
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, action=action, **kw))
    assert exc.value.code == "degenerate_zero_action"
    assert not os.path.exists(payload["new_root"])


def test_constant_action_episode_is_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    const = np.tile(np.linspace(-0.5, 0.5, 14), (120, 1))
    state = np.empty((120, 14))
    state[:-1] = const[1:]
    state[-1] = const[-1]
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, state=state, action=const, **kw))
    assert exc.value.code == "degenerate_constant_action"


def test_too_few_frames_is_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    state, action, rgb = episode(frames=99)
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, state=state, action=action, rgb_frames=rgb, **kw))
    assert exc.value.code in ("too_few_frames", "size_floor")
    assert not os.path.exists(payload["new_root"])


def test_missing_camera_is_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    state, action, rgb = episode()
    rgb = {k: v for k, v in rgb.items() if k != "cam_right_wrist"}
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, state=state, action=action, rgb_frames=rgb, **kw))
    assert exc.value.code == "missing_camera"


def test_camera_frame_count_mismatch_is_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    state, action, rgb = episode()
    rgb["cam_head"] = rgb["cam_head"][:-1]
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, state=state, action=action, rgb_frames=rgb, **kw))
    assert exc.value.code == "bad_frame_count"


def test_non_jpeg_frame_bytes_are_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    state, action, rgb = episode()
    rgb["cam_head"] = [b"\x00" * 4096] * 120
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, state=state, action=action, rgb_frames=rgb, **kw))
    assert exc.value.code == "bad_frame_bytes"


def test_tiny_camera_frames_are_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    state, action, rgb = episode()
    rgb["cam_head"] = [jpeg(w=32, h=32) for _ in range(120)]
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, state=state, action=action, rgb_frames=rgb, **kw))
    assert exc.value.code == "frame_too_small"


def test_nonfinite_state_is_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    state = np.array(payload["state"], float)
    state[3, 4] = np.nan
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, state=state, **kw))
    assert exc.value.code == "nonfinite"


def test_misaligned_action_target_is_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    action = np.array(payload["action"], float)
    action[10, 0] += 0.5
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, action=action, **kw))
    assert exc.value.code == "misaligned_state_action"


def test_wrong_dimension_is_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, state=np.zeros((120, 13)), **kw))
    assert exc.value.code in ("bad_shape", "misaligned_state_action")


def test_empty_instruction_is_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, instruction="   ", **kw))
    assert exc.value.code == "bad_instruction"


def test_instruction_without_the_task_is_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, instruction="Move the red cube.", **kw))
    assert exc.value.code == "bad_instruction"


def test_bad_expert_version_sha_is_refused(tmp_path):
    W, payload, kw = write(tmp_path)
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, expert_version_sha="", **kw))
    assert exc.value.code == "bad_expert_version"


def test_size_cap_is_enforced_before_writing(tmp_path):
    W, payload, kw = write(tmp_path)
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**payload, max_bytes=10_000, **kw)
    assert exc.value.code == "size_cap"
    assert not os.path.exists(payload["new_root"])


def test_size_floor_rejects_placeholder_episodes(tmp_path):
    W, payload, kw = write(tmp_path)
    state, action, rgb = episode(frames=110, filler=8)
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, state=state, action=action, rgb_frames=rgb, **kw))
    assert exc.value.code == "size_floor"
    assert not os.path.exists(payload["new_root"]), "a rejected episode is not left on disk"


def test_no_tmp_residue_after_an_injected_write_failure(tmp_path, monkeypatch):
    W, payload, kw = write(tmp_path)

    real_rename = os.rename

    def boom(*a, **k):
        raise OSError("injected rename failure")

    monkeypatch.setattr(W.os, "rename", boom, raising=False)
    with pytest.raises(Exception):
        W.write_episode(**payload, **kw)
    monkeypatch.setattr(W.os, "rename", real_rename, raising=False)
    root = pathlib.Path(payload["new_root"])
    leftovers = list(root.rglob("*.tmp")) if root.exists() else []
    assert leftovers == [], "a .tmp file was left behind: %r" % leftovers


def test_build_existing_index_reads_reference_episodes(tmp_path):
    W, payload, kw = write(tmp_path)
    rep = W.write_episode(**payload, **kw)
    ref = tmp_path / "reference"
    os.makedirs(ref)
    os.link(rep["path"], os.path.join(str(ref), "episode_0000000.hdf5"))
    index = W.build_existing_index(str(ref))
    assert list(index) == ["episode_0000000.hdf5"]
    assert index["episode_0000000.hdf5"]["content_sha256"] == rep["content_sha256"]
    assert index["episode_0000000.hdf5"]["state_action_sha256"] == rep["state_action_sha256"]
    # and that index really blocks a duplicate
    with pytest.raises(W.EpisodeRejected) as exc:
        W.write_episode(**dict(payload, episode_id=1), existing_index=index,
                        allow_existing_root=True, **kw)
    assert exc.value.code == "hash_collision"


def test_real_encoder_jpeg_is_accepted(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    import io
    W, payload, kw = write(tmp_path)
    state, action, rgb = episode()
    buf = io.BytesIO()
    PIL.new("RGB", (96, 96), (30, 60, 90)).save(buf, format="JPEG")
    raw = buf.getvalue()
    rgb["cam_head"] = [raw] * 120
    rep = W.write_episode(**dict(payload, state=state, action=action, rgb_frames=rgb, **kw))
    assert rep["verified"] is True
    h, w = W.jpeg_size(raw)
    assert (h, w) == (96, 96)


def test_accepts_plain_pose_free_meta_and_optional_ee_poses(tmp_path):
    W, payload, kw = write(tmp_path)
    ee = {"left": np.tile(np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]), (120, 1)),
          "right": np.tile(np.array([0.0, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0]), (120, 1))}
    rep = W.write_episode(**payload, ee_poses=ee, **kw)
    import h5py
    with h5py.File(rep["path"], "r") as h:
        assert h["state/left_ee_poses"].shape == (120, 7)
        assert h["action/left_ee_poses"].shape == (120, 7)
