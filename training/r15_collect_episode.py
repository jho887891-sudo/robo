#!/usr/bin/env python3
"""R15 reusable episode-collection driver (pure CPU; no isaacsim, no torch, no CUDA).

Mission: RoboDojo 12-task programme, spec section V ("M2 data"), gate P5_EXPERT_VALID.

The driver owns the only legal definition of a collected episode:

    reset -> loop { obs = env.get_obs(); act = expert.act(obs, privileged);
                    env.take_action(act); stop on the OFFICIAL verdict }
         -> write the episode through training/r11_collector_writer.py
         -> append the privileged-state provenance (prov/ group) and the audit block
         -> read the file back and account for it in the run ledger

Success is NEVER decided here.  It is read from exactly one place:

    RoboDojo/src/eval_client/eval_env.py:882-901  is_episode_end()
      -> RoboDojo/env/reward_manager/reward_manager.py:527-587  get_reward()
      -> env.success[<idx>] = True is written only at eval_env.py:897

Any expert-side claim of success (an attribute, a method, a return value) is ignored by
construction: the driver never asks the expert whether the task is done.

Fail-closed refusal codes (the episode is logged and NEVER written as a success):

    expert_no_action            expert.act() returned None
    expert_bad_action           wrong shape / non-finite / non-numeric action
    expert_error                expert.act() raised
    nonfinite_observation       obs state (or next state) is NaN/Inf
    misaligned_state_action     recorded |A[t] - S[t+1]| > align_tol
    obs_unavailable             get_obs() failed before the official end
    official_verdict_unavailable  neither is_episode_end() nor env.end_flag is readable
    official_success_unavailable  the official end fired but env.success is unreadable
    missing_provenance          a privileged value has no declared source getter
    undeclared_privileged       the expert consumed an undeclared privileged value
    missing_privileged          a declared privileged value was not produced
    missing_camera / bad_frame_bytes   camera stream absent or not a JPEG stream
    step_budget_exhausted       the driver budget ran out before the official end
    too_few_frames              fewer than min_frames control steps
    cap_accepted_per_task / cap_attempts_per_task / cap_attempts_total / cap_new_bytes
    writer_<code>               the writer refused the episode (its stable code)
    env_error / bad_observation / bad_config / bad_ledger / manifest_mismatch

The env is INJECTED and duck-typed, so the whole driver is exercised on CPU against a fake
environment double (tests/test_r15_collect_episode.py).  For the real simulator use
EvalEnvAdapter, which pins the exact official call sequence.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import time
from datetime import datetime, timezone

import numpy as np

__all__ = [
    "collect",
    "Collector",
    "CollectConfig",
    "CollectRefused",
    "config_from_task",
    "state_from_obs",
    "frames_from_obs",
    "action_dict_from_14d",
    "camera_inclusive_sha256",
    "summarize",
    "check_caps",
    "Ledger",
    "EpisodeIndex",
    "EvalEnvAdapter",
    "OFFICIAL_VERDICT",
    "CAPS",
    "BASE_DIM",
    "JOINT_GROUPS",
    "CAMERAS",
    "FREQUENCY_HZ",
    "ALIGN_TOL",
    "MIN_FRAMES",
    "COLLECT_VERSION",
]

COLLECT_VERSION = "r15-collect-1"
SCHEMA = "r15-collect-ledger-1"
MANIFEST_SCHEMA = "r15-collect-manifest-1"
DRIVER_REL = "training/r15_collect_episode.py"
WRITER_REL = "training/r11_collector_writer.py"
REGISTRY_REL = "training/r15_expert_registry.py"

BASE_DIM = 14
JOINT_GROUPS = (("left_arm", 6), ("left_ee", 1), ("right_arm", 6), ("right_ee", 1))
EE_INDEX = {"left": 6, "right": 13}
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
FREQUENCY_HZ = 25
GIB = 1024 ** 3
ALIGN_TOL = 1e-6
MIN_FRAMES = 100

EPISODES_DIR = "episodes"
REFUSED_DIR = "refused"
MANIFEST_NAME = "run_manifest.json"
LEDGER_NAME = "ledger.jsonl"
REPORT_NAME = "check_report.json"

CAPS = {
    "accepted_per_task": 100,
    "attempts_per_task": 250,
    "attempts_total": 3000,
    "new_bytes": 250 * GIB,
}

OFFICIAL_VERDICT = {
    "method": "RoboDojo/src/eval_client/eval_env.py:882-901 is_episode_end()",
    "reward": "RoboDojo/env/reward_manager/reward_manager.py:527-587 get_reward()",
    "flag": "env.success[i] is written only at RoboDojo/src/eval_client/eval_env.py:895-897",
    "rule": "reward > 1 - 1e-3 sets env.end_flag[i] and env.success[i] = True",
}

_STATE_KEYS = {
    "left_arm": ("left_arm_joint_state", "left_arm_joint_states", "left_joint_state"),
    "left_ee": ("left_ee_joint_state", "left_ee_joint_states", "left_gripper_joint_state"),
    "right_arm": ("right_arm_joint_state", "right_arm_joint_states", "right_joint_state"),
    "right_ee": ("right_ee_joint_state", "right_ee_joint_states", "right_gripper_joint_state"),
}
_CAMERA_PAYLOAD_KEYS = ("jpeg", "jpeg_bytes", "colors", "color")

_SIBLINGS = {}


class CollectRefused(RuntimeError):
    """An episode (or a run) was refused.  The code is stable and machine-checkable."""

    def __init__(self, code, message):
        super().__init__("[%s] %s" % (code, message))
        self.code = str(code)
        self.message = str(message)


def _refuse(code, message):
    raise CollectRefused(code, message)


# --------------------------------------------------------------------------- misc helpers

def _now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_sibling(name):
    """Load a sibling module of training/ by path (the repo has no packages/__init__)."""
    cached = _SIBLINGS.get(name)
    if cached is not None:
        return cached
    here = pathlib.Path(__file__).resolve().parent
    path = here / (name + ".py")
    if not path.is_file():
        _refuse("missing_module", "required module %r is missing at %s" % (name, path))
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _SIBLINGS[name] = mod
    return mod


def writer_module():
    return _load_sibling("r11_collector_writer")


def registry_module():
    return _load_sibling("r15_expert_registry")


def _vec14(value, name, dim_code="bad_state_dim", nan_code="nonfinite_observation"):
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        _refuse(dim_code, "%s is not numeric (%r): %r" % (name, value, exc))
    if arr.size != BASE_DIM:
        _refuse(dim_code, "%s must be %dD, got %d values" % (name, BASE_DIM, arr.size))
    if not np.isfinite(arr).all():
        _refuse(nan_code, "%s contains non-finite values: %r" % (name, arr.tolist()))
    return arr


# --------------------------------------------------------------------------- observation

def state_from_obs(obs):
    """Concatenate the four joint groups of an EvalEnv observation into the 14D state.

    Order: left_arm(6), left_ee(1), right_arm(6), right_ee(1) -- the dataset order.
    Sources (RoboDojo/env/observation_manager/obs_manager.py:149-191): obs["state"] holds
    "<arm>_joint_state" (6) and "<side>_ee_joint_state" (1) exactly in that layout.
    """
    if not isinstance(obs, dict):
        _refuse("bad_observation", "get_obs() must return a dict, got %r" % (type(obs).__name__,))
    state = obs.get("state")
    if not isinstance(state, dict):
        _refuse("bad_observation", "obs['state'] is missing or is not a dict (got %r)"
                % (type(state).__name__,))
    parts = []
    for group, dim in JOINT_GROUPS:
        found = None
        for key in _STATE_KEYS[group]:
            if key in state:
                found = state[key]
                break
        if found is None:
            _refuse("missing_state_key",
                    "obs['state'] has no key for %r (looked for %s); present keys: %s"
                    % (group, ", ".join(_STATE_KEYS[group]), sorted(state)))
        try:
            arr = np.asarray(found, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            _refuse("bad_state_dim", "obs['state'][%r] is not numeric: %r" % (group, exc))
        if arr.size != dim:
            _refuse("bad_state_dim", "obs['state'][%r] must have %d values, got %d"
                    % (group, dim, arr.size))
        parts.append(arr)
    return _vec14(np.concatenate(parts), "observation state")


def default_jpeg_encoder(array, quality=90):
    """Encode an HxWx3 uint8 frame to JPEG bytes (Pillow; the only encoder we trust)."""
    try:
        from PIL import Image
    except ImportError as exc:                              # pragma: no cover - env dependent
        _refuse("no_jpeg_encoder",
                "Pillow is required to JPEG-encode camera frames, and it is missing: %r"
                % (exc,))
    import io

    arr = np.asarray(array)
    if arr.ndim != 3 or arr.shape[2] < 3:
        _refuse("bad_frame_bytes", "camera frame must be HxWx3, got shape %r"
                % (tuple(arr.shape),))
    if arr.dtype != np.uint8:
        flat = arr.astype(np.float64)
        if not np.isfinite(flat).all():
            _refuse("bad_frame_bytes", "camera frame contains non-finite pixels")
        if flat.min() >= 0.0 and flat.max() <= 1.0:
            flat = flat * 255.0
        elif flat.min() < 0.0 or flat.max() > 255.0:
            _refuse("bad_frame_bytes", "camera frame range [%.3f, %.3f] is neither 0..1 nor "
                    "0..255" % (flat.min(), flat.max()))
        arr = np.clip(np.rint(flat), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr[..., :3]).save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def frame_bytes(cam_data, encoder=None, quality=90):
    """Return raw JPEG bytes for one camera entry of an observation."""
    encoder = encoder or default_jpeg_encoder
    if isinstance(cam_data, (bytes, bytearray, memoryview)):
        return bytes(cam_data)
    if isinstance(cam_data, dict):
        for key in _CAMERA_PAYLOAD_KEYS:
            if key in cam_data:
                value = cam_data[key]
                if isinstance(value, (bytes, bytearray, memoryview)):
                    return bytes(value)
                return encoder(np.asarray(value), quality)
        _refuse("missing_camera_payload",
                "camera entry has none of %s (present keys: %s)"
                % (", ".join(_CAMERA_PAYLOAD_KEYS), sorted(cam_data)))
    if isinstance(cam_data, np.ndarray) or hasattr(cam_data, "__array__"):
        return encoder(cam_data, quality)
    _refuse("missing_camera_payload", "camera entry is %r, not bytes or an image"
            % (type(cam_data).__name__,))


def frames_from_obs(obs, cameras=CAMERAS, encoder=None, quality=90):
    """Extract one JPEG byte string per camera from an observation."""
    if not isinstance(obs, dict):
        _refuse("bad_observation", "get_obs() must return a dict")
    vision = obs.get("vision")
    if not isinstance(vision, dict):
        _refuse("missing_camera", "obs['vision'] is missing (got %r)"
                % (type(vision).__name__,))
    out = {}
    for cam in cameras:
        if cam not in vision:
            _refuse("missing_camera", "camera %r is missing; obs['vision'] has %s"
                    % (cam, sorted(vision)))
        buf = frame_bytes(vision[cam], encoder, quality)
        if len(buf) < 4 or not bytes(buf[:2]) == b"\xff\xd8":
            _refuse("bad_frame_bytes", "camera %r frame is not a JPEG stream (starts %r)"
                    % (cam, bytes(buf[:4])))
        out[cam] = buf
    return out


def action_dict_from_14d(value):
    """Split a 14D absolute joint target into the official action dict."""
    arr = _vec14(value, "expert action", "expert_bad_action", "expert_bad_action")
    out = {}
    col = 0
    for group, dim in JOINT_GROUPS:
        out[group + "_joint_state"] = np.ascontiguousarray(arr[col:col + dim],
                                                           dtype=np.float64)
        col += dim
    return out


# --------------------------------------------------------------------------- hashing

def camera_inclusive_sha256(state, action, instruction, rgb_frames, cameras=CAMERAS):
    """SHA-256 over state + action + instruction + every camera byte.

    The existing writer's content_sha256 deliberately excludes the camera streams; this is
    the camera-inclusive digest the mission requires, so a run cannot swap pixels without
    being caught.  Definition: sha256 over, in order,

        b"r15-camera-inclusive-1"
        int64(shape(state)) . state.tobytes()          (little endian float64)
        int64(shape(action)) . action.tobytes()
        utf-8(instruction)
        for cam in cameras: utf-8(cam), int64(n_frames), then per frame int64(len) . bytes
    """
    state = np.ascontiguousarray(np.asarray(state, dtype=np.float64), dtype="<f8")
    action = np.ascontiguousarray(np.asarray(action, dtype=np.float64), dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"r15-camera-inclusive-1")
    digest.update(np.asarray(state.shape, dtype="<i8").tobytes())
    digest.update(state.tobytes())
    digest.update(np.asarray(action.shape, dtype="<i8").tobytes())
    digest.update(action.tobytes())
    digest.update(str(instruction).encode("utf-8"))
    for cam in cameras:
        stream = rgb_frames[cam]
        digest.update(str(cam).encode("utf-8"))
        digest.update(np.int64(len(stream)).tobytes())
        for buf in stream:
            raw = bytes(buf)
            digest.update(np.int64(len(raw)).tobytes())
            digest.update(raw)
    return digest.hexdigest()


# --------------------------------------------------------------------------- ledger + caps

class Ledger(object):
    """Append-only JSONL ledger.  One line per attempt (accepted or refused)."""

    def __init__(self, root):
        self.root = os.fspath(root)
        self.path = os.path.join(self.root, LEDGER_NAME)
        self.records = []

    @classmethod
    def open(cls, root):
        ledger = cls(root)
        ledger.load()
        return ledger

    def load(self):
        self.records = []
        if not os.path.isfile(self.path):
            return self.records
        with open(self.path, "r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError as exc:
                    _refuse("bad_ledger", "%s line %d is not JSON: %r"
                            % (LEDGER_NAME, lineno, exc))
                if not isinstance(rec, dict):
                    _refuse("bad_ledger", "%s line %d is not an object" % (LEDGER_NAME, lineno))
                if rec.get("schema") != SCHEMA:
                    _refuse("bad_ledger", "%s line %d has schema %r, expected %r"
                            % (LEDGER_NAME, lineno, rec.get("schema"), SCHEMA))
                if rec.get("status") not in ("accepted", "refused"):
                    _refuse("bad_ledger", "%s line %d has status %r"
                            % (LEDGER_NAME, lineno, rec.get("status")))
                if not isinstance(rec.get("task"), str) or not rec["task"]:
                    _refuse("bad_ledger", "%s line %d has no task" % (LEDGER_NAME, lineno))
                self.records.append(rec)
        return self.records

    def append(self, record):
        os.makedirs(self.root, exist_ok=True)
        line = json.dumps(record, sort_keys=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:                                 # pragma: no cover
                pass
        self.records.append(record)
        return record


def summarize(records):
    """Per-task / total attempt, accepted, success and byte counts from ledger records."""
    per_task = {}
    total = {"attempts": 0, "accepted": 0, "successes": 0, "refused": 0, "bytes": 0}
    for rec in records:
        task = str(rec.get("task"))
        slot = per_task.setdefault(task, {"attempts": 0, "accepted": 0, "successes": 0,
                                          "refused": 0, "bytes": 0})
        slot["attempts"] += 1
        total["attempts"] += 1
        if rec.get("status") == "accepted":
            slot["accepted"] += 1
            total["accepted"] += 1
            nbytes = int(rec.get("bytes") or 0)
            slot["bytes"] += nbytes
            total["bytes"] += nbytes
            if rec.get("official_success"):
                slot["successes"] += 1
                total["successes"] += 1
        else:
            slot["refused"] += 1
            total["refused"] += 1
    return {"per_task": per_task, "total": total}


def check_caps(stats, task, caps=None, add_attempts=0, add_accepted=0, add_bytes=0):
    """Enforce the mission caps.  Raises CollectRefused with a stable cap_* code."""
    caps = dict(CAPS if caps is None else caps)
    slot = stats["per_task"].get(str(task), {})
    total = stats["total"]
    attempts = int(slot.get("attempts", 0)) + int(add_attempts)
    accepted = int(slot.get("accepted", 0)) + int(add_accepted)
    nbytes = int(total.get("bytes", 0)) + int(add_bytes)
    if attempts > int(caps["attempts_per_task"]):
        _refuse("cap_attempts_per_task",
                "task %r would reach %d attempts > the %d cap"
                % (task, attempts, caps["attempts_per_task"]))
    if int(total.get("attempts", 0)) + int(add_attempts) > int(caps["attempts_total"]):
        _refuse("cap_attempts_total",
                "the run would reach %d attempts > the %d cap"
                % (int(total.get("attempts", 0)) + int(add_attempts),
                   caps["attempts_total"]))
    if accepted > int(caps["accepted_per_task"]):
        _refuse("cap_accepted_per_task",
                "task %r would reach %d accepted episodes > the %d cap"
                % (task, accepted, caps["accepted_per_task"]))
    if nbytes > int(caps["new_bytes"]):
        _refuse("cap_new_bytes",
                "the run would reach %d new bytes > the %d byte (%.0f GiB) cap"
                % (nbytes, caps["new_bytes"], caps["new_bytes"] / float(GIB)))
    return True


class EpisodeIndex(object):
    """In-run dedupe index handed to the writer (its existing_index hook).

    The writer's dedupe is opt-in; the driver always passes this index, so a duplicate
    content or state/action digest inside the run is refused as writer_hash_collision.
    """

    def __init__(self, extra=None):
        self.entries = {}
        if extra:
            for name, rec in dict(extra).items():
                if isinstance(rec, dict):
                    self.entries[str(name)] = {
                        "content_sha256": rec.get("content_sha256"),
                        "state_action_sha256": rec.get("state_action_sha256")}

    def add(self, name, report):
        self.entries[str(name)] = {
            "content_sha256": report.get("content_sha256"),
            "state_action_sha256": report.get("state_action_sha256")}

    def add_from_record(self, record):
        name = os.path.basename(str(record.get("episode_path") or ""))
        if name:
            self.entries[name] = {
                "content_sha256": record.get("content_sha256"),
                "state_action_sha256": record.get("state_action_sha256")}

    def as_dict(self):
        return dict(self.entries)


# --------------------------------------------------------------------------- config

class CollectConfig(object):
    """Everything the driver needs that is not the injected env/expert."""

    def __init__(self, run_id, task, instruction, step_lim, writer_task=None,
                 expert_name="unknown", expert_version_sha="0" * 64,
                 expert_status="IMPLEMENTED", provenance=None, privileged_keys=(),
                 predicate_ref="", caps=None, cameras=CAMERAS, frequency_hz=FREQUENCY_HZ,
                 min_frames=MIN_FRAMES, align_tol=ALIGN_TOL, verdict_mode="auto",
                 reset_seed_mode="list", jpeg_quality=90, env_idx=0,
                 reference_dirs=(), reference_index=None, extra_manifest=None):
        if not isinstance(run_id, str) or not run_id:
            _refuse("bad_config", "run_id must be a non-empty string")
        if not isinstance(task, str) or not task:
            _refuse("bad_config", "task must be a non-empty task id")
        if not isinstance(instruction, str) or len(instruction.strip()) < 8:
            _refuse("bad_config", "instruction must be the official task sentence, got %r"
                    % (instruction,))
        if isinstance(step_lim, bool) or not isinstance(step_lim, (int, np.integer)):
            _refuse("bad_config", "step_lim must be an integer, got %r" % (step_lim,))
        if int(step_lim) < 1:
            _refuse("bad_config", "step_lim must be >= 1")
        if not isinstance(expert_version_sha, str) or len(expert_version_sha) < 8:
            _refuse("bad_config", "expert_version_sha must be >= 8 hex characters")
        if any(c not in "0123456789abcdefABCDEF" for c in expert_version_sha):
            _refuse("bad_config", "expert_version_sha must be hex, got %r" % (expert_version_sha,))
        if verdict_mode not in ("auto", "official_method", "official_flag"):
            _refuse("bad_config", "verdict_mode must be auto|official_method|official_flag")
        if reset_seed_mode not in ("list", "scalar", "none"):
            _refuse("bad_config", "reset_seed_mode must be list|scalar|none")
        self.run_id = run_id
        self.task = task
        self.instruction = instruction
        self.writer_task = writer_task or task
        self.step_lim = int(step_lim)
        self.expert_name = str(expert_name)
        self.expert_version_sha = str(expert_version_sha)
        self.expert_status = str(expert_status)
        self.provenance = dict(provenance or {})
        self.privileged_keys = tuple(privileged_keys)
        self.predicate_ref = str(predicate_ref)
        self.caps = dict(CAPS if caps is None else caps)
        self.cameras = tuple(cameras)
        self.frequency_hz = int(frequency_hz)
        self.min_frames = int(min_frames)
        self.align_tol = float(align_tol)
        self.verdict_mode = verdict_mode
        self.reset_seed_mode = reset_seed_mode
        self.jpeg_quality = int(jpeg_quality)
        self.env_idx = int(env_idx)
        self.reference_dirs = tuple(reference_dirs)
        self.reference_index = dict(reference_index or {})
        self.extra_manifest = dict(extra_manifest or {})
        if not (self.writer_task.strip().lower() in self.instruction.lower()):
            _refuse("bad_config",
                    "writer_task %r must appear inside the instruction %r: the writer "
                    "refuses an instruction that does not name the task"
                    % (self.writer_task, self.instruction))
        for name in self.privileged_keys:
            if name not in self.provenance:
                _refuse("missing_provenance",
                        "privileged value %r is declared as used by the expert but has no "
                        "provenance entry (need source/file/line)" % (name,))

    def manifest(self):
        return {
            "schema": MANIFEST_SCHEMA,
            "collect_version": COLLECT_VERSION,
            "created_utc": _now_utc(),
            "run_id": self.run_id,
            "task": self.task,
            "writer_task": self.writer_task,
            "instruction": self.instruction,
            "step_lim": self.step_lim,
            "frequency_hz": self.frequency_hz,
            "cameras": list(self.cameras),
            "state_action": {"base_dim": BASE_DIM,
                             "order": [g for g, _ in JOINT_GROUPS],
                             "units": "absolute joint targets / joint positions",
                             "alignment": "A[t] == S[t+1]",
                             "align_tol": self.align_tol},
            "expert": {"name": self.expert_name,
                       "version_sha": self.expert_version_sha,
                       "status": self.expert_status},
            "predicate_ref": self.predicate_ref,
            "official_verdict": dict(OFFICIAL_VERDICT),
            "privileged": dict(self.provenance),
            "privileged_keys": list(self.privileged_keys),
            "caps": dict(self.caps),
            "driver": {"path": DRIVER_REL, "version": COLLECT_VERSION},
            "writer": {"path": WRITER_REL},
            "extra": dict(self.extra_manifest),
        }

    def check_manifest(self, manifest):
        """The manifest is the run's contract: refuse a run that changes under it."""
        problems = []
        for key in ("run_id", "task", "writer_task", "instruction", "step_lim"):
            if manifest.get(key) != getattr(self, key):
                problems.append("%s: manifest %r != config %r"
                                % (key, manifest.get(key), getattr(self, key)))
        if dict(manifest.get("privileged") or {}) != dict(self.provenance):
            problems.append("privileged provenance differs from the run manifest")
        if list(manifest.get("privileged_keys") or []) != list(self.privileged_keys):
            problems.append("privileged key list differs from the run manifest")
        if problems:
            _refuse("manifest_mismatch",
                    "%s does not match this collector: %s"
                    % (MANIFEST_NAME, "; ".join(problems)))
        return True

    def to_json(self):
        return {"run_id": self.run_id, "task": self.task, "writer_task": self.writer_task,
                "instruction": self.instruction, "step_lim": self.step_lim,
                "expert_name": self.expert_name,
                "expert_version_sha": self.expert_version_sha,
                "expert_status": self.expert_status,
                "privileged_keys": list(self.privileged_keys),
                "align_tol": self.align_tol, "min_frames": self.min_frames,
                "verdict_mode": self.verdict_mode,
                "reset_seed_mode": self.reset_seed_mode}


def config_from_task(task_id, **overrides):
    """Build a CollectConfig from the expert registry entry for task_id."""
    registry = registry_module()
    spec = registry.get_task(task_id)
    if not registry.is_implemented(task_id):
        _refuse("expert_blocked",
                "task %r is %s: %s" % (task_id, spec.status, spec.status_reason()))
    cfg = CollectConfig(
        run_id=overrides.pop("run_id", "r15_%s" % task_id),
        task=task_id,
        instruction=spec.instruction,
        step_lim=spec.step_lim,
        writer_task=spec.writer_task,
        expert_name=spec.expert_name,
        expert_version_sha=spec.expert_version_sha(),
        expert_status=spec.status,
        provenance=dict(spec.provenance),
        privileged_keys=tuple(spec.privileged_keys),
        predicate_ref=spec.predicate_ref,
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            _refuse("bad_config", "unknown config override %r" % (key,))
        setattr(cfg, key, value)
    return cfg


# --------------------------------------------------------------------------- official verdict

def _flag_value(obj, names, idx=0):
    """Read a per-env boolean flag exposed as attr/method/list/array."""
    for name in names:
        value = getattr(obj, name, None)
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except Exception:                            # noqa: BLE001
                continue
        try:
            if isinstance(value, (list, tuple)) or (
                    isinstance(value, np.ndarray) and value.ndim >= 1):
                if len(value) <= idx:
                    continue
                return bool(value[idx])
        except TypeError:
            pass
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
    return None


class EvalEnvAdapter(object):
    """Pin the official EvalEnv call sequence (RoboDojo/src/eval_client/eval_env.py).

    reset(seed)   ->  env.reset(seed=[int(seed)])          (:242, seed is a list)
    get_obs()     ->  env.get_obs()                        (:300, returns obs dict for env 0)
    take_action() ->  env.take_action(action)              (:388; it advances the sim and runs
                      the official is_episode_end()/reward check at :506, so the verdict is
                      already written into env.end_flag / env.success when it returns)
    end_flag()    ->  env.end_flag[env_idx]                (:886-901, only is_episode_end writes it)
    success()     ->  env.success[env_idx]                 (:895-897)

    NOTE (unverified on the GPU host): take_action_batch() re-runs is_episode_end() itself,
    so this adapter is what lets the driver read the verdict WITHOUT re-evaluating the
    reward a second time per step (verdict_mode="official_flag").
    """

    def __init__(self, env, env_idx=0):
        self.env = env
        self.env_idx = int(env_idx)

    def reset(self, seed=None):
        return self.env.reset(seed=[int(seed)] if seed is not None else [0])

    def get_obs(self):
        return self.env.get_obs()

    def take_action(self, action):
        return self.env.take_action(action)

    def end_flag(self):
        return bool(np.asarray(self.env.end_flag).reshape(-1)[self.env_idx])

    def success(self):
        return bool(np.asarray(self.env.success).reshape(-1)[self.env_idx])

    def is_episode_end(self):
        return bool(self.env.is_episode_end())


# --------------------------------------------------------------------------- collector

class Collector(object):
    """Owns one run directory: manifest, ledger, dedupe index and the writer."""

    def __init__(self, out_dir, cfg, writer=None, encoder=None, clock=None,
                 privileged_fn=None, state_fn=None, action_fn=None):
        self.out_dir = os.fspath(out_dir)
        self.cfg = cfg
        self.writer = writer or writer_module()
        self.encoder = encoder
        self.clock = clock or time.time
        self.privileged_fn = privileged_fn
        self.state_fn = state_fn
        self.action_fn = action_fn
        self.episodes_dir = os.path.join(self.out_dir, EPISODES_DIR)
        self.refused_dir = os.path.join(self.out_dir, REFUSED_DIR)
        os.makedirs(self.episodes_dir, exist_ok=True)
        os.makedirs(self.refused_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.out_dir, MANIFEST_NAME)
        self.ledger = Ledger.open(self.out_dir)
        self.index = EpisodeIndex(cfg.reference_index)
        for rec in self.ledger.records:
            if rec.get("status") == "accepted":
                self.index.add_from_record(rec)
        self.manifest = self._open_manifest()

    # -- run plumbing ------------------------------------------------------
    def _open_manifest(self):
        if os.path.isfile(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as handle:
                try:
                    manifest = json.load(handle)
                except ValueError as exc:
                    _refuse("manifest_mismatch", "%s is not JSON: %r"
                            % (MANIFEST_NAME, exc))
            self.cfg.check_manifest(manifest)
            return manifest
        manifest = self.cfg.manifest()
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:                                 # pragma: no cover
                pass
        os.replace(tmp, self.manifest_path)
        return manifest

    def stats(self):
        return summarize(self.ledger.records)

    def _check_attempt_caps(self):
        stats = self.stats()
        slot = stats["per_task"].get(self.cfg.task, {})
        if int(slot.get("accepted", 0)) >= int(self.cfg.caps["accepted_per_task"]):
            _refuse("cap_accepted_per_task",
                    "task %r already has %d accepted episodes, the cap is %d"
                    % (self.cfg.task, int(slot.get("accepted", 0)),
                       int(self.cfg.caps["accepted_per_task"])))
        check_caps(stats, self.cfg.task, self.cfg.caps, add_attempts=1)

    def _check_write_caps(self, estimate_bytes):
        check_caps(self.stats(), self.cfg.task, self.cfg.caps, add_attempts=0,
                   add_accepted=1, add_bytes=int(estimate_bytes))

    def _next_episode_id(self):
        used = -1
        prefix = self.cfg.run_id + "_ep"
        for name in os.listdir(self.episodes_dir):
            if name.startswith(prefix) and name.endswith(".hdf5"):
                try:
                    used = max(used, int(name[len(prefix):-len(".hdf5")]))
                except ValueError:
                    continue
        for rec in self.ledger.records:
            if rec.get("status") == "accepted" and rec.get("episode_id") is not None:
                used = max(used, int(rec["episode_id"]))
        return used + 1

    # -- env contract ------------------------------------------------------
    def _reset(self, env, seed):
        reset = getattr(env, "reset", None)
        if not callable(reset):
            _refuse("env_error", "env.reset() is missing")
        mode = self.cfg.reset_seed_mode
        try:
            if mode == "list":
                reset(seed=[int(seed)])
            elif mode == "scalar":
                reset(seed=int(seed))
            else:
                reset()
        except CollectRefused:
            raise
        except Exception as exc:                            # noqa: BLE001
            _refuse("env_error", "env.reset(seed=%r, mode=%s) failed: %r"
                    % (seed, mode, exc))

    def _obs(self, env):
        getter = getattr(env, "get_obs", None)
        if not callable(getter):
            _refuse("env_error", "env.get_obs() is missing")
        try:
            obs = getter()
        except Exception as exc:                            # noqa: BLE001
            _refuse("env_error", "env.get_obs() failed: %r" % (exc,))
        if not isinstance(obs, dict):
            _refuse("bad_observation", "get_obs() returned %r, not a dict"
                    % (type(obs).__name__,))
        return obs

    def _obs_or_none(self, env):
        try:
            obs = env.get_obs()
        except Exception:                                   # noqa: BLE001
            return None
        return obs if isinstance(obs, dict) else None

    def _take_action(self, env, action):
        take = getattr(env, "take_action", None)
        if not callable(take):
            _refuse("env_error", "env.take_action() is missing")
        try:
            take(action)
        except CollectRefused:
            raise
        except Exception as exc:                            # noqa: BLE001
            _refuse("env_error", "env.take_action() failed: %r" % (exc,))

    def _official_verdict(self, env):
        mode = self.cfg.verdict_mode
        if mode == "auto":
            readable = (_flag_value(env, ("end_flag", "end_flags"), self.cfg.env_idx)
                        is not None)
            mode = "official_flag" if readable else "official_method"
        if mode == "official_method":
            checker = getattr(env, "is_episode_end", None)
            if not callable(checker):
                _refuse("official_verdict_unavailable",
                        "verdict_mode=official_method needs env.is_episode_end(); the "
                        "official success signal must come from "
                        "eval_env.py:882-901")
            try:
                ended = checker()
            except Exception as exc:                        # noqa: BLE001
                _refuse("official_verdict_unavailable",
                        "env.is_episode_end() raised: %r" % (exc,))
            if not isinstance(ended, (bool, np.bool_)):
                _refuse("official_verdict_unavailable",
                        "env.is_episode_end() returned %r, not a bool" % (ended,))
            ended = bool(ended)
            if not ended:
                return False, None, mode
            success = _flag_value(env, ("success", "success_flag"), self.cfg.env_idx)
            if success is None:
                _refuse("official_success_unavailable",
                        "the official end fired but env.success[%d] is unreadable; the "
                        "verdict must come from eval_env.py:895-897"
                        % (self.cfg.env_idx,))
            return True, bool(success), mode
        ended = _flag_value(env, ("end_flag", "end_flags"), self.cfg.env_idx)
        if ended is None:
            _refuse("official_verdict_unavailable",
                    "verdict_mode=official_flag needs env.end_flag[%d], which is written "
                    "only by eval_env.py:882-901" % (self.cfg.env_idx,))
        if not ended:
            return False, None, mode
        success = _flag_value(env, ("success", "success_flag"), self.cfg.env_idx)
        if success is None:
            _refuse("official_success_unavailable",
                    "env.end_flag[%d] is set but env.success[%d] is unreadable"
                    % (self.cfg.env_idx, self.cfg.env_idx))
        return True, bool(success), mode

    # -- provenance --------------------------------------------------------
    def _validate_provenance(self):
        decl = dict(self.cfg.provenance)
        for name, entry in decl.items():
            if not isinstance(entry, dict):
                _refuse("missing_provenance", "provenance[%r] must be a dict, got %r"
                        % (name, type(entry).__name__))
            for field in ("source", "file", "line"):
                if field not in entry:
                    _refuse("missing_provenance",
                            "provenance[%r] has no %r; every privileged value must name "
                            "the getter and its file:line" % (name, field))
            if not str(entry["source"]).strip() or not str(entry["file"]).strip():
                _refuse("missing_provenance",
                        "provenance[%r] has an empty source or file" % (name,))
            try:
                if int(entry["line"]) < 1:
                    raise ValueError
            except (TypeError, ValueError):
                _refuse("missing_provenance", "provenance[%r]['line'] is not a line number"
                        % (name,))
        return decl

    def _privileged(self, env, obs):
        if self.privileged_fn is None:
            if self.cfg.privileged_keys:
                _refuse("missing_privileged",
                        "the expert declares privileged keys %r but no privileged_fn was "
                        "supplied" % (list(self.cfg.privileged_keys),))
            return {}
        try:
            raw = self.privileged_fn(env, obs)
        except CollectRefused:
            raise
        except Exception as exc:                            # noqa: BLE001
            _refuse("missing_privileged", "privileged_fn failed: %r" % (exc,))
        if not isinstance(raw, dict):
            _refuse("missing_privileged", "privileged_fn returned %r, not a dict"
                    % (type(raw).__name__,))
        missing = [k for k in self.cfg.privileged_keys if k not in raw]
        if missing:
            _refuse("missing_privileged", "privileged_fn did not produce %s"
                    % (", ".join(missing),))
        undeclared = [k for k in raw if k not in self.cfg.provenance]
        if undeclared:
            _refuse("undeclared_privileged",
                    "the expert consumed privileged values %s that have no declared "
                    "source getter" % (", ".join(sorted(undeclared)),))
        out = {}
        for name, value in raw.items():
            if isinstance(value, str):
                out[name] = value
                continue
            try:
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                _refuse("missing_privileged",
                        "privileged value %r is neither numeric nor a string: %r"
                        % (name, type(value).__name__))
            if not np.isfinite(arr).all():
                _refuse("missing_privileged",
                        "privileged value %r is non-finite: %r" % (name, arr.tolist()))
            out[name] = arr
        return out

    # -- expert ------------------------------------------------------------
    def _expert_act(self, expert, obs, privileged):
        act = getattr(expert, "act", None)
        if not callable(act):
            _refuse("expert_error", "the expert has no act(obs, privileged) method")
        try:
            value = act(obs, privileged)
        except CollectRefused:
            raise
        except Exception as exc:                            # noqa: BLE001
            _refuse("expert_error", "expert.act() raised: %r: %s"
                    % (type(exc).__name__, exc))
        if value is None:
            _refuse("expert_no_action",
                    "expert.act() returned None: the expert cannot produce an action, so "
                    "this episode is refused rather than ended on the expert's word")
        return value

    # -- refusal -----------------------------------------------------------
    def _refuse_episode(self, exc, attempt_id, attempt_no, seed, task_started,
                        steps, states, official, alignment, termination, expert):
        code = str(getattr(exc, "code", "") or "internal_error")
        message = str(getattr(exc, "message", "") or exc)
        info = {}
        if expert is not None:
            getter = getattr(expert, "info", None)
            if callable(getter):
                try:
                    info = getter()
                except Exception:                            # noqa: BLE001
                    info = {}
                if not isinstance(info, dict):
                    info = {"info": repr(info)}
        record = {
            "schema": SCHEMA,
            "attempt_id": attempt_id,
            "attempt": int(attempt_no),
            "task": self.cfg.task,
            "writer_task": self.cfg.writer_task,
            "seed": int(seed),
            "status": "refused",
            "code": code,
            "message": message,
            "steps": int(steps),
            "termination": termination or "refused_before_official_end",
            "success": None,
            "official_success": None,
            "official": dict(official),
            "official_source": dict(OFFICIAL_VERDICT),
            "alignment": dict(alignment),
            "instruction": self.cfg.instruction,
            "step_lim": self.cfg.step_lim,
            "expert": {"name": self.cfg.expert_name,
                       "version_sha": self.cfg.expert_version_sha,
                       "status": self.cfg.expert_status,
                       "info": info},
            "driver": {"path": DRIVER_REL, "version": COLLECT_VERSION},
            "utc": _now_utc(),
            "started_utc": task_started,
        }
        detail = dict(record)
        detail["error_type"] = type(exc).__name__
        detail["error_repr"] = repr(exc)
        path = os.path.join(self.refused_dir, attempt_id + ".json")
        tmp = path + ".tmp"
        os.makedirs(self.refused_dir, exist_ok=True)
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(detail, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:                                 # pragma: no cover
                pass
        os.replace(tmp, path)
        record["refusal_report"] = os.path.relpath(path, self.out_dir).replace(os.sep, "/")
        self.ledger.append(record)
        return record

    # -- audit block -------------------------------------------------------
    def _append_audit(self, path, privs, decl, official, cam_sha, alignment, stamps,
                      termination, meta):
        import h5py

        with h5py.File(path, "r+") as handle:
            prov = handle.require_group("prov")
            for name in sorted(privs):
                values = privs[name]
                try:
                    data = np.asarray(values, dtype=np.float64)
                except (TypeError, ValueError):
                    if not all(isinstance(v, str) for v in values):
                        _refuse("bad_privileged_value",
                                "privileged value %r cannot be stored as a numeric array "
                                "or a string array" % (name,))
                    vlen = h5py.special_dtype(vlen=bytes)
                    data = np.empty((len(values),), dtype=object)
                    data[:] = [v.encode("utf-8") for v in values]
                    ds = prov.create_dataset(name, data=data, dtype=vlen)
                else:
                    ds = prov.create_dataset(name, data=data)
                entry = decl[name]
                ds.attrs["source"] = str(entry["source"])
                ds.attrs["file"] = str(entry["file"])
                ds.attrs["line"] = int(entry["line"])
                ds.attrs["note"] = str(entry.get("note", ""))
                ds.attrs["steps"] = int(len(values))
            prov.attrs["declared_json"] = json.dumps(decl, sort_keys=True)
            prov.attrs["official_verdict"] = json.dumps(OFFICIAL_VERDICT, sort_keys=True)

            audit = handle.require_group("audit")
            audit.attrs["official_success"] = int(bool(official["success"]))
            audit.attrs["official_mode"] = str(official["mode"])
            audit.attrs["official_json"] = json.dumps(official, sort_keys=True)
            audit.attrs["official_source"] = json.dumps(OFFICIAL_VERDICT, sort_keys=True)
            audit.attrs["termination"] = str(termination)
            audit.attrs["camera_inclusive_sha256"] = str(cam_sha)
            audit.attrs["max_abs_action_minus_next_state"] = float(alignment["max_abs"])
            audit.attrs["align_tol"] = float(alignment["tol"])
            audit.attrs["alignment_pairs"] = int(alignment["pairs"])
            audit.attrs["terminal_state_available"] = int(bool(alignment["terminal_state"]))
            audit.attrs["meta_json"] = json.dumps(meta, sort_keys=True)
            audit.create_dataset("step_timestamp_unix_us",
                                 data=np.asarray(stamps, dtype="<i8"))
            handle.flush()
            try:
                os.fsync(handle.id.get_vfd_handle())
            except Exception:                               # noqa: BLE001, pragma: no cover
                pass
        return True

    # -- write -------------------------------------------------------------
    def _write_episode(self, attempt_id, attempt_no, seed, states, actions, frames,
                       privs, stamps, official, alignment, termination, expert_info,
                       task_started, episode_id=None):
        cfg = self.cfg
        state = np.ascontiguousarray(np.asarray(states, dtype=np.float64))
        action = np.ascontiguousarray(np.asarray(actions, dtype=np.float64))
        n_frames = int(state.shape[0])
        if n_frames < cfg.min_frames:
            _refuse("too_few_frames",
                    "the official end arrived after %d steps, below the %d frame floor"
                    % (n_frames, cfg.min_frames))
        payload = sum(len(buf) for cam in cfg.cameras for buf in frames[cam])
        estimate = n_frames * BASE_DIM * 8 * 2 + payload + 65536
        self._check_write_caps(estimate)

        decl = dict(cfg.provenance)
        hashes = self.writer.episode_hashes(state, action, cfg.instruction)
        cam_sha = camera_inclusive_sha256(state, action, cfg.instruction, frames,
                                          cfg.cameras)
        ep_id = int(self._next_episode_id() if episode_id is None else episode_id)
        meta = {
            "collect_version": COLLECT_VERSION,
            "run_id": cfg.run_id,
            "task": cfg.task,
            "writer_task": cfg.writer_task,
            "seed": int(seed),
            "steps": n_frames,
            "step_lim": cfg.step_lim,
            "termination": termination,
            "official": {"success": bool(official["success"]),
                         "ended": bool(official["ended"]),
                         "mode": str(official["mode"]),
                         "source": dict(OFFICIAL_VERDICT),
                         "global_step": None},
            "alignment": {"max_abs": float(alignment["max_abs"]),
                          "tol": float(alignment["tol"]),
                          "pairs": int(alignment["pairs"]),
                          "terminal_state_available": bool(alignment["terminal_state"])},
            "privileged": decl,
            "privileged_keys": list(cfg.privileged_keys),
            "camera_inclusive_sha256": cam_sha,
            "expert": {"name": cfg.expert_name,
                       "version_sha": cfg.expert_version_sha,
                       "status": cfg.expert_status,
                       "info": expert_info},
            "predicate_ref": cfg.predicate_ref,
            "driver": {"path": DRIVER_REL, "version": COLLECT_VERSION},
            "control_hz": cfg.frequency_hz,
            "attempt_id": attempt_id,
        }
        report = self.writer.write_episode(
            new_root=self.episodes_dir,
            run_id=cfg.run_id,
            episode_id=ep_id,
            task=cfg.writer_task,
            instruction=cfg.instruction,
            state=state,
            action=action,
            rgb_frames={cam: list(frames[cam]) for cam in cfg.cameras},
            seed=int(seed),
            expert_version_sha=cfg.expert_version_sha,
            meta=meta,
            frequency_hz=cfg.frequency_hz,
            min_frames=cfg.min_frames,
            align_tol=cfg.align_tol,
            cameras=cfg.cameras,
            existing_dirs=cfg.reference_dirs,
            existing_index=self.index.as_dict(),
            allow_existing_root=True,
        )
        path = report["path"]
        self._append_audit(path, privs, decl, official, cam_sha, alignment, stamps,
                           termination, meta)
        post = self.writer.verify_episode(path, min_frames=cfg.min_frames,
                                          frequency_hz=cfg.frequency_hz,
                                          align_tol=cfg.align_tol, cameras=cfg.cameras)
        size = int(os.path.getsize(path))
        file_sha = _sha256_file(path)
        rel = os.path.relpath(path, self.out_dir).replace(os.sep, "/")
        record = {
            "schema": SCHEMA,
            "attempt_id": attempt_id,
            "attempt": int(attempt_no),
            "task": cfg.task,
            "writer_task": cfg.writer_task,
            "seed": int(seed),
            "status": "accepted",
            "code": None,
            "steps": n_frames,
            "termination": termination,
            # "success" is an alias of the official verdict and nothing else: it is copied
            # from env.success[i] which eval_env.py:895-897 writes from reward_manager
            # get_reward(); no expert-side claim can reach this field.
            "success": bool(official["success"]),
            "official_success": bool(official["success"]),
            "official": dict(official),
            "official_source": dict(OFFICIAL_VERDICT),
            "episode_id": ep_id,
            "episode_path": rel,
            "bytes": size,
            "frames": int(post["frames"]),
            "frequency_hz": int(cfg.frequency_hz),
            "content_sha256": hashes["content_sha256"],
            "state_action_sha256": hashes["state_action_sha256"],
            "camera_inclusive_sha256": cam_sha,
            "file_sha256": file_sha,
            "max_abs_action_minus_next_state": float(alignment["max_abs"]),
            "alignment": dict(alignment),
            "privileged": decl,
            "privileged_keys": list(cfg.privileged_keys),
            "instruction": cfg.instruction,
            "step_lim": cfg.step_lim,
            "expert": {"name": cfg.expert_name,
                       "version_sha": cfg.expert_version_sha,
                       "status": cfg.expert_status,
                       "info": expert_info},
            "driver": {"path": DRIVER_REL, "version": COLLECT_VERSION},
            "utc": _now_utc(),
            "started_utc": task_started,
        }
        self.index.add(os.path.basename(path), report)
        self.ledger.append(record)
        record["report"] = {k: post.get(k) for k in
                            ("file", "bytes", "frames", "frequency", "instruction",
                             "content_sha256", "state_action_sha256", "verified")}
        return record

    # -- the loop ----------------------------------------------------------
    def collect_episode(self, env, expert, seed, max_steps=None, episode_id=None):
        """Collect exactly one episode; return the ledger record (accepted or refused)."""
        cfg = self.cfg
        if cfg.expert_status != "IMPLEMENTED":
            return self._refuse_episode(
                CollectRefused("expert_blocked",
                               "task %r has no implemented expert (status %s)"
                               % (cfg.task, cfg.expert_status)),
                "%s_a%05d" % (cfg.run_id, len(self.ledger.records)),
                len(self.ledger.records), seed, _now_utc(), 0, [], {}, {}, None, expert)
        try:
            self._check_attempt_caps()
        except CollectRefused as exc:
            return self._refuse_episode(exc, "%s_a%05d" % (cfg.run_id,
                                                           len(self.ledger.records)),
                                        len(self.ledger.records), seed, _now_utc(), 0, [],
                                        {}, {}, None, expert)

        attempt_no = len(self.ledger.records)
        attempt_id = "%s_a%05d" % (cfg.run_id, attempt_no)
        state_fn = self.state_fn or state_from_obs
        action_fn = self.action_fn or action_dict_from_14d
        limit = int(cfg.step_lim if max_steps is None
                    else min(int(max_steps), int(cfg.step_lim)))
        started = _now_utc()
        states, actions = [], []
        frames = {cam: [] for cam in cfg.cameras}
        privs = {}
        stamps = []
        official = {"success": None, "ended": False, "mode": None,
                    "source": dict(OFFICIAL_VERDICT)}
        alignment = {"max_abs": 0.0, "tol": float(cfg.align_tol), "pairs": 0,
                     "terminal_state": False}
        termination = None
        expert_info = {}

        def expert_snapshot():
            getter = getattr(expert, "info", None)
            if callable(getter):
                try:
                    value = getter()
                except Exception:                            # noqa: BLE001
                    return {}
                if isinstance(value, dict):
                    return value
            return {}

        try:
            if limit < 1:
                _refuse("bad_step_budget", "max_steps resolves to %d; nothing to do" % limit)
            self._validate_provenance()
            self._reset(env, seed)
            obs = self._obs(env)
            state = state_fn(obs)
            for _step in range(limit):
                frames_t = frames_from_obs(obs, cfg.cameras, self.encoder,
                                           cfg.jpeg_quality)
                priv_t = self._privileged(env, obs)
                act = self._expert_act(expert, obs, priv_t)
                action = _vec14(act, "expert action", "expert_bad_action",
                                "expert_bad_action")
                self._take_action(env, action_fn(action))
                ended, success, mode = self._official_verdict(env)
                official.update({"ended": bool(ended), "success": success, "mode": mode})
                obs_next = self._obs_or_none(env)
                if obs_next is None:
                    if not ended:
                        _refuse("obs_unavailable",
                                "get_obs() failed after take_action() and the official end "
                                "has not fired")
                else:
                    state_next = state_fn(obs_next)
                    misalign = float(np.abs(action - state_next).max())
                    alignment["pairs"] += 1
                    alignment["max_abs"] = max(alignment["max_abs"], misalign)
                    if misalign > cfg.align_tol:
                        _refuse("misaligned_state_action",
                                "max|A[%d] - S[%d]| = %.6g exceeds %.6g: the applied "
                                "absolute joint target is not the next observed state"
                                % (len(actions), len(actions) + 1, misalign,
                                   cfg.align_tol))
                    if ended:
                        alignment["terminal_state"] = True
                states.append(state)
                actions.append(action)
                stamps.append(int(self.clock() * 1e6))
                for cam in cfg.cameras:
                    frames[cam].append(frames_t[cam])
                for name, value in priv_t.items():
                    privs.setdefault(name, []).append(value)
                if ended:
                    termination = ("official_success" if success
                                   else "official_failure")
                    break
                obs = obs_next
                state = state_next
            else:
                _refuse("step_budget_exhausted",
                        "%d control steps without the official end (step_lim %d)"
                        % (limit, cfg.step_lim))
            expert_info = expert_snapshot()
            return self._write_episode(attempt_id, attempt_no, seed, states, actions,
                                       frames, privs, stamps, official, alignment,
                                       termination, expert_info, started,
                                       episode_id=episode_id)
        except Exception as exc:                            # noqa: BLE001
            return self._refuse_episode(exc, attempt_id, attempt_no, seed, started,
                                        len(actions), states, official, alignment,
                                        termination, expert)

    def collect(self, env, expert, seed, max_steps=None, episode_id=None):
        return self.collect_episode(env, expert, seed, max_steps=max_steps,
                                    episode_id=episode_id)


def collect(task, env, expert, seed, out_dir, max_steps=None, *, cfg=None,
            episode_id=None, writer=None, encoder=None, clock=None,
            privileged_fn=None, state_fn=None, action_fn=None, **cfg_overrides):
    """Reusable one-episode collection entry point.

    task        mission task id (e.g. "push_T"); when cfg is None the official instruction,
                step_lim, privileged provenance and expert identity come from
                training/r15_expert_registry.py
    env         an injected duck-typed env: reset(seed=[s]), get_obs(), take_action(dict),
                and the official verdict (is_episode_end()/end_flag + success)
    expert      an injected duck-typed expert: act(obs, privileged) -> 14D target or None
    seed        episode seed
    out_dir     run directory (manifest, ledger, episodes/, refused/)
    max_steps   hard driver budget; clipped to the official step_lim (default: step_lim)

    Returns the ledger record: {status: accepted|refused, success, steps, termination,
    code, ...}.  Nothing is ever written to disk as a success unless the official verdict
    fired and every fail-closed check passed.
    """
    if cfg is None:
        cfg = config_from_task(task, **cfg_overrides)
    elif cfg_overrides:
        _refuse("bad_config", "cfg= was given together with config overrides %r"
                % (sorted(cfg_overrides),))
    collector = Collector(out_dir, cfg, writer=writer, encoder=encoder, clock=clock,
                          privileged_fn=privileged_fn, state_fn=state_fn,
                          action_fn=action_fn)
    return collector.collect_episode(env, expert, seed, max_steps=max_steps,
                                    episode_id=episode_id)
