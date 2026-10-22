#!/usr/bin/env python3
"""R15 collection quality gate (post-hoc, pure CPU, no simulator).

    python tools/r15_collect_check.py <run_dir> [--report PATH] [--json]

Exit codes:  0 = PASS    3 = FAIL    2 = the check itself could not run

What a PASS means, for every episode the run claims to have accepted:

  1. the file exists, its byte size and whole-file sha256 match the ledger;
  2. it reads back as an HDF5 with the freeze schema: state/action [T, 14] in the
     left_arm/left_ee/right_arm/right_ee order, frequency 25, three JPEG camera streams with
     T frames each, an instruction that is the official task sentence;
  3. the freeze hashes are recomputed here from scratch (this tool does NOT import the
     writer's hash helper) and match the ledger and the recorded metadata;
  4. the camera-inclusive sha256 is recomputed from the stored JPEG bytes and matches --
     closing the writer's documented gap that content_sha256 excludes camera bytes;
  5. provenance is present and names its source: every declared privileged value has a
     prov/<name> dataset of length T, each with non-empty source/file/line attributes that
     agree with the run manifest, and nothing undeclared was recorded;
  6. the official success flag is re-read independently from three places (ledger record,
     HDF5 attrs meta_json, HDF5 audit group) and all three must agree, and the recorded
     source must be the official one:
     RoboDojo/src/eval_client/eval_env.py:882-901 -> reward_manager.py:527-587;
  7. A[t] == S[t+1] holds on disk within the manifest tolerance;
  8. the mission caps hold for the run's own ledger: <=100 accepted per task, <=250
     attempts per task, <=3000 attempts across tasks, <=250 GiB of new bytes;
  9. no unledgered episode file and no .tmp residue is left in the run directory.

The tool never re-evaluates the reward (it cannot: no simulator, no GPU) -- it verifies
that the recorded verdict is the official verdict, is internally consistent, and was not
edited after the fact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

__all__ = ["run_check", "main", "FAIL", "PASS"]

PASS = "PASS"
FAIL = "FAIL"
EXIT_PASS = 0
EXIT_FAIL = 3
EXIT_ERROR = 2

ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "training" / "r15_collect_episode.py"

BASE_DIM = 14
JOINT_GROUPS = (("left_arm", 6), ("left_ee", 1), ("right_arm", 6), ("right_ee", 1))
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
FREQUENCY_HZ = 25
MIN_FRAMES = 100
GIB = 1024 ** 3
MIN_FRAME_SIDE = 64
_SOF_MARKERS = (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF)


# --------------------------------------------------------------------------- helpers

def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_driver():
    if not DRIVER_PATH.is_file():
        raise RuntimeError("the collection driver is missing at %s" % DRIVER_PATH)
    spec = importlib.util.spec_from_file_location("r15_collect_episode", str(DRIVER_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jpeg_size(buf):
    """(height, width) from a JPEG SOF marker, or None."""
    b = bytes(buf)
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
            return ((b[i + 3] << 8) | b[i + 4], (b[i + 5] << 8) | b[i + 6])
        if marker == 0xDA:
            return None
        i += seg_len
    return None


# --------------------------------------------------------------------------- hashes

def content_hashes(state, action, instruction):
    """The freeze pipeline's hash definition, re-implemented here (independent code)."""
    state = np.ascontiguousarray(np.asarray(state, dtype=np.float64), dtype="<f8")
    action = np.ascontiguousarray(np.asarray(action, dtype=np.float64), dtype="<f8")
    content = hashlib.sha256()
    content.update(state.tobytes())
    content.update(action.tobytes())
    content.update(str(instruction).encode("utf-8"))
    content.update(np.int64(len(state)).tobytes())
    state_action = hashlib.sha256(state.tobytes() + action.tobytes()).hexdigest()
    return {"content_sha256": content.hexdigest(), "state_action_sha256": state_action}


def camera_inclusive_sha256(state, action, instruction, streams, cameras=CAMERAS):
    """Same definition the driver records; recomputed from the stored JPEG bytes."""
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
        frames = streams[cam]
        digest.update(str(cam).encode("utf-8"))
        digest.update(np.int64(len(frames)).tobytes())
        for buf in frames:
            raw = bytes(buf)
            digest.update(np.int64(len(raw)).tobytes())
            digest.update(raw)
    return digest.hexdigest()


# --------------------------------------------------------------------------- reading

def _concat(handle, group):
    parts = [handle["%s/%s_joint_states" % (group, name)][()] for name, _ in JOINT_GROUPS]
    return np.concatenate(parts, axis=1).astype(np.float64)


def read_episode(path):
    """Read everything the gate needs, independently of the writer module."""
    import h5py

    with h5py.File(path, "r") as handle:
        state = _concat(handle, "state")
        action = _concat(handle, "action")
        instruction = handle["instruction"][()]
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8", "replace")
        frequency = int(handle["additional_info/frequency"][()])
        frames = int(state.shape[0])
        streams = {}
        for cam in CAMERAS:
            key = "vision/%s/colors" % cam
            if key not in handle:
                raise KeyError("missing camera stream %s" % cam)
            ds = handle[key]
            streams[cam] = [bytes(ds[t]) for t in range(int(ds.shape[0]))]
        attrs = dict(handle.attrs)
        meta = {}
        raw_meta = attrs.get("meta_json")
        if isinstance(raw_meta, bytes):
            raw_meta = raw_meta.decode("utf-8", "replace")
        if raw_meta:
            try:
                meta = json.loads(raw_meta)
            except ValueError:
                meta = {}
        audit = {}
        if "audit" in handle:
            audit = dict(handle["audit"].attrs)
            for key in list(audit):
                if isinstance(audit[key], bytes):
                    audit[key] = audit[key].decode("utf-8", "replace")
        prov = {}
        if "prov" in handle:
            prov_group = handle["prov"]
            for name in prov_group:
                ds = prov_group[name]
                data = [bytes(ds[i]) if isinstance(ds[i], (bytes, bytearray)) else ds[i]
                        for i in range(int(ds.shape[0]))]
                prov[name] = {"data": data, "attrs": dict(ds.attrs),
                              "declared": prov_group.attrs.get("declared_json")}
            declared = prov_group.attrs.get("declared_json")
            if isinstance(declared, bytes):
                declared = declared.decode("utf-8", "replace")
        else:
            declared = None
    if isinstance(attrs.get("task"), bytes):
        attrs["task"] = attrs["task"].decode("utf-8", "replace")
    if isinstance(attrs.get("expert_version_sha"), bytes):
        attrs["expert_version_sha"] = attrs["expert_version_sha"].decode("utf-8", "replace")
    return {"state": state, "action": action, "instruction": instruction,
            "frequency": frequency, "frames": frames, "streams": streams,
            "attrs": attrs, "meta": meta, "audit": audit, "prov": prov,
            "prov_declared": declared}


def read_ledger(run_dir):
    path = os.path.join(run_dir, "ledger.jsonl")
    if not os.path.isfile(path):
        raise RuntimeError("no ledger at %s" % path)
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError as exc:
                raise RuntimeError("ledger line %d is not JSON: %r" % (lineno, exc))
            if not isinstance(rec, dict):
                raise RuntimeError("ledger line %d is not an object" % lineno)
            rec["__line__"] = lineno
            records.append(rec)
    return records


def read_manifest(run_dir):
    path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.isfile(path):
        raise RuntimeError("no run manifest at %s" % path)
    with open(path, "r", encoding="utf-8") as handle:
        try:
            manifest = json.load(handle)
        except ValueError as exc:
            raise RuntimeError("%s is not JSON: %r" % (path, exc))
    if not isinstance(manifest, dict):
        raise RuntimeError("%s is not an object" % path)
    return manifest


# --------------------------------------------------------------------------- caps

def summarize(records):
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


# --------------------------------------------------------------------------- check

class _Report(object):
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.failures = []
        self.episodes = []
        self.warnings = []

    def fail(self, code, detail, episode=None):
        entry = {"code": code, "detail": detail}
        if episode:
            entry["episode"] = episode
        self.failures.append(entry)

    def warn(self, code, detail, episode=None):
        entry = {"code": code, "detail": detail}
        if episode:
            entry["episode"] = episode
        self.warnings.append(entry)


def _check_episode(report, run_dir, manifest, rec, caps, driver):
    name = str(rec.get("episode_path") or "")
    entry = {"episode": name, "attempt_id": rec.get("attempt_id"), "codes": []}

    def bad(code, detail):
        entry["codes"].append(code)
        report.fail(code, detail, episode=name)

    if not name:
        bad("episode_path_missing", "an accepted ledger record has no episode_path")
        report.episodes.append(entry)
        return entry
    path = os.path.join(run_dir, name)
    if not os.path.isfile(path):
        bad("episode_missing", "%s is in the ledger but not on disk" % name)
        report.episodes.append(entry)
        return entry

    size = int(os.path.getsize(path))
    entry["bytes"] = size
    if int(rec.get("bytes") or -1) != size:
        bad("size_mismatch", "%s is %d bytes but the ledger says %r"
            % (name, size, rec.get("bytes")))
    file_sha = _sha256_file(path)
    if rec.get("file_sha256") != file_sha:
        bad("file_sha256_mismatch", "%s whole-file sha256 is %s, ledger says %r (the file "
            "changed after it was accepted)" % (name, file_sha, rec.get("file_sha256")))

    try:
        data = read_episode(path)
    except Exception as exc:                            # noqa: BLE001
        bad("readback_failed", "%s cannot be read back: %r" % (name, exc))
        report.episodes.append(entry)
        return entry

    state, action = data["state"], data["action"]
    frames = int(data["frames"])
    entry["frames"] = frames
    align_tol = float((manifest.get("state_action") or {}).get("align_tol") or 1e-6)
    instruction = data["instruction"]
    meta = data["meta"]
    audit = data["audit"]

    if state.shape != action.shape or state.shape[1:2] != (BASE_DIM,):
        bad("readback_schema", "%s state %r / action %r are not [T, %d]"
            % (name, state.shape, action.shape, BASE_DIM))
        report.episodes.append(entry)
        return entry
    if frames < MIN_FRAMES:
        bad("readback_frames", "%s has %d frames (< %d)" % (name, frames, MIN_FRAMES))
    if int(rec.get("frames", -1)) != frames:
        bad("frames_mismatch", "%s has %d frames, the ledger says %r"
            % (name, frames, rec.get("frames")))
    if not (np.isfinite(state).all() and np.isfinite(action).all()):
        bad("nonfinite_state", "%s state/action contain non-finite values" % name)
    misalign = float(np.abs(action[:-1] - state[1:]).max()) if frames > 1 else 0.0
    entry["max_abs_action_minus_next_state"] = misalign
    if misalign > align_tol:
        bad("alignment_failed", "%s max|A[t]-S[t+1]| = %.6g > %.6g"
            % (name, misalign, align_tol))
    recorded_align = float(rec.get("max_abs_action_minus_next_state") or 0.0)
    if recorded_align > align_tol:
        bad("alignment_recorded", "%s the ledger recorded a misalignment of %.6g > %.6g"
            % (name, recorded_align, align_tol))

    if data["frequency"] != int((manifest.get("frequency_hz") or FREQUENCY_HZ)):
        bad("frequency_mismatch", "%s frequency %d != %r"
            % (name, data["frequency"], manifest.get("frequency_hz")))
    if str(instruction) != str(manifest.get("instruction")):
        bad("instruction_mismatch", "%s instruction %r != the official %r"
            % (name, instruction, manifest.get("instruction")))
    if str(rec.get("instruction")) != str(manifest.get("instruction")):
        bad("ledger_instruction_mismatch", "%s ledger instruction != the official sentence"
            % name)
    if str(data["attrs"].get("task")) != str(manifest.get("writer_task")):
        bad("task_attr_mismatch", "%s HDF5 task attr %r != %r"
            % (name, data["attrs"].get("task"), manifest.get("writer_task")))
    if int(rec.get("seed", -1)) != int(data["attrs"].get("seed", -2)):
        bad("seed_mismatch", "%s seed attr %r != ledger %r"
            % (name, data["attrs"].get("seed"), rec.get("seed")))
    expert_sha = str((manifest.get("expert") or {}).get("version_sha"))
    if str(data["attrs"].get("expert_version_sha")) != expert_sha:
        bad("expert_sha_mismatch", "%s expert_version_sha %r != manifest %r"
            % (name, data["attrs"].get("expert_version_sha"), expert_sha))
    if str((manifest.get("expert") or {}).get("status")) != "IMPLEMENTED":
        bad("expert_status_invalid", "%s the manifest does not declare an IMPLEMENTED expert "
            "(%r)" % (name, (manifest.get("expert") or {}).get("status")))

    for cam in CAMERAS:
        stream = data["streams"].get(cam)
        if stream is None:
            bad("camera_missing", "%s has no %r stream" % (name, cam))
            continue
        if len(stream) != frames:
            bad("camera_frame_count", "%s camera %s has %d frames, expected %d"
                % (name, cam, len(stream), frames))
        for t, buf in enumerate(stream):
            if not buf or bytes(buf[:2]) != b"\xff\xd8":
                bad("camera_frame_invalid", "%s camera %s frame %d is not a JPEG stream"
                    % (name, cam, t))
                break
            size_px = _jpeg_size(buf)
            if size_px is None:
                bad("camera_frame_invalid", "%s camera %s frame %d has no SOF header"
                    % (name, cam, t))
                break
            if min(size_px) < MIN_FRAME_SIDE:
                bad("camera_frame_small", "%s camera %s frame %d is %dx%d (< %dpx)"
                    % (name, cam, t, size_px[0], size_px[1], MIN_FRAME_SIDE))
                break

    hashes = content_hashes(state, action, instruction)
    entry["content_sha256"] = hashes["content_sha256"]
    entry["state_action_sha256"] = hashes["state_action_sha256"]
    if hashes["content_sha256"] != rec.get("content_sha256"):
        bad("content_sha256_mismatch", "%s content_sha256 recomputes to %s, ledger says %r"
            % (name, hashes["content_sha256"], rec.get("content_sha256")))
    if hashes["state_action_sha256"] != rec.get("state_action_sha256"):
        bad("state_action_sha256_mismatch", "%s state_action_sha256 recomputes to %s, ledger "
            "says %r" % (name, hashes["state_action_sha256"],
                         rec.get("state_action_sha256")))
    cam_sha = camera_inclusive_sha256(state, action, instruction, data["streams"])
    entry["camera_inclusive_sha256"] = cam_sha
    if cam_sha != rec.get("camera_inclusive_sha256"):
        bad("camera_sha256_mismatch", "%s camera-inclusive sha256 recomputes to %s, ledger "
            "says %r (camera bytes are not covered by content_sha256, so this is the only "
            "pixel tamper check)" % (name, cam_sha, rec.get("camera_inclusive_sha256")))
    if str(audit.get("camera_inclusive_sha256")) != cam_sha:
        bad("camera_sha256_attr_mismatch", "%s audit group says %r, recomputed %s"
            % (name, audit.get("camera_inclusive_sha256"), cam_sha))

    # ---- provenance
    manifest_prov = dict(manifest.get("privileged") or {})
    manifest_keys = list(manifest.get("privileged_keys") or [])
    prov = data["prov"]
    entry["provenance_keys"] = sorted(prov)
    if not prov:
        bad("prov_missing", "%s has no prov/ group: privileged state was not recorded"
            % name)
    for key in manifest_keys:
        if key not in prov:
            bad("prov_missing", "%s has no prov/%s dataset (declared as consumed by the "
                "expert)" % (name, key))
            continue
        node = prov[key]
        attrs = node["attrs"]
        for field in ("source", "file", "line"):
            if field not in attrs:
                bad("prov_incomplete", "%s prov/%s has no %r attribute"
                    % (name, key, field))
        for field in ("source", "file"):
            value = attrs.get(field)
            if isinstance(value, bytes):
                value = value.decode("utf-8", "replace")
            if not value or not str(value).strip():
                bad("prov_incomplete", "%s prov/%s has an empty %r attribute"
                    % (name, key, field))
        if len(node["data"]) != frames:
            bad("prov_length_mismatch", "%s prov/%s has %d values for %d frames"
                % (name, key, len(node["data"]), frames))
        declared = manifest_prov.get(key) or {}
        for field in ("source", "file", "line"):
            got = attrs.get(field)
            if isinstance(got, bytes):
                got = got.decode("utf-8", "replace")
            want = declared.get(field)
            if field == "line" and got is not None and want is not None:
                try:
                    if int(got) != int(want):
                        bad("prov_declaration_mismatch",
                            "%s prov/%s line %r != manifest %r" % (name, key, got, want))
                except (TypeError, ValueError):
                    bad("prov_declaration_mismatch", "%s prov/%s line %r is not an integer"
                        % (name, key, got))
            elif field != "line" and want is not None and str(got) != str(want):
                bad("prov_declaration_mismatch", "%s prov/%s %s %r != manifest %r"
                    % (name, key, field, got, want))
        if not str(attrs.get("source", "") or "").strip():
            bad("prov_source_unnamed", "%s prov/%s does not name its source getter"
                % (name, key))
    undeclared = [k for k in prov if k not in manifest_keys]
    if undeclared:
        bad("prov_undeclared", "%s records privileged values the manifest does not declare: "
            "%s" % (name, ", ".join(sorted(undeclared))))
    if list(rec.get("privileged_keys") or []) != manifest_keys:
        bad("ledger_privileged_mismatch", "%s ledger privileged keys %r != manifest %r"
            % (name, rec.get("privileged_keys"), manifest_keys))

    # ---- the official verdict, re-read three ways
    ledger_success = rec.get("official_success")
    meta_success = ((meta.get("official") or {}).get("success"))
    audit_success = audit.get("official_success")
    try:
        audit_success = bool(int(audit_success))
    except (TypeError, ValueError):
        audit_success = None
    entry["official_success"] = ledger_success
    if not isinstance(ledger_success, bool):
        bad("verdict_invalid", "%s ledger official_success is %r, not a bool"
            % (name, ledger_success))
    if not isinstance(meta_success, bool):
        bad("verdict_invalid", "%s HDF5 meta_json has no official.success bool" % name)
    if audit_success is None:
        bad("verdict_invalid", "%s HDF5 audit group has no official_success" % name)
    if isinstance(ledger_success, bool) and isinstance(meta_success, bool) \
            and ledger_success != meta_success:
        bad("verdict_mismatch", "%s ledger success %r != HDF5 meta success %r"
            % (name, ledger_success, meta_success))
    if isinstance(ledger_success, bool) and audit_success is not None \
            and ledger_success != audit_success:
        bad("verdict_mismatch", "%s ledger success %r != HDF5 audit success %r"
            % (name, ledger_success, audit_success))
    expected_termination = ("official_success" if ledger_success
                            else "official_failure")
    if rec.get("termination") != expected_termination:
        bad("termination_mismatch", "%s termination %r does not match the official verdict "
            "%r" % (name, rec.get("termination"), ledger_success))

    official_source = rec.get("official_source") or {}
    source_text = " ".join(str(official_source.get(k, "")) for k in
                           ("method", "reward", "flag"))
    if "eval_env.py:882-901" not in source_text or "reward_manager.py:527-587" not in \
            source_text:
        bad("official_source_invalid", "%s does not record the official verdict source "
            "(eval_env.py:882-901 / reward_manager.py:527-587): %r"
            % (name, official_source))
    manifest_source = manifest.get("official_verdict") or {}
    for key in ("method", "reward", "flag"):
        if str(manifest_source.get(key)) != str(official_source.get(key)):
            bad("official_source_invalid", "%s official_source.%s differs from the run "
                "manifest" % (name, key))
    meta_source = (meta.get("official") or {}).get("source") or {}
    if str(meta_source.get("method")) != str(official_source.get("method")):
        bad("official_source_invalid", "%s HDF5 meta official source differs from the "
            "ledger" % name)

    report.episodes.append(entry)
    return entry


def run_check(run_dir, registry_path=None, caps=None):
    """Run the gate on a collection run directory and return the report dict."""
    run_dir = os.fspath(run_dir)
    report = _Report(run_dir)
    driver = _load_driver()
    caps = dict(driver.CAPS if caps is None else caps)
    result = {
        "schema": "r15-collect-check-1",
        "checked_utc": _utc_now(),
        "run_dir": os.path.abspath(run_dir),
        "caps": caps,
        "counts": {},
        "episodes": [],
        "failures": [],
        "warnings": [],
    }
    try:
        manifest = read_manifest(run_dir)
        records = read_ledger(run_dir)
    except Exception as exc:                            # noqa: BLE001
        report.fail("run_unreadable", "%r" % (exc,))
        result.update({"verdict": FAIL, "exit_code": EXIT_FAIL,
                       "failures": report.failures, "episodes": report.episodes})
        return result

    if manifest.get("schema") != "r15-collect-manifest-1":
        report.fail("manifest_bad", "manifest schema is %r" % (manifest.get("schema"),))
    if not manifest.get("instruction") or len(str(manifest.get("instruction"))) < 8:
        report.fail("manifest_bad", "manifest has no official instruction")
    if not manifest.get("official_verdict"):
        report.fail("manifest_bad", "manifest has no official verdict declaration")
    if not manifest.get("privileged"):
        report.fail("manifest_bad", "manifest declares no privileged provenance")

    stats = summarize(records)
    result["counts"] = stats

    # ---- task metadata against the registry (if it is reachable)
    registry_summary = None
    try:
        if registry_path is None:
            reg_path = ROOT / "training" / "r15_expert_registry.py"
        else:
            reg_path = pathlib.Path(registry_path)
        if reg_path.is_file():
            spec = importlib.util.spec_from_file_location("r15_expert_registry",
                                                          str(reg_path))
            registry = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(registry)
            task_id = str(manifest.get("task"))
            task = registry.get_task(task_id)
            registry_summary = {"task": task_id, "status": task.status,
                                "predicate_ref": task.predicate_ref}
            if task.status != registry.IMPLEMENTED:
                report.fail("task_not_implemented",
                            "the registry says %s is %s" % (task_id, task.status))
            if str(task.instruction) != str(manifest.get("instruction")):
                report.fail("instruction_not_official",
                            "manifest instruction %r != registry %r"
                            % (manifest.get("instruction"), task.instruction))
            if int(task.step_lim) != int(manifest.get("step_lim") or -1):
                report.fail("step_lim_not_official",
                            "manifest step_lim %r != registry %r"
                            % (manifest.get("step_lim"), task.step_lim))
            if str(task.writer_task) != str(manifest.get("writer_task")):
                report.fail("writer_task_mismatch",
                            "manifest writer_task %r != registry %r"
                            % (manifest.get("writer_task"), task.writer_task))
            if list(task.privileged_keys) != list(manifest.get("privileged_keys") or []):
                report.fail("privileged_keys_mismatch",
                            "manifest privileged keys %r != registry %r"
                            % (manifest.get("privileged_keys"),
                               list(task.privileged_keys)))
        else:
            report.warn("registry_missing", "no registry at %s; skipped the official "
                        "instruction/step_lim cross-check" % reg_path)
    except Exception as exc:                            # noqa: BLE001
        report.fail("registry_check_failed", "%r" % (exc,))
    result["registry"] = registry_summary

    # ---- caps from the run's own ledger
    total = stats["total"]
    for task, slot in sorted(stats["per_task"].items()):
        if slot["accepted"] > int(caps["accepted_per_task"]):
            report.fail("cap_accepted_per_task",
                        "task %r has %d accepted episodes > %d"
                        % (task, slot["accepted"], caps["accepted_per_task"]))
        if slot["attempts"] > int(caps["attempts_per_task"]):
            report.fail("cap_attempts_per_task",
                        "task %r has %d attempts > %d"
                        % (task, slot["attempts"], caps["attempts_per_task"]))
    if total["attempts"] > int(caps["attempts_total"]):
        report.fail("cap_attempts_total", "the run has %d attempts > %d"
                    % (total["attempts"], caps["attempts_total"]))
    if total["bytes"] > int(caps["new_bytes"]):
        report.fail("cap_new_bytes", "the run wrote %d bytes > %d (%.0f GiB)"
                    % (total["bytes"], caps["new_bytes"], caps["new_bytes"] / float(GIB)))
    if total["accepted"] == 0:
        report.fail("no_accepted_episodes", "the run accepted no episode: nothing to ship")
    if total["attempts"] != len(records):
        report.fail("ledger_inconsistent", "%d attempts but %d ledger records"
                    % (total["attempts"], len(records)))

    seen_ids = set()
    accepted_names = set()
    for rec in records:
        if rec.get("schema") != driver.SCHEMA:
            report.fail("ledger_schema", "ledger line %s has schema %r"
                        % (rec.get("__line__"), rec.get("schema")))
        attempt_id = rec.get("attempt_id")
        if attempt_id in seen_ids:
            report.fail("attempt_id_duplicate", "duplicate attempt_id %r" % (attempt_id,))
        seen_ids.add(attempt_id)
        if rec.get("status") == "refused":
            if rec.get("code") in (None, ""):
                report.fail("refusal_without_code", "attempt %r was refused without a code"
                            % (attempt_id,))
            if rec.get("episode_path"):
                path = os.path.join(run_dir, str(rec["episode_path"]))
                if os.path.exists(path):
                    report.fail("refused_has_episode", "attempt %r was refused but %s exists"
                                % (attempt_id, rec["episode_path"]))
            continue
        if rec.get("status") != "accepted":
            report.fail("ledger_status", "attempt %r has status %r"
                        % (attempt_id, rec.get("status")))
            continue
        accepted_names.add(os.path.basename(str(rec.get("episode_path") or "")))
        _check_episode(report, run_dir, manifest, rec, caps, driver)

    # ---- no stray episode files or .tmp residue
    episodes_dir = os.path.join(run_dir, "episodes")
    if os.path.isdir(episodes_dir):
        for name in sorted(os.listdir(episodes_dir)):
            if name.endswith(".tmp"):
                report.fail("tmp_residue", "episodes/%s is a leftover temporary file"
                            % name)
            elif name.endswith(".hdf5") and name not in accepted_names:
                report.fail("unledgered_episode",
                            "episodes/%s is not accounted for by any accepted ledger "
                            "record" % name)

    result["episodes"] = report.episodes
    result["failures"] = report.failures
    result["warnings"] = report.warnings
    result["verdict"] = PASS if not report.failures else FAIL
    result["exit_code"] = EXIT_PASS if not report.failures else EXIT_FAIL
    result["successes"] = total["successes"]
    return result


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="R15 collection quality gate: verify a run directory end to end.")
    parser.add_argument("run_dir", help="collection run directory (manifest + ledger + "
                                        "episodes/)")
    parser.add_argument("--report", default=None,
                        help="where to write the JSON report (default: <run_dir>/"
                             "check_report.json)")
    parser.add_argument("--registry", default=None, help="path to r15_expert_registry.py")
    parser.add_argument("--json", action="store_true", help="print the whole report as JSON")
    args = parser.parse_args(argv)

    try:
        result = run_check(args.run_dir, registry_path=args.registry)
    except Exception as exc:                            # noqa: BLE001
        print("CHECK ERROR: %r" % (exc,), file=sys.stderr)
        return EXIT_ERROR

    report_path = args.report or os.path.join(args.run_dir, "check_report.json")
    try:
        tmp = report_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, report_path)
    except OSError as exc:
        print("WARNING: could not write %s: %r" % (report_path, exc), file=sys.stderr)

    counts = result.get("counts") or {}
    total = counts.get("total") or {}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("R15 collect check: %s" % result["verdict"])
        print("  run dir      : %s" % result["run_dir"])
        print("  attempts     : %d (accepted %d, refused %d, official successes %d)"
              % (total.get("attempts", 0), total.get("accepted", 0),
                 total.get("refused", 0), total.get("successes", 0)))
        print("  new bytes    : %d (cap %d)" % (total.get("bytes", 0),
                                                result["caps"]["new_bytes"]))
        print("  episodes     : %d checked" % len(result["episodes"]))
        print("  failures     : %d" % len(result["failures"]))
        for failure in result["failures"][:20]:
            print("    - [%s] %s%s" % (failure["code"], failure["detail"],
                                       (" (%s)" % failure["episode"])
                                       if failure.get("episode") else ""))
        if len(result["failures"]) > 20:
            print("    ... %d more" % (len(result["failures"]) - 20))
        for warning in result["warnings"][:5]:
            print("  warning      : [%s] %s" % (warning["code"], warning["detail"]))
        print("  report       : %s" % report_path)
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
