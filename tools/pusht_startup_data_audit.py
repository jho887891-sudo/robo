#!/usr/bin/env python3
"""Read-only Push-T startup/data audit for the M2 step-2000 checkpoint.

CPU only. Reads HDF5, frozen splits and checkpoint metadata; never loads model weights into
Torch, never initializes CUDA, never changes HDF5/checkpoints/WAL, and never performs training.
It reconstructs the exact consumed sampler prefix, then compares the first second of every
episode with the actual TRAIN-SEEN / TRAIN-UNSEEN / DEV frame groups.

Interpretation guardrails:
- HDF5 action A[t] is specified as the next-step absolute target S[t+1].
- Residual R[t] = (A[t] - S[t]) - (S[t] - S[t-1]); valid t is 2..T-2.
- Kinematic stillness is a threshold-sensitivity diagnostic, not a task-phase label.
- HDF5 RGB has frame indices but no per-stream capture timestamps; index alignment is not
  proof of wall-clock synchronization.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (
    os.path.join(REPO, "training"),
    os.path.join(REPO, "code/xpolicylab/policy/OpenVLA_R42"),
    os.path.join(REPO, "code/xpolicylab/policy/OpenVLA_Pure"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ARM_DIMS = np.asarray(list(range(6)) + list(range(7, 13)), dtype=np.int64)
GRIPPER_DIMS = np.asarray([6, 13], dtype=np.int64)
STILL_THRESHOLDS_RAD_PER_SAMPLE = (1e-5, 1e-4, 1e-3, 1e-2)
DEFAULT_START_WINDOW = 25
ALIGN_ATOL = 1e-6


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arrays(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str(a.dtype).encode("ascii"))
        h.update(json.dumps(list(a.shape), separators=(",", ":")).encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


def distribution(values):
    """Finite scalar distribution summary; returns null quantiles for an empty input."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0, "mean": None, "std": None, "q50": None, "q90": None,
                "q95": None, "q99": None, "max": None}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q50": float(np.quantile(x, 0.50)),
        "q90": float(np.quantile(x, 0.90)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
    }


def vector_distribution(values):
    """Per-dimension absolute-value quantiles and signed means."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 14 or x.shape[0] == 0:
        return {"n": int(x.shape[0]) if x.ndim else 0,
                "mean_signed_14d": None, "mean_abs_14d": None,
                "abs_q50_14d": None, "abs_q90_14d": None, "abs_q99_14d": None}
    ax = np.abs(x)
    return {
        "n": int(x.shape[0]),
        "mean_signed_14d": np.mean(x, axis=0).tolist(),
        "mean_abs_14d": np.mean(ax, axis=0).tolist(),
        "abs_q50_14d": np.quantile(ax, 0.50, axis=0).tolist(),
        "abs_q90_14d": np.quantile(ax, 0.90, axis=0).tolist(),
        "abs_q99_14d": np.quantile(ax, 0.99, axis=0).tolist(),
    }


def analyze_arrays(state, action, start_window=DEFAULT_START_WINDOW):
    """Pure NumPy audit for one episode. Exposes arrays for group aggregation in the caller."""
    s = np.asarray(state)
    a = np.asarray(action)
    if s.ndim != 2 or a.ndim != 2 or s.shape != a.shape or s.shape[1] != 14:
        raise ValueError("state/action must have identical [T,14] shapes; got %s / %s"
                         % (s.shape, a.shape))
    if s.shape[0] < 4:
        raise ValueError("episode must have at least 4 rows for history-safe targets")
    if not np.isfinite(s).all() or not np.isfinite(a).all():
        raise ValueError("state/action contains NaN or Inf")

    # A[t] is expected to be S[t+1]. The final action row has no next-state row to compare.
    alignment_diff = np.asarray(a[:-1], dtype=np.float64) - np.asarray(s[1:], dtype=np.float64)
    alignment_abs = np.abs(alignment_diff)
    aligned_rows = np.all(alignment_diff == 0.0, axis=1)
    over_tolerance_rows = np.any(alignment_abs > ALIGN_ATOL, axis=1)

    s64 = np.asarray(s, dtype=np.float64)
    a64 = np.asarray(a, dtype=np.float64)
    transition = s64[1:] - s64[:-1]               # transition indexed by source t: S[t+1]-S[t]
    action_delta = a64[:-1] - s64[:-1]            # A[t]-S[t]
    arm_rms_transition = np.sqrt(np.mean(transition[:, ARM_DIMS] ** 2, axis=1))
    grip_abs_transition = np.max(np.abs(transition[:, GRIPPER_DIMS]), axis=1)
    n_start = min(int(start_window), int(transition.shape[0]))

    # Residual is defined only for t=2..T-2 inclusive.
    valid_t = np.arange(2, s.shape[0] - 1, dtype=np.int64)
    velocity = s64[valid_t] - s64[valid_t - 1]
    delta = a64[valid_t] - s64[valid_t]
    residual = delta - velocity
    arm_rms_velocity = np.sqrt(np.mean(velocity[:, ARM_DIMS] ** 2, axis=1))
    arm_rms_delta = np.sqrt(np.mean(delta[:, ARM_DIMS] ** 2, axis=1))
    arm_rms_residual = np.sqrt(np.mean(residual[:, ARM_DIMS] ** 2, axis=1))
    grip_abs_velocity = np.max(np.abs(velocity[:, GRIPPER_DIMS]), axis=1)
    grip_abs_delta = np.max(np.abs(delta[:, GRIPPER_DIMS]), axis=1)
    grip_abs_residual = np.max(np.abs(residual[:, GRIPPER_DIMS]), axis=1)

    first_moves = {}
    still_sensitivity = {}
    for threshold in STILL_THRESHOLDS_RAD_PER_SAMPLE:
        key = "%g" % threshold
        moving = np.flatnonzero(arm_rms_transition > threshold)
        first_moves[key] = (None if not moving.size else {
            "source_transition_t": int(moving[0]),
            "time_seconds_from_episode_start": float(moving[0] / 25.0),
        })
        still_sensitivity[key] = {
            "all_transitions_fraction_below_or_equal": float(
                np.mean(arm_rms_transition <= threshold)),
            "first_window_fraction_below_or_equal": float(
                np.mean(arm_rms_transition[:n_start] <= threshold)),
        }

    n_res_start = min(int(start_window), int(valid_t.size))
    return {
        "T": int(s.shape[0]),
        "state_dtype": str(s.dtype),
        "action_dtype": str(a.dtype),
        "state_action_array_sha256": sha256_arrays(s, a),
        "action_next_state_alignment": {
            "compared_rows": int(alignment_diff.shape[0]),
            "exactly_equal_rows": int(np.count_nonzero(aligned_rows)),
            "rows_over_atol_1e-6": int(np.count_nonzero(over_tolerance_rows)),
            "max_abs_error": float(np.max(alignment_abs)) if alignment_abs.size else 0.0,
            "mean_abs_error": float(np.mean(alignment_abs)) if alignment_abs.size else 0.0,
        },
        "valid_residual_t_range_inclusive": [2, int(s.shape[0] - 2)],
        "valid_residual_rows": int(valid_t.size),
        "startup_window": {
            "requested_transitions": int(start_window),
            "actual_transitions": int(n_start),
            "arm_rms_state_transition_rad_per_sample": distribution(
                arm_rms_transition[:n_start]),
            "gripper_max_abs_state_transition": distribution(grip_abs_transition[:n_start]),
            "first_action_delta_arm_rms": (float(np.sqrt(np.mean(
                action_delta[0, ARM_DIMS] ** 2))) if action_delta.size else None),
            "first_action_delta_gripper_max_abs": (float(np.max(
                np.abs(action_delta[0, GRIPPER_DIMS]))) if action_delta.size else None),
            "first_residual_window_arm_rms": distribution(arm_rms_residual[:n_res_start]),
            "still_threshold_sensitivity": still_sensitivity,
            "first_motion_transition_by_threshold": first_moves,
        },
        "all_valid_rows": {
            "velocity_arm_rms_rad_per_sample": distribution(arm_rms_velocity),
            "action_delta_arm_rms": distribution(arm_rms_delta),
            "residual_arm_rms": distribution(arm_rms_residual),
            "velocity_gripper_max_abs": distribution(grip_abs_velocity),
            "action_delta_gripper_max_abs": distribution(grip_abs_delta),
            "residual_gripper_max_abs": distribution(grip_abs_residual),
            "velocity_14d": vector_distribution(velocity),
            "action_delta_14d": vector_distribution(delta),
            "residual_14d": vector_distribution(residual),
            "residual_to_action_delta_arm_rms_median_ratio": (
                float(np.median(arm_rms_residual) / np.median(arm_rms_delta))
                if np.median(arm_rms_delta) > 0 else None),
        },
        "_arrays": {
            "valid_t": valid_t,
            "velocity": velocity,
            "action_delta": delta,
            "residual": residual,
            "arm_rms_velocity": arm_rms_velocity,
            "arm_rms_delta": arm_rms_delta,
            "arm_rms_residual": arm_rms_residual,
            "transition_arm_rms": arm_rms_transition,
        },
    }


def aggregate_rows(rows, thresholds=STILL_THRESHOLDS_RAD_PER_SAMPLE):
    """Aggregate selected history-safe rows without conflating episode-level and frame-level stats."""
    if not rows:
        return {"n_rows": 0}
    velocity = np.concatenate([r["velocity"] for r in rows], axis=0)
    delta = np.concatenate([r["action_delta"] for r in rows], axis=0)
    residual = np.concatenate([r["residual"] for r in rows], axis=0)
    arm_v = np.concatenate([r["arm_rms_velocity"] for r in rows])
    arm_d = np.concatenate([r["arm_rms_delta"] for r in rows])
    arm_r = np.concatenate([r["arm_rms_residual"] for r in rows])
    return {
        "n_rows": int(velocity.shape[0]),
        "velocity_arm_rms_rad_per_sample": distribution(arm_v),
        "action_delta_arm_rms": distribution(arm_d),
        "residual_arm_rms": distribution(arm_r),
        "velocity_14d": vector_distribution(velocity),
        "action_delta_14d": vector_distribution(delta),
        "residual_14d": vector_distribution(residual),
        "still_threshold_sensitivity": {
            "%g" % threshold: float(np.mean(arm_v <= threshold))
            for threshold in thresholds
        },
        "residual_to_action_delta_arm_rms_median_ratio": (
            float(np.median(arm_r) / np.median(arm_d)) if np.median(arm_d) > 0 else None),
    }


def _bare_episode(M2, task, value):
    value = str(value)
    if "/" in value:
        got_task, episode = M2.split_qualified(value)
        if got_task != task:
            raise ValueError("split entry %r is not in task %s" % (value, task))
        return episode
    return value


def reconstruct_seen_prefix(checkpoint, tasks, split_map, indexes, splits_path, normalizer_path):
    """Reconstruct the exact consumed sampler prefix, checking it against checkpoint contracts."""
    import m2_multitask_data as M2
    import m2_train_shared as M2T
    import r15_resume as RS

    _manifest, trainer = RS.verify_and_load_manifest(checkpoint)
    if int(trainer.get("global_step", -1)) < 0:
        raise ValueError("checkpoint trainer_state has invalid global_step")
    if list(trainer.get("tasks") or []) != list(tasks):
        raise ValueError("checkpoint task list does not match expected tasks: %r" %
                         (trainer.get("tasks"),))
    task_mix = trainer.get("task_mix") or {}
    required = ("floor", "seed", "schedule_sha256", "counts", "horizon")
    if any(k not in task_mix for k in required):
        raise ValueError("checkpoint task_mix missing required fields")
    _doc, loaded_splits, _payload, _sp, _np = M2T.load_m2_artifacts(
        tasks, splits_path, normalizer_path)
    if set(loaded_splits) != set(split_map):
        raise ValueError("loaded split task set differs")
    frames = {t: M2T.task_train_frames(indexes[t], loaded_splits[t], t) for t in tasks}
    mix = M2T.plan_task_mix(
        tasks, {t: len(frames[t]) for t in tasks},
        floor=float(task_mix["floor"]), seed=int(task_mix["seed"]),
        horizon=int(task_mix["horizon"]))
    if mix["schedule_sha256"] != task_mix["schedule_sha256"]:
        raise ValueError("reconstructed sampler schedule SHA256 mismatch")
    if mix["counts"] != task_mix["counts"] or mix["horizon"] != task_mix["horizon"]:
        raise ValueError("reconstructed task mix counts/horizon mismatch")
    slots = M2T.build_slot_frames(mix, frames)
    with open(os.path.join(checkpoint, "sampler.json"), "r") as fh:
        saved_sampler = json.load(fh)
    sampler = RS.FrameSampler.from_state(slots, saved_sampler)
    position = int(sampler.position)
    global_step = int(trainer["global_step"])
    if position != global_step or not 0 <= position <= len(slots):
        raise ValueError("sampler.position/global_step mismatch or sampler wrapped an epoch")
    actual_order_hash = sampler.order_hash()
    recorded_order_hash = trainer.get("sampler_order_hash")
    if recorded_order_hash and actual_order_hash != recorded_order_hash:
        raise ValueError("reconstructed sampler order hash differs from trainer_state")
    seen = set()
    duplicates = 0
    for idx in sampler.perm[:position]:
        qualified, tick = slots[int(idx)]
        task, episode = M2.split_qualified(qualified)
        key = (task, os.path.basename(str(episode)), int(tick))
        if key in seen:
            duplicates += 1
        seen.add(key)
    return {
        "position": position,
        "global_step": global_step,
        "slot_pool_size": len(slots),
        "unique_consumed_frame_keys": len(seen),
        "duplicate_consumed_frame_keys": int(duplicates),
        "sampler_order_sha256": actual_order_hash,
        "checkpoint_manifest_sha256": sha256_file(os.path.join(checkpoint, "MANIFEST.json")),
        "seen_keys": seen,
    }


def _image_audit_for_episode(h, episode, sample_indices, startup_window, cv2, camera_order):
    result = {}
    for camera in camera_order:
        ds = h["vision/" + camera + "/colors"]
        digest = hashlib.sha256()
        decoded = {}
        decode_failures = 0
        malformed_jpeg = 0
        for t in sorted(sample_indices):
            raw = bytes(ds[int(t)])
            digest.update(str(int(t)).encode("ascii") + b"\0")
            digest.update(raw)
            if not (raw.startswith(b"\xff\xd8") and raw.endswith(b"\xff\xd9")):
                malformed_jpeg += 1
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                decode_failures += 1
            else:
                decoded[int(t)] = image
        adjacent_mae = []
        exact_duplicate_pairs = 0
        prev_raw = None
        for t in range(min(int(startup_window), int(ds.shape[0]))):
            raw = bytes(ds[t])
            if prev_raw is not None and raw == prev_raw:
                exact_duplicate_pairs += 1
            prev_raw = raw
            if t > 0 and (t - 1) in decoded and t in decoded:
                left, right = decoded[t - 1], decoded[t]
                if left.shape == right.shape:
                    adjacent_mae.append(float(np.mean(np.abs(
                        right.astype(np.int16) - left.astype(np.int16)))))
        result[camera] = {
            "sampled_frame_count": int(len(sample_indices)),
            "decoded_frame_count": int(len(decoded)),
            "malformed_jpeg_count": int(malformed_jpeg),
            "decode_failure_count": int(decode_failures),
            "exact_duplicate_adjacent_jpeg_pairs_in_startup_window": int(exact_duplicate_pairs),
            "startup_adjacent_pixel_mae": distribution(adjacent_mae),
            "sampled_jpeg_bytes_sha256": digest.hexdigest(),
        }
    return result


def run_audit(args):
    import cv2
    import h5py
    import m2_multitask_data as M2
    import m2_train_shared as M2T
    import r42_train_contract as TC

    with open(args.splits, "r") as fh:
        split_doc = json.load(fh)
    split_map = split_doc.get("splits", split_doc)
    task = "push_T"
    if task not in split_map:
        raise ValueError("split file has no push_T entry")
    task_split = split_map[task]
    train_eps = sorted({_bare_episode(M2, task, e) for e in task_split["train"]})
    dev_eps = sorted({_bare_episode(M2, task, e) for e in task_split["dev"]})
    if set(train_eps) & set(dev_eps):
        raise ValueError("TRAIN and DEV episode lists overlap")
    if not train_eps or not dev_eps:
        raise ValueError("TRAIN or DEV split is empty")

    index = M2.load_task_index(task, args.hdf5_root, verify_targets=False)
    available = set(index["episodes"])
    if not set(train_eps + dev_eps) <= available:
        raise ValueError("split references missing HDF5 episodes")
    if set(train_eps + dev_eps) != available:
        raise ValueError("split does not cover the available task episode inventory")

    # The checkpoint is two-task, so reconstruct both task indices and its actual consumed prefix.
    _doc, all_splits, _payload, splits_path, normalizer_path = M2T.load_m2_artifacts(
        ["push_T", "stack_bowls"], args.splits, args.normalizer)
    indexes = {"push_T": index}
    indexes["stack_bowls"] = M2.load_task_index("stack_bowls", args.hdf5_root,
                                                  verify_targets=False)
    exposure = reconstruct_seen_prefix(
        args.checkpoint, ["push_T", "stack_bowls"], all_splits, indexes,
        splits_path, normalizer_path)
    seen_push = {key for key in exposure["seen_keys"] if key[0] == task}
    seen_ep_t = {}
    for _task, ep, tick in seen_push:
        seen_ep_t.setdefault(ep, set()).add(int(tick))

    split_episodes = {"train": train_eps, "dev": dev_eps}
    group_rows = {
        "train_seen_all_valid": [], "train_unseen_all_valid": [],
        "train_seen_startup": [], "train_unseen_startup": [],
        "dev_all_valid": [], "dev_startup": [],
    }
    episode_reports = []
    image_groups = {
        "train": {c: [] for c in M2.CAMERA_ORDER},
        "dev": {c: [] for c in M2.CAMERA_ORDER},
    }
    integrity = {
        "episode_errors": [],
        "action_alignment_episodes_over_atol": [],
        "image_decode_failures": 0,
        "malformed_jpeg_count": 0,
    }
    timestamp_paths = set()
    pose_like_paths = set()

    for split_name, episodes in split_episodes.items():
        for ep in episodes:
            path = os.path.join(index["hdf5_dir"], ep)
            with h5py.File(path, "r") as h:
                state = TC.read14(h, "state")
                action = TC.read14(h, "action")
                report = analyze_arrays(state, action, args.start_window)
                arrays = report.pop("_arrays")
                report["episode"] = ep
                report["split"] = split_name
                report["gradient_exposed_valid_frames"] = (
                    sum(1 for t in arrays["valid_t"] if (task, ep, int(t)) in seen_push)
                    if split_name == "train" else 0)
                report["is_gradient_exposed_episode"] = bool(
                    split_name == "train" and report["gradient_exposed_valid_frames"] > 0)
                report["file_size_bytes"] = int(os.path.getsize(path))
                report["file_mtime_ns"] = int(os.stat(path).st_mtime_ns)
                if report["action_next_state_alignment"]["rows_over_atol_1e-6"] > 0:
                    integrity["action_alignment_episodes_over_atol"].append(ep)

                valid_t = arrays["valid_t"]
                mask_seen = np.asarray([(task, ep, int(t)) in seen_push for t in valid_t],
                                       dtype=bool)
                row = {
                    "velocity": arrays["velocity"],
                    "action_delta": arrays["action_delta"],
                    "residual": arrays["residual"],
                    "arm_rms_velocity": arrays["arm_rms_velocity"],
                    "arm_rms_delta": arrays["arm_rms_delta"],
                    "arm_rms_residual": arrays["arm_rms_residual"],
                }
                if split_name == "train":
                    for name, mask in (("train_seen_all_valid", mask_seen),
                                       ("train_unseen_all_valid", ~mask_seen)):
                        group_rows[name].append({k: v[mask] for k, v in row.items()})
                    start_mask = valid_t < int(args.start_window)
                    group_rows["train_seen_startup"].append(
                        {k: v[mask_seen & start_mask] for k, v in row.items()})
                    group_rows["train_unseen_startup"].append(
                        {k: v[(~mask_seen) & start_mask] for k, v in row.items()})
                else:
                    group_rows["dev_all_valid"].append(row)
                    group_rows["dev_startup"].append(
                        {k: v[valid_t < int(args.start_window)] for k, v in row.items()})

                # Decode only the first startup window and exact TRAIN-SEEN image indices.
                sampled_indices = set(range(min(int(args.start_window), int(state.shape[0]))))
                if split_name == "train":
                    sampled_indices.update(seen_ep_t.get(ep, set()))
                sampled_indices = {t for t in sampled_indices if 0 <= t < state.shape[0]}
                image_report = _image_audit_for_episode(
                    h, ep, sampled_indices, args.start_window, cv2, M2.CAMERA_ORDER)
                report["image_audit"] = image_report
                for camera, block in image_report.items():
                    image_groups[split_name][camera].append(block)
                    integrity["image_decode_failures"] += int(block["decode_failure_count"])
                    integrity["malformed_jpeg_count"] += int(block["malformed_jpeg_count"])

                def visitor(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        low = name.lower()
                        if "timestamp" in low or "time_ns" in low or low.endswith("/time"):
                            timestamp_paths.add(name)
                        if any(token in low for token in
                               ("object_pose", "target_pose", "goal_pose", "object_position",
                                "target_position", "object_state", "success_state")):
                            pose_like_paths.add(name)
                h.visititems(visitor)
                episode_reports.append(report)

    def combine(name):
        rows = group_rows[name]
        nonempty = [r for r in rows if len(r["velocity"])]
        if not nonempty:
            return {"n_rows": 0}
        return aggregate_rows(nonempty)

    per_camera = {}
    for split_name, cameras in image_groups.items():
        per_camera[split_name] = {}
        for camera, blocks in cameras.items():
            per_camera[split_name][camera] = {
                "episodes": int(len(blocks)),
                "sampled_frames": int(sum(x["sampled_frame_count"] for x in blocks)),
                "decoded_frames": int(sum(x["decoded_frame_count"] for x in blocks)),
                "decode_failures": int(sum(x["decode_failure_count"] for x in blocks)),
                "malformed_jpeg_count": int(sum(x["malformed_jpeg_count"] for x in blocks)),
                "exact_duplicate_adjacent_jpeg_pairs_in_startup_windows": int(sum(
                    x["exact_duplicate_adjacent_jpeg_pairs_in_startup_window"] for x in blocks)),
                "episode_median_startup_adjacent_pixel_mae": distribution([
                    x["startup_adjacent_pixel_mae"]["q50"] for x in blocks
                    if x["startup_adjacent_pixel_mae"]["q50"] is not None]),
            }

    errors = []
    if integrity["action_alignment_episodes_over_atol"]:
        errors.append("one or more A[t] vs S[t+1] alignments exceed 1e-6")
    if integrity["image_decode_failures"] or integrity["malformed_jpeg_count"]:
        errors.append("one or more sampled RGB frames failed JPEG validation/decode")
    result = {
        "schema": "pusht-startup-data-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DATA_INTEGRITY_PASS" if not errors else "DATA_INTEGRITY_FAIL",
        "errors": errors,
        "no_model_loaded": True,
        "no_cuda_initialized": True,
        "no_optimizer_steps": 0,
        "scope": {
            "task": task,
            "start_window_frames": int(args.start_window),
            "start_window_seconds_at_25hz": float(args.start_window / 25.0),
            "stillness_thresholds_rad_per_sample": list(STILL_THRESHOLDS_RAD_PER_SAMPLE),
            "thresholds_are_exploratory_not_canonical_labels": True,
            "hdf5_root_basename": os.path.basename(os.path.normpath(args.hdf5_root)),
            "split_file_sha256": sha256_file(args.splits),
            "normalizer_file_sha256": sha256_file(args.normalizer),
            "checkpoint_manifest_sha256": exposure["checkpoint_manifest_sha256"],
            "checkpoint_global_step": exposure["global_step"],
            "sampler_position": exposure["position"],
            "sampler_pool_size": exposure["slot_pool_size"],
            "unique_consumed_frame_keys_all_tasks": exposure["unique_consumed_frame_keys"],
            "duplicate_consumed_frame_keys_all_tasks": exposure["duplicate_consumed_frame_keys"],
            "sampler_order_sha256": exposure["sampler_order_sha256"],
            "pushT_gradient_exposed_unique_frames": len(seen_push),
            "train_episodes": len(train_eps),
            "dev_episodes": len(dev_eps),
            "train_dev_episode_overlap": 0,
            "camera_order": list(M2.CAMERA_ORDER),
            "nominal_frequency_hz": 25,
            "alignment_atol": ALIGN_ATOL,
            "action_target_contract": "A[t] == S[t+1] for t=0..T-2",
            "residual_contract": "R[t]=(A[t]-S[t])-(S[t]-S[t-1]), valid t=2..T-2",
            "timestamps_found": sorted(timestamp_paths),
            "pose_or_goal_like_dataset_paths_found": sorted(pose_like_paths),
            "task_phase_ground_truth_available_from_known_contract": False,
            "time_alignment_limit": (
                "Same-index array lengths and A[t]/S[t+1] are checked; no per-stream timestamps "
                "were found, so wall-clock camera/state synchronization is not established."),
        },
        "frame_group_metrics": {
            name: combine(name) for name in (
                "train_seen_all_valid", "train_unseen_all_valid", "train_seen_startup",
                "train_unseen_startup", "dev_all_valid", "dev_startup")
        },
        "image_integrity_by_split_and_camera": per_camera,
        "integrity_counts": integrity,
        "episodes": sorted(episode_reports, key=lambda x: (x["split"], x["episode"])),
        "interpretation_guardrails": [
            "This report describes dataset/coverage; it does not prove the model learned or failed to learn.",
            "The checkpoint's 2000 consumed frames are not 2000 full-dataset passes.",
            "R24 TRAIN diagnostic frames must be divided into actual TRAIN-SEEN and TRAIN-UNSEEN groups.",
            "Image pixel differences are not object-motion labels and cannot replace privileged pose/contact labels.",
            "No training, model inference, or closed-loop execution occurred.",
        ],
        "tool": {
            "path": os.path.relpath(__file__, REPO),
            "sha256": sha256_file(__file__),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "h5py": h5py.__version__,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    print(json.dumps({
        "status": result["status"],
        "output": os.path.abspath(args.out),
        "checkpoint_global_step": exposure["global_step"],
        "pushT_gradient_exposed_unique_frames": len(seen_push),
        "frame_groups": {
            name: {
                "n_rows": result["frame_group_metrics"][name].get("n_rows", 0),
                "residual_arm_rms_q50": result["frame_group_metrics"][name].get(
                    "residual_arm_rms", {}).get("q50"),
            } for name in result["frame_group_metrics"]
        },
        "integrity_counts": integrity,
        "no_optimizer_steps": 0,
    }, indent=2))
    return 0 if result["status"] == "DATA_INTEGRITY_PASS" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Read-only M2 full-state checkpoint, e.g. step2000")
    parser.add_argument("--hdf5-root", required=True,
                        help="Dataset root accepted by training/m2_multitask_data.py::task_dir")
    parser.add_argument("--splits", default=os.path.join(REPO, "data/m2_splits.json"))
    parser.add_argument("--normalizer", default=os.path.join(
        REPO, "data/m2_task_normalizer_2task.json"))
    parser.add_argument("--start-window", type=int, default=DEFAULT_START_WINDOW,
                        help="Number of initial transitions to audit (default 25 = 1 second at 25 Hz)")
    parser.add_argument("--out", required=True,
                        help="New JSON output path; existing files are never overwritten")
    args = parser.parse_args()
    if args.start_window < 2:
        parser.error("--start-window must be >= 2")
    if os.path.exists(args.out):
        parser.error("output already exists; choose a new path to preserve prior evidence")
    return run_audit(args)


if __name__ == "__main__":
    sys.exit(main())
