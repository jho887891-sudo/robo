#!/usr/bin/env python3
"""Inference-only GPU smoke for the R42 official Model call path.

This script never starts a simulator/client, sends an action to a robot, trains, writes
checkpoints, or modifies HDF5. It uses deterministic synthetic RGB/state input and writes
one JSON report outside the repository. The forced-rate phase replaces only the token
decoder output after a real model load, so it tests the official rate-limit path without
misrepresenting forced tokens as model predictions.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_ROOT = os.path.join(ROOT, "code", "xpolicylab", "policy")
POLICY_DIR = os.path.join(POLICY_ROOT, "OpenVLA_R42")
PURE_DIR = os.path.join(POLICY_ROOT, "OpenVLA_Pure")
for _path in (POLICY_ROOT, POLICY_DIR, PURE_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import r42_adapter as R  # noqa: E402
import OpenVLA_R42.model as M  # noqa: E402

CAMERAS = tuple(R.CAMERA_ORDER)
REPORT_SCHEMA = "r42-rate-limit-gpu-smoke/v1"


class SmokeBlocked(RuntimeError):
    """A precondition was not met; no real inference should be attempted."""


class SmokeFailure(RuntimeError):
    """A required smoke assertion failed."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path, chunk_size=4 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args):
    try:
        result = subprocess.run(
            ["git"] + list(args), cwd=ROOT, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return None


def record(report, name, ok, detail=None):
    row = {"name": str(name), "result": "PASS" if ok else "FAIL"}
    if detail is not None:
        row["detail"] = detail
    report["checks"].append(row)
    print("[%s] %s %s" % (row["result"], name,
                           json.dumps(detail, sort_keys=True, default=str)
                           if detail is not None else ""), flush=True)
    return bool(ok)


def require(report, name, ok, detail=None):
    if not record(report, name, ok, detail):
        raise SmokeFailure(name)


def resolve_path(value):
    if not value:
        return None
    value = os.path.expanduser(os.path.expandvars(str(value)))
    return os.path.abspath(value if os.path.isabs(value) else os.path.join(ROOT, value))


def artifact_record(path):
    return {
        "path": path,
        "exists": bool(path and os.path.isfile(path)),
        "size_bytes": os.path.getsize(path) if path and os.path.isfile(path) else None,
        "sha256": sha256_file(path) if path and os.path.isfile(path) else None,
    }


def save_report(path, report):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def synthetic_observation(state, control_hz):
    """Deterministic synthetic RGB frames. No camera or environment is opened."""
    h, w = 480, 640
    images = {}
    for camera, level in zip(CAMERAS, (96, 127, 160)):
        image = np.full((h, w, 3), level, dtype=np.uint8)
        # Small deterministic channel gradients exercise ordinary RGB preprocessing.
        image[:, :, 0] = np.arange(w, dtype=np.uint16)[None, :] % 256
        image[:, :, 1] = (np.arange(h, dtype=np.uint16)[:, None] + level) % 256
        images[camera] = np.ascontiguousarray(image)
    return {
        "vision": images,
        "state": np.asarray(state, dtype=np.float32).copy(),
        "additional_info": {"frequency": float(control_hz)},
        "env_idx": 0,
        "timestamp": time.time(),
        "camera_names": list(CAMERAS),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deploy-yml",
        default=os.path.join(POLICY_DIR, "deploy.yml"),
        help="R42 deployment YAML used by the real model path")
    parser.add_argument("--out", required=True,
                        help="new JSON report path outside the repository; existing files are refused")
    parser.add_argument("--min-free-gb", type=float, default=24.0,
                        help="minimum free CUDA VRAM required before loading weights")
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = os.path.abspath(os.path.expanduser(args.out))
    if os.path.exists(out_path):
        print("BLOCKED: refusing to overwrite existing report: " + out_path, file=sys.stderr)
        return 2
    if os.path.commonpath([ROOT, out_path]) == ROOT:
        print("BLOCKED: --out must be outside the repository to keep the checkout read-only",
              file=sys.stderr)
        return 2
    if not math.isfinite(args.min_free_gb) or args.min_free_gb <= 0:
        print("BLOCKED: --min-free-gb must be finite and positive", file=sys.stderr)
        return 2

    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "source": {
            "repo_root": ROOT,
            "git_sha": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("branch", "--show-current"),
            "git_worktree_status": git_value("status", "--short"),
        },
        "scope": {
            "real_model_inference_steps": 1,
            "simulator_started": False,
            "robot_connected_or_commanded": False,
            "training_or_optimizer_steps": 0,
            "checkpoint_or_hdf5_mutation": False,
            "real_observations_used": False,
            "synthetic_rgb_state_used": True,
        },
        "checks": [],
        "artifacts_before": {},
        "artifacts_after": {},
    }
    exit_code = 1
    model = None
    torch = None
    server = None
    original_act = None
    original_decode = None

    try:
        deploy_path = resolve_path(args.deploy_yml)
        require(report, "deploy_yaml_exists", bool(deploy_path and os.path.isfile(deploy_path)),
                {"path": deploy_path})
        try:
            import yaml
        except ImportError as exc:
            raise SmokeBlocked("PyYAML is unavailable: %s" % exc)
        with open(deploy_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise SmokeBlocked("deploy YAML did not decode to a mapping")
        report["deploy_yaml"] = deploy_path
        report["deployment_config"] = {
            "policy_name": config.get("policy_name"),
            "task_name": config.get("task_name"),
            "device": config.get("device"),
            "control_hz": config.get("control_hz"),
            "checkpoint_dir": resolve_path(config.get("checkpoint_dir")),
            "base_model_path": resolve_path(config.get("base_model_path")),
        }
        if str(config.get("device", "")).lower() != "cuda":
            raise SmokeBlocked("the frozen deployment config does not select CUDA")
        control_hz = float(config.get("control_hz", 0.0))
        if not math.isfinite(control_hz) or abs(control_hz - 25.0) > 1e-6:
            raise SmokeBlocked("expected frozen R15 control_hz=25, got %r" % control_hz)

        try:
            import torch as _torch
            torch = _torch
        except ImportError as exc:
            raise SmokeBlocked("PyTorch is unavailable: %s" % exc)
        cuda_available = bool(torch.cuda.is_available())
        require(report, "cuda_available", cuda_available, {"available": cuda_available})
        if not cuda_available:
            raise SmokeBlocked("CUDA is unavailable; no model weights were loaded")
        device_name = torch.cuda.get_device_name(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = float(free_bytes) / 1e9
        total_gb = float(total_bytes) / 1e9
        report["gpu"] = {
            "device_index": 0,
            "device_name": device_name,
            "free_vram_gb_before_load": round(free_gb, 3),
            "total_vram_gb": round(total_gb, 3),
            "minimum_free_gb": float(args.min_free_gb),
        }
        record(report, "minimum_free_vram", free_gb >= args.min_free_gb,
               report["gpu"])
        if free_gb < args.min_free_gb:
            raise SmokeBlocked(
                "free VRAM %.3f GB is below the %.3f GB gate; nothing was loaded"
                % (free_gb, args.min_free_gb))

        checkpoint = resolve_path(config.get("checkpoint_dir"))
        base_model = resolve_path(config.get("base_model_path"))
        lora_dir = resolve_path(config.get("adapter_dir")) or (
            os.path.join(checkpoint, "lora_adapter") if checkpoint else None)
        projector = resolve_path(config.get("proprio_projector_path")) or (
            os.path.join(checkpoint, "proprio_projector_42.pt") if checkpoint else None)
        required_paths = {
            "checkpoint_manifest": os.path.join(checkpoint, "MANIFEST.json") if checkpoint else None,
            "lora_weights": os.path.join(lora_dir, "adapter_model.safetensors") if lora_dir else None,
            "lora_config": os.path.join(lora_dir, "adapter_config.json") if lora_dir else None,
            "proprio_projector": projector,
            "base_model_config": os.path.join(base_model, "config.json") if base_model else None,
            "frozen_normalizer_contract": os.path.join(ROOT, "evidence", "r9_gate_20261020",
                                                       "frozen_contract_64train.json"),
            "physical_bounds": os.path.join(ROOT, "data", "push_T_bounds.npz"),
            "episode_split": os.path.join(ROOT, "data", "split80.json"),
        }
        missing = [name for name, path in required_paths.items()
                   if not path or not os.path.isfile(path)]
        record(report, "required_local_artifacts_present", not missing,
               {"missing": missing, "paths": required_paths})
        if missing:
            raise SmokeBlocked("required local checkpoint/base/contract files are missing: %s"
                               % ", ".join(missing))
        report["artifacts_before"] = {
            name: artifact_record(path) for name, path in required_paths.items()
            if name != "physical_bounds" and name != "episode_split"
        }
        report["artifact_hash_policy"] = (
            "Hashes small checkpoint/config artifacts and MANIFEST; does not re-hash the full "
            "multi-GB base-model weight shards. No source/checkpoint file is written.")

        # Metadata bridge validation happens in Model.__init__ before the backend is loaded.
        model = M.Model(config)
        meta = model.metadata
        limits = meta.get("safety_limits") or {}
        lo = np.asarray(limits.get("amin"), dtype=np.float32).reshape(-1)
        hi = np.asarray(limits.get("amax"), dtype=np.float32).reshape(-1)
        rate = np.asarray(limits.get("max_rate"), dtype=np.float32).reshape(-1)
        require(report, "metadata_has_valid_14d_rate_contract",
                lo.shape == hi.shape == rate.shape == (14,)
                and np.isfinite(lo).all() and np.isfinite(hi).all()
                and np.isfinite(rate).all() and np.all(hi > lo) and np.all(rate >= 0),
                {"amin_dim": int(lo.size), "amax_dim": int(hi.size), "max_rate_dim": int(rate.size)})
        require(report, "metadata_rate_and_control_period_frozen",
                abs(float(model.control_dt) - 0.04) <= 1e-6
                and abs(float(model.control_hz) - 25.0) <= 1e-6,
                {"control_hz": model.control_hz, "control_dt": model.control_dt,
                 "max_rate14": rate.tolist()})
        report["metadata_contract"] = {
            "task_id": meta.get("task_id"),
            "codec_mode": meta.get("codec_mode"),
            "action_token_window": meta.get("action_token_window"),
            "proprio_dim": meta.get("proprio_dim"),
            "normalizer_sha256": meta.get("normalizer_sha256"),
            "lora_sha256": meta.get("lora_sha256"),
            "projector_sha256": meta.get("projector_sha256"),
            "max_rate14": rate.tolist(),
            "amin14": lo.tolist(),
            "amax14": hi.tolist(),
            "model_contract": model.contract(),
        }

        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        model.warmup()
        torch.cuda.synchronize()
        report["model_load_seconds"] = round(time.perf_counter() - load_started, 3)
        server = model._server
        require(report, "real_torch_backend_loaded",
                server is not None and type(server.backend).__name__ == "_TorchBackend",
                {"backend_type": type(server.backend).__name__ if server else None})
        server.assert_frozen()
        require(report, "all_loaded_parameters_frozen", True,
                {"backend_type": type(server.backend).__name__})
        require(report, "real_processor_and_prompt_loaded",
                server.processor is not None and len(server.prompt_ids()) > 0,
                {"processor_type": type(server.processor).__name__ if server.processor else None,
                 "prompt_tokens": len(server.prompt_ids()) if server else 0})

        calls = []
        original_act = server.act

        def traced_act(images, state, env_idx=0, timestamp=None, prev_cmd=None, dt=None,
                       record=False):
            out = original_act(images, state, env_idx=env_idx, timestamp=timestamp,
                               prev_cmd=prev_cmd, dt=dt, record=record)
            calls.append({
                "prev_cmd": None if prev_cmd is None else np.asarray(prev_cmd, np.float32).copy(),
                "dt": dt,
                "action14": np.asarray(out["action"], np.float32).reshape(-1).copy(),
                "info": dict(out.get("info") or {}),
                "latency_ms": dict(out.get("latency_ms") or {}),
                "ids": np.asarray(out.get("ids", []), np.int64).reshape(-1).copy(),
            })
            return out

        server.act = traced_act
        state = ((lo + hi) * 0.5).astype(np.float32)
        observation = synthetic_observation(state, control_hz)
        real_started = time.perf_counter()
        model.update_obs(observation)
        payload = model.get_action()
        torch.cuda.synchronize()
        real_seconds = time.perf_counter() - real_started
        require(report, "one_real_model_forward_completed",
                len(calls) == 1 and len(calls[0]["ids"]) == 14,
                {"server_calls": len(calls),
                 "token_count": len(calls[0]["ids"]) if calls else 0})
        call = calls[-1]
        action = np.asarray(payload["action14"], dtype=np.float32).reshape(-1)
        info = call["info"]
        max_step = rate * float(model.control_dt)
        require(report, "real_inference_action_is_finite_14d",
                action.shape == (14,) and np.isfinite(action).all(),
                {"shape": list(action.shape), "all_finite": bool(np.isfinite(action).all())})
        require(report, "real_inference_tokens_are_14_legal_ids",
                len(payload["ids"]) == 14
                and all(R.CODEC_WINDOW[0] <= int(token) <= R.CODEC_WINDOW[1]
                        for token in payload["ids"]),
                {"n_action_tokens": payload.get("n_action_tokens"),
                 "token_min": min(payload["ids"]) if payload["ids"] else None,
                 "token_max": max(payload["ids"]) if payload["ids"] else None})
        require(report, "real_inference_decoder_received_dt",
                call["dt"] is not None and abs(float(call["dt"]) - model.control_dt) <= 1e-6
                and abs(float(info.get("dt", -1.0)) - model.control_dt) <= 1e-6,
                {"passed_dt": call["dt"], "decoder_info_dt": info.get("dt"),
                 "expected_dt": model.control_dt})
        require(report, "real_inference_action_satisfies_physical_bounds",
                np.all(action >= lo - 1e-5) and np.all(action <= hi + 1e-5),
                {"min_action": float(action.min()), "max_action": float(action.max())})
        require(report, "real_inference_action_satisfies_rate_bound",
                call["prev_cmd"] is not None
                and np.all(np.abs(action - call["prev_cmd"]) <= max_step + 1e-5),
                {"rate_limited_dims": info.get("rate_limited"),
                 "max_observed_delta": float(np.max(np.abs(action - call["prev_cmd"]))),
                 "max_allowed_delta": max_step.tolist()})
        require(report, "real_inference_decoder_has_no_safety_conflict",
                info.get("unsafe") is None and not info.get("rate_conflict_dims"),
                {"unsafe": info.get("unsafe"),
                 "rate_conflict_dims": info.get("rate_conflict_dims")})
        report["real_inference"] = {
            "input_kind": "synthetic RGB/state, not a real observation",
            "wall_seconds": round(real_seconds, 3),
            "action14": action.tolist(),
            "previous_command14": call["prev_cmd"].tolist(),
            "decoder_info": info,
            "latency_ms": call["latency_ms"],
            "n_action_tokens": len(payload["ids"]),
            "token_ids": [int(x) for x in payload["ids"]],
        }

        # Force one extreme legal token vector after the real model is loaded. This is not a
        # model prediction: it is a deterministic integration stress test of the same official
        # Model -> server.act -> decode_with_metadata path.
        model.reset()
        normalizer = R.normalizer_for_task(meta["normalizer"], meta.get("task_id"))
        q01 = np.asarray(normalizer["q01"], dtype=np.float32).reshape(14)
        q99 = np.asarray(normalizer["q99"], dtype=np.float32).reshape(14)
        candidates = []
        for dim in range(14):
            if dim in R.GRIPPER_DIMS:
                continue
            width = float(hi[dim] - lo[dim])
            step = float(rate[dim] * model.control_dt)
            if width * 0.25 <= step + 1e-5:
                continue
            endpoint = float(q99[dim] if abs(float(q99[dim])) >= abs(float(q01[dim]))
                             else q01[dim])
            candidates.append((abs(endpoint), dim, endpoint, width, step))
        if not candidates:
            raise SmokeFailure("no non-gripper dimension has room for a rate-limit stress case")
        _, dim, endpoint, width, step = max(candidates, key=lambda row: row[0])
        offset = min(width * 0.25, max(0.1, step * 4.0 + 0.01))
        if offset <= step + 1e-5:
            raise SmokeFailure("could not construct an in-bounds prior command beyond max_rate*dt")
        direction = 1.0 if endpoint >= 0.0 else -1.0
        previous = state.copy()
        previous[dim] = state[dim] - direction * offset
        if not (lo[dim] <= previous[dim] <= hi[dim]):
            raise SmokeFailure("synthetic prior command fell outside physical bounds")
        forced_bins = [
            255 if abs(float(q99[i])) >= abs(float(q01[i])) else 0
            for i in range(14)
        ]
        forced_ids = [int(R.CODEC_WINDOW[0] + value) for value in forced_bins]
        original_decode = server._decode_tokens
        server._decode_tokens = lambda pixel_values, feat: (
            list(forced_ids), [{"smoke_forced_token": True} for _ in forced_ids])
        model._rate_prev_cmd[0] = previous.copy()
        model.update_obs(synthetic_observation(state, control_hz))
        forced_payload = model.get_action()
        torch.cuda.synchronize()
        forced_call = calls[-1]
        forced_action = np.asarray(forced_payload["action14"], dtype=np.float32).reshape(-1)
        forced_info = forced_call["info"]
        forced_delta = np.abs(forced_action - previous)
        require(report, "forced_token_rate_branch_activated",
                int(forced_info.get("rate_limited", 0)) > 0,
                {"rate_limited_dims": forced_info.get("rate_limited"),
                 "forced_dimension": dim, "forced_endpoint_residual": endpoint,
                 "synthetic_previous_command_offset": offset})
        require(report, "forced_token_action_respects_rate_bound",
                np.all(forced_delta <= max_step + 1e-5),
                {"max_observed_delta": float(forced_delta.max()),
                 "max_allowed_delta": max_step.tolist()})
        require(report, "forced_token_action_respects_physical_bounds",
                np.all(forced_action >= lo - 1e-5) and np.all(forced_action <= hi + 1e-5),
                {"min_action": float(forced_action.min()),
                 "max_action": float(forced_action.max())})
        require(report, "forced_token_decoder_has_no_safety_conflict",
                forced_info.get("unsafe") is None and not forced_info.get("rate_conflict_dims"),
                {"unsafe": forced_info.get("unsafe"),
                 "rate_conflict_dims": forced_info.get("rate_conflict_dims")})
        report["forced_rate_path_test"] = {
            "is_model_prediction": False,
            "uses_loaded_real_model_instance": True,
            "decoder_output_overridden_for_test": True,
            "forced_bins14": forced_bins,
            "forced_token_ids14": forced_ids,
            "forced_dimension": int(dim),
            "endpoint_residual": float(endpoint),
            "synthetic_previous_command14": previous.tolist(),
            "measured_state14": state.tolist(),
            "action14": forced_action.tolist(),
            "decoder_info": forced_info,
            "max_allowed_delta14": max_step.tolist(),
        }

        # Restore all instance hooks before hashing artifacts or cleaning up.
        server._decode_tokens = original_decode
        original_decode = None
        server.act = original_act
        original_act = None
        torch.cuda.synchronize()
        report["gpu"]["free_vram_gb_after_smoke"] = round(
            float(torch.cuda.mem_get_info()[0]) / 1e9, 3)
        report["gpu"]["peak_allocated_gb"] = round(
            float(torch.cuda.max_memory_allocated()) / 1e9, 3)
        report["artifacts_after"] = {
            name: artifact_record(path) for name, path in required_paths.items()
            if name != "physical_bounds" and name != "episode_split"
        }
        require(report, "checkpoint_and_config_artifacts_unchanged",
                report["artifacts_before"] == report["artifacts_after"],
                {"before": report["artifacts_before"], "after": report["artifacts_after"]})
        report["completed_at_utc"] = utc_now()
        report["status"] = "PASS"
        exit_code = 0

    except SmokeBlocked as exc:
        report["status"] = "BLOCKED"
        report["blocked_reason"] = str(exc)
        exit_code = 2
        print("BLOCKED: " + str(exc), file=sys.stderr, flush=True)
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc()[-12000:],
        }
        exit_code = 1
        print("FAIL: %s: %s" % (type(exc).__name__, exc), file=sys.stderr, flush=True)
    finally:
        try:
            if server is not None and original_decode is not None:
                server._decode_tokens = original_decode
            if server is not None and original_act is not None:
                server.act = original_act
        except Exception:
            pass
        try:
            if model is not None:
                model._server = None
                model._inputs = None
            if torch is not None and torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass
        report["completed_at_utc"] = report.get("completed_at_utc") or utc_now()
        report["exit_code"] = int(exit_code)
        report["report_path"] = out_path
        try:
            save_report(out_path, report)
            print("REPORT: " + out_path, flush=True)
            print("FINAL_STATUS: " + str(report["status"]), flush=True)
        except Exception as exc:
            print("FATAL: could not persist report: %s" % exc, file=sys.stderr, flush=True)
            return 4
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
