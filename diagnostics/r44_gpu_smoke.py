#!/usr/bin/env python3
"""R44 one-step GPU smoke: measure real 3-camera + 42D + joint_delta_v1 training memory.

This is NOT formal training. It performs exactly one disposable optimizer step in RAM,
exports no model/checkpoint, and requires explicit smoke authorization plus a confirmed GPU
window. The process exits after writing an evidence report.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAINING = REPO / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

import r44_train as R44  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="R44 disposable one-step GPU smoke; no model export")
    ap.add_argument("--splits", default=str(REPO / "data/m2_splits.json"))
    ap.add_argument("--hdf5-root", default=os.environ.get("M2_HDF5_ROOT", ""))
    ap.add_argument("--feature-normalizer", default=os.environ.get("M2_NORMALIZER_JSON", ""))
    ap.add_argument("--delta-normalizer", default="")
    ap.add_argument("--dataset-manifest", default="")
    ap.add_argument("--preflight-report", default="",
                    help="same-commit R44 CPU preflight report.json; required for GPU smoke")
    ap.add_argument("--base-model", default="")
    ap.add_argument("--oft-root", default=os.environ.get(
        "PUSHT_OFT", os.path.join(os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime"),
                                  "openvla_oft_src")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-smoke-authorized", action="store_true")
    ap.add_argument("--gpu-window-confirmed", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--include-task-token", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--checkpoint-every-steps", type=int, default=5000)
    ap.add_argument("--warmup-fraction", type=float, default=0.01)
    ap.add_argument("--cams", type=int, default=3)
    ap.add_argument("--expected-train-samples", type=int, default=R44.DEFAULT_TRAIN_SAMPLES)
    ap.add_argument("--expected-dev-samples", type=int, default=R44.DEFAULT_DEV_SAMPLES)
    ap.add_argument("--expected-gpu-uuid", default=R44.EXPECTED_GPU_UUID)
    ap.add_argument("--expected-peak-gpu-memory-gb", type=float, default=24.0)
    ap.add_argument("--min-free-gpu-gb", type=float, default=12.0)
    ap.add_argument("--resource-mode", choices=("time-window", "shared-memory"), default="time-window")
    ap.add_argument("--accept-shared-gpu-risk", action="store_true")
    ap.add_argument("--max-process-memory-gb", type=float, default=None)
    ap.add_argument("--ledger-dir", default=os.environ.get("R15_LEDGER_DIR", None))
    ap.add_argument("--gpu-lock-wait-seconds", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--projector-lr-multiplier", type=float, default=0.1)
    ap.add_argument("--max-wall-seconds", type=float, default=1800.0)
    return ap.parse_args(argv)


def _compute_process_snapshot():
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        return {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as exc:
        return {"error": repr(exc)}


def _attest_base_model(base_model: str) -> dict:
    import r15_resume as RS

    path = Path(base_model).expanduser().resolve()
    identity = RS.attest_base_model(str(path))
    hashes = {}
    for shard in sorted(path.glob("*.safetensors")):
        before = R44.stat_identity(shard)
        digest = R44.sha256_file(shard)
        if R44.stat_identity(shard) != before:
            raise RuntimeError(f"base-model shard changed while hashing: {shard}")
        hashes[shard.name] = digest
    if not hashes:
        raise RuntimeError(f"no .safetensors base-model shards found under {path}")
    return {**identity, "_weight_shard_sha256": hashes}


def main(argv=None) -> int:
    args = parse_args(argv)

    # This gate runs before runtime imports, model loading, or any CUDA operation.
    if not args.gpu_smoke_authorized or not args.gpu_window_confirmed:
        print("BLOCKED_AUTHORIZATION: require --gpu-smoke-authorized and --gpu-window-confirmed", flush=True)
        return 2
    if args.cams != 3:
        print("BLOCKED_CONFIG: R44 smoke requires exactly three cameras", flush=True)
        return 2
    if not (math.isfinite(args.max_wall_seconds) and args.max_wall_seconds > 0
            and math.isfinite(args.gpu_lock_wait_seconds) and args.gpu_lock_wait_seconds >= 0):
        print("BLOCKED_CONFIG: wall limit must be finite/positive and lock wait finite/non-negative", flush=True)
        return 2
    if args.epochs != 5 or args.checkpoint_every_steps < 1:
        print("BLOCKED_CONFIG: R44 smoke contract requires 5 epochs and positive checkpoint cadence", flush=True)
        return 2
    if not (math.isfinite(args.warmup_fraction) and 0.0 < args.warmup_fraction < 1.0):
        print("BLOCKED_CONFIG: warmup fraction must be finite and in (0,1)", flush=True)
        return 2
    for name in ("hdf5_root", "feature_normalizer", "delta_normalizer", "dataset_manifest", "preflight_report", "base_model"):
        if not getattr(args, name):
            print(f"BLOCKED_CONFIG: --{name.replace('_', '-')} is required", flush=True)
            return 2
    if args.resource_mode == "shared-memory":
        if not args.accept_shared_gpu_risk or args.max_process_memory_gb is None:
            print("BLOCKED_SHARED_MODE: explicit shared-memory risk and allocator cap are required", flush=True)
            return 2
        if args.max_process_memory_gb < args.expected_peak_gpu_memory_gb:
            print("BLOCKED_SHARED_MODE: memory cap is below the expected peak; use a measured configuration", flush=True)
            return 2
    elif args.max_process_memory_gb is not None:
        print("BLOCKED_CONFIG: allocator cap is accepted only in shared-memory mode", flush=True)
        return 2

    out_dir = Path(args.out).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"BLOCKED_OUTPUT: output directory must be new or empty: {out_dir}", flush=True)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "r44-gpu-smoke/v1",
        "status": "RUNNING",
        "utc_start": R44.utcnow(),
        "host": platform.node(),
        "source_sha": R44.current_git_sha(),
        "smoke_optimizer_steps_max": 1,
        "formal_training": False,
        "weights_exported": False,
        "checkpoint_exported": False,
        "cuda_context_created": False,
        "resource_mode": args.resource_mode,
        "gradient_checkpointing": args.gradient_checkpointing,
        "expected_peak_gpu_memory_gb": args.expected_peak_gpu_memory_gb,
        "max_process_memory_gb": args.max_process_memory_gb,
    }
    gpu_lock = None
    smoke_start = time.monotonic()
    try:
        if report["source_sha"] == "UNKNOWN_RUNTIME_SHA":
            raise RuntimeError("cannot run GPU smoke without a verifiable Git commit SHA")
        gpu_lock = R44.acquire_gpu_lane_lock(args)
        modules = R44.load_runtime_modules(args)
        plan = R44.load_and_validate_cpu_plan(args, modules)
        preflight_info = R44.validate_preflight_report(
            args.preflight_report, plan, source_sha=report["source_sha"],
            expected_epochs=args.epochs, expected_budget=None,
            expected_lr=args.lr,
            expected_projector_lr_multiplier=args.projector_lr_multiplier,
            expected_warmup_fraction=args.warmup_fraction,
            expected_seed=args.seed,
            expected_checkpoint_every_steps=args.checkpoint_every_steps,
        )
        plan["preflight_report_path"] = preflight_info["path"]
        plan["preflight_report_sha256"] = preflight_info["sha256"]
        if time.monotonic() - smoke_start > args.max_wall_seconds:
            raise TimeoutError("smoke wall limit exceeded during CPU contract preflight")
        report["cpu_contracts"] = {
            "train_samples": plan["n_train"],
            "dev_samples": plan["n_dev"],
            "train_samples_sha256": plan["train_samples_sha256"],
            "dev_samples_sha256": plan["dev_samples_sha256"],
            "split_sha256": plan["split_sha256"],
            "feature_normalizer_sha256": plan["feature_normalizer_sha256"],
            "delta_normalizer_sha256": plan["delta_normalizer_sha256"],
            "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
            "preflight_report_sha256": preflight_info["sha256"],
        }
        report["base_model_identity"] = _attest_base_model(args.base_model)
        report["gpu_before"] = R44.gpu_snapshot(args)
        report["compute_processes_before"] = _compute_process_snapshot()

        import numpy as np
        import torch
        from training.r44_safe_stop import configure_torch_memory_fraction

        report["torch_version"] = str(torch.__version__)
        report["torch_cuda_version"] = str(torch.version.cuda)
        report["cuda_device_name"] = torch.cuda.get_device_name(0)
        report["cuda_context_created"] = True
        report["memory_cap"] = configure_torch_memory_fraction(args.max_process_memory_gb, device=0)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        proc, vla, projector, model_info = R44.build_model(args, modules)
        if time.monotonic() - smoke_start > args.max_wall_seconds:
            raise TimeoutError("smoke wall limit exceeded while loading the model")
        prompts = {t: modules[3].TASK_PROMPTS[t] for t in R44.TASKS}
        prompt_ids = {
            t: R44.build_prompt_ids(proc, t, modules, args.include_task_token)
            for t in R44.TASKS
        }
        model_info["prompt_text_sha256"] = {
            t: __import__("hashlib").sha256(
                f"What action should the robot take to {modules[3].build_prompt(t, include_token=args.include_task_token)['prompt']}?".encode()
            ).hexdigest() for t in R44.TASKS
        }
        model_info["prompt_token_ids_sha256"] = {
            t: R44.sha256_json(prompt_ids[t]) for t in R44.TASKS
        }

        D = modules[5]
        order = D.EpochSampler(plan["n_train"], args.seed, 0).order
        sample_index = int(order[0])
        task, episode, t = plan["train_samples"][sample_index]
        episode_path = plan["episode_paths"][f"{task}/{episode}"]
        TC, M2, TN = modules[2], modules[3], modules[4]

        prepare_start = time.perf_counter()
        states, actions = R44.read_state_action(episode_path, TC)
        row = M2.build_row(
            states, actions, int(t), TN.contract_for(plan["feature_payload"], task),
            task, episode=episode,
        )
        state_vec = np.asarray(states[int(t)], dtype=np.float32)
        action_vec = np.asarray(actions[int(t)], dtype=np.float32)
        lab, _delta = R44.build_runtime_labels(
            prompt_ids[task], state_vec, action_vec, task, plan["delta_payload"], modules,
        )
        pixel_values = R44.episode_images(proc, episode_path, int(t), args.cams, modules)
        proprio = torch.from_numpy(np.asarray(row["features"], dtype=np.float32)).reshape(1, 42).to("cuda")
        input_ids = torch.tensor([lab["input_ids"]], dtype=torch.long, device="cuda")
        labels = torch.tensor([lab["labels"]], dtype=torch.long, device="cuda")
        torch.cuda.synchronize()
        prepare_seconds = time.perf_counter() - prepare_start

        if time.monotonic() - smoke_start > args.max_wall_seconds:
            raise TimeoutError("smoke wall limit exceeded before the disposable optimizer step")
        trainable_lora = [p for p in vla.parameters() if p.requires_grad]
        optimizer, scheduler, scheduler_info = R44.make_optimizer_and_scheduler(
            vla, projector, args, total_steps=plan["n_train"] * args.epochs
        )
        vla.train()
        projector.train()
        baseline_allocated_gb = float(torch.cuda.memory_allocated()) / (1024**3)
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pixel_values=pixel_values,
                labels=labels,
                proprio=proprio,
                proprio_projector=projector,
            )
            loss = out.loss
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss in disposable GPU smoke step")
        loss.backward()
        params = trainable_lora + list(projector.parameters())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        if not math.isfinite(grad_norm):
            raise FloatingPointError("non-finite gradient norm in disposable GPU smoke step")
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_start

        # Measure the exact resumable state serialization in RAM. No checkpoint file is written.
        checkpoint_state = R44.checkpoint_state(
            torch, vla, projector, optimizer, scheduler,
            epoch=0, cursor=1, order=order, global_step=1,
            plan=plan, base_identity=report["base_model_identity"], model_info=model_info,
            prompts=prompts, prompt_ids=prompt_ids,
            run_state={"schema": "r44-run-state/v1", "dev_history": [],
                       "best_dev_mae": None, "stale_epochs": 0,
                       "completed_epochs": 0, "global_step": 1},
        )
        serialization_start = time.perf_counter()
        checkpoint_buffer = io.BytesIO()
        torch.save(checkpoint_state, checkpoint_buffer)
        checkpoint_state_bytes = int(len(checkpoint_buffer.getbuffer()))
        checkpoint_serialization_seconds = time.perf_counter() - serialization_start
        checkpoint_buffer.close()
        del checkpoint_state
        torch.cuda.synchronize()

        peak_allocated_gb = float(torch.cuda.max_memory_allocated()) / (1024**3)
        peak_reserved_gb = float(torch.cuda.max_memory_reserved()) / (1024**3)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_after_gb = float(free_bytes) / (1024**3)
        peak_within_expected = peak_allocated_gb <= args.expected_peak_gpu_memory_gb
        peak_within_cap = (
            None if args.max_process_memory_gb is None
            else peak_reserved_gb <= args.max_process_memory_gb
        )
        memory_pass = (
            peak_within_expected
            and peak_within_cap is not False
            and free_after_gb >= args.min_free_gpu_gb
        )
        checkpoint_count_estimate = (
            math.ceil(plan["n_train"] * args.epochs / args.checkpoint_every_steps)
            + args.epochs + 1
        )
        estimated_checkpoint_storage_gb = (
            checkpoint_state_bytes * checkpoint_count_estimate / (1024**3)
        )

        current_sha = R44.current_git_sha()
        if current_sha != report["source_sha"]:
            raise RuntimeError("repository HEAD changed during GPU smoke")
        report.update({
            "status": "PASS_SMOKE" if memory_pass else "FAIL_GPU_SMOKE",
            "utc_end": R44.utcnow(),
            "source_sha": current_sha,
            "sample": {"index": sample_index, "task": task, "episode": episode, "t": int(t)},
            "optimizer_steps_executed": 1,
            "loss": float(loss.detach().float().cpu()),
            "grad_norm": grad_norm,
            "trainable_lora_parameter_count": sum(p.numel() for p in trainable_lora),
            "projector_parameter_count": sum(p.numel() for p in projector.parameters()),
            "data_and_preprocess_seconds": prepare_seconds,
            "forward_backward_optimizer_seconds": step_seconds,
            "measured_peak_allocated_gb": peak_allocated_gb,
            "measured_peak_reserved_gb": peak_reserved_gb,
            "baseline_allocated_gb_after_model_load": baseline_allocated_gb,
            "global_free_memory_after_step_gb": free_after_gb,
            "global_total_memory_gb": float(total_bytes) / (1024**3),
            "peak_within_expected_peak": peak_within_expected,
            "peak_within_allocator_cap": peak_within_cap,
            "memory_budget_verdict": "PASS_WITHIN_EXPECTED_PEAK" if memory_pass else "FAIL_MEMORY_GATE",
            "checkpoint_size_status": "MEASURED",
            "checkpoint_state_bytes": checkpoint_state_bytes,
            "checkpoint_serialization_seconds": checkpoint_serialization_seconds,
            "checkpoint_serialization_mode": "in-memory only; no checkpoint file written",
            "checkpoint_every_steps": int(args.checkpoint_every_steps),
            "epochs_max": int(args.epochs),
            "checkpoint_count_estimate": int(checkpoint_count_estimate),
            "estimated_checkpoint_storage_gb": estimated_checkpoint_storage_gb,
            "scheduler_info": scheduler_info,
            "model_info": model_info,
            "gpu_after_model_cleanup": None,
            "compute_processes_after": None,
        })
        # The disposable optimizer update and all model state are intentionally discarded.
        del optimizer, out, vla, projector, proc, pixel_values, proprio, input_ids, labels
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        report["gpu_after_model_cleanup"] = R44.gpu_snapshot(args, enforce_headroom=False)
        report["compute_processes_after"] = _compute_process_snapshot()
        report["cuda_peak_includes_one_adamw_step"] = True
        report["persistent_weight_changes"] = False
        report["persistent_optimizer_steps"] = 0
        if not memory_pass:
            report["error"] = (
                "measured peak/global-free-memory gate failed; do not launch formal training "
                "with this memory configuration"
            )
        R44.atomic_json(out_dir / "report.json", report)
        R44.atomic_json(out_dir / "SHA256SUMS.json", {
            "report.json": R44.sha256_file(out_dir / "report.json"),
        })
        print(json.dumps(report, indent=2, sort_keys=True, default=str), flush=True)
        return 0 if memory_pass else 3
    except BaseException as exc:
        report.update({
            "status": "FAIL_GPU_SMOKE",
            "utc_end": R44.utcnow(),
            "error_type": type(exc).__name__,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "compute_processes_at_failure": _compute_process_snapshot(),
        })
        try:
            R44.atomic_json(out_dir / "report.json", report)
            R44.atomic_json(out_dir / "SHA256SUMS.json", {
                "report.json": R44.sha256_file(out_dir / "report.json"),
            })
        except Exception:
            pass
        print(json.dumps(report, indent=2, sort_keys=True, default=str), file=sys.stderr, flush=True)
        if isinstance(exc, KeyboardInterrupt):
            return 130
        return 3
    finally:
        if gpu_lock is not None:
            gpu_lock.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
