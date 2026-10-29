#!/usr/bin/env python3
"""R46 real OpenVLA five-sample smoke child; only the supervisor may launch this."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _read_json(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _build_r44_argv(forwarded: list[str], *, manifest_path: Path,
                    output_path: Path, approved_gpu_uuid: str) -> list[str]:
    """Build only options accepted by R44; the wall limit belongs to this child."""
    return list(forwarded) + [
        "--dataset-manifest", str(manifest_path),
        "--gpu-window-confirmed",
        "--out", str(output_path),
        "--expected-gpu-uuid", approved_gpu_uuid,
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--approval-file", required=True)
    ap.add_argument("--source-sha", required=True)
    ap.add_argument("--sample-plan", required=True)
    ap.add_argument("--dataset-manifest", required=True)
    ap.add_argument("--model-identity", required=True)
    ap.add_argument("--approved-gpu-uuid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-wall-seconds", required=True, type=float)
    outer, forwarded = ap.parse_known_args(argv)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    # This boundary runs before importing R44, torch, Transformers, PEFT, or OpenCV.
    parent_text = os.environ.get("R46_SUPERVISOR_PID", "")
    try:
        parent_pid = int(parent_text)
    except ValueError:
        print("BLOCKED_PARENT: missing supervisor identity", file=sys.stderr)
        return 2
    if parent_pid != os.getppid() or os.name != "posix" or os.getpgid(0) != os.getpid():
        print("BLOCKED_PARENT: child is not in the supervisor-owned process group", file=sys.stderr)
        return 2
    if os.environ.get("CUDA_VISIBLE_DEVICES") != outer.approved_gpu_uuid:
        print("BLOCKED_GPU_IDENTITY: child GPU visibility differs from approval", file=sys.stderr)
        return 2
    if not math.isfinite(outer.max_wall_seconds) or not 0 < outer.max_wall_seconds <= 1800:
        print("BLOCKED_BUDGET: invalid wall limit", file=sys.stderr)
        return 2

    phase = "approval_and_admission"
    actual_sha = None
    normalized = None
    sample_rows = []
    cuda_device_identity = None
    try:
        from diagnostics.r46_smoke_supervisor import (
            _read_json as read_object, probe_gpu, sha256_file, validate_approval,
            model_identity_sha_list, verify_model_identity,
        )
        from diagnostics.r46_smoke_contract import validate_sample_plan, validate_smoke_report
        actual_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                    capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        if actual_sha != outer.source_sha:
            raise ValueError("source SHA changed before child execution")
        approval = read_object(outer.approval_file, "approval")
        sample_path = Path(outer.sample_plan).resolve()
        manifest_path = Path(outer.dataset_manifest).resolve()
        identity_path = Path(outer.model_identity).resolve()
        identity = read_object(str(identity_path), "model identity")
        model_file_list = model_identity_sha_list(identity)
        expected_hashes = {
            "approved_original_manifest_sha256": sha256_file(manifest_path),
            "approved_smoke_samples_sha256": sha256_file(sample_path),
            "approved_model_sha_list": model_file_list,
        }
        normalized = validate_approval(
            approval, source_sha=actual_sha, requested_wall_seconds=outer.max_wall_seconds,
            expected_hashes=expected_hashes,
        )
        if normalized["approved_gpu_uuid"] != outer.approved_gpu_uuid:
            raise ValueError("approved GPU UUID mismatch")
        admission_before_cuda = probe_gpu(outer.approved_gpu_uuid,
                                          float(approval["approved_min_total_gib"]),
                                          float(approval["approved_min_free_gib"]))

        # Runtime-heavy imports remain behind approval, identity and resource checks.
        if str(REPO / "training") not in sys.path:
            sys.path.insert(0, str(REPO / "training"))
        import r44_train as R44
        args = R44.parse_args(_build_r44_argv(
            forwarded,
            manifest_path=manifest_path,
            output_path=Path(outer.out).resolve(),
            approved_gpu_uuid=outer.approved_gpu_uuid,
        ))
        if args.cams != 3 or not args.base_model or not args.dataset_manifest:
            raise ValueError("R44 runtime args must supply the base model, strict manifest, and 3 cameras")
        if args.seed != 0 or args.include_task_token or args.gradient_checkpointing \
                or args.formal_training_authorized or args.resume_from or args.preflight_only \
                or args.epochs != 5 or args.min_epochs != 2 or args.lr != 5e-4 \
                or args.projector_lr_multiplier != 0.1:
            raise ValueError("forwarded runtime options differ from frozen R46 smoke defaults")
        if Path(args.dataset_manifest).resolve() != manifest_path:
            raise ValueError("forwarded dataset manifest differs from approved manifest")
        modules = R44.load_runtime_modules(args)
        cpu_plan = R44.load_and_validate_cpu_plan(args, modules)
        phase = "sample_contract"
        split_doc = _read_json(args.splits)
        manifest_doc = _read_json(str(manifest_path))
        sample_doc = _read_json(str(sample_path))
        from diagnostics.r46_smoke_contract import sha256_file as file_sha
        split_sha = file_sha(Path(args.splits))
        from diagnostics.r46_smoke_contract import SELECTION_SOURCE_SHA
        sample_check = validate_sample_plan(
            sample_doc, split_doc=split_doc, manifest=manifest_doc,
            split_sha256=split_sha, manifest_sha256=expected_hashes["approved_original_manifest_sha256"],
            hdf5_root=Path(args.hdf5_root), source_sha=SELECTION_SOURCE_SHA, verify_files=True,
        )

        phase = "base_model_shard_attestation"
        actual_shards = verify_model_identity(identity, Path(args.base_model))
        gpu_lock = R44.acquire_gpu_lane_lock(args)
        phase = "cuda_initialization"
        import torch
        import numpy as np
        if torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA_VISIBLE_DEVICES must expose exactly the one approved GPU")
        props = torch.cuda.get_device_properties(0)
        cuda_device_identity = {
            "visible_device_count": torch.cuda.device_count(),
            "visible_device_index": 0,
            "name": str(props.name),
            "approved_uuid_from_cuda_visible_devices": outer.approved_gpu_uuid,
            "reported_uuid": str(getattr(props, "uuid", "unavailable")),
        }
        if getattr(props, "uuid", None) is not None and str(props.uuid) != outer.approved_gpu_uuid:
            raise RuntimeError("CUDA runtime selected a GPU UUID different from the approved UUID")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.synchronize()
        load_start = time.perf_counter()
        phase = "openvla_model_load"
        proc, vla, projector, model_info = R44.build_model(args, modules)
        torch.cuda.synchronize()
        model_load_s = time.perf_counter() - load_start
        model_memory_baseline = {
            "allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        }
        model_info.update({
            "base_model": "OpenVLA-7B",
            "lora_config": {"r": 32, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
            "projector_input_dim": 42,
            "camera_order": list(modules[3].CAMERA_ORDER),
        })
        prompt_ids = R44.build_prompt_ids(proc, "push_T", modules, args.include_task_token)
        paths = cpu_plan["episode_paths"]
        samples = sample_doc["samples"]
        if len(samples) != 5:
            raise ValueError("fixed sample plan must have exactly five records")
        trainable = [p for p in vla.parameters() if p.requires_grad]
        projector_parameters = list(projector.parameters())
        optimizer = torch.optim.AdamW(
            [{"params": trainable, "lr": args.lr},
             {"params": projector_parameters, "lr": args.lr * args.projector_lr_multiplier}],
            weight_decay=0.0,
        )
        reader = R44.R45DATA.RuntimeReader(cache_bytes=0)
        vla.train()
        projector.train()
        phase = "five_sample_steps"
        measured_end_to_end_s = []
        for index, sample in enumerate(samples):
            role = "warmup" if index == 0 else "measured"
            if sample.get("role") != role or sample.get("order_index") != index:
                raise ValueError("sample role/order differs from frozen 1+4 sequence")
            task, episode, t = sample["task"], sample["episode"], int(sample["t"])
            path = paths[f"{task}/{episode}"]
            total_start = time.perf_counter()
            read_start = time.perf_counter()
            states, actions = R44.read_state_action(path, modules[2])
            torch.cuda.synchronize()
            hdf5_read_s = time.perf_counter() - read_start
            feature_start = time.perf_counter()
            feature_row = modules[3].build_row(
                states, actions, t, modules[4].contract_for(cpu_plan["feature_payload"], task),
                task, episode=episode,
            )
            lab, _delta = R44.build_runtime_labels(
                prompt_ids, np.asarray(states[t], dtype=np.float32),
                np.asarray(actions[t], dtype=np.float32), task, cpu_plan["delta_payload"], modules,
            )
            if lab["labels"][:lab["n_prompt"]] != [-100] * lab["n_prompt"] \
                    or [int(x) for x in lab["labels"][-15:-1]] != sample["action_token_ids"] \
                    or int(lab["labels"][-1]) != 2:
                raise ValueError("runtime labels differ from the frozen sample plan")
            torch.cuda.synchronize()
            state_label_s = time.perf_counter() - feature_start
            pixel_values, image_timing = reader.load_images(
                proc, path, t, 3, modules[3].CAMERA_ORDER, device="cuda",
            )
            if list(pixel_values.shape) != sample["processor_output_shape"]:
                raise ValueError("runtime OpenVLA processor output differs from frozen sample plan")
            transfer_start = time.perf_counter()
            proprio = torch.from_numpy(np.asarray(feature_row["features"], dtype=np.float32)).reshape(1, 42).to("cuda")
            input_ids = torch.tensor([lab["input_ids"]], dtype=torch.long, device="cuda")
            labels = torch.tensor([lab["labels"]], dtype=torch.long, device="cuda")
            torch.cuda.synchronize()
            cpu_to_gpu_s = time.perf_counter() - transfer_start + float(image_timing["cpu_to_gpu_s"])
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            forward_start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = vla(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                             pixel_values=pixel_values, labels=labels, proprio=proprio,
                             proprio_projector=projector)
                loss = output.loss
            torch.cuda.synchronize()
            forward_s = time.perf_counter() - forward_start
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite OpenVLA loss")
            backward_start = time.perf_counter()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable + projector_parameters, 1.0))
            torch.cuda.synchronize()
            backward_s = time.perf_counter() - backward_start
            if not math.isfinite(grad_norm):
                raise FloatingPointError("non-finite trainable gradient norm")
            optimizer_start = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize()
            optimizer_s = time.perf_counter() - optimizer_start
            total_s = time.perf_counter() - total_start
            if index:
                measured_end_to_end_s.append(total_s)
            sample_rows.append({
                "role": role, "id": {"task": task, "episode": episode, "t": t},
                "latency": {"hdf5_read_s": hdf5_read_s,
                            "jpeg_preprocess_s": image_timing["jpeg_preprocess_s"],
                            "state_label_s": state_label_s, "host_to_gpu_s": cpu_to_gpu_s,
                            "forward_s": forward_s, "backward_s": backward_s,
                            "optimizer_s": optimizer_s, "end_to_end_s": total_s},
                "memory": {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30},
                "loss": float(loss.detach().float().cpu()), "grad_norm": grad_norm,
                "exit_status": "PASS_STEP", "cache_hit": image_timing["cache_hit"],
            })
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        gpu_after = probe_gpu(outer.approved_gpu_uuid, 0, 0)
        report = {
            "schema": "r46-openvla-five-sample-smoke/v1", "status": "PASS_SMOKE",
            "source_sha": actual_sha, "approval_id": normalized["approval_id"],
            "sample_plan_sha256": sample_check["sample_plan_sha256"],
            "manifest_sha256": expected_hashes["approved_original_manifest_sha256"],
            "base_model_weight_sha256": actual_shards,
            "model_identity_sha_list": model_file_list, "model_info": model_info,
            "model_load_s": model_load_s, "gpu_admission_before_cuda": admission_before_cuda,
            "gpu_after_steps": gpu_after, "optimizer_steps_executed": len(sample_rows),
            "model_memory_baseline": model_memory_baseline,
            "cuda_device_identity": cuda_device_identity,
            "warmup_count": 1, "measured_count": len(measured_end_to_end_s),
            "smoke_interval_samples_per_second_excludes_model_load_and_warmup": (
                len(measured_end_to_end_s) / sum(measured_end_to_end_s)),
            "throughput_scope": "preliminary_smoke_only_not_stable_training_throughput",
            "throughput_excludes_model_load": True,
            "throughput_excludes_warmup_sample": True,
            "samples": sample_rows, "weights_exported": False, "checkpoint_exported": False,
            "optimizer_state_exported": False, "formal_training": False,
            "strict_cpu_plan": {"train_samples": cpu_plan["n_train"], "dev_samples": cpu_plan["n_dev"],
                                "train_samples_sha256": cpu_plan["train_samples_sha256"],
                                "dev_samples_sha256": cpu_plan["dev_samples_sha256"]},
        }
        validate_smoke_report(report)
        out = Path(outer.out).resolve()
        phase = "write_success_report"
        out.mkdir(parents=True, exist_ok=False)
        (out / "samples.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sample_rows),
                                           encoding="utf-8")
        (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        import hashlib
        sums = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                for name in ("samples.jsonl", "report.json")}
        (out / "SHA256SUMS.json").write_text(json.dumps(sums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        print(f"FAIL_R46_SMOKE: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        failure = {
            "schema": "r46-openvla-five-sample-smoke/v1", "status": "FAIL_R46_SMOKE",
            "phase": phase, "source_sha": actual_sha, "error_type": type(exc).__name__,
            "error": repr(exc), "traceback": traceback.format_exc(),
            "optimizer_steps_completed": len(sample_rows), "samples_completed": sample_rows,
            "cuda_device_identity": cuda_device_identity,
            "cuda_context_created": bool("torch" in locals() and torch.cuda.is_initialized()),
            "weights_exported": False, "checkpoint_exported": False,
            "formal_training": False,
        }
        try:
            failure_out = Path(outer.out).resolve()
            failure_out.mkdir(parents=True, mode=0o700, exist_ok=True)
            failure_path = failure_out / "failure.json"
            with failure_path.open("x", encoding="utf-8") as stream:
                json.dump(failure, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            import hashlib
            sums_path = failure_out / "SHA256SUMS.json"
            with sums_path.open("x", encoding="utf-8") as stream:
                json.dump({"failure.json": hashlib.sha256(failure_path.read_bytes()).hexdigest()},
                          stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except (OSError, FileExistsError):
            pass
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
