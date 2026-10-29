# Standard OpenVLA-7B GPU Smoke plan

STATUS: PLANNED ONLY. No GPU authorization is implied.
Target: standard OpenVLA-7B base + LoRA r=32, alpha=16, q_proj/v_proj + existing 42D projector. No OFT model or weights.

## Purpose and scope

Verify one real local OpenVLA model load and a tiny bounded training-path smoke. Use five fixed, preselected TRAIN examples from one task: one warm-up and four measured. Each has the three ordered cameras, 42D feature and 14 action-token targets plus STOP. Run real forward/loss/backward, finite-gradient checks and ephemeral optimizer updates for LoRA/projector only. Do not save/export weights or checkpoints. This is not formal training or a stable throughput benchmark.

## CPU preconditions

1. Pin exact code SHA; clean isolated worktree; unique evidence directory.
2. Verify frozen config/index and all three weight shard SHA256 values, Prismatic API and frozen dynamic-module SHA.
3. Verify selected TRAIN rows against the frozen manifest; no DEV data. Check camera order cam_head, cam_left_wrist, cam_right_wrist, RGB 480x640x3, state [1,42], legal 14-token targets plus STOP.
4. Assert LoRA r32/alpha16/q_proj-v_proj, frozen base, 42D projector input, and only adapter/projector parameters trainable.
5. Recheck NumPy/OpenCV in the same environment. The real CPU JPEG sample passed, but OpenCV metadata still requires NumPy>=2; explicitly resolve or accept that package metadata mismatch before GPU approval.
6. Confirm unique output path, disk space, process timeout supervisor and stop-signal logging before CUDA visibility is enabled.

## GPU admission gate

Only after explicit approval, check the agreed time window, exact GPU UUID, total/free memory, utilization and active compute processes. Require exclusive availability; abort if another process is using the selected GPU. Never kill another user's process.

Proposed conservative floor: at least 42 GiB free on a GPU with at least 48 GiB total. This is a proposed admission threshold, not a measured model requirement; confirm it for the approved hardware. If the threshold is not met, stop and request a decision; do not silently change precision, batch size or model configuration.

After admission, expose only the approved GPU, verify UUID and one visible device, then create the CUDA context. Do not initialize CUDA before approval and the resource check.

## Load and measurement

- Load the actual local three-shard OpenVLA-7B checkpoint in BF16 with the registered standard OpenVLA/Prismatic config and processor, trust_remote_code=True and local_files_only=True. Record hashes, load wall time, processor/model classes and active modeling source SHA.
- Freeze base weights; attach the specified LoRA and projector. Verify trainable names/counts and that base gradients remain absent.
- Use the formal TRAIN sample reader/labels, micro-batch 1, exactly three cameras. No attention/backend changes or gradient accumulation.
- Record load time separately. After load, reset CUDA peak counters and record model-resident baseline plus allocated/reserved peak memory.
- For each measured step, synchronize timing boundaries. Record end-to-end time from HDF5/JPEG read and preprocessing through transfer, forward, backward and ephemeral optimizer update. Also record synchronized CUDA-event compute time.
- Exclude one warm-up from throughput. Report 4 measured samples divided by total synchronized end-to-end time. This is a preliminary smoke metric only; do not extrapolate it to stable throughput or a five-epoch ETA.
- Write logs/JSON/PID/start/deadline/stop reason and hashes to the unique evidence directory. No checkpoint, model export, deployment or data mutation.

## Hard stops

Hard wall-clock cap: 30 minutes total, including load and cleanup. Suggested stage limits: model load 15 minutes, smoke steps 10 minutes, cleanup/evidence 5 minutes. A parent supervisor starts the timer before process launch, sends SIGINT at deadline and SIGKILL after a 10-second grace period. Record signals and exit code; no extension.

Stop on any SHA change, model/config/API mismatch, bad camera/state/label/LoRA contract, wrong GPU UUID, insufficient memory, unexpected GPU process, OOM, non-finite loss/gradient, base-model gradient, deadline or manual interrupt. Do not automatically retry or save partially updated adapter/projector state.

## Tool readiness and authorization

diagnostics/r45_gpu_bench.py currently returns BLOCKED_NOT_IMPLEMENTED before CUDA because real OpenVLA forward/backward/optimizer timing is not wired. This plan is not an executable command; implement and CPU-test a dedicated real smoke entry point before scheduling it. Stable throughput benchmarking requires a separate approved matrix and budget.

This document defines scope only. Do not initialize CUDA, load weights on GPU, run Smoke/benchmark or formal training until the research lead explicitly approves the exact device, time window and 30-minute cap.
