# R44: bounded OpenVLA inference latency attribution

**Status:** SPEC ONLY. No production behavior changed by this document.  
**Date:** 2026-10-26  
**Frozen implementation/source SHA:** `d99635c4cebff1a5b8ee459ad60aea7695408bc6`  
**Prior GPU smoke evidence commit:** `034d4f9eff69b9750050b8e1d655c9c2a535874c`  
**Prior approved execution plan:** Issue #46 comment `6408387411`  
**Main protection:** `main` must remain `329614f54d3090b9b6ab120855ba50fbc93bdd1e`.

## 1. Question and evidence boundary

The R42 GPU smoke report records one first-observed inference on synthetic RGB/state:
- total: 11,899.14 ms
- preprocessing: 1,043.92 ms
- forward: 10,842.17 ms
- decode: 12.84 ms

This is **one request, not a warmed steady-state latency distribution**. Current R15 Stage 300 P50/P95/P99 remains UNKNOWN.

At the frozen source:
- `R42_KV_CACHE=0` and `R42_VISION_CACHE=0` are the required baseline flags.
- With KV cache disabled, `R42DeployServer._decode_tokens()` performs 14 calls to `backend.next_token_logits(prefix, pixel_values, feat)`, recomputing the multimodal forward for each action token.
- The immediate task is to measure this exact baseline and attribute the latency. **Do not run a cache-path experiment in R44.**
- Existing KV-cache evidence comes from another checkpoint, includes free-running token differences, and is not authorization to enable caching. Any cache experiment requires a separate exact-checkpoint parity task and explicit review.
- The vision-cache flag alone does not prove that the vision cache was installed. Historical evidence on a different checkpoint showed exact output but approximately 1.0002x speedup; do not prioritize it here.

## 2. Goal

Measure steady-state latency and per-token cost on the exact current R42 implementation and R15 Stage 300 checkpoint, while preserving the default full-recompute path. Confirm repeated-input determinism, safety-contract integrity, runtime/checkpoint identity, and GPU memory behavior.

This is diagnosis only, not a deployment or closed-loop gate.

## 3. Frozen inputs and identity

Use the exact smoke checkout/source:
- production source SHA: `d99635c4cebff1a5b8ee459ad60aea7695408bc6`
- base model: `/home/dgut/pusht_runtime/model_cache/openvla-7b-pristine`
- checkpoint: `/home/T7/ojh/robot_actual/r11/runs/r15_stage_300_20261021T030423Z/checkpoints/step300`
- policy/task: `OpenVLA_R42` / `push_T`
- cameras: `cam_head, cam_left_wrist, cam_right_wrist`
- action IDs: 14 tokens, legal window 31744..31999
- control: 25 Hz, `dt=0.04`
- runtime flags: `R42_POLICY=''`, `R42_KV_CACHE=0`, `R42_VISION_CACHE=0`

Use the same deterministic synthetic observation generator and midpoint state as `diagnostics/r42_rate_limit_gpu_smoke.py`. Record the exact RGB/state fixture or its SHA256. Do not use real observations, another checkpoint, or a different prompt.

Before loading, verify that the existing repository remote is the expected `jho887891-sudo/robo`, the production checkout is exactly at the frozen SHA, and the working tree is clean. Do not use Git wrappers or alter Git's reported state. If the source, path, runtime, or checkpoint differs, stop BLOCKED.

## 4. Instrumentation scope

Use the official path:
`Model.update_obs -> Model.get_action -> R42DeployServer.act -> _decode_tokens`.

Do not edit production policy/server/model files. Use a temporary external driver or isolated diagnostic-only file. Wrap `backend.next_token_logits` externally to record, for each call:
- request index and action-token position k (0..13);
- prefix length;
- wall-clock milliseconds for the call;
- legal output-window bounds and returned argmax token;
- request-level preprocess/forward/decode/total times.

The backend returns logits via `.cpu().numpy()`, which synchronizes the result transfer. Record the timing method and CUDA synchronization points; synchronize before and after each end-to-end request. Do not attribute model loading to action latency. The instrumentation must not change token selection, input tensors, cache flags, model weights, or action decoding.

## 5. Fixed execution budget and gates

1. Capture GPU model, driver/CUDA, `nvidia-smi`, compute-process ownership, free/total VRAM, Python/PyTorch/Transformers versions, and source/checkpoint identity.
2. Require at least 24 GB free CUDA VRAM before model load. If free memory is below threshold, GPU ownership is unclear, or another process makes the run unsafe, record BLOCKED and exit without loading. Never kill/suspend processes.
3. Load the model once and record load time separately.
4. Run exactly **one warm-up action** through the official path. Exclude it from summaries.
5. Run exactly **20 measured action requests**, each with the same deterministic synthetic RGB/state fixture. Reset episode history and previous-command latch before each request so every request starts with the same first-step contract.
6. Hard wall-clock budget: 15 minutes from model-load start. No automatic retry, threshold lowering, or package installation.
7. Stop on OOM, CUDA/device error, non-finite output, illegal token, physical/rate contract violation, changed checkpoint hash, unexpected side effect, or budget overrun. Preserve partial evidence and report BLOCKED/PARTIAL/FAIL honestly.

No training, optimizer steps, cache-path experiment, HDF5 writes, checkpoint writes, simulator launch, robot connection/command, deployment, package installation, or source-policy modification.

## 6. Required report and acceptance

Report request-level P50/P95/P99, mean, max, requests/second, and ratio to the 40 ms control period for total, preprocessing, forward and decode latency. Report per-token P50/P95/P99 and position-wise mean/max; clearly state that P99 from 20 samples is a screening estimate only.

For each measured action verify:
- exactly 14 action tokens, each in 31744..31999;
- finite 14D output;
- physical position bounds satisfied;
- `abs(action14 - prev_cmd14) <= max_rate14 * 0.04 + 1e-5`;
- `dt=0.04`, no `unsafe` or rate-conflict signal;
- repeated identical input produces the same tokens and action, or any variation is explicitly reported.

If any request exceeds 40 ms, label 25 Hz real-time compliance **NOT MET**. Do not imply that correct `dt` wiring proves the server can execute at 25 Hz. The output is not evidence of closed-loop Push-T success.

## 7. Hashes and evidence

Hash the actual base-model weight shards (`*.safetensors`) plus base config, checkpoint manifest, LoRA adapter/config, proprio projector, normalizer/split contract, and relevant production source files before and after the run. Record full path, byte size and SHA256. If hashing the full base weights cannot finish within the budget, keep full base-weight identity UNKNOWN and explain why; do not label it verified based only on the config file.

Write to a new unique directory outside the repository, never overwrite existing evidence. Include:
- `report.json` with schema/version, UTC timestamps, frozen production SHA, diagnostic driver identity, full source/checkpoint/config paths, hashes, dependency versions, GPU/process snapshots, flags, warmup count, measured count, per-trial/per-token records and summary statistics;
- raw `console.log`;
- `environment.txt`;
- `nvidia-smi-before.csv` and `nvidia-smi-after.csv`, including process ownership if supported;
- `SHA256SUMS` covering every evidence file except itself;
- explicit `exit_code.txt`.

## 8. GitHub delivery and stop rules

Use an isolated branch `evidence/r42-perf-diagnosis-<UTCSTAMP>` based on the frozen source SHA. Commit only the diagnostic driver (if needed) and evidence directory; do not change production source/policy, deployment config, `main`, PR #47 or PR #48. Preserve the prior smoke evidence unchanged.

- **BLOCKED:** pre-load gate fails, source/checkpoint identity mismatch, dirty checkout, or runtime differs from the frozen environment.
- **PARTIAL:** a valid run starts but the hard budget or environment interrupts it; preserve completed samples and all raw logs.
- **FAIL:** runtime or contract failure, non-finite/illegal action, safety violation, hash mismatch, or malformed evidence.
- **PASS (diagnostic only):** 1 warm-up + 20 measured baseline requests complete with valid evidence and all contracts intact. This is not a 25 Hz or rollout PASS.

Post branch name/SHA, exit code, completed sample count, evidence path, and VERIFIED/BLOCKED/UNKNOWN conclusions to Issue #46. Keep PR #47 and #48 as drafts and Issue #46 open. Do not run a simulator or robot episode after this task.
