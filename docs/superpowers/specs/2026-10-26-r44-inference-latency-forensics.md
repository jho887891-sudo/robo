# R44: OpenVLA inference latency forensics

**Status:** SPEC ONLY. No production behavior changed by this document.  
**Date:** 2026-10-26  
**Frozen implementation base:** `d99635c4cebff1a5b8ee459ad60aea7695408bc6`  
**GPU smoke evidence commit:** `034d4f9eff69b9750050b8e1d655c9c2a535874c`  
**Evidence report:** `evidence/r42_rate_limit_gpu_smoke_20261025T093237Z/report.json`  
**Main protection:** `main` must remain `329614f54d3090b9b6ab120855ba50fbc93bdd1e` and must not be modified.

## 1. Question and current evidence

The R42 GPU smoke proves one real OpenVLA-7B forward/decode request on an NVIDIA RTX A6000 using synthetic RGB/state. The single request reports 11,899.14 ms total (10,842.17 ms forward, 1,043.92 ms preprocessing, 12.84 ms decode). This is a **single first-observed inference**, not a steady-state latency distribution. Steady-state latency for the current R15 Stage 300 checkpoint is UNKNOWN.

At the frozen source:
- `R42_KV_CACHE` defaults to `0`.
- With KV cache disabled, `R42DeployServer._decode_tokens()` loops over 14 action tokens and calls `next_token_logits(prefix, pixel_values, feat)` for every token, recomputing the multimodal prefix each time.
- The existing KV path performs one multimodal prefill plus 13 incremental language-model steps.
- Prior KV parity and latency evidence was collected on a different checkpoint. It is useful as a hypothesis only, not as evidence for the current Stage 300 checkpoint.
- The vision-only cache patch persists by replacing `vision_backbone.forward`. Do not benchmark it in the same process unless the original method is explicitly restored and tested. It is not the primary R44 hypothesis.

## 2. Goal

On the exact current R42 implementation and exact Stage 300 checkpoint, determine:
1. warmed full-recompute latency;
2. warmed KV-cache latency;
3. whether free-running token differences occur and their physical action impact;
4. whether the KV path preserves the existing 14D physical-bound and max-rate contracts;
5. GPU memory headroom and reproducibility metadata.

This is a bounded diagnostic, **not permission to enable cache by default or to run an episode**.

## 3. Frozen inputs and artifacts

Use the deployment config already recorded in the smoke report:
- Base model: `/home/dgut/pusht_runtime/model_cache/openvla-7b-pristine`
- Checkpoint: `/home/T7/ojh/robot_actual/r11/runs/r15_stage_300_20261021T030423Z/checkpoints/step300`
- Policy: `OpenVLA_R42`, task `push_T`
- Three camera order: `cam_head, cam_left_wrist, cam_right_wrist`
- Action contract: 14 tokens, legal token IDs 31744..31999
- Control contract: 25 Hz, `dt=0.04`
- Same synthetic observation generator and midpoint state as `diagnostics/r42_rate_limit_gpu_smoke.py`; record the exact generated RGB/state bytes or their SHA256. Do not substitute real observations or another checkpoint.

Before execution, re-check that the local deployment config resolves to these paths and that the checkpoint manifest, LoRA adapter, projector, normalizer contract and base-model configuration match the smoke report. Hash the base model weight shards too if feasible; if not, explicitly retain full base-weight identity as UNKNOWN. Never rewrite or re-save any checkpoint.

## 4. Implementation scope

Create a new standalone diagnostic entry point, e.g. `diagnostics/r44_inference_latency_forensics.py`, plus CPU tests. Do not change production inference code, default environment flags, `deploy.yml`, action codec, safety limits, checkpoint files or `main`.

Use one loaded model instance and compare only these two modes:
- **A / FULL:** `backend.kv_cache_enabled=False`, `backend.vision_cache_enabled=False`; no vision monkey-patch may be installed.
- **B / KV:** `backend.kv_cache_enabled=True`, `backend.vision_cache_enabled=False`; use the existing one-prefill + 13-step path.

Do not run the vision-only cache arm in this process. If a later task needs it, isolate it in a separate process or add and test an explicit unpatch/restore operation first.

For each measured trial:
- restore the model's episode history and previous-command latch to a clean reset state;
- feed identical RGB/state, prompt, timestamp and metadata;
- synchronize CUDA immediately before starting and after finishing each timed region;
- record the 14 legal token IDs/bins, decoded action14, latency phases, rate-limited dimensions, max absolute action delta, peak allocated and reserved CUDA memory;
- time each full-forward or incremental step separately where practical, without changing numerical behavior;
- ensure the cache is created and discarded within that one action and cannot survive into another trial.

Use the same loaded model and exact weights for both modes. Do not reload or change weights between arms. If the source cannot safely toggle modes on the same backend, stop and report BLOCKED rather than weakening the comparison.

## 5. Fixed budget and order

1. Read-only environment/GPU gate. Require at least 24 GB free CUDA memory before loading. Never kill or suspend other processes.
2. Load the model exactly once. Record load duration separately from inference duration.
3. Run 2 warmup actions in FULL and 2 in KV. Exclude warmups from latency summaries.
4. Run 10 measured actions in each mode, interleaved by round (FULL then KV per round), resetting history and latches before every action.
5. Hard wall-clock budget: 10 minutes after model loading. Hard inference budget: 24 total action calls including warmups. Stop on OOM, CUDA/device error, non-finite output, illegal token, safety-contract violation, unexpected cache persistence, or any need to mutate weights.
6. No training, optimizer steps, HDF5 writes, simulator launch, robot connection/command, deployment, package installation, or source/checkpoint mutation.

If the GPU precondition fails, record BLOCKED and exit without loading the model.

## 6. Required comparisons and acceptance

For both modes, report P50/P95/P99 for total, preprocessing, forward and decode latency, plus per-token timing and CUDA allocated/reserved/peak memory.

Correctness:
- All 14 IDs per action must lie in 31744..31999.
- All actions must be finite and satisfy the frozen physical bounds.
- All actions must satisfy `abs(action14 - prev_cmd14) <= max_rate14 * 0.04 + 1e-5`.
- No unsafe/rate-conflict signal.
- Report exact free-running token equality rate, differing token indices/bins, maximum bin delta, max absolute action-coordinate delta and normalized/physical residual delta if recoverable from the frozen decoder.
- Do not demand bit-identical BF16 logits as the sole acceptance criterion; token and action effects must be reported explicitly.
- A cache speed claim is supported only if both modes use the same loaded weights, identical inputs, the same process/GPU and the same warmup/timing method.
- A speed improvement is **not** sufficient to authorize deployment. If any free-running token differs, preserve the result and mark semantic parity as measured-but-not-identical; do not silently change defaults.

Performance interpretation:
- 25 Hz requires an end-to-end policy request within 40 ms per control step. If the measured P50/P95 is above 40 ms, label the 25 Hz requirement FAIL/NOT MET; do not redefine the target or imply that a faster cache meets real-time control if it does not.
- A single cold request is not a steady-state distribution.
- The output is a diagnostic only, not evidence of closed-loop Push-T success.

## 7. Evidence and GitHub delivery

Create a new unique output directory outside the repository, never overwrite earlier evidence. Include:
- `report.json` with schema/version, UTC timestamps, source SHA/branch/worktree status, full config and checkpoint paths, all relevant SHA256 hashes, dependency versions, GPU/driver/CUDA data, flags, warmup count, measured count, per-trial records and summary statistics;
- raw `console.log`;
- `environment.txt`;
- `nvidia-smi-before.csv` and `nvidia-smi-after.csv`;
- `SHA256SUMS` covering every evidence file except itself;
- explicit `exit_code.txt`.

Preserve the original R42 smoke report and commit unchanged. Commit code/tests/evidence-manifest references only on an isolated R44 branch; do not modify `main`, merge PR #47/#48, or convert either draft to ready. Report branch name, final SHA, exact command, test counts and evidence path. Do not claim success if the report, hashes or raw logs are missing.

## 8. Stop and decision rules

- **BLOCKED:** GPU below threshold, source/checkpoint identity mismatch, dirty/unverified checkout, or unable to reset/toggle paths without side effects.
- **FAIL:** runtime exception, illegal tokens, non-finite actions, safety violation, stale cache, malformed evidence, or exceeded hard budget.
- **PASS (diagnostic only):** both arms complete with intact evidence and all safety contracts pass. This does not mean 25 Hz is met.
- If KV is materially faster and token/action differences remain bounded, present the measured trade-off for review. Do not change the default from FULL to KV without a separate explicit decision.
- If both arms remain far above 40 ms, stop optimizing this narrow path and report that the 25 Hz requirement needs a separate architecture-level decision (e.g. action chunking or a smaller/distilled policy). Do not start such work or training without authorization.
