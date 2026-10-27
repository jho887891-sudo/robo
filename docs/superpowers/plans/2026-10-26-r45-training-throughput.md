# R45 Training Throughput Implementation Plan

## Scope and guardrails

Work only in /tmp/r45-wt on experiment/r45-training-throughput-20261026, whose HEAD is the required R44 code SHA. Preserve all four pre-existing untracked R45 files and review them in place. Do not touch main, R44 evidence, historical weights/HDF5, or run GPU code. Every production change follows a failing CPU test first.

## File map

- Create training/r45_contract.py for immutable R45 configuration and fail-closed 42D [S,V,Acc] validation.
- Modify training/r45_data.py for indexed manifests, worker-local handles, bounded cache, image/state/action alignment, and efficient per-file reads.
- Modify training/r45_train.py for full-data epoch semantics, model seam, optimizer/scheduler accounting, safe stop, recovery, and evidence.
- Modify diagnostics/r45_gpu_bench.py for an independent hard-deadline benchmark; it must never import the formal trainer.
- Modify tests/test_r45_pipeline.py; add focused contract tests where needed.
- Create diagnostics/r45_cpu_audit.py and docs/r45/CPU_AUDIT.md for static evidence and measured/unknown timing classification.
- Create docs/r45/GPU_BUDGET_REQUEST.md and a deterministic budget-calculation helper only after CPU gates pass.
- Generate only under a new evidence/r45/ directory.

## Task 1: Freeze the contract and baseline audit

Tests first: add failing tests for exact constants (joint_delta_v1, 14D action, 42D input, 3 cameras, LoRA r=32/alpha=16), accepted [S,V,Acc] transform, rejection of 14D/28D fallbacks, and rejection of unknown normalization/time alignment.

Implementation: add R45Config/StateHistory42 with explicit layout, normalization provenance, timestamp alignment, camera order, LoRA target choice, and dataset split. No default may silently select q_proj/v_proj or shrink the state width. Add a CPU audit command that records commit SHA, diff status, Python/PyTorch/Transformers/PEFT versions, attention implementation, and static trainable-module facts; unknown timings remain UNKNOWN.

Verification: run the focused tests red then green; run the audit on CPU and confirm it emits VERIFIED, IMPLEMENTED, BLOCKED, and HYPOTHESIS classifications without GPU initialization.

## Task 2: Make the indexed data pipeline correct and bounded

Tests first: add/repair tests for index counts, deterministic epoch permutations, TRAIN/DEV disjointness, one-worker vs multi-worker equality, all three camera names/order, RGB uint8 shape, state/action time alignment, history shape, and cache byte ceiling. Include a synthetic HDF5 fixture so tests do not depend on the live dataset path.

Implementation: retain one HDF5 handle per worker/process and close it at worker shutdown; avoid rereading whole state/action arrays for every sample; use an index manifest with content hash and exact counts; implement bounded LRU cache by bytes; preserve deterministic order and fail on missing/corrupt frames. Expose a 42D feature builder only through the approved contract and record its source timestamps.

Verification: run the focused tests in red/green order, then the complete CPU data/contract suite. Report live-data counts only when the manifest proves 108,780 TRAIN and 28,268 DEV.

## Task 3: Implement checkpoint-safe formal trainer on CPU seam

Tests first: add failing tests for effective batch and optimizer-step accounting, five-epoch no-replacement coverage, gradient accumulation accounting, LR schedule logging, signal/stop-file checkpointing, checkpoint hash/config consistency, and resume cursor/order/RNG validation.

Implementation: keep r45_train.py as a distinct entry point. Separate dataset iteration from model execution; make micro_batch, grad_accum, workers, precision, scheduler, and LoRA target explicit. Ensure the complete run accesses every TRAIN index exactly once per epoch, while optimizer_steps is reported separately. Implement atomic checkpoint directories containing model, optimizer, scheduler, RNG, epoch order hash, cursor, config, environment, and stop reason. Add only a CPU toy model seam for tests; real OpenVLA loading remains gated and must enforce the approved 42D projector width, frozen base, LoRA r=32/alpha=16, and explicit target module choice.

Verification: run the CPU trainer tests, inspect emitted metrics and coverage proof, and confirm interrupted/resumed runs match uninterrupted sample order and cumulative step accounting.

## Task 4: Build independent bounded benchmark and evidence

Tests first: add failing tests that benchmark code does not import r45_train, refuses missing approval, refuses output overwrite, records PID, stops at a hard deadline, handles SIGINT/SIGTERM, and writes a failure report.

Implementation: benchmark only a fixed sample set and explicit model/config seam; add warm-up/stable separation, batch time, samples/sec, peak memory, utilization fields, numerical checks, raw log path, process identity, and stop reason. Keep optimization switches independent: loader, attention/SDPA, and micro-batch 1/2/4. Do not claim any speedup from placeholder timing.

Verification: execute only CPU refusal/deadline tests. Do not run the approved-GPU path until the research lead separately authorizes the budget matrix.

## Task 5: CPU gate, report, and budget request

Run the full repository CPU test command plus R45 tests. Review git diff, branch identity, and untracked-file provenance. Produce docs/r45/CPU_AUDIT.md with evidence classes, bottleneck timings (measured vs UNKNOWN), all-linear vs q_proj/v_proj parameter/cost comparison, attention findings, and unresolved 42D status. Produce a minimal GPU budget request with fixed samples, matrix, 15-minute-per-cell cap, max six cells, PID/stop behavior, and explicit no-formal-training scope. Do not submit it as approved and do not run GPU.

## Review focus

- A 42D history must never silently become 14D/28D: covered by Task 1 contract tests.
- HDF5 worker handles must not cross process boundaries: covered by Task 2 multi-worker test.
- Full data coverage must not be confused with optimizer updates: covered by Task 3 accounting/coverage tests.
- A checkpoint must not resume with a changed index/order/config: covered by Task 3 tamper tests.
- The benchmark must not accidentally become training or run unbounded: covered by Task 4 isolation/deadline tests.

## Self-review gaps

The real OpenVLA model construction and exact live HDF5 42D provenance remain BLOCKED until the repository evidence identifies the approved source and compatibility path. This plan intentionally does not authorize GPU execution or dependency changes.

