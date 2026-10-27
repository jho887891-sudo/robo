# R45 GPU Benchmark Plan

Status: IMPLEMENTED preparation only; no CUDA context or GPU benchmark executed.

Authoritative code baseline: 7598a7449652a5d8dc9b6cca883fac3af0d2d56c
Prepared branch: experiment/r45-integrated-throughput-20261027

## Unauthorized guard

The benchmark must be invoked only after explicit approval. Without approval:

python3 diagnostics/r45_gpu_bench.py --out /tmp/r45-gpu-unauthorized

Expected: exit code 2, no output directory, no CUDA initialization.

## Proposed authorized matrix

Each cell uses the same fixed sample manifest and model/config:
- baseline loader, micro-batch 1
- R45 bounded loader, micro-batch 1
- R45 bounded loader, micro-batch 2
- R45 bounded loader, micro-batch 4
- compatible attention/SDPA switch, micro-batch 1
- one evidence-backed combination

Suggested command shape after approval:

python3 diagnostics/r45_gpu_bench.py --out <unique-evidence-dir> --approve-hours 0.5 --configs '<json>' --steps-per-config <bounded-count>

The approved budget request is a maximum of 30 minutes on one GPU wall-clock time. This is a request ceiling, not authorization. The tool must stop at the cap or on SIGINT/SIGTERM and retain PID, raw log, config, commit SHA, phase timings, samples/sec, peak memory, utilization, and stop reason.

## Separation of claims

GPU smoke: one bounded real-model forward/backward validation for 3 cameras, [B,42] projector, frozen base, q_proj/v_proj LoRA, and peak memory.

Stable throughput: fixed samples, warm-up excluded, stable samples/sec and phase timings.

Formal training: 543,900 TRAIN accesses across five complete epochs; not part of this preparation and not authorized.
