## Intent

R45 will make the R44 full-data OpenVLA-7B + LoRA path reproducible, interruptible, and measurable before any GPU run. The deliverable is CPU-verified infrastructure and an evidence-producing GPU gate, not a training result. The implementation must start at R44 code SHA `fd0a7b87813dd55bb330a0f722a8ba6f69cbf5e3` on branch `experiment/r45-training-throughput-20261026`; historical branches, weights, HDF5 files, and evidence remain untouched.

## Goals and non-goals

- Preserve `joint_delta_v1`, `Delta_t = A_t - S_t`, TRAIN-only normalization, three cameras, and the approved state-input contract.
- Cover exactly 108,780 TRAIN samples once per epoch for five epochs (543,900 accesses), while reporting effective batch, optimizer steps, scheduler, and samples/sec separately.
- Add indexed HDF5 loading with worker-local handles, bounded caching, deterministic no-replacement sampling, checkpoint/resume, coverage proofs, environment/config/hash manifests, and safe-stop evidence.
- Add a bounded, independently invocable GPU benchmark that cannot enter formal training.
- Keep `all-linear` and `q_proj/v_proj` as explicit configurations; do not silently choose between them.

Non-goals: GPU execution, downloading a model, formal training, deployment, changing the target/normalizer/split, dependency upgrades, or claiming an acceleration without measurement.

## Known constraints and unresolved contract

R44 proves the target contract and epoch permutation but does not provide a formal full-data trainer. The repository's approved `proprio_history.py` defines a 28D `[S_t,S_(t-1)]` history, while the route documents a separate 42D `[S,V,Acc]` contract. R45 must fail closed unless the source data and approved metadata prove the 42D layout, normalization, and time alignment. No 14D or 28D fallback is allowed. This is a potential BLOCKED item for model integration and must be recorded as such if evidence is absent.

## Approaches

1. Reuse and rename the Push-T trainer. Rejected: it carries old dataset and target assumptions and would weaken the contract boundary.
2. Build a single monolithic trainer around the existing model. Rejected: it makes data, model, checkpoint, and benchmark failures hard to isolate.
3. Recommended: separate contract/data, training/checkpoint, and benchmark/evidence modules behind explicit interfaces, with CPU tests first and GPU execution behind a budget gate. This keeps each optimization independently switchable and makes rollback and evidence reviewable.

## Proposed boundaries

- `training/r45_contract.py`: immutable configuration and validation for dimensions, target, LoRA, camera order, dataset split, and the 42D history; no fallback.
- `training/r45_data.py`: index manifest, worker-local HDF5 handles, bounded cache, image preprocessing, and deterministic batch records.
- `training/r45_train.py`: formal five-epoch loop, mixed precision options, optimizer/scheduler, signal handling, checkpoint save/restore, and structured metrics.
- `diagnostics/r45_gpu_bench.py`: fixed-sample benchmark with hard deadline, PID/signal handling, warm-up/stable phases, and raw failure evidence; never calls the formal training loop.
- `tests/test_r45_pipeline.py` plus focused contract tests: single/multi-worker equivalence, no duplicates/omissions, image/state/action alignment, 42D width, checkpoint consistency, and benchmark deadline behavior.
- `docs/superpowers/plans/2026-10-26-r45-training-throughput.md`: implementation plan after this design is approved.
- `evidence/r45/...`: generated run evidence only; no historical evidence paths are overwritten.

Existing untracked R45 files in the worktree (`training/r45_data.py`, `training/r45_train.py`, `diagnostics/r45_gpu_bench.py`, `tests/test_r45_pipeline.py`) are preserved and must be reviewed before any edits; they are not assumed correct or complete.

## CPU verification gates

Before GPU authorization: run the existing R44 CPU suite and new tests; verify exact sample counts and epoch coverage; verify DEV is never opened by TRAIN loaders; inspect SDPA/attention path and trainable parameter names statically; record dependency versions; and produce a CPU audit report marking measured timings as VERIFIED and unknown timings as UNKNOWN. Any 42D contract ambiguity blocks model-training implementation but does not block independent data-index tests.

## GPU budget proposal (approval required later)

Request only after CPU gates pass: one fixed 256-sample warm-up plus 1,024 stable samples for each approved matrix cell, hard wall-clock cap 15 minutes per cell, maximum 6 cells, and one 5-minute cooldown/cleanup allowance. Candidate cells are baseline, data-loader optimization, attention optimization (only if compatible), and micro-batch 1/2/4; combinations are limited to one evidence-backed winner. Every cell records PID, config, commit SHA, samples/sec, batch time, peak memory, utilization, numerical checks, and stop reason. No formal five-epoch run is included in this budget.

## Acceptance classification

Every result is labeled `VERIFIED`, `IMPLEMENTED`, `BLOCKED`, or `HYPOTHESIS`. CPU success cannot be reported as GPU validation. A GPU request is not authorization; it remains pending until the research lead explicitly approves the matrix and cap.

