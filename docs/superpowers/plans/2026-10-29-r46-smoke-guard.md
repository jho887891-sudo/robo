# R46 OpenVLA Smoke Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prepare a fail-closed, parent-supervised 1-warmup+4-measured OpenVLA smoke path and its CPU evidence without invoking CUDA.

**Architecture:** Keep the existing R44 one-step entrypoint intact. Add an independent supervisor that validates an externally supplied approval, validates the frozen sample-plan and source identity before starting a child, and enforces the wall limit with process-group signals. Add a separate R46 runner that reuses R44/R45 model/data/label helpers; all testable validation is pure CPU and runtime access stays behind the supervisor/child admission gate.

**Tech Stack:** Python 3.10 standard library, existing PyTorch/Transformers/PEFT and R45 reader at runtime, `unittest`, GitHub Actions.

**Spec:** user-provided `R46_R47_R48_SPEC.zip` and GitHub Issue #52. The approval contract is exactly `ACCEPTANCE/APPROVAL_TEMPLATE.json` (`r47-gpu-approval-template/v1`); no alternate/automatic approval schema is permitted.

## Global Constraints

- Base source SHA: `ba563d0f8c638b089f588027dd0af41ff290e1f4`.
- Standard OpenVLA-7B, LoRA r=32 alpha=16 q_proj/v_proj, 42D projector, three cameras, 14 action tokens + STOP, `joint_delta_v1`.
- R46 may use CPU tests only. No CUDA initialization, GPU smoke, benchmark, training, deployment, or main mutation.
- Parent wall cap is at most 1800 seconds and governs the complete child lifetime.
- Never synthesize approval, sample IDs, data hashes, GPU measurements, or PASS evidence.

## Review Focus

- Invalid/missing authorization must stop before child creation; test with child marker and CUDA import guard.
- Child hangs/ignores SIGINT must be bounded by the parent process group; test normal exit, cooperative signal, and hard kill.
- Wrong source/config/sample SHA, DEV/duplicate sample, or altered HDF5 identity must block before model/GPU code.
- R44 single-step behavior and benchmark refusal must remain unchanged; retain existing CPU tests.
- Windows development host must not be mistaken for Linux GPU-run semantics; execute POSIX signal tests on the authorized Linux CPU environment.

---

### Task 1: Freeze and record the R46-A source audit

**Files:** Create `evidence/r46-cpu-20261029/AUDIT.md`, `evidence/r46-cpu-20261029/r46_audit.json`, and command/output/exit/SHA files.

- [x] Record exact source SHA and clean worktree state.
- [x] Map R44 smoke, formal trainer, reader, placeholder benchmark, and existing tests to functions/lines.
- [x] Record frozen model/data contract and the parent-timeout/five-sample gaps without changing implementation.
- [x] Run AST parse and the existing R45 runtime gate; preserve raw stdout/stderr and exit codes.

### Task 2: Add CPU-tested approval and parent timeout supervisor

**Files:** Create `diagnostics/r46_smoke_supervisor.py`; create `tests/test_r46_smoke_supervisor.py`.

**Interface:** `validate_approval(doc, *, source_sha, now_utc, requested_wall_seconds)` returns a normalized validated approval or raises `ApprovalError`. CLI accepts an approval path, exact source SHA, unique output directory, hard wall seconds, and a fixed R46 child entrypoint plus child arguments.

- [x] Test missing/invalid approval refuses before child marker creation and output directory creation.
- [x] Test source SHA, GPU UUID, time window, max duration, exclusivity, memory minima, model/manifest/sample hashes fail closed.
- [x] Test child normal exit, SIGINT handling, SIGINT ignored then SIGKILL, and output-dir collision under POSIX.
- [x] Implement append+fsync JSONL lifecycle events, child PID/PGID and return code, and final report even on timeout.
- [x] Run CPU tests on Windows and POSIX Linux; explicitly mark unsupported platform behavior if needed.

### Task 3: Add fixed real TRAIN sample-plan contract and R46 smoke runner

**Files:** Create `diagnostics/r46_openvla_smoke.py`; create `tests/test_r46_smoke_contract.py`; add `config/r46_smoke_samples.json` only after server-side real-data selection succeeds.

**Interface:** `validate_sample_plan(plan, *, manifest, train_index, expected_sha)` validates exactly five unique TRAIN records, one warmup/four measured, sample/HDF5 identities, camera/proprio/action contract, and plan SHA. Runner uses one warmup and four measured disposable optimizer updates and emits `report.json` and `samples.jsonl`; it must not persist weights/checkpoints.

- [x] CPU tests reject wrong roles, duplicate samples, DEV membership, bad file SHA, invalid dimensions/camera order, illegal token count, and altered frozen hyperparameters.
- [x] Add runner/report schema tests with injected CPU fakes; assert no torch CUDA API is invoked by tests.
- [x] Implement actual model path by reusing `R44.build_model`, R45 `RuntimeReader`, `build_prompt_ids`, and `build_runtime_labels`; keep all GPU calls behind child-only execution after supervisor admission.
- [x] On Linux CPU server, select/verify five actual TRAIN HDF5 samples against the already-attested strict manifest, decode all three cameras, and freeze the plan SHA. If server access or contracts fail, stop this task as BLOCKED; do not fabricate IDs.

### Task 4: Full CPU gate, docs, and GitHub delivery

**Files:** Modify `.github/workflows/r45-runtime-cpu.yml`, `docs/PROJECT_STATUS.md`; add sanitized R46 evidence under `evidence/r46-cpu-20261029/`.

- [x] Run existing R45 tests plus new R46 tests and syntax checks with command/stdout/stderr/exit files tied to the final source SHA.
- [x] Verify `r45_gpu_bench.py` still returns `BLOCKED_NOT_IMPLEMENTED` and `r44_gpu_smoke.py` remains authorization-blocked without creating output/CUDA.
- [x] `git diff --check`; verify frozen contract files unchanged except required test/CI documentation.
- [x] Commit on this isolated branch, push, open a draft PR; never merge.
- [x] Stop at `STOP_WAIT_GPU_APPROVAL`. R47/R48 GPU-dependent tasks remain NOT_RUN until separate research-lead authorization.

## Integration ruling

The ZIP template has exact `approved_*` fields and explicitly marks itself template-only/unapproved. The initial local draft's schema name was corrected before acceptance; tests must prove only an externally completed copy is accepted. The three shard hashes are identity inputs, not GPU authorization. No sample IDs may be committed until the real server-side TRAIN selection succeeds.
