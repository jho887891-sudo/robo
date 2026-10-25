# R30 Runtime Audit Trace Implementation Plan

> **Execution record:** Implemented inline with Superpowers TDD on an isolated feature branch. The history includes intentional RED checkpoints for the missing trace writer, failure-path evidence, shared run IDs, artifact collisions, report parsing/gates, missing trace reports, and reset-order validation. Final CI status is recorded below.

**Goal:** Persist enough server-boundary evidence for an authorized R15 run to compare the observed 14D state, model output, and server-generated response, then automatically produce a report that says which evidence gates are open or blocked.

**Architecture:** An opt-in standard-library JSONL writer fails closed when enabled but unable to persist. The R42 model boundary records observation/action/error events. The R15 driver shares one run ID. The launcher uses collision-resistant artifact names and a finalizer that stops the server and validates the trace on every post-launch exit path. A read-only report validator never repairs source evidence. Schema v2 adds the exact per-step rate reference so the report can independently recompute rate compliance; schema v1 remains readable but cannot certify that gate.

**Tech Stack:** Python 3.10+, standard library, NumPy already used by the R42 model, pytest, Bash, GitHub Actions CPU runner.

**Spec:** `docs/superpowers/specs/2026-10-25-r30-runtime-audit-trace-design.md`

## Global Constraints

- Base source SHA: `9b94823d7af6909811bd3e7039db23d855c09559`.
- Preserve R30 evidence branch at `0b974aea73f4ecf55c503d8528297c1e084f2045`.
- Never change `main`, rewrite historical evidence, or merge this PR.
- Never store raw RGB pixels; camera evidence is SHA256 + shape + dtype only.
- Never claim a server-generated response is client-acknowledged or executed.
- No model load, inference, training, optimizer step, CUDA, HDF5/checkpoint mutation, simulator, or robot execution during implementation/CI.
- A future closed-loop run requires a separate explicit authorization.

## Review Focus

- Existing trace/report paths must never be truncated or overwritten.
- Missing or unsafe run ID must be rejected before policy backend loading.
- Non-finite or unserializable records fail closed.
- An observation is persisted before inference; backend/inference/response-contract errors link back to it.
- Docker run ID must be authoritative even if stale `ROBODOJO_RUN_ID` is supplied in `--docker-env`.
- A valid report must not overstate client acknowledgement, camera/state synchronization, or robot actuation.
- Missing trace input must still produce a durable failure report rather than leaving only stderr.

---

### Task 1: Durable JSONL writer

**Files:**
- Create: `code/xpolicylab/policy/OpenVLA_R42/r42_runtime_audit.py`
- Test: `tests/test_r42_runtime_audit.py`

- [x] Test ordered records, shared run ID, exclusive create, invalid run IDs, JSON safety, and missing parent directory.
- [x] Observe RED on the missing writer module.
- [x] Implement exclusive creation, mode 0600, optional O_NOFOLLOW, schema/version metadata, JSON validation, flush/fsync, and fail-closed errors.
- [x] Verify writer regressions: **5 passed**.

### Task 2: R42 model-boundary instrumentation

**Files:**
- Modify: `code/xpolicylab/policy/OpenVLA_R42/model.py`
- Test: `tests/test_openvla_r42_server.py`

**Environment contract:** `R42_AUDIT_LOG_PATH`, `R42_AUDIT_RUN_ID`.

**Events:** `run_start`, `episode_reset`, `observation_received`, `backend_ready`, `action_error`, `action_step`.

- [x] Test state14, features42, token IDs/bins, residual14, decoder info, response action, camera hashes, reset indices, previous response vs next observation, disabled-by-default behavior, and no-overwrite behavior.
- [x] Observe RED before model instrumentation existed.
- [x] Persist validated observation state/camera hashes before inference.
- [x] Link backend-load, server-act and response-contract errors to the exact observation record and observation index.
- [x] Measure observation audit overhead (camera hashing + durable observation write) in memory and attach it to the linked action/error record without adding another fsync.
- [x] Include the frozen configured `max_rate[14]`, `control_dt` and `control_hz` in run-start metadata for later safety-gate verification.
- [x] Rename trace timing fields to `pre_audit_return_wall_utc` / `server_span_pre_audit_ms`; explicitly state that the action-step fsync is excluded and client receive time is not measured.
- [x] Keep the action contract and returned action unchanged; label actions as server-generated.
- [x] Keep client acknowledgement and camera/state synchronization UNKNOWN.
- [x] Verify model trace and official rate-limit regressions: **9 passed**.

### Task 3: Shared run ID in the R15 driver

**Files:**
- Modify: `tools/r15_loop_dev.py`
- Test: `tests/test_r15_loop_dev.py`

- [x] Add `--run-id`, validate against `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`, and preserve generated IDs when omitted.
- [x] Add dry-run tests for propagation and unsafe-ID rejection.
- [x] Remove any stale `ROBODOJO_RUN_ID` from Docker env and append the chosen run ID exactly once.
- [x] Verify driver regressions: **10 passed**, including a dry-run proving stale IDs are overridden.

### Task 4: Collision-safe launcher and automatic finalizer

**Files:**
- Modify: `scripts/r15_loop_pushT.sh`
- Test: `tests/test_r15_loop_dev.py`

- [x] Add a regression test for unique artifact names and automatic report generation.
- [x] Observe RED because the launcher had no report path/finalizer.
- [x] Include Bash PID in the UTC run stamp so concurrent launches cannot share the same second-level paths.
- [x] Preflight-check server log, trace, report, and official result paths; refuse existing artifacts.
- [x] On every exit after the server starts, stop the server, run the validator, and persist a report even when the trace file is missing.
- [x] Preserve the original client/launch failure code; if the original exit was zero but the audit gate fails, return exit code 5.
- [x] Verify `bash -n scripts/r15_loop_pushT.sh` and launcher static regression tests.

### Task 5: Read-only runtime trace report and evidence gates

**Files:**
- Create: `tools/r42_runtime_audit_report.py`
- Create: `tests/test_r42_runtime_audit_report.py`
- Modify: `.github/workflows/r30-runtime-audit-cpu.yml`

- [x] Define report schema `r42-runtime-audit-report/v2` with input SHA256, event counts, trace integrity, run outcome, unresolved observations, and explicit gate statuses.
- [x] Write trace schema `r42-runtime-audit/v2` with exact `rate_reference_action14`, `rate_reference_source`, `rate_limit_source`, and `control_dt`; continue reading v1 traces but do not certify rate compliance without the v2 evidence.
- [x] Validate JSONL line termination, supported schema versions, contiguous record indices, shared run ID, event contracts, vector sizes/finite values, camera fingerprints, and observation-to-action/error references.
- [x] Treat missing/incomplete trace data as FAIL/PARTIAL; never rewrite the input JSONL.
- [x] Keep client acknowledgement, camera/state wall-clock sync and robot actuation UNKNOWN; keep demo-S0 comparison BLOCKED unless a verified reference vector is supplied.
- [x] Block the first-response gate when the first observation has no successful action, and block evidence gates when trace integrity is not PASS.
- [x] Require the reset record to precede the first observation in the same episode.
- [x] Cross-check that the four-key action dict flattens to the recorded action14 and that the previous-response delta equals `state14 - previous_server_response_action14`.
- [x] Add `rate_limit_branch_active`: require a valid configured 14D rate vector and positive decoder `dt` matching the configured control period for every successful action. A null `dt` blocks the report/launcher gate.
- [x] Add `rate_limit_compliance`: independently verify every action delta against `configured_max_rate14 * control_dt + tolerance` using the recorded reference vector, and verify the reference is either the same-step measured state or the prior response for the same environment.
- [x] Include a latency caveat: per-record fsync affects runtime; `server_span_pre_audit_ms` includes observation audit overhead but excludes action-step fsync, and actual client receive time is not measured.
- [x] Refuse to overwrite an existing report; if trace input is missing, still write a machine-readable failure report.
- [x] CLI exit codes: 0 valid complete evidence with no action errors and first-action gate PASS; 2 partial/action-error/blocked gate; 3 invalid or unavailable trace; 4 report write refusal/failure.
- [x] Verify report regressions: **20 passed**.

### Task 6: CI, cross-audit, and handoff

- [x] Focused CPU workflow runs writer **5 passed**, model trace/rate integration **9 passed**, standalone RoboDojo adapter **36 passed**, R15 driver **10 passed**, report validator **20 passed**, Python compilation and Bash syntax.
- [x] R4 CPU contracts: **PASS**.
- [x] R5 CPU contracts: **PASS**.
- [x] Recompute SHA256 for all 8 files listed by the R30 evidence directory's `SHA256SUMS.txt`: **8/8 match**. This verifies file integrity, not a second HDF5 execution.
- [x] Track Issue #46 as a separate behavior-change PR (#47); this branch changes action behavior only in the isolated model path and does not authorize rollout.
- [x] PR #45 was merged into `audit/pusht-startup-import-fix-20261024` at `0f1f98366e8a400d530a62cc5998c4eefdb43fe0`; `main` and the R30 evidence branch remain unchanged.
- [x] No model weights, CUDA, training, optimizer step, HDF5/checkpoint mutation, simulator, or robot execution occurred.

## Rulings

- **Ruling:** instrument the existing policy-server boundary instead of patching the external RoboDojo client checkout — the needed values already exist in memory, while this avoids an unversioned client mutation. Cost if wrong: the trace still cannot prove client acknowledgement, so that gate remains UNKNOWN.
- **Ruling:** enable trace logging only through explicit environment variables, set by the official R15 launcher — direct callers remain unchanged unless opted in. Cost if wrong: alternate launchers must set both variables to obtain trace coverage.
- **Ruling:** add Bash PID to the timestamp — the prior seconds-only stamp plus a preflight existence check was not atomic under concurrent launch. Cost if wrong: artifact names are longer, but evidence collisions are less likely.
- **Ruling:** make report generation part of the launcher finalizer — otherwise startup failures could leave no machine-readable audit report. Cost if wrong: a failed audit gate changes a zero launcher exit into code 5, intentionally preventing silent promotion.
- **Ruling:** the report gate explicitly checks whether the decoder's rate-limiting branch was entered. It does not change action behavior; the known bypass remains BLOCKED and must be fixed/reviewed before a real rollout is authorized.
- **Ruling:** timing fields describe the actual measurement boundary: the observation record is fsynced before inference, while the action record is fsynced after the pre-audit response timestamp is sampled. The report warns that audit-enabled timing is not directly comparable to unaudited runs. Cost if wrong: an extra caveat, but no false claim about client receive latency.
- **Ruling:** leave the adapter's rate-limiter behavior untouched — the current `Model.get_action -> R42DeployServer.act` call supplies no `prev_cmd` or `dt`, so the configured rate-limit branch appears inactive on this path. This is a separate safety-sensitive change requiring its own contract, tests and authorization.
