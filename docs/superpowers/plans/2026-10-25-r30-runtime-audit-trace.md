# R30 Runtime Audit Trace Implementation Plan

> **For agentic workers:** This plan was executed inline with Superpowers TDD and an isolated GitHub feature branch. Each production behavior was preceded by tests; the CI log records both expected RED checkpoints and subsequent GREEN results.

**Goal:** Persist enough server-boundary evidence for a future authorized R15 run to compare the observed 14D state, model output, and server-generated action response.

**Architecture:** A standard-library JSONL writer is opt-in via environment variables and fails closed if configured but unable to persist. The existing model boundary records state/features/tokens/residual/decoder output and the generated action; the launcher assigns unique artifact names and shares one run ID with the client ledger. No action semantics are changed.

**Tech Stack:** Python 3.10+, standard library, NumPy already used by the R42 model, pytest, Bash, GitHub Actions CPU runner.

**Spec:** `docs/superpowers/specs/2026-10-25-r30-runtime-audit-trace-design.md`

## Global Constraints

- Base source SHA: `9b94823d7af6909811bd3e7039db23d855c09559`.
- Preserve R30 evidence branch at `0b974aea73f4ecf55c503d8528297c1e084f2045`.
- Never change `main`, rewrite historical evidence, or merge this PR.
- Never store raw RGB pixels; camera evidence is SHA256 + shape + dtype only.
- Never claim a server-generated response is client-acknowledged or executed.
- No model load, inference, training, optimizer step, CUDA, HDF5/checkpoint mutation, simulator, or robot execution during implementation/CI.

## Review Focus

- Existing trace path: must fail rather than truncate prior evidence.
- Missing or unsafe run ID: must be rejected before policy backend loading.
- Non-finite or unserializable record: must fail closed without writing a malformed partial record.
- Episode reset: must clear previous-response comparison state and restart episode-local step index.
- Docker client run ID: must be explicitly forwarded via `docker exec -e`; host environment alone is insufficient.

---

### Task 1: Durable JSONL writer

**Files:**
- Create: `code/xpolicylab/policy/OpenVLA_R42/r42_runtime_audit.py`
- Test: `tests/test_r42_runtime_audit.py`

**Interfaces:**
- `RuntimeAuditWriter(path, run_id, run_metadata=None)`
- `write(record: dict) -> dict`
- `close() -> None`

- [x] Write tests for ordered records, shared run ID, exclusive create, invalid run IDs, JSON safety, and missing parent directory.
- [x] Observe RED on the missing writer module at the initial CPU workflow.
- [x] Implement exclusive creation, mode 0600, optional O_NOFOLLOW, schema/version metadata, JSON validation, flush/fsync, and fail-closed errors.
- [x] Verify writer regressions: **5 passed**.

### Task 2: R42 model boundary instrumentation

**Files:**
- Modify: `code/xpolicylab/policy/OpenVLA_R42/model.py`
- Test: `tests/test_openvla_r42_server.py`

**Interfaces:**
- Opt-in environment variables: `R42_AUDIT_LOG_PATH`, `R42_AUDIT_RUN_ID`.
- Events: `run_start`, `episode_reset`, `observation_received`, `backend_ready`, `action_error`, `action_step`.

- [x] Write tests for 14D state, 42D features, token IDs/bins, residual, decoder info, response action, camera hashes, reset indices, previous response vs next observed state, disabled-by-default behavior, and no-overwrite behavior.
- [x] Observe RED: model lacked `_audit_trace`; explicit audit path did not yet refuse an existing file.
- [x] Integrate trace writes after existing action-contract validation, with no changes to `server.act()` arguments or returned action.
- [x] Persist validated observation state/camera hashes before inference, so backend/inference failure still leaves the received input.
- [x] Link backend-load, server-act and response-contract errors to the exact observation record; add unique episode-local observation indices.
- [x] Verify model trace regressions: **5 passed, 19 deselected** at the last completed model test run before the newest backend-failure case; latest run pending.
- [x] Record response semantics explicitly as server-generated; keep client acknowledgement and camera/state synchronization UNKNOWN.

### Task 3: Shared run ID in the R15 driver

**Files:**
- Modify: `tools/r15_loop_dev.py`
- Test: `tests/test_r15_loop_dev.py`

- [x] Add `--run-id`, validate it against `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`, and preserve generated IDs when omitted.
- [x] Add dry-run tests for propagation and unsafe-ID rejection.
- [x] Make the driver authoritative for Docker run-ID propagation: remove any stale ROBODOJO_RUN_ID from docker-env and append the chosen run ID exactly once.
- [x] Verify driver regressions: **10 passed**, including a Docker dry-run proving stale run IDs are overridden.

### Task 4: Non-overwriting launcher integration

**Files:**
- Modify: `scripts/r15_loop_pushT.sh`
- Test: `tests/test_r15_loop_dev.py`

- [x] Add a static regression test for timestamped server logs, trace path, unique result path, shared run ID, and explicit Docker environment forwarding.
- [x] Observe RED: baseline launcher failed the `RUN_STAMP` assertion.
- [x] Use one UTC stamp/run ID across server trace, server log, output JSON, host ledger and Docker client.
- [x] Keep Docker run-ID forwarding in the driver, not duplicated in the launcher.
- [x] Refuse pre-existing output paths and remove the port-only server log naming that could truncate old logs.
- [x] Validate shell syntax with `bash -n scripts/r15_loop_pushT.sh`.

### Task 5: CI, cross-check, and handoff

**Files:**
- Create: `.github/workflows/r30-runtime-audit-cpu.yml`
- Create: `docs/superpowers/specs/2026-10-25-r30-runtime-audit-trace-design.md`
- Modify: this plan.

- [x] Add focused CPU-only CI; no simulator or model weights.
- [x] Verify focused suite on the implementation before the launcher-specific test: writer **5 passed**, model integration **3 passed**, driver **9 passed**, shell syntax **PASS**.
- [x] Observe launcher regression RED, then restore implementation and re-run the workflow.
- [x] Keep PR #45 draft/open/unmerged.
- [x] Final runtime CI on code SHA `b9c92351259ed4b15ca7978daff455ce6cb4d6a3`: writer 5 passed, model trace 5 passed, R15 driver 10 passed, Python compile + Bash syntax passed; R4/R5 CPU contracts passed.
- [x] Final diff review: PR #45 changes only the R42 policy boundary trace, R15 run-ID/unique artifact wiring, tests, CPU workflow and Superpowers docs. No main/evidence branch changes and no rollout.
- [x] Opened separate follow-up issue #46 for the rate-limiter call-path concern; do not alter action behavior in this instrumentation PR.

## Rulings

- **Ruling:** log at the existing server boundary instead of patching the external RoboDojo checkout — the needed vectors already exist in memory and this avoids an unversioned client mutation. Cost if wrong: the trace still cannot prove client acknowledgement, so the schema explicitly preserves that limitation.
- **Ruling:** enable tracing by default only through the official R15 launcher; direct model callers remain unchanged unless they set the explicit path. Cost if wrong: any alternate launcher must set the two environment variables to obtain trace coverage.
- **Ruling:** leave rate-limiter behavior untouched — the current `Model.get_action -> R42DeployServer.act` call supplies no `prev_cmd` or `dt`, so the adapter's rate-limiting branch appears inactive on this path. This is a separate safety/behavior change and needs its own spec and tests.
