# R26 M2 Serving Safety Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Tasks must use strict RED→GREEN with real GitHub Actions run IDs.

**Goal:** prevent the preflight-only M2 CPU stub/diagnostic interfaces from silently producing apparently deployable robot commands, and leave a verifiable entrypoint audit.

**Architecture:** retain frozen R15 starter. Restrict ONLY R25 M2 model/server classes, add explicit test/diagnostic capability flags and runtime rejection, certify externally observable behavior using CPU fakes and existing real two-task normalizer. Preserve all prior evidence by additive commits.

**Tech stack:** Python 3.10, numpy 1.26.4, pytest 8.3.5, GitHub Actions (CPU).

**Spec:** `docs/superpowers/specs/2026-10-22-r26-m2-serving-boundary-audit.md`

## Global constraints

- Base SHA: `274b851818b76a3f875b0a1a401a4bdb66358948`; review branch `review/r26-m2-safety-entrypoint-20261022`.
- main/R15 launcher/legacy R15 `model.py`, `r42_serve.py`, checkpoint/HDF5/WAL/historical R24/PRs must NOT change.
- Formal optimizer step budget: **0**; no real CUDA, no RoboDojo deployment.
- No self-signed safety metadata may authorize real robot actuation.
- All tests must be verifiable from immutable SHA and GitHub Actions raw jobs.

## Review Focus

1. Injected CPU backend with no test intent must be rejected before robot action or GPU initialization.
2. A mismatched policy mode inherited through environment must not silently convert VLA to CV.
3. Replay without recorded RGB must fail when a real processor is attached; avoid fake physical data.
4. R15 path must stay compatible; 32 historical R42 metadata tests remain stable.
5. Standard R42 launcher must remain identified as legacy-only, not falsely advertised as M2-ready.

### Task 1: Failing safety regression pack (RED)

**Files:** Create `tests/test_r26_m2_runtime_guards.py`; add `.github/workflows/r26-m2-safety-gates.yml`.

**Interfaces:** Consumes `OpenVLA_R42.model_m2.Model`, `m2_deploy_server.M2TaskDeployServer`, old R25 fixture `case`.

- [ ] Add test `test_unauthorised_cpu_backend_fails_before_action`: existing M2 session with `backend=FakeBackend,device=cpu` but absent `allow_stub_action_for_contract_test` must raise an M2ModelError.
- [ ] Add test `test_cpu_fake_requires_optin_even_if_allow_real7b_smoke`: `allow_real_7b_smoke=True` must NOT whitelist CPU injected backend.
- [ ] Add test `test_cv_environment_requires_explicit_diagnostic_optin`: `R42_POLICY=cv` without explicit server `allow_cv_diagnostic` must raise, not emit tokens; explicit opt-in must return the frozen task's correct q01/q99 CV.
- [ ] Add test `test_real_processor_replay_refused_without_rgb`: replaying a captured FakeBackend record with processor!=None must fail, rather than pass `pixel_values=None` to a real visual model.
- [ ] Add static launcher test proving `r42_serve.py` imports legacy `OpenVLA_R42.model` and `deploy.yml` currently remains PushT. It's a *blocking negative fact*, not a M2 deployment pass.
- [ ] Run GitHub Actions, record actual failure assertion lines and exit code, distinguishing intended RED from unrelated test syntax/import failures.

### Task 2: Minimal default-deny M2 Model fix (GREEN)

**Files:** Modify `code/xpolicylab/policy/OpenVLA_R42/model_m2.py` + old R25 test helper in `tests/test_r25_m2_model_integration.py`.

- [ ] Add constructor guard: any `backend is not None` requires `device=="cpu"` **and** explicit `allow_stub_action_for_contract_test is True`; no truthy string.
- [ ] In `get_action()` reassert the guard even if `model_cfg` was modified after initialization. Real backend remains categorically blocked.
- [ ] Old CPU stub fixture sets `allow_stub_action_for_contract_test=True` explicitly; all previously valid old tests stay green.
- [ ] Run full R25 new tests + existing R42 regressions in GitHub Actions, compare stdout.

### Task 3: CV and replay fail-closed (GREEN)

**Files:** Modify `m2_deploy_server.py`, `tests/test_r25_m2_deploy_server.py` (for explicit opted-in CV fixture).

- [ ] Add `allow_cv_diagnostic=False` parameter, require **literal bool True** plus `R42_POLICY=cv`, fail-closed when overridden silently.
- [ ] Expose `policy_mode` in per-action output; when enabled explicitly it must match CV task-specific token codec; other modes must remain VLA and no bypass.
- [ ] `replay` rejects use when `processor is not None` and complete pixel input was not preserved; stub-only accepted and tagged.
- [ ] No changes to original R15 mode or environment behavior.
- [ ] Run CI; fix any regression by minimal targeted test and source changes, never relax incorrect tests.

### Task 4: Independent evidence and PR

**Files:** Create `report/R26_M2_SERVING_SAFETY_AUDIT_20261022.md`.

- [ ] Report exact line-level findings and expected vs observed behavior, CI RED/GREEN run IDs, branch HEAD, and explicit remaining deployment blockers.
- [ ] Check changed filenames from GitHub compare: only new audit files and M2 path plus fixture tests, **zero edits to frozen R15 paths**.
- [ ] GitHub Draft PR base: `fix/r25-m2-deployment-contract`; do not merge into main or rewrite R25 PR.
- [ ] Final status: `VERIFIED` only for GitHub CI tested CPU guards, `BLOCKED` for real checkpoint/GPU/Isaac Sim/closed loop.

**If budget/time fails:** deliver completed tasks and leave partial line(s) explicitly BLOCKED, never simulate logs or claim future execution.
