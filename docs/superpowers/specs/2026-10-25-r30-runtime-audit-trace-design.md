# R30 Runtime Audit Trace Design

**Status:** Implementation committed on an isolated feature branch and ready for review. No rollout has been authorized or started.

## Context

R30 (evidence branch SHA `0b974aea73f4ecf55c503d8528297c1e084f2045`) verified the Push-T dataset start state but could not compare it with R15 deployment state because the old run did not persist per-request 14D proprio, model token/bin output, decoded residual, or server response action. The former episode artifact directory under `/tmp/eval_result` is no longer present.

The source checkout at `9b94823d7af6909811bd3e7039db23d855c09559` already has all policy-boundary values in memory inside `OpenVLA_R42.model.Model.get_action()` and `R42DeployServer.act()`. Persisting these values at that boundary is less invasive than changing the external RoboDojo client checkout.

## Goal

Make the next separately authorized R15 evaluation produce durable, step-level evidence and an automatically generated gate report to compare:
1. the 14D proprio observation received by the policy server;
2. the normalized 42D features and motion features used for inference;
3. the 14 token IDs / bins and decoded residual;
4. the final 14D action returned to the client after the existing decode/safety logic;
5. the next observed state relative to the previous server-generated response;
6. whether the JSONL is structurally complete, observations are linked to actions/errors, and the first post-reset response exists.

## Alternatives considered

- **Server-boundary trace (chosen):** the current server already holds the state, features, tokens, residual, decoder info and response action. This is the smallest versioned change and does not require the external simulator/client source.
- **Patch the RoboDojo client/environment:** could prove which command the simulator accepted, but the client checkout is external to this repo and not versioned in the current source tree. Do not silently modify that checkout as part of this task.
- **Infer from episode videos or scalar motion magnitude:** cannot recover the 14D proprio or the actual command. Rejected.

## Record contract

Trace file: `runs/r15_loop_dev/runtime_audit_push_T_stage<stage>_<UTC stamp>-pid<PID>.jsonl`
Report file: `runs/r15_loop_dev/runtime_audit_report_push_T_stage<stage>_<UTC stamp>-pid<PID>.json`

One JSON object per line; current schema `r42-runtime-audit/v2` (the validator remains able to read v1 traces, but v1 traces cannot pass the new rate-compliance gate without the explicit rate-reference evidence). Each record includes a shared run ID, monotonically increasing record index, UTC recording time and process-monotonic timestamp.

Required events:
- run_start: checkpoint/model metadata including checkpoint MANIFEST SHA256 and embedded-normalizer payload fingerprint, normalizer/adapter/projector hashes, state and camera order, Python/runtime info, and explicit statements that raw images are not stored and wall-clock sensor synchronization remains UNKNOWN.
- episode_reset: reset count and episode index; clear previous-response comparison and episode-local observation/action indices.
- observation_received: persist the validated 14D state and per-camera SHA256/shape/dtype before inference, with a unique episode-local observation index. This survives a later backend or inference failure.
- backend_ready: backend class, device, checkpoint and metadata source.
- action_error: failure stage/type/message linked to the observation record/index. Capture backend-load, inference and response-contract failures. The trace writer itself remains fail-closed.
- action_step: reference the observation record/index and record velocity [14], normalized features [42], source timestamp if present, token IDs [14], bins [14], decoded residual [14], decoder safety metadata, final server response [14] and four-key action dict, latency, and the previous server response plus the current observed-state difference from it when available. The trace uses `pre_audit_return_wall_utc` and `server_span_pre_audit_ms` for timestamps sampled before the action-step fsync, and records `observation_audit_overhead_ms` (camera hashing + observation record persistence).
- Schema v2 additionally records the exact `rate_reference_action14`, its source (`measured_state` on first step after reset or `previous_server_response` later for that environment), `rate_limit_source`, and `control_dt`. The report independently verifies the reference linkage and checks every `abs(action14 - rate_reference_action14) <= configured_max_rate14 * control_dt + tolerance`.

The response must be labeled **server-generated**, not client-acknowledged or confirmed as executed. The next observation provides a comparison point but is not proof of actuation. Camera/state wall-clock synchronization remains `UNKNOWN`.

Raw RGB pixel arrays and privileged state/goal labels must never be persisted by this logger.

## Durability and failure semantics

- Logging is disabled if `R42_AUDIT_LOG_PATH` is absent, preserving the existing model path for callers that do not opt in.
- If the path is present, `R42_AUDIT_RUN_ID` is mandatory.
- Open the file exclusively (`O_CREAT | O_EXCL`, mode 0600, and `O_NOFOLLOW` when available). Never truncate an existing trace.
- Parent directory must already exist. The R15 launcher creates it before starting the server.
- Each complete JSONL record is flushed and fsynced. Non-JSON-safe values or write/fsync errors raise and fail the request closed; the run must not silently continue without trace evidence.
- Server log, trace, audit report, and official result filenames share one UTC timestamp plus the Bash process ID. The launcher refuses existing paths before launch; the PID suffix avoids the seconds-only collision where two concurrent launchers could both pass a preflight existence check.
- The launcher installs a finalizer after starting the server. On every later exit path, it stops the server, runs the report validator, and preserves a report even if the trace file is missing. If the client exits successfully but the audit gate fails, the launcher returns a nonzero audit-gate exit code.
- The same run ID is passed to the server trace and host-side ledger; the driver forwards it into the Docker client via `docker exec -e ROBODOJO_RUN_ID=...`.

## Post-run validator and gates

`tools/r42_runtime_audit_report.py --input <trace.jsonl> --out <report.json>` reads the source trace without repairing or rewriting it. The report contains the trace SHA256, record counts, schema/reference errors, unresolved observations, and explicit gates.

- `trace_integrity`: PASS only when JSONL/schema/record indices/references validate; PARTIAL for an incomplete tail; FAIL for invalid or unavailable trace input.
- `first_observation` and `post_reset_observation`: require a valid observation and a reset record that precedes it in the same episode.
- `first_server_response`: requires a successful `action_step` linked to the first observation.
- `rate_limit_branch_active`: requires a valid configured 14D `max_rate` vector and every successful action's decoder `dt` to be positive and equal to the configured control period. A null `dt` means the decoder skipped the rate branch and blocks the gate. This is a post-run diagnostic gate, not a substitute for reviewing/fixing a known unsafe call path before authorizing a rollout.
- The validator checks that flattening the four-key action dictionary in the frozen joint order reproduces `server_response.action14`, and that the recorded previous-response delta equals `state14 - previous_server_response_action14`.
- Every report includes a `latency_caveat`: the writer fsyncs each record, `server_span_pre_audit_ms` includes observation audit overhead but excludes the action-step fsync, and the actual client receive time is not measured. Do not compare these timings directly with unaudited runs.
- `client_acknowledgement`, `camera_state_sync`, and `robot_actuation`: always remain UNKNOWN unless another independently verified source is added.
- `reset_vs_demo_S0`: remains BLOCKED until a separately verified demo-start vector is supplied; the report does not invent it.

Exit codes: 0 means structurally valid, complete evidence with no action errors and required first-action/rate-limit gates PASS; 2 means partial evidence, action errors, or a blocked required gate; 3 means invalid/unavailable trace; 4 means the report output cannot be written safely. Existing report paths are never overwritten.

## Non-goals

- No change to the 14D action contract, action decoder, normalization, safety limits, policy architecture or reset behavior.
- No claim that the action was executed just because the server generated it.
- No model loading, inference, training, optimizer step, CUDA, HDF5/checkpoint mutation, simulator execution or robot control during implementation/CI.
- Do not merge to main or alter the R30 evidence branch.

## Acceptance criteria

1. Unit tests prove exclusive creation, stable run IDs, ordered records, required event names, non-finite JSON rejection, and preservation of an existing file.
2. Stub-backend model tests prove that state, features, camera hashes, tokens/bins/residual, decoder info, server response and next-observation comparison are captured without raw image bytes.
3. R15 driver tests prove a caller-supplied safe run ID is propagated and unsafe IDs are rejected.
4. Launcher tests prove process-unique artifact names, report generation on all post-launch exits, and a nonzero final gate when the trace is missing or incomplete.
5. Report-validator tests cover complete traces, action errors, missing files, incomplete tails, malformed identities, broken references, reset ordering, and report no-overwrite behavior.
6. GitHub Actions CPU tests pass; shell syntax and Python compilation pass.
7. PR remains unmerged. A real run requires a separate explicit decision after code review.
