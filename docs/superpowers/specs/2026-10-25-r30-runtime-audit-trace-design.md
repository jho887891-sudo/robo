# R30 Runtime Audit Trace Design

**Status:** Implementation in progress on an isolated feature branch. No rollout has been authorized or started.

## Context

R30 (evidence branch SHA `0b974aea73f4ecf55c503d8528297c1e084f2045`) verified the Push-T dataset start state but could not compare it with R15 deployment state because the old run did not persist per-request 14D proprio, model token/bin output, decoded residual, or server response action. The former episode artifact directory under `/tmp/eval_result` is no longer present.

The source checkout at `9b94823d7af6909811bd3e7039db23d855c09559` already has all policy-boundary values in memory inside `OpenVLA_R42.model.Model.get_action()` and `R42DeployServer.act()`. Persisting these values at that boundary is less invasive than changing the external RoboDojo client checkout.

## Goal

Make the next separately authorized R15 evaluation produce durable, step-level evidence to compare:
1. the 14D proprio observation received by the policy server;
2. the normalized 42D features and motion features used for inference;
3. the 14 token IDs / bins and decoded residual;
4. the final 14D action returned to the client after the existing decode/safety logic;
5. the next observed state relative to the previous server-generated response.

## Alternatives considered

- **Server-boundary trace (chosen):** the current server already holds the state, features, tokens, residual, decoder info and response action. This is the smallest versioned change and does not require the external simulator/client source.
- **Patch the RoboDojo client/environment:** could prove which command the simulator accepted, but the client checkout is external to this repo and not versioned in the current source tree. Do not silently modify that checkout as part of this task.
- **Infer from episode videos or scalar motion magnitude:** cannot recover the 14D proprio or the actual command. Rejected.

## Record contract

File: `runs/r15_loop_dev/runtime_audit_push_T_stage<stage>_<UTC stamp>.jsonl`

One JSON object per line; schema `r42-runtime-audit/v1`. Each record includes a shared run ID, monotonically increasing record index, UTC recording time and process-monotonic timestamp.

Required events:
- run_start: checkpoint/model metadata, normalizer/adapter/projector hashes, state and camera order, Python/runtime info, and explicit statements that raw images are not stored and wall-clock sensor synchronization remains UNKNOWN.
- episode_reset: reset count and episode index; clear previous-response comparison and episode-local observation/action indices.
- observation_received: persist the validated 14D state and per-camera SHA256/shape/dtype before inference, with a unique episode-local observation index. This survives a later backend or inference failure.
- backend_ready: backend class, device, checkpoint and metadata source.
- action_error: failure stage/type/message linked to the observation record/index. Capture backend-load, inference and response-contract failures. The trace writer itself remains fail-closed.
- action_step: reference the observation record/index and record velocity [14], normalized features [42], source timestamp if present, observation/reply timestamps, token IDs [14], bins [14], decoded residual [14], decoder safety metadata, final server response [14] and four-key action dict, latency, and the previous server response plus the current observed-state difference from it when available.

The response must be labeled **server-generated**, not client-acknowledged or confirmed as executed. The next observation provides a comparison point but is not proof of actuation. Camera/state wall-clock synchronization remains `UNKNOWN`.

Raw RGB pixel arrays and privileged state/goal labels must never be persisted by this logger.

## Durability and failure semantics

- Logging is disabled if `R42_AUDIT_LOG_PATH` is absent, preserving the existing model path for callers that do not opt in.
- If the path is present, `R42_AUDIT_RUN_ID` is mandatory.
- Open the file exclusively (`O_CREAT | O_EXCL`, mode 0600, and `O_NOFOLLOW` when available). Never truncate an existing trace.
- Parent directory must already exist. The R15 launcher creates it before starting the server.
- Each complete JSONL record is flushed and fsynced. Non-JSON-safe values or write/fsync errors raise and fail the request closed; the run must not silently continue without trace evidence.
- Server log, trace, and official result filenames share one unique UTC run stamp. The same run ID is passed to the server trace, the host-side ledger, and the Docker client via explicit `docker exec -e ROBODOJO_RUN_ID=...`.

## Non-goals

- No change to the 14D action contract, action decoder, normalization, safety limits, policy architecture or reset behavior.
- No claim that the action was executed just because the server generated it.
- No model loading, inference, training, optimizer step, CUDA, HDF5/checkpoint mutation, simulator execution or robot control during implementation/CI.
- Do not merge to main or alter the R30 evidence branch.

## Acceptance criteria

1. Unit tests prove exclusive creation, stable run IDs, ordered records, required event names, non-finite JSON rejection, and preservation of an existing file.
2. Stub-backend model tests prove that state, features, camera hashes, tokens/bins/residual, decoder info, server response and next-observation comparison are captured without raw image bytes.
3. R15 driver tests prove a caller-supplied safe run ID is propagated and unsafe IDs are rejected.
4. Launcher tests prove the trace and server log paths are timestamp-unique and the Docker client receives the same run ID.
5. GitHub Actions CPU tests pass; shell syntax and Python compilation pass.
6. PR remains unmerged. A real run requires a separate explicit decision after code review.
