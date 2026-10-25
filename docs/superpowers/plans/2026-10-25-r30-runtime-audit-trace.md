# R30 Runtime Audit Trace: implementation plan

Date: 2026-10-25
Base: `9b94823d7af6909811bd3e7039db23d855c09559`
Branch: `feature/r30-runtime-audit-trace-20261025`
Related evidence: `evidence/pusht-startup-data-audit@0b974aea73f4ecf55c503d8528297c1e084f2045`

## Problem and evidence

R30 verified the dataset start state but could not compare it with the real R15 deployment because the policy-server process did not persist per-request proprio, decoded tokens/residual, or the response action. Existing R15 episode output stores the official task verdict, not the 14D policy boundary values. The client episode directory formerly under `/tmp/eval_result` is gone.

Code inspection at the fixed source base shows `OpenVLA_R42.model.Model.update_obs()` already receives the 14D state and three RGB arrays; `get_action()` receives the decoded 14D command, token IDs/bins, residual, velocity, 42D features, and decoder safety info from `R42DeployServer.act()`. These values can be logged at the server boundary without changing inference or action computation.

A second evidence-preservation risk was found: `scripts/r15_loop_pushT.sh` currently names server logs only by port and opens them with shell truncation (`>`), so reusing a port can overwrite an earlier server log.

## Chosen approach

Add an opt-in-by-path, fail-closed JSONL writer on the server boundary, enabled by the official R15 launcher. Do not modify RoboDojo's external simulator/client checkout, change the action contract, save image pixels, or run any rollout.

The trace will distinguish:
- observed 14D state arriving at the policy server;
- velocity and normalized 42D features used by the policy;
- model token IDs/bins and decoded residual;
- final 14D absolute action returned to the client, plus the four-key wire action dict and decoder safety metadata;
- next-step observed state compared with the previous server response (a diagnostic comparison, not proof that the command was applied);
- camera input SHA256 hashes, not raw images;
- episode/reset, sequence, timestamp, checkpoint/normalizer provenance and latency.

Every record is JSONL, flushed and fsynced. File creation is exclusive; an existing path, missing run ID, serialization error, or write error refuses the audited run rather than silently losing evidence. With no audit path set, the model's existing behavior remains unchanged.

The launcher will generate one unique run stamp and one run ID shared by the policy-server trace and the client ledger, and will use unique server log/output filenames so existing evidence cannot be truncated.

## Implementation tasks

1. Add CPU tests first for the writer: run-start record, ordered append-only events, no overwrite, required run ID, invalid JSON values refused, and exact SHA256/image metadata fields.
2. Add model integration tests with the existing stub backend: first state/action trace, token/bin/residual/feature fields, camera hashes without image bytes, reset episode indexing, previous-response vs next-state comparison, and opt-out preserving behavior.
3. Implement `r42_runtime_audit.py` using only the standard library. Open with `O_CREAT|O_EXCL`, mode 0600; use a schema version; add UTC/monotonic sequence metadata; flush/fsync each record.
4. Integrate in `OpenVLA_R42/model.py` behind `R42_AUDIT_LOG_PATH` and `R42_AUDIT_RUN_ID`. Log only after the action payload passes the existing 14-token contract. Do not pass new arguments into `server.act()` or change its result.
5. Add an optional `--run-id` to `tools/r15_loop_dev.py`; validate a conservative identifier format and use it for `ROBODOJO_RUN_ID`.
6. Update `scripts/r15_loop_pushT.sh` to produce unique, non-truncating server-log/output/audit names and pass the same run ID to server and client.
7. Run focused CPU tests, the full existing R42/R15 relevant CPU suite, compile checks, shell syntax check, and inspect GitHub Actions. Open a PR against `audit/pusht-startup-import-fix-20261024`; do not merge it.

## Acceptance / stop conditions

- No model weights, CUDA, optimizer, HDF5, checkpoint, or simulator touched during implementation.
- No changes to `main`, the R30 evidence branch, or historical evidence files.
- The logger never writes raw RGB pixels and never labels a server-generated action as an acknowledged/executed command.
- The next observation is compared against the previous server response, but any causal claim about actuator execution remains UNKNOWN without client/robot acknowledgement.
- If test infrastructure cannot run the required suite, report exactly which checks ran and which remain BLOCKED. Do not claim full verification from a subset.
