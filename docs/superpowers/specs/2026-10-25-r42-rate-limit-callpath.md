# R42 Rate-Limit Call-Path Contract

**Status:** Source-audited specification only. No action behavior has been changed. The first implementation must remain blocked until the unit contract and command-source semantics are reviewed.

**Audited base SHA:** `9b94823d7af6909811bd3e7039db23d855c09559`

**Related issue:** #46, “Audit: R42 policy path appears to bypass decoder max-rate limiting”.

## 1. VERIFIED source findings

1. `model.py::Model.get_action()` calls `server.act(images, state, env_idx=..., timestamp=...)` without `prev_cmd` or `dt`.
2. `r42_deploy_server.py::R42DeployServer.act(..., prev_cmd=None, dt=None)` forwards those arguments to `r42_adapter.decode_with_metadata()`.
3. `r42_adapter.py::decode_command()` explicitly skips rate limiting when both `prev_cmd` and `dt` are `None`; if only one is present, or the metadata lacks `max_rate`, it raises.
4. `Model` already validates `control_hz` and stores `control_dt = 1 / control_hz`; the current R15 config sets 25 Hz, i.e. a nominal 0.04 s period.
5. `r42_robodojo_adapter.py::R42RoboDojoAdapter` separately implements a per-environment previous-command latch. It seeds the first rate step from the observed state, applies the configured rate, and then stores the emitted command. The official `r15_loop_pushT.sh -> tools/r15_loop_dev.py -> robodojo.sh client` path does not invoke this wrapper.
6. Existing CPU tests cover the decoder rate branch and the standalone RoboDojo adapter, but do not prove that the official `Model.get_action -> R42DeployServer.act` path activates the branch.
7. `r42_metadata_bridge.py` defaults `max_rate` to a scalar 0.5 expanded across 14 dimensions. The frozen state layout is 12 arm-joint coordinates plus gripper dimensions 6 and 13; `r42_env_io.py` says gripper values are normalized/clipped to [0, 1]. The `r42_robodojo_adapter.py` docstring calls the whole 14D override “rad/s”, which is not a consistent unit description for the gripper dimensions.

## 2. Decision required before implementation

The code proves that the configured rate rule is bypassed on the official model path. It does **not** by itself prove which rate vector is safe for every dimension. Do not silently tune `max_rate` while wiring it into the live path.

Recommended contract:

- **Rate source:** use the checkpoint's frozen `metadata["safety_limits"]["max_rate"]`; do not fit or tune it on DEV/held-out episodes.
- **Units:** define each component as coordinate-units per second. Arm-joint dimensions must use the simulator's verified joint-coordinate unit; gripper dimensions use normalized-gripper units per second. If the checkpoint metadata cannot express or document this distinction, fix the schema/docs before claiming physical-unit semantics.
- **Time step:** use the frozen control period `dt = 1 / control_hz` only if the simulator contract confirms one policy response corresponds to one held control period. If the actual command interval is different, measure and define that contract before choosing elapsed-wall-time semantics.
- **Previous command:** seed the first step of each episode from the measured 14D state; subsequent steps use the previous rate-limited command emitted by the same policy environment. This matches the existing wrapper's semantics. A server-generated command is not proof of client acknowledgement or actuator execution.
- **Reset:** clear the held command and seed again from the first post-reset measured state. Never carry a previous command across episodes.
- **Single enforcement point:** the official server/model path must not depend on an uninvoked client wrapper. Choose one rate-limiting owner and explicitly prevent accidental double limiting if the wrapper path is used elsewhere.
- **Failure:** out-of-box previous command, non-finite/non-positive `dt`, malformed rate vector, or incompatible physical/rate bounds must fail closed. Do not silently fall back to “physical bounds only” in the official deployment path when rate limiting is required.

## 3. Minimal implementation proposal

**Preferred minimal patch:** wire the existing `Model.get_action -> R42DeployServer.act -> decode_with_metadata -> decode_command` path using the already validated `control_dt` and a reset-scoped previous-command value. Update the held command only after the action passes the existing output contract and is successfully prepared for return. Keep the 14D output shape, joint order, normalization, token codec and physical bounds unchanged.

The separate `R42RoboDojoAdapter` path should remain behaviorally consistent, but it must not add a second rate limiter on top of a server path that already limits the command. Decide whether the server is the single source of truth or whether the wrapper remains the only limiter before implementing either route.

## 4. Required tests (TDD, CPU only)

1. **First action:** stub the server, provide measured state (S_0), and prove the actual model call receives `prev_cmd=S_0` and `dt=control_dt).
2. **Second action:** prove the actual model call receives the previous rate-limited 14D command, not the raw model target or a stale episode command.
3. **Rate bound:** for every dimension, assert (|A_t - A_{t-1}| leq max\_rate \cdot dt + tolerance).
4. **Physical bound:** simultaneously assert `amin <= A_t <= amax` for all 14 dimensions.
5. **Reset:** after reset, the first rate reference is the new measured state; no previous episode command leaks.
6. **Failure modes:** invalid `dt`, malformed/missing `max_rate`, out-of-range previous command and incompatible bounds fail closed with the documented exception/hold policy.
7. **No double limiting:** prove the official model path applies the rate rule exactly once; separately prove the wrapper path cannot accidentally apply a second limiter.
8. **Trace:** log the rate source, control `dt`, previous-command vector, `rate_limited` count and final 14D response, while retaining the existing distinction between server-generated output and actual execution.
9. **Regression:** existing action codec, normalizer, 42D shape, gripper fallback and physical-limit tests remain unchanged and pass.

## 5. Acceptance gates

- **Gate A, source contract:** rate units, `dt`, reset semantics and previous-command source are explicit in code/docs.
- **Gate B, CPU:** all tests above pass on the exact commit; no model weights, CUDA, simulator or robot needed.
- **Gate C, real-model smoke:** one GPU inference smoke proves the real metadata carries a valid 14D rate vector and the live model/server path invokes the limiter. This is not a rollout.
- **Gate D, controlled evaluation:** exactly one separately authorized R15 episode, using the intended checkpoint, with audit trace/report inspected before any expansion. Stop if the first-response or trace-integrity gate is blocked, or if action bounds/rate checks fail.
- **Forbidden:** no main-branch change, no checkpoint/HDF5 mutation, no training, no rollout without separate authorization.

## 6. Evidence boundary

This document is a source-derived proposal, not an implementation or proof of closed-loop safety. The current source path appears to skip the adapter's rate limiter; other simulator/controller limits may still exist and remain UNKNOWN. No action behavior was changed while preparing this spec.
