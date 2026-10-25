# R42 Rate-Limit Call-Path Contract

**Status:** Frozen implementation contract for the isolated Issue #46 branch. CPU-only implementation; no rollout authorized.

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

## 2. Frozen contract used by this implementation

The call-path bug is VERIFIED. The contract below wires the existing checkpoint rate rule into the official model path without tuning its values. This is a software contract, **not** proof that the configured rates are physically safe for every task.

- **Rate source:** use the checkpoint's frozen `metadata["safety_limits"]["max_rate"]`; the current R15 bridge default expands `0.5` to all 14 dimensions. Do not fit or tune it on DEV/held-out episodes.
- **Units:** interpret each component as *that coordinate's native state units per second*. The 12 arm-joint coordinates and two normalized gripper coordinates do not share one physical unit. The code does not infer or claim radians for all 14 dimensions. The existing checkpoint vector remains unchanged.
- **Time step:** use `dt = 1 / control_hz`. The R15 config sets 25 Hz (`0.04 s`); `r42_env_io.py` rejects observations whose declared frequency differs from the configured control rate, and the standalone adapter contract says each returned command is held for one control period.
- **Previous command:** the first step after reset is seeded from the measured 14D state. Later steps use the previous server-generated, rate-limited 14D response for the same environment. This is not proof that the client accepted or the robot executed that response; those remain UNKNOWN.
- **Reset:** `Model.reset()` resets all environment histories and clears all held commands. The next observed state seeds the next episode's first rate step.
- **Single enforcement point:** the official `Model.get_action -> R42DeployServer.act -> decode_with_metadata -> decode_command` path applies the decoder's limiter. `R42RoboDojoAdapter` remains a separate standalone wrapper path; do not stack it around the official `Model` path.
- **Failure:** missing/malformed `max_rate`, missing or mismatched decoder `dt`, non-finite action, physical-bound violation, rate-bound violation, or a reported decoder conflict fails closed. No silent fallback to physical bounds alone on the official path.

## 3. Minimal implementation proposal

**Implementation:** `Model` now requires a valid 14D checkpoint rate vector, passes the reset-scoped previous command and configured `control_dt` to `server.act`, checks decoder branch evidence and independently verifies physical/rate bounds, then latches the returned command only after the response contract and audit persistence succeed. The 14D output shape, joint order, normalization, token codec, physical bounds and checkpoint rate values are unchanged.

The standalone `R42RoboDojoAdapter` still owns rate limiting only when used directly; the official model path owns it inside the decoder. The documentation now describes per-coordinate units per second instead of calling the heterogeneous 14D vector uniformly `rad/s`.

## 4. Required tests (TDD, CPU only)

1. **First action:** prove the official model call passes the measured state as `prev_cmd` and `dt=control_dt`.
2. **Second action:** prove the official model call receives the previous rate-limited 14D response, not the raw target or a stale episode command.
3. **Rate bound:** for every dimension, assert `abs(A_t - A_{t-1}) <= max_rate * dt + tolerance`.
4. **Physical bound:** simultaneously assert `amin <= A_t <= amax` for all 14 dimensions.
5. **Reset:** after reset, the first rate reference is the new measured state; no previous episode command leaks.
6. **Failure modes:** missing `max_rate` and missing/mismatched decoder `dt` fail closed; existing decoder tests cover invalid `dt`, malformed vectors, out-of-range previous command and incompatible bounds.
7. **No double limiting:** the official path consumes the decoder's final action without a second wrapper clamp; the standalone wrapper remains a separate integration path.
8. **Trace:** R30 logs the configured rate vector, control period, decoder `dt` / `rate_limited` count and final 14D server response, while retaining the distinction between server-generated output and actual execution.
9. **Regression:** existing action codec, normalizer, 42D shape, gripper fallback and physical-limit tests remain unchanged and pass.

## 5. Acceptance gates

- **Gate A, source contract:** rate units, `dt`, reset semantics and previous-command source are explicit in code/docs.
- **Gate B, CPU:** the targeted official-path tests and existing R4/R5 contracts pass on the exact commit; no model weights, CUDA, simulator or robot needed.
- **Gate C, real-model smoke:** BLOCKED pending separate authorization. A single GPU inference smoke must prove the real checkpoint metadata and live model/server path invoke the limiter; this is not a rollout.
- **Gate D, controlled evaluation:** exactly one separately authorized R15 episode, using the intended checkpoint, with audit trace/report inspected before any expansion. Stop if the first-response or trace-integrity gate is blocked, or if action bounds/rate checks fail.
- **Forbidden:** no main-branch change, no checkpoint/HDF5 mutation, no training, no rollout without separate authorization.

## 6. Evidence boundary

This document now describes the CPU implementation and its explicit contract, but it is not proof of closed-loop safety. The previous official call path bypassed rate limiting; that code path is fixed on this branch. Other simulator/controller limits, client acknowledgement, actual actuation and physical safety remain UNKNOWN. No model load, CUDA, training, checkpoint/HDF5 mutation, simulator or robot execution occurred.
