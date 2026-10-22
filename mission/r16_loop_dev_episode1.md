# Official DEV closed loop — 1 episode, recovered verdict (2026-10-22)

This is the first time the pipeline produced a **complete official episode** rather than a timeout.

## Setup (all recorded by the supervisor job)

| item | value |
|---|---|
| supervisor job | `loop_pushT_stage1_r7` |
| checkpoint | `r15_s2b_1000_20261021T0803Z/checkpoints/step1000` (LoRA `de93b7fe…`) |
| task / stage / episodes | push_T / DEV stage 1 / **1** |
| official client | `docker exec rd_setup … robodojo.sh client … --eval-num 1 --seed 0` |
| driver wall | **3842.9 s** (driver timeout 5400 s, supervisor 9000 s) |
| control steps executed | **600** (the task's `step_lim`), 601 frames per camera video |

## Official result (the client's own `_result.json`, copied out of the container)

```json
{"success_rate": 0.0, "eval_time": 1, "score": 0.0,
 "details": {"0": {"layout_id": 0, "success": false, "score": 0.0}}}
```

Recovered record: `runs/r15_loop_dev/push_T_stage1_reparsed_20261022T0400Z.json` —
`verdict=OFFICIAL_FAILURE`, 1 episode observed, 0 successes, success rate 0.0,
Wilson 95% CI **[0.0, 0.7935]**, and the client's own stdout tally (`Success nums: 0, Fail nums: 1,
Unstable nums: 0`) agrees. Two independent reads, same answer.

## Measured inference wall (same run, in-process, 20-action windows)

| window | total p50 | p95 | p99 | forward p50 | vs 40 ms |
|---|---|---|---|---|---|
| 100 actions | 6749 ms | 8168 | 9172 | 5453 ms | 169x |
| 300 actions | 5585 ms | 7360 | 8825 | 5126 ms | 140x |
| **600 actions** | **5529 ms** | **7040** | **8166** | **5111 ms** | **138x** |

The cost settles around 5.5 s per control step once warm (the first window is 7.7 s). This is
**OFFLINE_SIM_ONLY**: the simulator advances in lock-step behind a synchronous policy call, so wall clock
inflates but the simulated timestep and control semantics are unaffected — which is why a bounded DEV
episode is still meaningful evidence. It is *not* a realtime claim and must never be quoted as one.

## Root cause of the failure

The policy ran its full 600-step budget without satisfying the Push-T predicate. This is consistent with,
not contradicted by, the offline evidence: M1 G5_1000 = **FAIL**, SAFE macro L1 0.0025385 against the
constant-velocity baseline's 0.0020475 (-23.98%), 0/4 motion strata won. A model that does not beat
constant velocity offline does not solve the task online. The 0/1 outcome is the expected consequence of a
measured weakness, and it is stated here as a data point (n=1) rather than a rate.

## Harness defect found and fixed

The driver recorded this run as **NO_RESULT** ("no _result.json was produced") although the client had
written one. Two causes, both fixed:

1. discovery searched the **host** tree while `--docker` runs write into the **container's**
   eval_result;
2. inside the container `eval_result` is a **symlink** to `/tmp/eval_result`, and `find` does not
   descend into a symlinked start point without `-L`.

`tools/r15_loop_dev.py` now discovers the result inside `--docker` with `find -L`, reads it there,
cross-checks the client's stdout tally, and gained `--reparse-only`, which recovers a verdict from an
already-completed run **without launching anything** and marks the record `reparse_only: true` so it can
never be mistaken for a fresh episode. Regression: `tests/test_r15_loop_result_recovery.py` (5 tests).

## What is NOT claimed

* Not a success rate: n=1, CI [0, 0.79].
* Not realtime: 138x over budget.
* No FINAL episode was touched; this is DEV, and the model was chosen earlier, never on FINAL.

## Addendum: the supervisor state for this job still says FAILED

`loop_pushT_stage1_r7` is recorded FAILED / exit 4 in the supervisor state, which was the driver's
NO_RESULT code **before** the result-discovery fix. The episode itself completed and its official verdict
was recovered from the client's own `_result.json` (see above): 1 episode, 0 successes,
Wilson [0.0, 0.7935]. The FAILED state is therefore a HARNESS verdict about parsing, not about the run, and
it is left in place (append-only) rather than rewritten.
