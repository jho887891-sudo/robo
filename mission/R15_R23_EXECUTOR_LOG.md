# R15–R23 executor log (append-only)

One line per real event. Times are UTC. Every claim here is tied to a file, a WAL event or a command
whose output is quoted in the linked evidence. Nothing here is a plan.

## 2026-10-21 — turn 1: state of the world as found

| when | what | evidence |
|---|---|---|
| 07:22Z | Remote repo at `feature/r15-true-continuation` HEAD `2598962`; WAL totals gpu_s=3268.4, formal steps=964, events=973 | `p0_out.txt` |
| 07:23Z | Spec V2 + DAG V2 extracted from git object `9f45d2e` (they are NOT in the working tree); spec 17713 B, DAG 10081 B | `/home/dgut/authoritative/` |
| 07:23Z | Disk: `/` 9.6 G free (97% used), `/home/T7` 1.1 T free on **ntfs-3g** | `p1_out.txt` |
| 07:23Z | GPU: 2.4 GB used by another user's Isaac job; kit PID 3111650 and a second agent (Codex) active on the box — not touched | `ps` snapshot |
| 07:48Z | Supervisor hardened and smoke-tested: real thread parallelism (max-cpu=4), FAILED->BLOCKED propagation, process-group timeout kill, atomic state file, per-job logs | `x_sup1_out.txt` (s1/s2 concurrent 3 s, s4 BLOCKED, s6 TIMEOUT) |

## Attempts and their real causes (kept, not hidden)

| # | what was tried | outcome | cause | evidence |
|---|---|---|---|---|
| 1 | `m1_700` via supervisor | FAILED exit=2 in 4 s, **0 GPU seconds** | the trailing platform token `arx_x5_push_T` is rejected by argparse; the trainer appends it itself in `force_arx_platform()` | `logs/supervisor/attempt1_failed_20261021T0750Z/` |
| 2 | `m1_700_r2` via supervisor | STALLED, killed at 0 GPU seconds after ~11 min | supervisor and trainer both take the same `GpuJobLock` flock -> the trainer waited for a lock its own supervisor held | `logs/supervisor/attempt2_stalled_20261021T0751Z/` |
| 3 | trainer started while the supervisor was being fixed | **ADOPTED** (not restarted) | restarting would have re-booked ~80 already-executed steps and polluted lineage accounting | `runs/r15_s2_1000_20261021T0803Z` |

## Corrections to the mission brief (verified against the box)

* **There is no `checkpoints/step963`.** The invalid segment's run `r15-s1000-20261021T061554Z` stopped at
  `global_step=963` with `verdict=PARTIAL` and its **last saved checkpoint is step950**; steps 951–963 were never
  persisted. Independently re-derived by audit lane H (`lanes/H_audit.md`), including four independent proofs that
  the resume consumed one sampler sample (`perm[300]=431` skipped in favour of `perm[301]=15645`).
* **"6.03% out-of-bounds" is a RAW coordinate rate on 56% of gate frames, not a deployment hazard**: max excursion
  0.03 rad, mean clamp magnitude 7e-5. RAW and SAFE are now reported separately and the gate is judged on SAFE.
* **G4_300 PASSES under the v2 definitions**, but under the SAFE口径 VLA@300 is ~1.0% *worse* than constant
  velocity (0.0020678 vs 0.0020475), i.e. G5 cannot pass at 300 by construction — as the pre-registration says.

## 2026-10-21 measurement that changes the schedule

`dd` on `/home/T7` (ntfs-3g): **2.9 MB/s** (256 MiB in 93.7 s), while `/` (ext4) does **632 MB/s**.
A 640 MB full-state checkpoint therefore costs **4–8 minutes**; step 350 took 7 min 21 s
(08:06:08 → 08:13:29). Consequences adopted: resume granularity raised to `--save_every 200` for later
segments, per-job wall estimates raised so the guard does not cut a segment between save points, and
stage 2 is run as 2a (300 → last saved step) + 2b (last saved step → 1000).

## 2026-10-21 08:45Z — stage 2 split into 2a/2b (measured, not planned)

| when | what | evidence |
|---|---|---|
| 07:54:40Z | 2a `job_begin`: `start_global_step=300`, 700 steps requested | WAL seq 994 |
| 08:03:04Z | resume verified in the log: `resumed from ... step300 at global_step=300 position=300  # peek only, never sampler.next()` | `logs/r15_s2_1000_20261021T0803Z.log` |
| 08:03:04Z | `sampler_order_hash = f1024210752cb42e0fbf60dd3270692c8d15555bb38b161eb87d22a032f48081` — **identical to the stage-1 run**, i.e. the sample order is the same as an uninterrupted continuation | `runs/r15_s2_1000_*/stage_evidence.json` |
| 08:13:29Z | save step350 took **7 min 21 s** (1.1 GB at ~1.5 MB/s) | log |
| 08:42:55Z | `stopping at global_step=500: bucket/time guard (2894s used of 2980s)` -> `R15 PARTIAL`, last saved **step500** | log + evidence |
| 08:42:56Z | `job_end` books `gpu_seconds=2269.106 = wall 2894.5 - booked steps 625.4`, i.e. **the load/save overhead is now billed** (audit lane H finding 2b) | WAL seq 1201 |
| 08:45:08Z | 2b launched by the supervisor: `500 -> 1000 (500 steps), estimate 6120s`, `--save_every 200` | `logs/r15_s2b_1000_20261021T0803Z.log` |

Stage 2 therefore produced exactly **+200 optimizer steps** in 2a and 2b continues the same lineage;
the discarded steps are the ones between the last save and the guard (none in 2a: it stopped exactly
on a save point at step500).

### supervisor bug found and fixed by evidence, not by inspection
`adopt_wait_trainer_exit` failed with exit 8 (`NO CHECKPOINT SAVED`) because the queue entry pointed at
`runs/r15_s2_1000_20261021T0758Z/checkpoints` — the stamp of the *failed attempt 1* — instead of
`..._0803Z`. The dependency chain then correctly marked 2b/dev-preds/gate/stage-3 BLOCKED, which is the
designed behaviour for a broken dependency. Fixed by enqueueing fresh ids with the right paths; the
BLOCKED ids are kept in the state file as evidence.

## Lane results (all with real evidence, deliverables under `lanes/`)

| lane | verdict | headline number |
|---|---|---|
| H audit | WAL chain PASS (0 breaks), step300/350/950 MANIFEST 27/27 reproduce, **no `step963` checkpoint exists** (run stopped partial at 963, last save 950) | 1014 step_commit events cover 963 distinct global_steps; 301 booked 3x |
| C TaskSpec | 12/12 installed and traceable, 0 BLOCKED_INTERFACE | `generalization` = exactly the 12 ids (`task_inventory.py:25-39`), 25 Hz control, dual_x5 = 14D |
| D expert | 6 realistic / 4 hard-but-rigid / **2 BLOCKED_DATA** (fold_clothes, pour_liquid) | success verdict only at `eval_env.py:895-897`; `k1_*` scripts are NOT legal experts |
| B R16 wiring | WS+msgpack, obs/action dict contract, gaps G1-G12 | **official FINAL is 25 episodes, not 30** (`_task.yml:105-107` + `main.py:325-329`) |
| M2 data | 3 tasks structurally identical, 0 schema work, language present | 2-task = 160 train eps / 61,171 rows; blocker = single normalizer has no task id |

## Mechanism result that frames the whole G5 question

`tools/r15_mechanism_diagnostics.py` on the frozen DEV frames (CPU, no model):

| model | DEV L1 | DEV R^2 |
|---|---|---|
| constant velocity (= predicting R = 0) | **0.002110** | 0 (by construction) |
| TRAIN-mean residual | 0.002333 | -0.0354 |
| TRAIN-only linear map on the 42D feature | 0.003429 | -0.1412 |

Because `A[t] == S[t+1]`, `|CV - A_t| = |R_t|` exactly, so the CV baseline IS the mean residual
magnitude. The 42D [S,V,Acc] features do not linearly predict R on DEV (negative R^2), so beating CV
requires the 3 RGB frames to carry residual information the proprio history does not. That is the
falsifiable claim G5 actually tests at step 1000 — and the reason a negative G5 is a data/residual
finding rather than a training bug.

## 2026-10-21 09:28Z — step 1000 reached (M1 training chain complete)

| when | what | evidence |
|---|---|---|
| 09:28:04Z | `r15-s2b-1000-…0803Z`: **REACHED_TARGET**, `500 -> 1000`, 2576.3 s, 3.039 s/step | `logs/r15_s2b_1000_*.log` |
| 09:28:02Z | step1000 MANIFEST written, 9 files hashed, `global_step=1000`, `position=1000` | MANIFEST.json + sampler.json |
| 09:28:04Z | `sampler_order_hash = f1024210…` identical to stage 1 | stage_evidence.json |
| 09:36:59Z | DEV greedy predictions at step 1000 written (32 frozen frames, global_step=1000) | `runs/r15_dev_gate_1000_20261021T0803Z.npz` |

The M1 formal budget is now **300 + 200 + 500 = 1000 optimizer steps** on one lineage whose parent is the
verified step300 checkpoint.

Two path bugs found and fixed by evidence (both fail closed, so nothing was silently scored):
* numpy's `savez_compressed` APPENDS `.npz`, so the archive was `…0803Z.json.npz` while the gate looked
  for `…0803Z.json` -> the first gate run correctly returned CANNOT_EVALUATE (exit 4), not a score;
* the supervisor DAG's `adopt_wait_trainer_exit` pointed at the failed attempt-1 directory.

## 2026-10-21 09:47Z — G5_1000 machine verdict: **FAIL**

`tools/r15_gate_check.py --gate G5_1000` on the frozen 32-frame gate set (frames_sha256 67b4320…):

| check | measured | passed |
|---|---|---|
| cohort.frozen_frames | matches pre-registration | yes |
| G5.macro_relative_5pct | VLA SAFE 0.0025385 vs CV 0.0020475 = **-23.98 %** | **no** |
| G5.strata_3_of_4 | 0 of 4 strata won (q1 -14.5 %, q2 +0.9 %, q3 -3.1 %, q4 -40.0 %) | **no** |
| G5.no_collapse | still_rate 0.0, delta magnitude 1.0508 | yes |
| G5.sign_mag_not_worse_20pct | delta sign 0.6897 vs CV 0.7857; magnitude 1.0508 vs 1.0403 | yes |
| **G5.pass_rule** | — | **no -> VERDICT FAIL (exit 3)** |

Progression across the lineage: step300 SAFE 0.0020678 -> step1000 SAFE 0.0025385. The model became
**23 % worse** with 700 more steps, while its decoded motion sign agreement stayed at 0.6897 — the
direction behaviour did not change, the magnitude error grew. `correction_r2` went +0.026 -> -0.121.

Per the pre-registered action the +2000 stage is **BLOCKED** (`m1_2000_seg3_to_3000` in the supervisor
state) and the checkpoint is kept. The mechanism evidence is in
`/home/dgut/r15_lanes/mechanism_diagnostics.json` and summarised in `report/R15_R23_STATUS.md`.

## 2026-10-21 10:17Z — closed loop: two real integration failures, then attempt 3

| attempt | outcome | root cause (from the log, not guessed) | fix |
|---|---|---|---|
| 1 (09:37) | exit 141, NO_RESULT | the container died importing `third_party/curobo`: `setuptools-scm was unable to detect version` | pass `SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0` + `CUROBO_KERNEL_BACKEND=cuda_core` with `docker exec -e` (what RoboDojo's own `_run_sim_client.sh` exports) |
| 2 (09:49) | exit 141 after 1541 s, NO_RESULT | the server answered `WsError: No module named 'prismatic'` because the host server lacked the OFT runtime on PYTHONPATH; and nothing had forced the ARX platform before the prismatic import (which would have decoded 14D actions against the LIBERO 7D table) | `PYTHONPATH` += OFT + vla-scripts; `force_arx_platform()` in `r42_serve.py`. Also fixed: a crashed run no longer consumes the 20-episode DEV budget |
| 3 (10:17) | running | — | — |

Isaac Sim reaches "Resetting all environments" ~530 s after launch, so a single-episode loop costs
roughly 25-30 min of wall clock before the policy is even asked for an action. That is why the driver's
own timeout is 3000 s and the supervisor's job timeout is 7200 s.
