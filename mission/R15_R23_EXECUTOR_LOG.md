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
