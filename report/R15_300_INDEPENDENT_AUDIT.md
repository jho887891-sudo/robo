# R15 300-step independent audit (2026-10-21) | source + immutable evidence

Base executor commit: feature/r15-true-continuation@ca62c490f3655a3aa1ce30a2ebde046ee93bb475.
Previous executor: feature/r11-p0-closeout@69031eda6ee32439aa75d71b227a66a7dcb78aa2.
Previous independent audit: review/r11-r15-resume-ledger-gate-20261021@988b6bbe0ab1decbc768b8af558aadcc2f7a454c.
No GPU, optimizer step, HDF5 generation, sim rollout, or old-result rewrite was done by this audit. DRAFT executor PR#20, audit PR to be created.

## VERIFIED from raw source and archived data

- Genuine 300 formal optimizer updates: evidence/r15_stage300_20261021T034119Z/raw_train.log and stage_evidence.json show a real 7B load, global_step=300, checkpoints 50/100/150/200/250/300 with real SHA and AdamW state claim. P0 resume CPU equivalence tests exist and include 12 uninterrupted vs 5+save+7. **This only independently verifies the recorded artifacts/source and CPU claims, not that the remote GPU state or all checkpoint bytes were downloaded/reloaded by the auditor.**
- The fixed DEV comparison covers **32 frames /16 DEV episodes**, with 2 frames each; TRAIN/DEV at episode level distinct by manifest. On the identical DEV cohort, CV A=S+V macro physical14D L1=0.002110443304, R42@300=0.002134356982, VLA worse by 0.000023913679 (~1.133%). correction R² R42=-0.023977 versus CV=-0.021606; no demonstrated residual benefit. The strong CV baseline on mostly smooth joint trajectories is an important mechanism finding; 300 steps alone do not prove failure to solve Push-T closed loop.
- G4 criteria were committed to GitHub **before** training: mission/R15_GATE_PREREGISTRATION.json first introduced by 7fcbc32a2fd2fb0d82e24cecdbdd6726fb9d8fb2 on 2026-10-21 02:59:53 UTC; training raw starts 03:04:25 UTC, DEV prediction logged 03:39:30 UTC. The file misleadingly states written_utc=2026-10-21T11:05:00Z, likely a UTC+8 wall-clock incorrectly labeled Z. Treat original GitHub commit as stronger chronological evidence and append a timestamp erratum; do not silently rewrite original freeze.
- Expert data still BLOCKED_EXPERT, no newly collected25, no new offline FINAL; R16 not executed.

## Critical P0 before 300 -> 1000; unchanged gate status until repaired

### P0-SAMPLER-CONSUME: resume moves sampler by one silently

training/r11_train_pushT_r42.py::run_r15 code after RS.load_full_state:
```
sampler = out["sampler"]
log("resumed from %s at global_step=%d (next frame %s)"
    % (args.resume_from, start_step, sampler.next()))
sampler = out["sampler"]
```
This **calls sampler.next() for logging**, mutating the restored sampler position. The second assignment is the same sampler instance; it does NOT undo the mutation. The first formal update after step300 will skip the next frozen frame and thus violate the proven CPU standalone resume equivalence. The 12-step unit test in training/r15_resume.py does not cover this full 7B training entry integration. Fix the log to use a non-mutating sampler preview or order_hash, and add a CPU regression on the actual continuation function path (injecting fake backend as necessary). Assert next sampled (episode,t) exactly equals continuous step301 under the same seed and restored RNG.

### P0-BUDGET-CHARGE: only optimizer step seconds billed; load/save/eval unbilled

training/r11_train_pushT_r42.py::run_r15 records `gpu_seconds=time.time()-st` on each `step_commit`, but `job_end` has **gpu_seconds=0.0**. Evidence: stage wall_clock_s=757.7 vs sum(step WAL seconds)=576.967; unbilled job overhead **180.733 seconds**, including model load, checkpoint saves and other in-job GPU time. The user's original delegated budget explicitly counts all GPU-associated job time INCLUDING loads and saves; so 1604.5s reported as spent is an undercount for R15.
Furthermore `run_dev_preds` does actual real 7B inference on GPU for 32 DEV frames; raw_devpreds.log timestamps 03:34:07→03:39:30 UTC, ~323s of process walltime; there is NO R15_Ledger job_begin/commit/end for this evaluation. Book its actual start/end as evidenced rather than invented GPU time; avoid double booking overlapping subprocess intervals. Continue only after a separate immutably logged `audit_adjustment` for historical billable seconds and a budget-aware per-job wrapper that charges all future GPU job duration (or reconciles incremental step charges with wall clock as final adjustment). Crash, OOM, SIGTERM and timeout need failure-safe time charging, not only happy-path job_end. No deleting/re-writing old WAL/events. Current remaining global budget cannot be trusted without this correction.

### P0-DEV-METRIC: magnitude/sign metrics are wrong for next-state ACTION assessment

diagnostics/r15_dev_eval.py::metrics currently computes `magnitude_ratio=mean(abs(pred)/abs(target))` and `sign_agreement=mean(sign(pred)==sign(target))`. Here pred and target are *absolute joint positions*, NOT action deltas. These metrics can be ~1 even for identity A=S with still_action_rate=1 (observed identity ratio~1/sign~0.993). They do NOT show learned motion magnitude or action direction. Fix/create additional `delta_magnitude_ratio` and `delta_sign_agreement` on `pred-state` versus `target-state`, plus residual R sign/magnitude and arm-vs-gripper; use nonzero magnitude masks and test identity baseline zero direction. Keep old fields as legacy compatibility with clear `absolute_position_*` labels. The frozen G4/G5 gates refer vaguely to magnitude/sign; write a prospective signed protocol ERRATA before step1000 result and retain original prereg unchanged. Do not tune a criterion from step300 model outcome.

### P0-DEV-SAFETY-GATE: original G4 “every decoded command inside joint box” is contradicted by DEV stats

At step300 evidence/r15_stage300_20261021T034119Z/dev_eval.json reports `vla_r42.metrics.joint_clip_rate=0.060267857...` i.e. ~6.03% of (32x14) unbounded raw predicted joint entries are out of the box. Source training/r11_train_pushT_r42.py::run_dev_preds calls:
```
cmd, _r, _i = R.decode_command(bins, q01, q99, row["state"], row["velocity"])
```
without `limits`, `prev_cmd` or `dt`. Thus the recorded DEV metric is **raw, not post-safety-clamp**. G4 preregistration says every decoded14D command inside the joint box, yet report marks G4 PASS by referencing safe-clipped commands from another context. The program must explicitly produce both RAW and actual DEPLOY-SAFE commands, use the same physical/rate bounds, valid previous command and real control period semantics, and report violations and intervention rates separately. An out-of-bounds RAW prediction is not proof of unsafe deployed action if the deployed guard intervenes, but the reported G4 cannot be independently accepted as fully PASS until this is resolved. No future FINAL evaluation before physical safety gate.

### P1: tiny DEV sampling

2 frames/DEV episode =32 frames. The comparison favors no unnecessary extra compute but is insufficient for robust model selection; preserve the frozen primary 32-frame gate and optionally design a **separately preregistered** expanded DEV cohort for sensitivity analysis, strictly among existing 16 DEV episodes (never old20/new FINAL). Report per-episode paired differences/Wilson-like uncertainty on episodes for L1 improvement rather than treating frames independent.

## Action and stop policy

- Status: R15_300_STEPS_RECORDED / G4_FINAL_AUDIT=DISPUTED_PENDING_SAFETY_AND_BUDGET. It is **not** necessary to throw away the 300-step checkpoint, restart training, or invalidate all DEV measurements.
- Block only the **next R15 GPU job (300→1000)** until sampler, GPU bill, metric and safety contracts above are minimally repaired with red/green tests and immutable audit corrections. Other CPU work including scripted Push-T expert and R16 interface can continue.
- Under previous total ≤32 GPU-hour, ≤14 GPU-hour formal R15 bucket, ≤3000 formal steps, original R14 100-step separate; no new approval necessary after repairs. Pending new GPU action is +700 steps **from verified step300 full-state checkpoint**, not fresh1000.
- Keep the *original* prereg and reports with provenance; append a clearly labeled erratum and report new supplementary metric protocol without retrospectively moving thresholds. The old 20 repeatedly selected holdout is not FINAL; 25 new offline episodes still missing.
- Real claim must include fixed GitHub source SHA, actual remote checksums of full-state checkpoint, ledger WAL and projections, stdout/stderr/exit, sim/DEV actual values and independent recomputation. Source-code audit is not a new GPU execution.

## Source references

- feature/r15-true-continuation@ca62c490: training/r11_train_pushT_r42.py L576-750 and L963-1018, diagnostics/r15_dev_eval.py L126-169; training/r15_ledger.py.
- evidence/r15_stage300_20261021T034119Z/stage_evidence.json (300 steps; wall757.7), ledger_projection.json (step300 and 576.967s), raw_train.log, raw_devpreds.log (03:34→03:39 UTC).
- mission/R15_GATE_PREREGISTRATION.json committed 7fcbc32a2fd2 at 02:59:53 UTC.
