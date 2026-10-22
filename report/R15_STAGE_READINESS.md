# R15 stage readiness and first real stage result

Branch `feature/r15-true-continuation` (from `feature/r11-p0-closeout@69031ed`).
Authoritative budget ledger: WAL at `/home/T7/ojh/robot_actual/r15_ledger` (outside any worktree).

## Lane T — the two P0 defects are fixed and gated on CPU

* **P0-RESUME-1**: `training/r15_resume.py` saves/restores LoRA + 42D projector + AdamW moments +
  LR scheduler + python/numpy/torch-CPU/torch-CUDA RNG + a resumable permutation sampler +
  global_step + the full data/normalisation/codec contract + base attestation, into a new
  directory via tmp→fsync→rename with a per-file sha256 MANIFEST. Loading verifies every hash,
  the schema, the optimizer format version, the parent global_step, the base identity and the data
  contract, and fails closed otherwise. A lossy-serialisation bug was found by the equivalence
  gate: `MultiStepLR.milestones` is a `Counter`, and a JSON round trip with `default=str`
  degraded it to a string so the resumed run never decayed the LR — the scheduler is now pickled.
* **P0-LEDGER-2**: `training/r15_ledger.py` is an append-only chained-hash WAL under an exclusive
  flock with fsync per event and an atomically rewritten projection that is cross-checked against
  the WAL. It guards GPU seconds (aggregate + bucket) **and** cumulative formal steps (3000), and
  refuses duplicate events, out-of-order seq, truncated/tampered WAL, a finished run_id and an
  externally edited projection. The 1027.5 s already spent is imported as a read-only event with
  its source sha256 — never reset.

CPU gates: **25 tests** (`tests/test_r15_ledger.py`, `tests/test_r15_resume.py`), including the
end-to-end equivalence *12 continuous steps == 5 + save + restore + 7 steps* on weights, AdamW
moments, LR, scheduler epoch and the next sampled frame, plus a control showing that a
weights-only reload (the old behaviour) does **not** reproduce it.

## Lane E — frozen DEV evaluator and pre-registered gates

`diagnostics/r15_dev_eval.py` (7 tests): 16-DEV cohort from the frozen split, one hashed 32-frame
gate list shared by every model, TRAIN-only affine/MLP fits, identical masks, episode-macro
metrics, and a frame-hash guard that refuses predictions made on a different frame list.
`mission/R15_GATE_PREREGISTRATION.json` was written **before any VLA DEV prediction existed**.

## R15 stage 1 (0 → 300 formal steps) — REACHED_TARGET

757.7 s, 1.922 s/step, peak 23.06 GB, full-state checkpoints at 50/100/150/200/250/300, ledger
booked 577.0 s and 300 formal steps (2700 remaining). Loss 1.3–2.3 with finite gradient norms
throughout.

### DEV result against the pre-registered baselines (same 32 gate frames)

| model | macro L1 | arm | gripper | correction R² | mag ratio | sign | still | clip |
|---|---|---|---|---|---|---|---|---|
| identity A=S | 0.007734 | 0.008477 | 0.003274 | -7.55 | 1.000 | 0.993 | 1.000 | 0.0045 |
| **constant velocity A=S+V** | **0.002110** | 0.002384 | 0.000469 | -0.022 | 1.001 | 0.998 | 0.000 | 0.0067 |
| TRAIN-only affine | 0.003526 | 0.003609 | 0.003023 | -0.231 | 1.000 | 0.837 | 0.000 | 0.1183 |
| TRAIN-only MLP | 0.017453 | 0.018110 | 0.013510 | -17.80 | 1.040 | 0.768 | 0.000 | 0.1250 |
| **VLA r42 @ step300** | **0.002134** | 0.002412 | 0.000469 | -0.024 | 1.001 | 0.855 | 0.000 | 0.0603 |

**G4 (pre-registered): PASS.** Finite loss, 100% legal action tokens, commands inside the joint box
after the safe clip, DEV macro L1 0.002134 strictly better than identity 0.007734 (-72%), no
collapse (still 0.000, magnitude ratio 1.001).

**Honest reading**: the model is at parity with constant velocity (-1.1%, i.e. marginally worse)
and its correction head has essentially not learned yet (R² -0.024). The 300-step model has
learned "predict a small action close to the velocity baseline", not the residual. The
pre-registered G5 bar for step 1000 is macro L1 **<= 0.002005**; reaching it requires the
correction to start carrying signal.

## Not done in this window

* stage 2 (300→1000, only 700 more) and stage 3 (1000→3000, only 2000 more) — the machinery is in
  place and the run_id/step plan is enforced, but the GPU wall clock was not available here;
* Lane D geometric expert: the Push-T success criterion is now known exactly from
  `/workspace/RoboDojo/task/RoboDojo/tasks/push_T.py` — `is_qpos_close(t, target_t, 7)` AND
  `xy_distance(t, target_t) < 0.007` AND `all_robot_back_to_origin()` — which makes a scripted
  expert implementable, but it was not built or attempted here (COLLECTOR bucket untouched, 0/40
  attempts);
* R16 closed-loop wiring and latency measurement.
