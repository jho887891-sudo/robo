# R5 source-based R4 audit, distribution-aware sampling and honest baselines

Base: `audit/r4-unified-action-target-ab@808abfa3866f193c71ffddf9338e5c2e8a02aa72`.

## 1. Grounded conclusion from R4 committed evidence

The original absolute vs residual 300-step comparison is not fabricated: its
`target_comparison.json`, `*_status.json`, and `*_config.json` agree:

- On 20 held-out frames, absolute position error = **0.5633775**,
  residual = **0.0143133** and hold-state = **0.0099364**, all in
  decoded physical 14D joint/gripper units.
- The residual is 39.36x better than absolute, but **still 1.44x worse
  than holding the current state**. This does NOT establish learned skill.
- R4 training selected 32 frames from only **5 of 80 TRAIN episodes**
  (9+6+9+7+1 frames). Its fixed 20 held-out frames were from only
  **10 of the 20 VAL episodes**, not 20 independent episodes.
- In the 20 residual validation free-generation rows, **dimensions 0
  and 1 each use precisely one action-bin value throughout** (127 and 139).
  Thus even 20 distinct 14D output rows do not prove every joint has
  meaningful state-dependent residual control. Other dimensions vary.
- The R4 double baseline `action[t-1]` and `S[t]` numerically coincide
  in this dataset. They are not two independent alternatives.

The standalone reproducible check is `diagnostics/audit_r4_evidence.py`;
the read-only GitHub Actions run executes this against committed R4 evidence.
Do not change historical experimental numbers.

## 2. R5 source improvements

- `training/r5_sampling.py`: deterministic TRAIN-only manifest from a
  motion-aware pool covering diverse episodes rather than the first few
  lexicographically sorted recordings. Selection cycles left-motion,
  right-motion, joint left+right activity, and timeline separation.
  It does **not** claim to guarantee all classes exist; an activity
  coverage manifest exposes which are unavailable.
- `spread_validation_frames()`: 60 frames from all 20 VAL episodes,
  approximately three distinct time-spread frames per episode instead of
  20 fixed frames from 10 episodes. Always checks episode disjointness.
  This is NOT equivalent to full-episode rollout or all-frame evaluation.
- `training/r5_metrics.py`: per-left/per-right and per-joint physical L1;
  per-episode macro L1 (every validation episode equally weighted);
  model, hold-state, constant-velocity `S[t]+(S[t]-S[t-1])` and
  constant **training-median** residual baseline `S[t]+median_train(A-S)`.
  Each baseline is calculated without peeking at validation action labels
  except to **score** predictions. Track true generated action-token
  uniqueness for each of 14 dimensions and predicted/expert displacement ratio.
- `training/overfit_pr1.py`: opt-in flags `--sample-strategy diverse`,
  `--val-strategy spread`, and up to 1000 updates, while preserving R4
  32-frame/300-update flags as defaults and keeping separated fresh output
  directories.
- `training/compare_target_runs.py`: rejects different sample manifests,
  learning budget, dynamic source snapshot, validation strategies,
  constant-delta baseline, or episode-macro baselines.
  On R5 runs its result records episode macro, per-arm metrics, constant
  velocity and median-delta comparisons, and token diversity.
- `scripts/run_r5_ab.sh`: read-only `--check` default, explicit
  `--execute` gate, 200 TRAIN frames across the 80 TRAIN episodes +
  60 VAL frames across 20 VAL episodes, two independent 800-update
  absolute/residual runs and strict comparator. New output directory;
  does not touch online RoboDojo, main, previous checkpoints or old R4 evidence.
- `.github/workflows/r5-cpu-contracts.yml`: syntax compilation, bash
  check, original regression contracts, 12 R5 CPU tests and R4 evidence audit.

## 3. Meaning and limitations

R4 is an under-representative 32-example *memorization* experiment, so the
39.4x advantage largely reflects how strongly absolute regression fails
on unseen joint-position ranges. Whether residual VLA actually predicts
visual-conditioned, task-progressing motions better than a
constant-velocity or training-median state-only baseline remains unknown.
Residual output diversity must be measured **before adding S**, not by the
number of unique final absolute joint commands.

This is also a mixed 14D physical L1 over rotational and gripper channels.
Do not describe all dimensions as radians without checking the raw units.
Use per-joint and per-arm results, and consider the BOUNDS-normalized L1
as an additional diagnostic if comparing across robot types.

## 4. Preregistered R5 success criteria

Before any larger 3000-step run or RoboDojo rollout:

1. A/B records the same 200 TRAIN examples and the same 60 VAL
   time-spread examples from all 20 episodes. No train/val overlap.
2. Compare models using identical decoded **absolute joint positions**
   and **per-episode macro L1**, not incompatible normalized targets.
3. On VAL, residual must beat `hold_state` globally AND must beat
   the train-median and constant-velocity baselines. Report each right-arm
   channel vs hold-state; a persistent right-arm deficit means
   **not accepted** even if aggregate metric looks good.
4. Both arms must exhibit **nontrivial, input-dependent generated residual
   token variation**, accompanied by displacement and per-frame action
   direction diagnostics. Token diversity alone cannot prove causation
   or task success; it only catches trivial constant output.
5. Any claimed improvement must be consistent across multiple
   validation episodes; one episode cannot dominate a frame-weighted mean.
6. No claims about RoboDojo closed-loop task success from R5 offline L1.

These criteria are deliberately stricter than simply observing that residual
beats the very poor R4 absolute baseline.

## 5. CPU validation, not GPU performance

GitHub Actions `R5 CPU contracts` at commit `9c59e5d1` passed:
- 13 target contract tests
- 4 A/B comparator legacy tests
- 9 R3 sampling/logging tests
- 12 R5 sampling/metric tests
- 20 action-codec tests
**58 unit tests passed**, shell syntax and Python compile passed, and
read-only `diagnostics/audit_r4_evidence.py` printed the grounded R4
5-train/10-val episode coverage and 1 unique bin for residual dimensions 0/1.

**No R5 A6000 experiment, no 800-step training, no simulation, no deployed
source synchronization has been performed.** CPU unit tests use synthetic
data; they do not establish HDF5 coverage or temporal dynamics on the
actual remote filesystem.

## 6. Commands for an approved remote operator

After checking out a NEW clean worktree on this R5 branch:

```bash
# This is read-only preflight; no model training.
bash scripts/run_r5_ab.sh --check

# Only with explicit approval and sufficient GPU memory:
bash scripts/run_r5_ab.sh --execute
```

The explicit execution mode logs all outputs under a NEW UTC-timestamped
`$PUSHT_RUNTIME/r5_ab_jobs/` directory, runs absolute and residual 800
updates independently, then compares them. The actual remote HDF5 sampling
manifest is recorded and fail-closed on train/validation overlap.

**Do not reuse or overwrite R4 result directories, train the 6000-step
model, change RoboDojo live policy, or merge this draft PR automatically.**
