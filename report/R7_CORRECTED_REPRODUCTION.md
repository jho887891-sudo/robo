# R7 corrected reproduction — verified run report

- Repository: https://github.com/jho887891-sudo/robo
- Audited branch: review/r7-state-dynamics-audit-20261020
- Audited commit: 3faed920421f02ac23e7a6a41050f4b5e93a762b (PR #8, state open / not merged)
- Remote host: jxxy (dgut@172.31.68.251)
- Isolated worktree on host: /home/dgut/r7_verify/repo_3faed92
- Run output directory: /home/dgut/pusht_runtime/r7_verify_r7_corrected_verified_20261020T054929Z
- This branch: audit/r8-pretraining-readiness

## 1. Provenance and preflight

The host has no GitHub egress, so the tree was shipped as a git archive and
re-verified rather than trusted:

| Check | Value |
|---|---|
| upstream commit | 3faed92 |
| upstream tree object | 1f381876aa7eb75ceb34395a4d8c08d3bf4aa857 |
| archive sha256 (local == remote) | 3e7040468313964b714be800659e98e857d3eee76664f82aaaca6896fb8576ff |
| shipped tree after git add -Af | 1f381876aa7eb75ceb34395a4d8c08d3bf4aa857 (TREE_MATCH=YES) |
| shipped local commit | 603990af542d8ed510e8298db3085ffd7c574376 (same tree, new parent) |
| worktree state before run | clean (0 dirty), branch review/r7-state-dynamics-audit-20261020 |

Data and environment:

| Item | Value |
|---|---|
| HDF5 | /home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data — 100 episodes, 7.3 GB |
| split file | /home/dgut/pusht_runtime/split80.json — 80 train / 20 val, split_seed 0 |
| pipeline partition | 64 train / 16 dev / 20 holdout (dev = sorted(train)[::5][:16]) |
| independent rebuild | identical to the pipeline lists; no episode overlap |
| python | /home/dgut/pusht_runtime/venv_oft/bin/python — 3.10.12, torch 2.4.1+cu121, h5py 3.16.0 |
| device | CPU only; no GPU job and no OpenVLA training started |

Other sessions were running on this host during the verification and were not
touched: PID 1359063 (training/r8_coverage_ablation.py inside
/home/T7/ojh/robot_actual/r8/wt), an Isaac Sim eval client, and an XPolicyLab
policy server. Every output path below is new and disjoint from theirs.

## 2. Stage results (all real runs, exit code 0)

Command: bash scripts/run_r7.sh --all with
R7_RUNS=/home/dgut/pusht_runtime/r7_verify_<ts>.

### 2.1 Unit tests and oracle diagnostics

| Step | Result |
|---|---|
| tests/test_r6_representable_oracle.py | passed |
| tests/test_r7_state_dynamics.py | passed |
| diagnostics/r6_representable_oracle.py | written |
| diagnostics/r7_oracle_compare.py — nearest 256-bin oracle | bound 0.0023258 > constant velocity 0.0018283 (ratio 1.272); bound below the floor encoder error 0.0034438 |
| diagnostics/r6_history_ceiling.py | written (history_ceiling_matched.json) |

### 2.2 Holdout baselines (20 episodes, 6,711 legal frames)

| Baseline | L1 |
|---|---|
| identity (S) | 0.007043 |
| constant velocity (S+V) | 0.001828 |
| train-fitted affine [S,V] | 0.002207 |
| train-fitted affine [V] | 0.001997 |
| train-fitted affine [S] | 0.008429 |

### 2.3 MLP grid on all legal training frames (3 seeds, mean)

| feature / target | L1 | macro L1 | R2 | corr(Rhat,R) | sign(R) | direction sign | episodes beating CV |
|---|---|---|---|---|---|---|---|
| SV / R | 0.001727 | 0.001724 | 0.311 | 0.564 | 0.663 | 0.957 | 14 / 20 |
| SV / delta | 0.001787 | 0.001780 | 0.301 | 0.555 | 0.657 | 0.956 | 10 / 20 |
| S / R | 0.002367 | 0.002347 | 0.012 | 0.232 | 0.574 | 0.943 | 0 / 20 |
| S / delta | 0.006546 | 0.006479 | -4.376 | 0.213 | 0.584 | 0.623 | 0 / 20 |
| V / R | 0.001761 | 0.001761 | 0.218 | 0.469 | 0.627 | 0.956 | 13 / 20 |
| V / delta | 0.001812 | 0.001811 | 0.199 | 0.453 | 0.618 | 0.955 | 12 / 20 |

Per-seed spread of SV/R L1 is 1e-5 (0.001715–0.001740), i.e. the seeds agree.

### 2.4 Three-frame acceleration arm (t >= 2 restricted, 6,691 frames)

| run | L1 | macro L1 | R2 | direction sign | episodes beating CV |
|---|---|---|---|---|---|
| SVAcc / R | 0.001204 | 0.001203 | 0.522 | 0.973 | 20 / 20 (all three seeds) |
| SV / R (same frame set) | 0.001696 | 0.001690 | 0.298 | 0.958 | 14 / 16 / 19 |

Improvement of SVAcc over constant velocity: 32.9 % (0.001204 vs 0.001795 on this
frame set). Paired per-episode test: 20 / 20 episodes improve, mean difference
-5.61e-4 (recomputed independently, matches the probe).

### 2.5 200-frame pilot (fixed epoch budget)

| seed | L1 | R2 | epochs executed | selection |
|---|---|---|---|---|
| 0 | 0.023284 | -42.06 | 377 | last |
| 1 | 0.028404 | -56.39 | 377 | last |
| 2 | 0.022758 | -40.90 | 377 | last |

Full-frame SV/R is 14.4x better than the 200-frame regime (0.001727 vs
0.024816). The pilot samples 80 episodes while full training uses 64, so this is
an observational coverage comparison, as the pipeline itself states.

### 2.6 DEV-selected channel hybrid (exploratory, not a gate)

DEV selection chose model-corrected dims [0,1,2,3,4,5,7,8,9,10,11,12] (it kept
constant velocity for dims 6 and 13, the two gripper channels). Holdout L1 with
that hybrid: 0.001208 vs model 0.001233 and CV 0.001795. The holdout was not used
to choose anything; the pipeline labels this arm exploratory.

## 3. The four audited fixes — verified

A. Direct-delta metric fix. reconstruct_prediction() returns the SAME ground truth
for both target modes: R_gt = (A - S) - V, and for the delta mode
R_hat = raw - V (r7_state_mlp.py:85-98). Independent check on all nine archived
sidecars: max |R_gt - ((gt - state) - V)| = 0.0 (float32 storage, exact).

B. 200-frame double-subtraction fix. A_hat = S + V + R_hat for every sidecar:
max |cmd - (state + V + R_hat)| = 0.0 across all nine files, including the pilot.

C. Fixed epoch checkpoint. The pilot trains with select_best=False, the epoch
budget is the median best epoch of the full-frame SV runs (377), the executed
count is recorded per seed (377/377/377) and checkpoint_selection = last for all
three. No epoch was chosen by looking at the 200 frames.

D. Gripper channel selection on TRAIN/DEV only. choose_hybrid_dims_on_dev() takes
dev predictions only; the resulting dims are recorded in the run JSON, and the
holdout evaluation happens once after the choice.

## 4. Pipeline self-verification: 17 / 18 claims pass

The single failing claim is pilot200_R_and_fixed_budget_verified_all_seeds:

| seed | L1 from sidecar | logged L1 | reconstruction ok |
|---|---|---|---|
| 0 | 0.0230980 | 0.0232844 | true |
| 1 | 0.0279598 | 0.0284044 | true |
| 2 | 0.0228084 | 0.0227582 | true |

### Root cause (measured, not inferred)

r7_state_mlp.py is invoked twice per run — once on all legal frames and once with
--restrict-t2 — and both invocations write the same sidecar names
(pred_SV_R_<seed>.npz at line 309, pred_pilot200_R_<seed>.npz at line 365). The
second invocation runs last, so its 6,691-row arrays overwrite the main
invocation 6,711-row arrays. The claim then compares the stale file with the main
JSON and fails by 1e-4.

Direct evidence:

| file | rows | L1 on disk | logged in state_mlp.json | logged in state_mlp_acc.json |
|---|---|---|---|---|
| pred_SV_R_0.npz | 6691 | 0.001721 | 0.001740 | 0.001721 |
| pred_SV_R_1.npz | 6691 | 0.001708 | 0.001725 | 0.001708 |
| pred_SV_R_2.npz | 6691 | 0.001659 | 0.001715 | 0.001659 |
| pred_pilot200_R_0.npz | 6691 | 0.023098 | 0.023284 | 0.023098 |
| pred_pilot200_R_1.npz | 6691 | 0.027960 | 0.028404 | 0.027960 |
| pred_pilot200_R_2.npz | 6691 | 0.022808 | 0.022758 | 0.022808 |

The earlier session run (r7_review_r7_corrected_verified_20261020T053550Z) fails
the same claim with the identical numbers to the last digit, so this is
deterministic, not flaky. Six files collide (three SV_R, three pilot200); the
SVAcc sidecars are unaffected because only the second invocation produces them.

Consequence: the numbers in the JSONs are internally consistent and independently
reproducible (see section 5), but the archived sidecars for the main invocation
are gone. Anything that wants to re-check the main-invocation SV_R or pilot200
predictions must re-run the pipeline with unique sidecar names.

## 5. Independent recomputation (verify_r7_independent.py, this branch)

The tool imports nothing from the training code; it reads the npz sidecars and
recomputes L1, macro L1, R2, correlation, sign agreement and the episode count
from scratch, then resolves every sidecar to the JSON it actually belongs to.

| Check | Result |
|---|---|
| split is 64/16/20 | true |
| no episode overlap | true |
| independent split rebuild equals the pipeline lists | true |
| R_gt identity (fix A) | true for all 9 sidecars |
| cmd identity (fix B) | true for all 9 sidecars |
| every prediction comes from a holdout episode | true |
| every sidecar resolves to a reported run | true |
| reported L1/R2 match recomputation (resolved sidecars) | true, < 1e-9, all 9 rows |
| sidecar collisions detected | 6 (three SV_R, three pilot200) |
| fix C budget enforced | true (377 executed, selection last) |
| fix C no train-set selection | true |

## 6. Direct delta versus R — the answer to question 2

Same inputs (SV), same budget, same seeds, same evaluation:

| metric | SV/R | SV/delta | winner |
|---|---|---|---|
| physical L1 | 0.001727 | 0.001787 | R by 3.4 % |
| macro L1 | 0.001724 | 0.001780 | R |
| R2 of the velocity correction | 0.311 | 0.301 | R |
| direction sign agreement | 0.957 | 0.956 | tie |
| episodes beating constant velocity | 14 / 20 | 10 / 20 | R, clearly |

R wins, but the L1 margin is small (3.4 %, ~6.5e-5). The qualitative difference is
coverage: with the delta target only half the holdout episodes beat constant
velocity, whereas R beats it in 14 of 20. This reproduces the audited claim
(R_target_better_than_delta_target, 3.36 % improvement).

## 7. Stage 2 — strict data-volume ablation (new experiment)

Fixed across every arm: the same 64 TRAIN episodes, the same 16 DEV / 20 holdout
episodes, the audited MLP (128,128), Adam lr 1e-3 (one sweep at 1e-4), batch 512,
standardisation from the training rows, target R, evaluation on all 6,691
t-eligible holdout frames, seeds 0/1/2. Only the number of training frames and the
optimisation budget change. Driver: training/r8_volume_ablation_agent.py.

Three budgets are reported separately because data volume and optimisation budget
cannot be varied at once:
equal_updates = 8000 optimiser steps, equal_epochs = 200 passes, dev_best = early
stop on DEV (patience 40, max 400 epochs).

| features / sampling | frames | budget | L1 | R2 | direction sign | episodes beating CV | mean|Rhat| / mean|R| |
|---|---|---|---|---|---|---|---|
| SVAcc / stratified | 200 | equal_updates | diverged | -2.0e9 | 0.680 | 0/20 | 3.4e4 |
| SVAcc / stratified | 200 | equal_epochs | diverged | -5.2e10 | 0.603 | 0/20 | 1.9e5 |
| SVAcc / stratified | 200 | dev_best | diverged | -4.1e10 | 0.613 | 0/20 | 1.7e5 |
| SVAcc / stratified | 2000 | equal_updates | 0.025746 | -587.7 | 0.878 | 0/20 | 14.3 |
| SVAcc / stratified | 2000 | equal_epochs | 0.064858 | -2521 | 0.807 | 0/20 | 36.1 |
| SVAcc / stratified | 2000 | dev_best | 0.045320 | -1268 | 0.840 | 0/20 | 25.2 |
| SVAcc / stratified | all | equal_updates | 0.001255 | +0.510 | 0.973 | 20/20 | 0.905 |
| SVAcc / stratified | all | equal_epochs | 0.001252 | +0.506 | 0.973 | 20/20 | 0.899 |
| SVAcc / stratified | all | dev_best | 0.001204 | +0.522 | 0.973 | 20/20 | 0.878 |
| SV / stratified | 200 | equal_updates | diverged (83.1) | -5.0e9 | 0.680 | 0/20 | 4.6e4 |
| SV / stratified | 2000 | equal_updates | 0.025719 | -614 | 0.831 | 0/20 | 14.2 |
| SV / stratified | all | equal_updates | 0.001866 | +0.284 | 0.956 | 7.3/20 | 0.784 |
| SV / stratified | all | dev_best | 0.001696 | +0.298 | 0.958 | 16.3/20 | 0.743 |
| SV / uniform | 200 | equal_updates | 0.011590 | -9.85 | 0.774 | 0/20 | 6.28 |
| SV / uniform | 2000 | equal_updates | 0.003009 | -0.289 | 0.931 | 0/20 | 1.49 |
| SVAcc / stratified, lr 1e-4 | 200 | equal_updates | diverged (427) | -8.0e10 | 0.594 | 0/20 | 2.4e5 |

Three things fall out of that table.

**Stability, not only coverage.** With the audited full-frame recipe the model is
well behaved, but at 200 and 2000 frames the same optimiser diverges: the
predictions become 4 orders of magnitude too large. Lowering the learning rate to
1e-4 does not help (L1 427), so this is not a step-size problem. The per-dimension
standardisation is computed from the sampled subset, and on a small stratified
subset some dimensions have a tiny spread, which turns ordinary inputs into
enormous z-scores. The audited pipeline avoids this only because its fixed 377
epoch budget on 200 frames stops before the blow-up (result 0.0233, stable but
18x worse than full data). Any small-data arm therefore needs a standardisation
guard, not just more epochs.

**The sampler matters more than the size.** Motion-phase stratified sampling makes
the small-data arms much worse than plain uniform sampling at the same size:
2000 frames gives 0.0257 (stratified) versus 0.0030 (uniform), and at 200 frames
stratified diverges while uniform stays finite (0.0116). Stratifying on |V|
over-weights the extreme-motion frames, which is exactly where the residual is
largest, so the fitted model over-predicts everywhere else.

**Only full data starts beating the baseline.** Every 200- and 2000-frame arm has
0 of 20 episodes better than constant velocity. The crossover to 20/20 happens
only with all 19,127 training frames.

### Answers to the five Stage-2 questions

1. Is 200 frames really insufficient to learn the velocity correction? **Yes.**
   Best stable 200-frame result is 0.0116 (uniform, 8000 steps) and the audited
   fixed-budget recipe gives 0.0233; full data gives 0.001204–0.001866. That is
   9x to 19x worse, and no 200-frame arm beats constant velocity in any episode.
2. Does 200 -> 2000 improve direction accuracy? **Yes, but not enough.** L1 falls
   0.0116 -> 0.0030 (3.9x) and direction sign rises 0.774 -> 0.931, yet the arm
   still loses to constant velocity on all 20 episodes and still over-predicts the
   correction magnitude by ~1.5x.
3. Does 2000 -> full continue to improve? **Yes.** 0.0030 -> 0.001866 (SV, same
   budget), and to 0.001252 / 0.001204 with the 42-dim features; R2 moves from
   -0.29 to +0.28/+0.52 and the episode count beating CV moves from 0 to 7/16/20.
4. How much do volume and updates each contribute? **At full data the budget is
   irrelevant** (0.001866 / 0.001252 / 0.001204 across the three budgets, within
   10 %); **at small data the budget decides stability** and the volume decides
   quality, and the two cannot be separated without the standardisation guard
   described above. This round therefore reports them separately and does not
   claim a causal split.
5. Does the model learn future velocity change rather than emitting zero?
   **At full data, yes**: mean|Rhat| / mean|R| is 0.74–0.92 with R2 0.28–0.52, so
   both the magnitude and the direction are learned. At small data it does not
   collapse to zero either — it over-predicts (1.5x to 14x) or diverges. A lower
   L1 alone would not have shown this; the ratio column is what distinguishes
   "learned" from "quiet".

Consequence for the R7 claim: "data coverage was the reason R6 failed" is
**supported but refined**. Coverage is the dominant factor, but two further
effects were hidden by the old 200-frame code: motion-stratified sampling can
make small-data training worse than uniform sampling, and small-subset
standardisation destabilises training regardless of the learning rate.

## 8. Where the evidence lives

All under evidence/r7_corrected_verified_20261020T054929Z/ in this branch:

- state_mlp.json, state_mlp_acc.json (all runs, per-seed detail)
- probe_main.json, probe_acc.json (paired tests, motion strata, correction structure)
- verification.json (the pipeline own 18 claims), verify.log
- oracle_compare.json, representable_oracle.json, history_ceiling_matched.json
- mlp_main.log, mlp_acc.log, probe_main.log, probe_acc.log, r7_all.log, r7_env.txt
- stage1_numbers.json (extraction of every headline number)
- independent_verification.json (my recomputation, including the six collisions)
- the nine prediction sidecars (17 MB, all seeds, as requested)
