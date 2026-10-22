# Why the 600-step Push-T run failed (workline A, 2026-10-22)

Status: **VERIFIED** — the mechanism is measured, not inferred, and it is not "the robot did nothing".

## 1. What the robot actually did

Three independent reads of the same official episode (step1000, layout 0, 600 steps,
`loop_pushT_stage1_r7`):

**(a) Frame-to-frame motion of the head camera** (601 frames, mean |grayscale delta| per step):

| steps | 0-100 | 100-200 | 200-300 | 300-400 | 400-500 | 500-600 |
|---|---|---|---|---|---|---|
| median motion | 0.11 | 0.09 | 0.07 | **0.02** | 0.03 | **0.01** |

The first ~10 frames carry the only large motion (1.0-1.9, the reset/settling). After step ~250 the scene
is at camera-noise level: **the robot moved briefly at the start and then held a static pose for the rest
of the episode.**

**(b) The frames themselves**: at t=0 the red T-object sits at the left of the table and the grey target
marker is at the centre; at t=240 the left arm has rotated/retracted and the object is untouched; at
t=600 the layout is the same. The object never travels toward the target.

**(c) The commands**, from the model's own DEV behaviour on the same checkpoint family (below): the
commanded 14D action is dominated by "keep the current velocity", with a residual correction that is far
too small to start a push.

## 2. The measured mechanism: the residual is under-commanded

Physical SAFE-14D audit (`tools/m2_physical_audit.py`, CPU, on the corrected action-window archive;
same 8 frozen DEV frames per task; SAFE = clip(S + V + R_hat, pmin, pmax)):

| task | model SAFE L1 | no-op (A=S) | constant velocity (A=S+V) | |R_model|/|R_true| | R² |
|---|---|---|---|---|---|
| push_T | **0.0014** | 0.0045 | **0.0009** | **0.47** | **-0.04** |
| stack_bowls | **0.0070** | 0.0092 | **0.0022** | 1.34 | **-1.28** |

Three facts, all needed to explain the loop:

1. the model does beat "hold still" on push_T (paired delta -0.0031, 95% CI [-0.0053, -0.0005]) — so it is
   not a pure no-op policy, which is why the arm does move at first;
2. it is **worse than a constant-velocity extrapolator on both tasks**, at every one of the ten
   checkpoints (below);
3. the predicted residual has about **half the magnitude** of the demonstration's residual and R² ≈ 0, i.e.
   the direction is essentially unpredictable from the inputs at this training scale.

Consequence in a control loop: the command A_hat = S + V + R_hat behaves like *coast with a weak,
randomly-directed nudge*. The arm coasts on the velocity it had, the residual cannot initiate a new push
toward the object, and as V decays the pose freezes. That is exactly the motion curve in §1a: a brief
initial move, then silence, with the object never displaced.

## 3. This is not a regression of the loop, and not a timing artefact

* the same checkpoint scores G5 **FAIL** offline (SAFE macro L1 0.0025385 vs CV 0.0020475, 0/4 motion
  strata), so the failure was predictable from DEV;
* the run used 600 steps of a synchronous policy at p50 5.5 s/action; the simulator advances one step per
  policy call, so the *simulated* trajectory is a valid 600-step rollout even though 64 minutes of wall
  clock were consumed;
* the SAFE clip fires on 12% of push_T dimensions, but clipping alone cannot explain a frozen arm: the
  unclipped RAW action already has L1 0.0009, the CV error.

## 4. What would have to change for a success

The gap is not "more steps of the same training". The physical error is flat from step 200 to step 2000
(§2 of the M2 report) while the probability CE keeps improving, so the current objective is not aligned
with the control quantity being judged. Candidate mechanisms, each with a distinguishing experiment:

| hypothesis | prediction | cheapest test |
|---|---|---|
| residual target too small to learn at this data scale (100 episodes/task) | |R_hat| stays ~0 and R² ~0 regardless of steps | already observed: flat across 10 checkpoints |
| 8-bit codec resolution limits the achievable L1 | achievable L1 floor ≈ one bin (~1.2e-4) | compare model L1 against the quantisation floor: 0.0014 vs 1.2e-4 = 12x, so **no** |
| proprio/history insufficient to infer the push direction | R² ~0 even with a linear probe on 42D | TRAIN-only linear residual fit was already measured on M1: DEV R² -0.14 |
| the task needs closed-loop visual feedback the dataset does not supervise (only 1-step imitation) | error grows with the horizon, DAgger-style relabelling needed | rollout with a scripted corrector, measure drift |

## 5. Deliverables tied to this report

* `mission/m2_physical_audit.json` — per-frame SAFE/RAW/no-op/CV L1, group split, motion strata, clip
  rate, residual magnitude and R², wrong-prompt counterfactual, paired CIs.
* `mission/m2_physical_curve.json` — the same quantities at all ten sealed checkpoints.
* the video-derived motion curve (`/tmp/ep1_motion.json` summary in this report) and the three frame
  stills used above.
* `tools/m2_physical_audit.py` — reproducible, CPU-only, no GPU and no re-inference.

## 6. The paired closed-loop control: the CV baseline also fails, and that refines the mechanism

A zero-parameter constant-velocity baseline was served through the **real** protocol (R42_POLICY=cv: the
tokens for a zero residual, so A_hat = S + V, same SAFE clip, same official client), job
`loop_pushT_cv_baseline_r10`:

| run | policy | official verdict | motion p50 / per-100-step medians | object moved? |
|---|---|---|---|---|
| `..._r7` | VLA step1000 (full path) | **0/1**, layout 0 | 0.046 : 0.106 / 0.092 / 0.071 / 0.020 / 0.031 / 0.009 | no |
| `..._cache_r8` | VLA step1000 (KV cache) | **0/1**, layout 0 | (same rollout) | no |
| `..._cv_baseline_r10` | constant velocity | **0/1**, layout 0 | 0.076 : 0.098 / 0.112 / 0.096 / 0.048 / 0.053 / 0.032 | no |

So **the trivial extrapolator that beats the model offline also fails closed-loop**. That is not a
contradiction, it is the most informative result of the round:

* offline L1 is measured on states **drawn from the demonstrations**, where V is non-zero by construction;
  a CV policy is therefore scored on the only state distribution where it is meaningful. A real episode
  starts from rest (V ≈ 0), where "continue at current velocity" is a *no-op* and cannot initiate a reach.
* the closed loop is governed by the distribution the policy **creates**, not the one it was scored on.
  Offline L1 on demo frames is a biased proxy; this is textbook covariate shift, measured rather than
  asserted.
* the VLA is still not rescued: it does initiate motion (26 steps above 0.2 vs 37 for CV, so it also
  decays), but its predicted residual has 0.47x the demonstration magnitude and R² ≈ 0, so what it
  initiates is not a push.

**Neither policy produces a reach-and-push trajectory.** The failure is not "the loop was too slow", not
"the cache", and not "the SAFE clip": it is that one-step imitation of this dataset, at this scale, does
not yield a controller that can generate the multi-step approach the task requires.

## 7. Consequences recorded for the next stage

* the 2->4 task expansion is **stopped** on the physical condition (see the M2 audit report), which is the
  pre-registered rule, not a judgement call;
* the metric that must gate future training is a **closed-loop** one (object displacement, minimum
  distance to target, sub-condition timeline), because the offline one has just been shown to prefer a
  policy that cannot start;
* the cheapest instrumented next step is the per-step privileged trace (object pose, contact, sub-conditions)
  described in workline A2, which needs an observation hook in the DEV rollout, not a new model.
