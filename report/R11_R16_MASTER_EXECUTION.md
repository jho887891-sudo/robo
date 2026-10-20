# R11–R16 master execution report (first execution window)

Window: 2026-10-20 19:20Z → 20:05Z. Branch `feature/r11-p0-closeout` (base `r11/p0-audit@8a444d1`).
Spec `missions/R11_R16_DELEGATED_EXECUTION_SPEC.md@83b5794`, budget
`mission/R11_R16_DELEGATED_BUDGET_AND_GATES.json@83b5794`, main `329614f` untouched.

**GPU used 1027.5 s of a 115200 s allowance (0.285 h). Formal optimizer steps 0/3000.
Overfit optimizer steps 100/100. Collector attempts 0/40. Closed-loop rollouts 0/20.**

| stage | status | evidence |
|---|---|---|
| P0 (source-level gates) | **VERIFIED_CPU** | `evidence/r11_p0/`, suite 250 passed / 3 skipped |
| R11 collector | **BLOCKED_EXPERT** | `evidence/r11_collector_discovery/DISCOVERY.md` |
| R13 zero-update 7B smoke | **VERIFIED_GPU_ZERO_UPDATE** | `evidence/r13_zero_update_smoke_20261020T194055Z/` |
| R14 ≤100-step overfit | **OVERFIT_SIGNAL_PASS** | `evidence/r14_overfit_<utc>/` |
| R15 staged training | NOT_STARTED (budget untouched) | entry + gates implemented |
| R16 closed loop / FINAL | NOT_STARTED | needs R15 |

## R13 — the first real 7B 42D forward/backward on this project

One pristine OpenVLA-7B (modeling `b5431a07…`), fresh Gaussian LoRA (110.83 M trainable) and a
42D proprio projector (16.96 M), one TRAIN-only batch (`episode_0000003`, t=5):

* teacher-forced loss **14.4278**, LoRA grad norm **56.75**, projector grad norm **16.17**;
* `optimizer_steps = 0`, optimizer state empty, `opt.step()` never called;
* 14 legal action tokens `31873…31999` (9 distinct bins), decoded command inside the joint box,
  gripper dims 6/13 on the S+V fallback;
* common-prefix: appending 3 tokens changed the earlier logits by **0.0** and kept the window
  argmax (the pure-autoregressive path recomputes the full prefix, so no KV cache can diverge);
* checkpoint → zero the projector → reload → **identical 14 tokens**; 21-field metadata and the
  normalizer SHA round-trip equal;
* real tokenizer: prompt length 36 ending in `[29901, 29871]` — the trailing-space rule that P0-1
  added is confirmed on the live processor; `pixel_values [1,18,224,224]`, `proprio [1,42]`;
* 99.7 s wall, peak 22.16 GB of 48 GB, GPU snapshots before/after recorded.

## R14 — the mechanism really learns

100 optimizer steps, 32 fixed frames from 32 distinct TRAIN episodes, lr 5e-4, milestones
0/20/40/60/80/100 with full checkpoints (LoRA + projector + metadata) and archives:

| step | teacher-forced CE on the 32 fixed frames |
|---|---|
| 0 | 14.8141 |
| 20 | 1.9080 |
| 40 | 1.3046 |
| 60 | 0.5410 |
| 80 | 0.3798 |
| 100 | **0.1250** |

Training loss head mean 5.63 → tail mean 0.40 (first 16.49, last 0.0288). The greedy 14-token
decode went from 9 distinct bins (step 0) to **13 distinct bins** (step 100) — the policy is not
degenerate and produces decodable, non-constant actions. 857.8 s, peak 23.05 GB.

Honest limit: this is memorisation of 32 frames with no DEV evaluation and no visual ablation yet,
so it is a mechanism gate, **not** evidence of generalisation.

## R11 — collector blocked on a real capability, not on effort

Read-only discovery (evidence in `evidence/r11_collector_discovery/`): the RoboDojo stack on this
box exposes `scripts/robodojo.sh {eval,server,client}` and drives `XPolicyLab/policy/<NAME>/deploy.py`
as the *policy under test*; the Push-T HDF5 sitting in
`/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data` came from
`scripts/RoboDojo/download_data.sh`, i.e. it was downloaded, not generated here. The only
"expert" artefact in the container is
`XPolicyLab/policy/openvla_flat3/deploy_pybullet_expert_backup.py`, which belongs to the pybullet
variant, not to the Isaac Sim task.

There is therefore **no executable, auditable expert-trajectory source** for Push-T in the Isaac
Sim container. Per the spec I did not fabricate one out of random or all-zero actions, and I did
not spend the COLLECTOR bucket. Producing the 25 episodes needs one of:
1. a scripted geometric Push-T expert (IK + contact push) inside the container, validated to a
   real success rate on a pre-registered seed list — the honest next step, est. 1–2 days of CPU
   work plus the 4 h COLLECTOR bucket;
2. the original data generator / expert checkpoint, which is not on this workstation.

## What is genuinely missing before M1 can be answered

1. R15 staged training (300 → 1000 → 3000) with the DEV gates — the machinery exists
   (`training/r11_train_pushT_r42.py --steps`, `--bucket R15_TRAIN_FORMAL`, ledger-capped);
2. a DEV evaluation tool producing macro L1 / correction R² / amplitude ratio per stage, and the
   A=S / A=S+V / TRAIN-only affine-MLP baselines in the same sample;
3. R16 closed loop: the RoboDojo client/server pair exists and `code/robodojo/push_T_progress.py`
   already logs object geometry per control step, but no policy server for the 42D checkpoint has
   been wired to it yet, and latency p50/p95/p99 for 14 autoregressive 7B forwards has not been
   measured;
4. a GitHub credential for the Draft PR (no `gh`, no API token on the box).

## Deliverable honesty statement

Nothing in this window reached a closed-loop success count. The concrete, verified progress is:
the training/deployment source contract is now fail-closed and bit-exact, a real 7B 42D
forward/backward has been executed with gradients on both the LoRA and the projector with zero
optimizer updates, and the training mechanism demonstrably learns and decodes non-degenerate
actions. GPU hours consumed: **0.285 of 32**; formal steps: **0 of 3000**.
