# R44 — Full-data LoRA training on the joint-displacement target

**STATUS: BLOCKED (GPU budget not approved)**

The data contract, the new target, the normalizer, the epoch-complete sampler, the offline decoder
and the checkpoint metadata contract are implemented and verified on the real 300-episode dataset.
**No optimizer step was taken**: the preflight budget says this plan needs **605 GPU-hours** on the
A6000 and no budget has been approved. Per section 7 of the task ("先报告实际需要的预算，不得静默缩短
训练") the run was not started and no epoch was truncated.

| field | value |
|---|---|
| evidence dir | `evidence/r44-full-data-delta-lora-20261026T065550Z/` |
| code baseline | `d99635c4cebff1a5b8ee459ad60aea7695408bc6` (recorded, not a branch name) |
| research branch | `experiment/r44-full-data-delta-lora-20261025` |
| contract id | `joint_delta_v1` |
| normalizer sha256 | `9f7c4f00fc2945825cc892493c21127b5e38d4f83a212bf91f7cf410776ba459` |
| TRAIN / DEV samples | **108,780 / 28,268** (240 / 60 episodes, 3 tasks) |
| optimizer steps taken | **0** |
| GPU used | none (no CUDA context was created) |

## 1. What is done and verified on real data

1. **A_t == S_{t+1} is bit-exact.** All 300 episodes (240 TRAIN + 60 DEV), every frame:
   `max |A_t - S_{t+1}| = 0.0`, 0 violations. The displacement target is therefore exactly the
   measured S→S' motion, with no model or interpolation in between.
2. **`joint_delta_v1` is a separate contract, not a re-parameterised residual.**
   `training/r44_delta_v1.py` carries the target id, a sha256 over the whole normalizer payload and a
   `refuse_legacy()` guard. A payload whose `target` is `velocity_residual`, or whose quantiles
   were edited after fitting, is rejected by `validate_normalizer()`; this is asserted by tests
   `test_legacy_residual_payload_is_refused` and `test_tampering_breaks_the_hash`.
3. **Normalizer fitted on TRAIN only**, independently per task:

   | task | TRAIN rows | q99-q01 span (max dim) | gripper span (dims 6/13) | target std (max, arm dims) | zero-delta fraction |
   |---|---|---|---|---|---|
   | push_T | 25,313 | 0.2111 | 0.2000 / 0.2000 | 0.0260 | 0.0000 |
   | stack_bowls | 35,858 | 0.2000 | 0.2000 / 0.2000 | 0.0262 | 0.0000 |
   | sweep_blocks | 47,609 | 0.2272 | 0.2000 / 0.2000 | 0.0295 | 0.0000 |

4. **Codec round trip.** 400 random TRAIN rows per task: encode → 14 legal tokens (31744..31999) →
   decode → recover. For values inside `[q01,q99]` the error is ≤ half a codec step
   (`0.00392` of the q99-q01 span, i.e. the theoretical bound). **1.68% of push_T elements fall
   outside `[q01,q99]` and are clipped by the [-1,1] normalisation** (recorded per task in
   `report.json`, not hidden): the 256-bin codec cannot represent them, which is a property of
   the codec, not of this adaptation.
5. **Epoch-complete, no-replacement sampling.** For epochs 0..4 on the real 108,780-sample list:
   `unique_seen == total == 108780`, `min_visits == max_visits == 1`, order hash for epoch 0
   `fa60fdaa1d629a1c4a5c73f0654f2eb51ed5ed65b326f78918c873299f202e58`. The order is seeded by
   `(base_seed, epoch)` so it is reproducible and different every epoch.
6. **Offline decoder for the new contract**: `A_hat = S + Delta_hat` — no velocity term and no
   residual term anywhere in the path. `recover_action()` is the only implementation and it is
   tested against `decode_delta()`.

## 2. Why 3 tasks and not 2

`data/m2_splits.json` (sha256 `0c0c65eb…`) is the current frozen split and lists three tasks.
All three pass the same checks (3 cameras, 14D state/action, bit-exact A_t == S_{t+1}), so all three
are in the TRAIN manifest: 240 episodes, **108,780** samples — not the 61,171 of the two-task R24
reference. The R24 number is quoted in the task as a budget reference only; it is not used here.

## 3. The budget (measured, not assumed)

`sec_per_step = 3.9135` comes from the last real run of this stack:
`m2_two_2000_20261021T123314Z` ran 2,000 optimizer steps between 12:33:22Z and 14:43:49Z, i.e.
7,827 s / 2,000 steps. DEV evaluation uses `0.36 s/frame` (R43 measured 359 ms per teacher-forced
14-token forward).

| quantity | value |
|---|---|
| samples per epoch | 108,780 |
| optimizer steps per epoch (1 sample / update, as specified) | 108,780 |
| steps for the 5-epoch maximum | 543,900 |
| steps for the 2-epoch minimum | 217,560 |
| wall clock per epoch | **118.3 h** |
| wall clock, 2 epochs (minimum) | **236.5 h** |
| wall clock, 5 epochs (maximum) | **591.3 h** |
| full DEV evaluation, per epoch / total | 2.83 h / 14.1 h |
| **total GPU-hours required (5 epochs)** | **605.4 h** |
| checkpoints (every 0.25 epoch + epoch end, 20 total) | 44 GB at ~2.2 GB each |
| approved GPU-hours | **0** |

Batching would cut the *step count* but not the *sample cost*: the same 108,780 forward/backward
passes per epoch are still paid, so the realistic gain from a larger batch is GPU-utilisation only
(≲20%), which does not change the order of magnitude. The A6000 is also shared: an IsaacSim process
(2.3 GB) is resident, so the box is not exclusively available for a 25-day job.

**This is the number the owner has to approve.** No epoch was shortened, no step cap from the old
M2 1,000/2,000/3,000 ledger was reused, and no checkpoint was written.

## 4. What is therefore NOT delivered

* 0 optimizer steps, so: no `train_metrics.jsonl`, no `dev_metrics.jsonl`, no trained LoRA /
  projector weights, no `checkpoint_manifest.json`, no DEV MAE, no comparison against the
  zero-motion / constant-velocity baselines, no best-checkpoint sha256.
* The trainer integration (R44 batching, optimizer groups, warmup/decay schedule sized to 543,900
  steps, resume-from-mid-epoch) is **not** claimed as working: it would be untested without a real
  run, and writing it first would have burned GPU time or produced a false "ready" claim.

## 5. Artifacts in this evidence directory

`preflight/report.json`, `preflight/budget.json`, `preflight/dataset_manifest.json`,
`preflight/dataset_hashes.json`, `preflight/normalizer.json`, `preflight/sample_coverage.json`,
`preflight/training_config.json`, `preflight/environment.txt`, `preflight/console.log`,
`preflight/exit_code.txt` (2 = BLOCKED), `preflight/SHA256SUMS`.

Code (research branch): `training/r44_delta_v1.py`, `diagnostics/r44_prepare_delta.py`,
`tests/test_r44_delta_v1.py` (14 CPU tests, all pass).

## 6. How to launch once a budget exists

```bash
cd <worktree of experiment/r44-full-data-delta-lora-20261025>
python diagnostics/r44_prepare_delta.py --out /tmp/r44-preflight-<stamp>        --gpu-hours-available <approved>          # exits 2 unless approved >= 605.4
# then (trainer to be completed against this contract):
python training/r44_train.py --contract joint_delta_v1        --normalizer /tmp/r44-preflight-<stamp>/normalizer.json        --epochs 5 --checkpoint-every 0.25
```

## 7. What this report does NOT claim

No training happened, so nothing here says how well the displacement target learns. The contract is
verified; the learning outcome is unknown.
