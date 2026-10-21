# M2 — the ONE shared multi-task trainer (gate P6_M2_TWO)

Lane: M2 / gate `P6_M2_TWO`. Additive only: five new files, no existing file touched, no git commit,
no push, no GPU work, no simulator.  Every number below was produced on the dev box
(`172.31.68.251`, repo `/home/T7/ojh/robot_actual/r11`) on **CPU only**; the 7B model was
never loaded and the running R15 job (PID 3111650 / 2737390) was never touched.

Repo HEAD moved under me while this lane ran — `97cd5e8` (brief) -> `c6c0059` (the M2 data-contract
commit) -> `e648c88` -> `6b7d058` — two other agents are committing in that worktree.  The lane's five
files are the only `m2_*` paths in `git status` that it added.

---

## 1. What is built

| file | lines | what it is |
|---|---|---|
| `training/m2_task_normalizer.py` | 658 | task-aware normalizer contract (versioned multi-task payload, per-task TRAIN-only fits, fail-closed validation, whole-payload sha256) |
| `training/m2_train_shared.py` | 1174 | the ONE shared trainer entry: task prompts/ids, balanced resumable sampler, per-task normalizers, per-task loss + per-task DEV CE, P6 evidence, hard caps, WAL booking, full-state resume |
| `tools/m2_gate_check.py` | 562 | machine verdict for `P6_M2_TWO`: exit 0 PASS / 3 FAIL / 4 CANNOT_EVALUATE + JSON report |
| `tests/test_m2_task_normalizer.py` | 441 | 17 CPU tests (payload completeness / fail-closed / provenance / DEV-fitted rejection / sha coverage) |
| `tests/test_m2_gate_check.py` | 472 | 19 CPU tests (gate PASS/FAIL/CANNOT paths, routing collapse, cap, forgetting, CE math, mix accounting) |

Everything below the model boundary is pure numpy; torch is imported only inside
`m2_train_shared.run()` and inside `r11_train_pushT_r42.build_model()`, which the trainer reuses
verbatim (LoRA r=32 all-linear, 42D projector, 14 action tokens + STOP(2), `force_arx_platform()` before any
prismatic import).

### 1.1 The task-aware normalizer contract (the blocking issue, solved without editing anything)

Payload (`training/m2_task_normalizer.SCHEMA = "m2-task-normalizer-v1"`):

```json
{"schema": "m2-task-normalizer-v1",
 "tasks":   {"push_T": <14D payload>, "stack_bowls": <14D payload>},
 "task_id": {"push_T": 0, "stack_bowls": 1},
 "fitted_on": {"push_T": {"train_episodes": [...80 names...], "n_train_episodes": 80,
                          "dev_episodes_excluded": [...20 names...], "n_dev_episodes_excluded": 20,
                          "train_episodes_sha256": "...", "split_sha256": "...",
                          "payload_sha256": "..."}}}
```

Per-task 14D payload fields: `normalization_mode, pmin, pmax, q01, q99, v_scale, a_scale,
motion_quantile, n_train_episodes, n_dev_episodes_excluded, fit_episodes, dev_episodes_excluded,
action_q01, action_q99, n_rows, fit_source, quantile_semantics, alignment_version,
target_representation, gripper_dims` (+ `task/task_id/task_token/prompt/proprio_mode/proprio_dim`).

**Field-name decision (documented because the repo has two naming schemes):**
`q01/q99` are the **velocity-residual** quantiles — byte-compatible with the frozen single-task
contract's `meta["q01"]/meta["q99"]` that `tools/make_frozen_contract.py` produces from
`r8_contract.velocity_residual_quantiles` and that `r42_adapter.build_labels()` already consumes.
`action_q01/action_q99` (the raw action quantiles from `TaskNormalizer.to_metadata()`) are carried
alongside for provenance only and never feed the codec.

API: `fit_task_normalizer(task, split, index=, hdf5_root=, min_rows=, hash_episodes=)`,
`fit_multi_normalizer(tasks, splits, indexes=)`, `build_multi_payload(blocks)`,
`payload_for(payload, task)`, `decode_context(payload, task)`, `contract_for(payload, task)`,
`assert_payload_complete(payload, tasks=, dev_lists=)`, `assert_payload_task_set`,
`normalizer_sha256(payload)`, `assert_same_normalizer`, `save_payload/load_payload`,
`describe`, plus a CLI that fits and writes the payload.

Fail-closed (each raises `M2NormalizerError`): unknown task; missing/extra task; wrong frozen task id
or duplicate id; `tasks/task_id/fitted_on` disagreeing; a block whose 14D vector has the wrong shape,
non-finite values, `q99<=q01`, `pmax<=pmin`, non-positive motion scales, or a drifted
`task/task_id`; a block with `n_dev_episodes_excluded == 0` (nothing held out => "TRAIN only"
unproven); a block whose `fit_episodes` intersect its own `dev_episodes_excluded` **or** the
authoritative DEV list the caller passes; a block whose recorded DEV list differs from that split; a
block edited after fitting (its `payload_sha256` no longer matches).  `normalizer_sha256()` refuses
a partial payload, so a checkpoint can only pin a complete task set.

### 1.2 The shared trainer

`python training/m2_train_shared.py --tasks push_T,stack_bowls --target-global-step 2000 --out DIR
  --resume-from CKPT --save_every 200 --max-tasks-stage 2 --est-seconds-per-step 3.5 --time-margin 300
  --seed 0 --cams 3 --bucket TWO_TASK_PILOT [--task-floor 0.30 --dev-frames-per-task 8
  --include-task-token --splits data/m2_splits.json --normalizer PAYLOAD.json --lr 5e-4
  --milestones 1000,2500 --log-every 10 --plan-only]`

* **Task conditioning** — one real instruction per task (`"What action should the robot take to <instr>?"`
  around the frozen `/instruction` string: push_T "Push the T-shaped block to align it precisely with the
  gray T-shaped pad.", stack_bowls "Stack the three bowls together.", sweep_blocks "Pick up the broom, hand
  it over to the right hand, then use the dustpan to sweep the blocks.").  `assert_prompts_distinct()`
  refuses a task set whose prompts collapse (identical strings, or identical tokenised ids once the
  processor is loaded).  The frozen numeric task id is the lookup key for the normalizer
  (`payload_for`) and is recorded in every checkpoint; the human-readable `<task:name#id>` token is
  available with `--include-task-token` (off by default: the dataset guarantees only the instruction).
* **Balanced sampling** — `M2_MIX_POLICY = "proportional_to_train_frames_with_floor"`:
  weight = max(floor, train_frames_t/total), renormalised; per-task slot counts by largest-remainder
  apportionment over a horizon of one epoch; slot order by a deterministic largest-deficit interleave;
  `schedule_sha256` + `FrameSampler.order_hash()` recorded in evidence and in every checkpoint, and
  `r15_resume.FrameSampler.from_state` compares `frames_sha256` on resume, so a changed mix cannot be
  resumed by accident.  The sampler frame list is the QUALIFIED `"<task>/<episode>"` name + t (bare names
  collide across tasks).
* **Per-task normalizers** — `row_for()`/`task_contract()` use that task's
  `NormalizationContract`; `labels_for()` calls `r42_adapter.build_labels` with that task's
  `q01/q99` from `decode_context()`.  There is exactly one code path and no global fallback.
* **Per-task logging** — `per_task_loss{task: [{global_step, loss, episode, t}]}`,
  `per_task_loss_summary{task: {n, first, last, first_window_mean, last_window_mean, min, max}}`,
  `steps_per_task`; every log line prints the last loss of *every* task so a task that stops learning is
  visible immediately.
* **Forgetting control** — at step 0 and at every save point the trainer evaluates each task's DEV
  teacher-forced CE (and, for the gate, the **counterfactual** CE under the other task's prompt) and appends
  `ce_curves{task: [{global_step, ce, ce_full, frames_sha256}]}` to the evidence.  The per-frame
  distributions go to `<out>/dev_eval_latest.npz`.
* **Hard caps** — `CAPS = {1: 1000, 2: 2000, 3: 3000}` cumulative multi-task optimizer steps per stage:
  a target above the cap, a task set larger than `--max-tasks-stage`, or a resume already above the cap
  raises BEFORE any GPU work, and the R15 ledger must have enough formal steps left.
* **Bookkeeping** — `r15_ledger.Ledger`: `job_begin` (with the task mix, cap, normalizer sha), one
  `step_commit` per optimizer step (with task, loss, grad-norm, lr, episode, t), `job_end`; bucket
  `TWO_TASK_PILOT`.  `r15_resume.save_full_state` per save point with
  `extra={tasks, task_ids, max_tasks_stage, stage_cap, task_mix, normalizer_sha256,
  per_task_normalizer_sha256, ce_curves, steps_per_task, base_model, modeling_sha256, target}` and a
  `data_contract` that pins tasks + per-task payload shas + mix + split sha.  Resume calls
  `RS.verify_and_load_manifest` then `assert_resume_compatible(...)` (task set, normalizer sha, mix sha,
  stage) then `RS.load_full_state(expected_contract=, expected_base=, expected_global_step=)`.

### 1.3 The gate

`python tools/m2_gate_check.py --evidence stage_evidence.json --eval dev_eval_latest.npz
  --normalizer PAYLOAD.json --splits data/m2_splits.json --out report.json
  [--ledger-dir ... --forget-tol 0.10 --routing-min-win-rate 0.9 --routing-min-ce-penalty 0.05]`

Checks (all must pass for PASS; any CANNOT input => exit 4 and no verdict):

| id | condition |
|---|---|
| `P6.dev_disjoint` | every scored frame is a DEV episode of the same task in `data/m2_splits.json` (a TRAIN episode => FAIL) |
| `P6.ce_vs_noop.<task>` | per-task DEV CE < the no-op baseline (A_hat = S, R_hat = -V) on the SAME frames |
| `P6.ce_vs_constant_velocity.<task>` | ... and < the constant-velocity baseline (A_hat = S+V, R_hat = 0) |
| `P6.routing_not_degenerate.own_prompt_wins` | on every frame the model is better with its OWN task prompt than with any other (ties are not wins) |
| `P6.routing_not_degenerate.wrong_prompt_penalty` | the mean CE paid for a wrong prompt exceeds a floor (>= 0.05 nats) |
| `P6.routing_not_degenerate.not_constant` | the model does not emit one constant bin vector for every frame of every task |
| `P6.no_catastrophic_forgetting.<task>` | every save-point CE <= that task's own step-0 CE + `--forget-tol` |
| `P6.cap.{task_count,cumulative_steps,stage_completed,booking_matches_steps,wal_steps}` | <= 2000 for 2 tasks, the stage reached its target, booked == executed, the WAL has one `step_commit` per step |

Baselines are **recomputed by the gate** from the HDF5 + the payload on exactly the archive's frames
(never copied), and the archive's own `bins_true` is re-derived and must match — otherwise CANNOT.

**Design note (a judgement call, see §9):** the repo's two baselines are point predictors scored with L1.
A CE gate needs a distribution, so each baseline is scored as the same proper score of the same target:
a hit on the exact quantisation bin costs 0 nats, a miss costs `ln 256 = 5.545` nats, i.e.
`CE_base = (1 - hit_rate) * ln 256`; hit rate and L1 are reported next to it.  Routing is tested with
**counterfactual prompts** rather than marginal distributions, because each task's targets are
standardised by its own quantiles and the marginals *should* look alike; the counterfactual test fails
exactly when the model ignores the task tag (win rate 0, penalty exactly 0).

---

## 2. Tests — real command, real tail

```
$ /home/dgut/pusht_runtime/venv_oft/bin/python -m pytest \
    tests/test_m2_task_normalizer.py tests/test_m2_gate_check.py -q
....................................                                     [100%]
36 passed in 30.51s
```

Per file, same interpreter, same run:

```
$ ... -m pytest tests/test_m2_task_normalizer.py -q
.................                                                        [100%]
17 passed in 34.56s
$ ... -m pytest tests/test_m2_gate_check.py -q
...................                                                      [100%]
19 passed in 10.51s
```

(The first cold run of the pair took `36 passed in 56.61s`.)  Full log: `/home/dgut/r15_lanes/M2T/pytest_m2_trainer.log`.
CPU only, no `torch` import in either file, no HDF5 model, no GPU.  Two tests are live-data tests
(`@real_only`) that now RUN on the box (the ARX-X5 root is mounted) and are skipped automatically
elsewhere.

Coverage highlights: missing/extra task, wrong frozen id, duplicate id, blocks disagreeing, 13D vector,
`q99<=q01`, `pmax<=pmin`, non-positive/NaN motion scale, zero held-out episodes, edited block
(provenance sha), DEV-fitted payload rejected three different ways, whole-payload sha sensitivity
(one task's scale changes it, provenance changes it, key order does not, partial payload cannot be
hashed), JSON round-trip, resume task-set guard; gate PASS with the real decision path, model worse than
both baselines, prompt-ignoring model (CE passes but routing FAILs — the collapse case), catastrophic
forgetting, 2001 steps, 3 tasks in a 2-task stage, unfinished stage, TRAIN episode in the DEV frame list,
missing archive, tampered frames, archive missing a task, payload missing a task, tampered `bins_true`,
wrong evidence schema, CLI report writing; and the trainer accounting: mix reproducibility + floor +
prefix drift <= 1 slot, qualified resumable slot list, cap table, resume guard, prompt distinctness,
exact teacher-forced CE / 256-bin window CE math.

---

## 3. CPU evidence produced on the real data (this lane, real numbers)

### 3.1 The real multi-task normalizer payload

`$ python training/m2_task_normalizer.py --tasks push_T,stack_bowls,sweep_blocks
    --out /home/dgut/r15_lanes/M2T/m2_task_normalizer.json`   (45.6 s, CPU)

```
normalizer_sha256 = 930213f515773f7ac5e7975f60054065fa6236b2fb881d7fa245bbceb4065927
residual_q99_spread_ratio = 2.442350799965546
  push_T        n_train=80 n_dev=20 rows=25313  residual q99 abs max 0.02127884328365326  v_scale max 0.12601086
  stack_bowls   n_train=80 n_dev=20 rows=35858  residual q99 abs max 0.05197039991617203  v_scale max 0.12130199
  sweep_blocks  n_train=80 n_dev=20 rows=47609  residual q99 abs max 0.04153493419289589  v_scale max 0.13643351
```
(2.44x spread — the audit's 0.0213 / 0.0520 / 0.0415 reproduced exactly, so a single global normalizer
remains unacceptable.)  The 2-task payload used by the stage is the sub-payload of the same fits:
`normalizer_sha256 = 43a166c30f7b8eac9a8bbd64d4e2e8224cc2c6e634379356136807c326f1f714`,
per-task block shas `7278affd…` (push_T) and `caa0916e…` (stack_bowls).

### 3.2 The real 2-task 2000-step plan (`--plan-only`, CPU, no model)

```
caps:  {cap: 2000, max_tasks_stage: 2, steps_requested: 2000, tasks: [push_T, stack_bowls]}
tasks: [push_T, stack_bowls]  task_ids {push_T: 0, stack_bowls: 1}
mix.policy       proportional_to_train_frames_with_floor
mix.floor        0.30
mix.weights      push_T 0.4138071962204313   stack_bowls 0.5861928037795687
mix.counts       push_T 25313                stack_bowls 35858        horizon 61171
schedule_sha256  22e2c116abfe282a37aa9c89a7421c806b602300af62580a7827bfbee6129478
normalizer_sha256 43a166c30f7b8eac9a8bbd64d4e2e8224cc2c6e634379356136807c326f1f714
dev_frames (8/task, fixed for every eval)
  push_T      ep_0000091@196 ep_0000002@210 ep_0000094@155 ep_0000041@240
              ep_0000048@209 ep_0000087@130 ep_0000016@143 ep_0000063@101
  stack_bowls ep_0000081@276 ep_0000004@140 ep_0000084@268 ep_0000037@204
              ep_0000051@177 ep_0000016@124 ep_0000080@309 ep_0000010@122
```
(Verbatim JSON: `/home/dgut/r15_lanes/M2T/m2_plan_2task_2000.json`.)  With 2 tasks the floor does not
bind (both weights exceed 0.30); with 3 tasks it does (push_T 0.2327 -> 0.30), which the tests cover.

### 3.3 The measured baselines on the real DEV frames

A **synthetic** archive (a fake model that emits the true bins — this proves the gate's real-data path and
measures the baselines; it proves nothing about training) was scored end to end on the real HDF5 with the
real 2-task payload:

```
$ python /home/dgut/r15_lanes/M2T/plumbing_check.py
DEV push_T      ce_window=0.0000 ce_full=0.0000 nats over 8 frames
DEV stack_bowls ce_window=0.0000 ce_full=0.0000 nats over 8 frames
verdict: PASS exit: 0
push_T      baseline_ce_noop=2.6241 (hit 0.527, L1 0.38057)  baseline_ce_cv=2.6736 (hit 0.518, L1 0.00094)
stack_bowls baseline_ce_noop=3.3667 (hit 0.393, L1 0.46287)  baseline_ce_cv=3.3172 (hit 0.402, L1 0.00220)
```

So the bar a trained shared model must clear on these 8-frame DEV sets is **2.6241 / 2.6736 nats for
push_T and 3.3667 / 3.3172 nats for stack_bowls**.  Note how different the two tasks' baselines are — a
single global normalizer/decision threshold across tasks would be visibly wrong.  Report:
`/home/dgut/r15_lanes/M2T/plumbing_gate_report.json`.

The same artefacts through the **CLI** (which additionally cross-checks the R15 ledger) return FAIL —
correctly, because that synthetic evidence claims 2000 optimizer steps that were never booked:

```
$ ... tools/m2_gate_check.py --evidence .../plumbing_evidence.json --eval .../plumbing_fake_eval.npz \
    --normalizer .../m2_task_normalizer_2task.json --splits data/m2_splits.json \
    --ledger-dir /home/T7/ojh/robot_actual/r15_ledger --out .../plumbing_gate_cli_report.json
gate_cli_exit=3
failed: ['P6.cap.wal_steps']   measured: {run_id: m2-plumbing-cpu-only, wal_steps: 0, executed: 2000}
```

and a **negative control** on the same artefacts with a 1-task normalizer returns
`CANNOT_EVALUATE` (exit 4, `inputs.consistent` false) instead of scoring a subset of the tasks:

```
$ ... --normalizer /tmp/one_task.json --out /tmp/cannot.json
cannot_exit=4 (expect 4)
VERDICT CANNOT_EVALUATE (P6_M2_TWO) -> /tmp/cannot.json
```


### 3.4 A data finding worth knowing

With only **4** push_T TRAIN episodes, dim 6 (left gripper) never moves, the 1%/99% state bounds collapse to
a point and `TaskNormalizer` correctly refuses the fit (`degenerate state bounds`).  From 8 episodes on all
14 dims are non-degenerate (measured, see the comment in the live test); the shipped payload uses all 80
TRAIN episodes.  Practical rule: never fit a task normalizer on fewer than ~8 TRAIN episodes.

---

## 4. The exact GPU command for the 2-task 2000-step stage

**NOT EXECUTED.**  Written out so the lead can run it unchanged; the GPU is busy with R15 and this lane was
forbidden to start GPU work.

```bash
cd /home/T7/ojh/robot_actual/r11
export PYTHONPATH=/home/dgut/pusht_runtime/openvla_oft_src:/home/dgut/pusht_runtime/openvla_oft_src/vla-scripts
export PUSHT_RUNTIME=/home/dgut/pusht_runtime
export R15_LEDGER_DIR=/home/T7/ojh/robot_actual/r15_ledger
export M2_NORMALIZER_JSON=/home/dgut/r15_lanes/M2T/m2_task_normalizer_2task.json
RUN=/home/T7/ojh/robot_actual/runs/m2_two_2000_$(date -u +%Y%m%dT%H%M%SZ)

# 0) CPU pre-flight: mix, caps, per-task normalizers, DEV frames (no model, no GPU)
/home/dgut/pusht_runtime/venv_oft/bin/python training/m2_train_shared.py \
  --tasks push_T,stack_bowls --target-global-step 2000 --out "$RUN" --plan-only \
  --max-tasks-stage 2 --normalizer "$M2_NORMALIZER_JSON" --splits data/m2_splits.json

# 1) the 2-task shared LoRA stage: 2000 cumulative multi-task optimizer steps
/home/dgut/pusht_runtime/venv_oft/bin/python training/m2_train_shared.py \
  --tasks push_T,stack_bowls \
  --target-global-step 2000 \
  --out "$RUN" \
  --save_every 200 \
  --max-tasks-stage 2 \
  --est-seconds-per-step 3.5 \
  --time-margin 300 \
  --seed 0 \
  --cams 3 \
  --bucket TWO_TASK_PILOT \
  --splits data/m2_splits.json \
  --normalizer "$M2_NORMALIZER_JSON" \
  --dev-frames-per-task 8 \
  --log-every 10

# 2) the gate verdict (exit 0 PASS / 3 FAIL / 4 CANNOT_EVALUATE)
/home/dgut/pusht_runtime/venv_oft/bin/python tools/m2_gate_check.py \
  --evidence "$RUN/stage_evidence.json" \
  --eval "$RUN/dev_eval_latest.npz" \
  --normalizer "$M2_NORMALIZER_JSON" \
  --splits data/m2_splits.json \
  --ledger-dir /home/T7/ojh/robot_actual/r15_ledger \
  --out /home/dgut/r15_lanes/M2T/P6_M2_TWO_report.json
echo "gate exit=$?"
```

Operational notes for whoever runs it:
* `TWO_TASK_PILOT` holds 7200 s.  2000 x 3.5 s + 300 s margin = 7300 s, so the time guard may stop a
  little short of 2000 and the gate's `P6.cap.stage_completed` will (correctly) FAIL.  Measure one step
  first (the ledger's `step_commit` events carry per-step seconds) and pass a realistic
  `--est-seconds-per-step`; if the stage is genuinely longer than the bucket, the lead has to decide
  between a bigger bucket and a smaller target — the trainer will not silently overrun.
* A 2000 -> 3000 continuation needs `--max-tasks-stage 3` (cap 3000) and
  `--resume-from "$RUN/checkpoints/step2000"`; the resume re-checks task set, normalizer sha and mix sha.

---

## 5. What only a GPU run can prove (honest list)

1. That 2000 optimizer steps fit the remaining `TWO_TASK_PILOT` seconds and the stage really reaches exactly
   step 2000 (the cap/completion checks are hard conditions, and nothing here has measured seconds/step).
2. That the ONE shared LoRA + 42D projector **learns both tasks**: every per-task DEV CE must fall below
   `2.62/2.67` (push_T) and `3.37/3.32` (stack_bowls) nats.  No model has seen these frames.
3. That neither task is sacrificed: every save-point CE must stay <= that task's step-0 CE + 0.10 nats.
   Nothing bounds how much of one task a 2000-step shared run gives up in practice.
4. That the prompt actually **routes**: the counterfactual wrong-prompt CE penalty must exceed 0.05 nats on
   real logits.  The mechanism is unit-tested with fake logits only.
5. That bf16 forward+backward with 3 cameras x 224 px, a 42D projector and 15 supervised tokens fits the
   card next to the running R15 job, without OOM or a non-finite loss.
6. That `r42_adapter.build_labels()` / the 256-bin codec / `decode_command()` agree bit-exactly with the
   per-task `q01/q99` on real logits and real residuals end to end (only fake-logit round trips were run).
7. That the full-state save/resume path works with the new multi-task `trainer_state` (task mix, per-task
   CE curves, counterfactual archive) on a real 7B checkpoint — the resume *guards* are tested, the weight
   restore is not.
8. That a checkpoint produced this way can be **deployed** — it cannot, until the patch in §6 is applied:
   the current deploy/trainer gate accepts exactly one 14D payload.
9. Any statement about P6 being passed at all.  This lane produced the machine, the caps and the gate; the
   verdict is a GPU-run artefact.

---

## 6. The minimal patch required in existing files (NOT applied — described precisely)

`code/xpolicylab/policy/OpenVLA_R42/r42_adapter.py`

* `normalizer_payload()` (lines 370-392): the hard failure
  `if int(n_train_episodes) != 64 or int(n_dev_episodes_excluded) != 16: raise` is what rejects an M2
  checkpoint (the M2 split is 80/20).  Minimal change: allow the caller to pass a *task block* obtained from
  `training/m2_task_normalizer.payload_for(payload, task)` and replace the 64/16 assertion with
  `n_train_episodes >= 1 and n_dev_episodes_excluded >= 1` (the block already records the exact episode
  lists; the DEV-disjointness proof lives in `m2_task_normalizer.assert_payload_complete`).  Keep the
  legacy 64/16 rule for payloads that carry no `schema`.
* `normalizer_sha256()/validate_normalizer()/build_metadata()/validate_metadata()` (lines 433-486):
  accept either a single 14D payload (legacy) or `{"schema": "m2-task-normalizer-v1", ...}`; for the
  multi-task case `normalizer_sha256` must be
  `m2_task_normalizer.normalizer_sha256(payload)` (the whole-payload hash, not the hash of one block), and
  `build_metadata(task_id=...)` must record `task_ids` (the map) instead of a single `task_id`.
* `decode_with_metadata()` (lines 502-510) and `validate_metadata(expect_task=)`: take the task to
  deploy and select `payload_for(meta["normalizer"], task)` before `decode_command(...)`;
  `expect_task` must be checked against `meta["task_ids"][task]`.

`training/r9_trainer_contract.py`

* `load_frozen_contract()` (lines 30-53): same dispatch.  If `meta.get("schema") == "m2-task-normalizer-v1"`,
  require a task argument, take that task's block, build the `NormalizationContract` from it and replace
  `require_64_16` with "the block records >= 1 DEV episode and its fit list is disjoint from it".  Otherwise
  fail closed exactly as today.

Both changes are one small function each; doing them inside this lane would have violated the additive-only
rule, so the M2 trainer/gate deliberately do not go through those two functions — they use
`m2_task_normalizer` directly and fail closed on anything else.

---

## 7. Where I had to guess (the repo did not answer)

1. **Cap table.**  Only "2 tasks -> 2000" was given.  `CAPS = {1:1000, 2:2000, 3:3000}` extrapolates
   linearly and lands on the existing R15 3000-step formal ceiling for 3 tasks.  If the lead wants a different
   1-task or 3-task row, it is one constant in `m2_train_shared.CAPS`.
2. **CE form of the baselines.**  The repo only defines the two baselines as point predictors with an L1
   metric.  I score them as `(1 - hit_rate) * ln 256` (see §1.3) and report hit rate + L1.
3. **"no-op" vs "constant velocity" mapping.**  Mapped to A_hat = S (R_hat = -V) and A_hat = S + V (R_hat = 0),
   matching `r15_dev_eval`'s `identity_A=S` / `constant_velocity_A=S+V` model names.
4. **Task mix.**  "proportional-to-frames with a floor per task" left both the floor value and the horizon
   open: floor 0.30, horizon = one epoch (sum of TRAIN frames), largest-remainder counts, deterministic
   largest-deficit order.  For 2 tasks the floor does not bind; it only bites at 3 tasks.
5. **Thresholds.**  `--forget-tol 0.10` nats, `--routing-min-win-rate 0.9`,
   `--routing-min-ce-penalty 0.05` nats, `--dev-frames-per-task 8` — all chosen, all CLI-overridable.
6. **Routing test shape.**  "must not collapse to one task's output distribution" is implemented as
   counterfactual prompt evaluation (n_tasks forwards per DEV frame per save point) because marginal
   comparisons are unsound when each task's targets are standardised by its own quantiles.  Cost is real:
   8 frames x 2 tasks x 2 prompts = 32 forwards per save point.
7. **Prompt template.**  "What action should the robot take to <instr>?" copied from the R15 single-task
   path; no M2 document specifies it.
8. **Task token.**  `<task:name#id>` exists (from the data contract) but is OFF by default, because only the
   instruction string is guaranteed to exist at deployment.
9. **Budget bucket.**  `TWO_TASK_PILOT` (7200 s) is the only bucket in `r15_ledger.BUCKET_CAPS_S` that
   fits a 2-task pilot; see the overrun note in §4.
10. **Splits authority.**  `data/m2_splits.json` (the audit artefact) is authoritative for TRAIN/DEV, and the
    normalizer's fit list must equal the split's TRAIN list exactly (stronger than "disjoint").  If a future
    stage wants a different split, both artefacts must be regenerated together.
11. **Repo HEAD.**  The brief said `97cd5e8`; the worktree had already advanced to `c6c0059` (the M2 data
    contract) and then to `6b7d058` while this lane ran.  I built on `c6c0059`'s `m2_multitask_data.py`
    (709 lines, unchanged) and re-verified the shipped files against the live tree at `6b7d058`.

---

## 8. Evidence index (`/home/dgut/r15_lanes/M2T/`)

| file | content |
|---|---|
| `m2_task_normalizer.py`, `m2_train_shared.py`, `m2_gate_check.py`, `test_m2_task_normalizer.py`, `test_m2_gate_check.py` | copies of the shipped files |
| `M2_trainer.md` | this document |
| `m2_task_normalizer.json` | the real 3-task payload (sha 930213f5…, spread 2.44x) |
| `m2_task_normalizer_2task.json` | the 2-task payload the stage and the gate use (sha 43a166c3…) |
| `m2_fit_3task.log` / `m2_fit_2task.log` | fit transcripts |
| `m2_plan_2task_2000.json` / `m2_plan_summary.log` | the real 2-task 2000-step plan (mix, caps, DEV frames) |
| `pytest_m2_trainer.log` | the pytest log quoted in §2 |
| `plumbing_check.py` / `plumbing_out.txt` / `plumbing_gate_report.json` | the real-data gate plumbing run with SYNTHETIC predictions + the measured baselines |
| `plumbing_cli_out.txt` / `plumbing_gate_cli_report.json` | the same artefacts through the CLI: FAIL on `P6.cap.wal_steps` only (the synthetic run booked no WAL steps) |
| `plumbing_fake_eval.npz`, `plumbing_evidence.json` | the synthetic archive/evidence behind that run |

Reproduce (from `E:\具身智能\_r11`): write the command file with LF endings, then
`python rsh.py <cmd.sh> <out.txt> <timeout>`; scripts used here: `m2t_run.sh`, `m2t_run2.sh`,
`m2t_final.sh`, `m2t_probe_deg.sh`, `m2t_plumbing.py`.

No git commit, no push, no GPU seconds of any kind were spent by this lane.
