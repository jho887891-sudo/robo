# R15–R23 status — what is true right now

Generated 2026-10-21T10:20Z. Every number was produced by a command on the server; the log of those
commands is `mission/R15_R23_EXECUTOR_LOG.md`, and the per-lane evidence is under `lanes/`.

## 1. HEADLINE: M1 Push-T offline gate G5_1000 = **FAIL** (machine verdict)

| quantity (frozen 32-frame DEV gate set, pre-registered) | step 300 | step 1000 |
|---|---|---|
| VLA macro L1, SAFE (what the robot executes) | 0.0020678 | **0.0025385** |
| VLA macro L1, RAW (as decoded) | 0.0021344 | 0.0026050 |
| constant velocity, SAFE | 0.0020475 | 0.0020475 |
| **relative improvement vs CV** | **-1.0 %** | **-23.98 %** |
| motion strata won (of 4, needs 3) | — | **0** (q4 loses by 40 %) |
| no collapse | pass | pass |
| delta sign / magnitude not worse than CV by >20 % | pass | pass (0.6897 vs 0.7857; 1.051 vs 1.040) |

Machine verdict: `mission/r15_gate_G5_1000.json` -> `G5.macro_relative_5pct=false`,
`G5.strata_3_of_4=false`, `G5.pass_rule=false`, **verdict FAIL**, exit 3.
The pre-registered action for a FAIL is "stop at 1000, keep the checkpoint, diagnose, never train to
3000 blindly" — so the +2000 stage is BLOCKED by design (`m1_2000_seg3_to_3000` = BLOCKED in the
supervisor state file) and the checkpoint is preserved.

### Why it fails (measured, not speculated)

Because `A[t] == S[t+1]`, the constant-velocity predictor's error **is** the residual magnitude:
`|CV - A_t| = |R_t|`. So CV is the "predict R = 0" model, and beating it requires predicting R better
than zero. `tools/r15_mechanism_diagnostics.py` on the frozen DEV frames:

| model on the same frames | DEV L1 | DEV R^2 |
|---|---|---|
| constant velocity (R = 0) | 0.002110 | 0 (by construction) |
| TRAIN-mean residual | 0.002333 | -0.0354 |
| TRAIN-only linear map, 42D feature -> R | 0.003429 | **-0.1412** |

The 42D proprio history does **not** carry the residual, so only the 3 RGB frames could. At 1000 steps
the model has not extracted it; the DEV L1 got 23 % worse than at 300 steps, with the decoded motion
sign agreement unchanged (0.6897 at both 300 and 1000) — i.e. the model's *direction* behaviour did not
change while its magnitude error grew. That is consistent with fitting TRAIN noise rather than a
residual mechanism, and it is the honest finding: **this R15 recipe does not beat a two-line baseline
on Push-T offline.**

## 2. What was executed for M1 training (all verifiable)

| segment | run | steps | verdict | evidence |
|---|---|---|---|---|
| 0 -> 300 | `r15-s300-…030423Z` | 300 | REACHED_TARGET | parent checkpoint, 9/9 MANIFEST sha256 verified by the independent audit |
| 300 -> 500 | `r15-s2-1000-…0803Z` | +200 | PARTIAL (wall guard at a save point) | `start_global_step=300`, `position=300` peek-only resume |
| 500 -> 1000 | `r15-s2b-1000-…0803Z` | +500 | REACHED_TARGET, 3.039 s/step | `position==global_step==1000` |

So `runs/r15_s2b_1000_20261021T0803Z/checkpoints/step1000` is **exactly +700 optimizer steps after the
verified step300 parent**, with `sampler_order_hash = f1024210…` identical to the stage-1 run (the same
sample order an uninterrupted run would have used) and the resume seam proven by
`position == global_step` at every saved checkpoint.

Ledger truth (`tools/r15_ledger_lineage.py`), which is deliberately honest about the wasted work:
1664 optimizer steps were **executed** and booked, but only **1000 distinct global steps** exist —
the 664 extra are the discarded invalid segment plus the 2a re-execution. 8764 s of GPU time is booked;
3979 s of load/save/inference overhead is now measurable, and every job started in this session books
its own overhead at `job_end` (the audit's finding 2b).

## 3. M1 closed loop (P3) — wire proven, runs in progress

The transport, the serving path and the official client are all real now:

* `OpenVLA_R42/model.py` + `r42_env_io.py` + `r42_metadata_bridge.py` + `r42_serve.py`
  (162 CPU tests). The metadata bridge reproduces **7/7** existing `r42_metadata.json` files
  byte-identically, and builds one for the R15 stage checkpoint that never had it.
* Measured on loopback with a stub backend (n=200, warm-up discarded): server total p50 24.6 /
  p95 46.0 / p99 53.8 ms; client RTT p50 31.2 / p95 57.6 / p99 69.5 ms against a 40 ms budget at 25 Hz.
* `tools/r15_loop_dev.py` drives the OFFICIAL client and reads ONLY the official `_result.json`.

Two real integration failures were found and fixed (each diagnosed from its log, not guessed):
1. the container died importing curobo because `SETUPTOOLS_SCM_PRETEND_VERSION` was unset;
2. the server answered every call with `No module named 'prismatic'` because the host server was
   started without the OFT runtime on `PYTHONPATH` — and, separately, nothing had forced the ARX
   platform before the prismatic import, which would have silently decoded 14D actions against the
   LIBERO 7D table. Both are fixed in `r42_serve.py` + the launcher.

Attempt 3 got further and failed at a third, precisely located defect: `split_with_sizes expects
split_sizes to sum exactly to 18 (input tensor's size at dimension 1), but got split_sizes=[3, 3]` —
the served vision backbone expects **6 images** while the request carried **2**. The training entry
calls `set_num_images_in_input(3)` and feeds 3 tiles; the serving path must do both. That fix is
confined to `OpenVLA_R42/model.py` / `r42_env_io.py`.

What attempt 3 DID prove: the container imports, the client connects over ws, the server asserts the
ARX constants before prismatic, loads the real 7B + LoRA + 42D projector
(`backend ready: _TorchBackend prompt_len=31 device=cuda`) and answers the `reset` call.
**No closed-loop success rate has been measured, and none is claimed.** Records for all three attempts
are kept: `runs/r15_loop_dev/` + `evidence/r15_loop_attempt{1,2}/`.

## 4. The 12-task programme (M2): what is now built and what is blocking

| piece | state | tests |
|---|---|---|
| 12/12 TaskSpec audit (P4) | **done, 0 BLOCKED_INTERFACE**: robot dual_x5, 25 Hz, 3 RGB cameras, official success predicate located | lane report |
| expert feasibility (P5) | 6 realistic, 4 hard-but-rigid, **2 BLOCKED_DATA** (fold_clothes cloth mesh, pour_liquid particles) | lane report |
| collection harness + quality gate | built, success taken ONLY from the official verdict, caps enforced | 44 |
| multi-task data contract | 3 tasks structurally identical (schema diff empty over 300 episodes, `A[t]==S[t+1]` bit-exact), per-task splits keyed `<task>/<episode>` | 22 |
| ONE shared task-conditioned trainer + P6 gate | built, per-task normalizers (2.44x residual scale spread handled), counterfactual routing test | 36 |
| R16 deployment path | built (see §3) | 162 |

2-task curriculum would consume push_T + stack_bowls = 160 TRAIN episodes / 61,171 rows. The measured
bars it must beat: push_T no-op CE 2.6241 / CV 2.6736; stack_bowls no-op 3.3667 / CV 3.3172.

**Blocking items, stated plainly:**
1. 9 of 12 tasks have **no data**; collection needs GPU simulator time and, for 2 tasks, is physically
   BLOCKED (cloth / fluid verdicts cannot be inverted).
2. No M2 checkpoint can be **deployed** until `r42_adapter.normalizer_payload` /
   `r9_trainer_contract.load_frozen_contract` accept the multi-task schema; the exact minimal patch is
   documented in `lanes/M2_trainer.md` and was deliberately not applied in the additive commits.
3. Scene-level TRAIN/eval disjointness is **unprovable** from the data (the HDF5 files carry no layout
   id), so any FINAL report must say file-level disjointness only.
4. The official FINAL is **25 episodes per task**, not 30 (`_task.yml:105-107` + `main.py:325-329`);
   the equivalent contract is pre-registered in `mission/R15_R23_FINAL_EVAL_CONTRACT.json`.

## 5. Engineering state

Branch `feature/r15-true-continuation` @ `8acede9` (pushed). Commits in this session: PR#21 findings
3-4 + supervisor + gate + lineage (`97cd5e8`), M2 data (`c6c0059`), loop driver + mechanism
(`c6e8590`), R16 deployment path (`5f99df8`), closed-loop wiring (`6b7d058`), M2 trainer (`8acede9`).
Tests added and passing in this session: **321** (162 deploy + 45 R15 core + 44 collection + 36 M2 +
22 M2 data + 7 loop driver + 5 stats). Nothing was merged to main, no official rule or success
predicate was modified, and no other user's process was touched.