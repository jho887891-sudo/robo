# HISTORY_PROPRIO_PILOT_RESULTS — R6 phase 2 (28D history-proprio A/B, 300 updates)

Branch `audit/r6-history-proprio-pilot` (base `f974aeb`, R5 audit tip). `main` untouched,
no R2–R5 checkpoint or dataset modified, no closed-loop run, no 3000/6000-step training,
no foreign GPU process touched.

Evidence: `evidence/history_proprio_pilot_20261019T161033Z/` —
`phase1_*.json` (data diagnostics), `smoke_*.json` (GPU interface gate),
`arm_comparison.json` (pre-registered gates), `replication_summary.json` (noise floor),
`token_calibration.json` (direction vs magnitude decomposition),
`history_ceiling.json` (non-parametric information ceiling),
`checkpoint_contract.json`, `config_*.json` (sample manifests), `metrics_*.jsonl`
(per-step loss + every generated token row), `train_*.log`, `gradient_trace_*.json`,
`environment.json`, `FAILURES.md`, `commit_sha.txt`.
Checkpoints and HDF5 stay on the machine and are never committed.

---

## 0. 结论摘要（中文）

**实验**：接口完全相同（28 维 ProprioProjector，16,900,096 参数）、相同基础模型 / LoRA /
残差动作目标 / 200 帧训练 + 60 帧验证 / 种子 / 优化器 / 学习率 / **300 次参数更新**，
唯一差别是 slot1：`duplicate = [S_t, S_t]`（无历史）vs `true = [S_t, S_{t-1}]`（真实历史）。
**每组各跑 3 次**（同配置重复，用于估计噪声）。

| 验证集（60 帧 / 20 回合，物理 L1） | step 0 | step 150 | step 300 |
|---|---:|---:|---:|
| duplicate（3 次，均值 [范围]） | 0.02086 | 0.01515 [0.01182, 0.01961] | 0.02422 [0.02312, 0.02605] |
| true（3 次，均值 [范围]） | 0.02083 | 0.01431 [0.01244, 0.01794] | 0.02047 [0.01846, 0.02183] |
| 参考：R5 原脚本 14 维 @300 步 | 0.02054 | 0.01749 | 0.02253 |
| identity / 恒速外推 | 0.00480 | 0.00480 | 0.00102 |

1. **已验证（最重要的否定结论）**：把每次运行的**生成 token 逐条还原成位移**后重新打分（重构与
   训练日志的 L1 完全一致），单个**在训练帧上拟合**的标量系数 α 就能让无历史组从 0.02422 降到
   **0.00817**，而有历史组是 0.02047 —— 也就是说所谓 A/B 差异**完全由位移幅度（过冲）解释**：
   对照组 4.86×、处理组 4.04× 的过冲，缩放到正确幅度后对照组反而更接近专家。
2. **已验证**：三组运行的方向指标全部处于随机水平：cosine(Δ̂, Δ) = −0.080 / −0.033，符号一致率
   0.502 / 0.475，池化相关 0.063 / 0.018。**"幅度对了、方向随机"** 是这 6 次 300 步运行的共同特征。
3. **已验证**：R5 的 800 步残差模型（0.00749）在同样分析下幅度比 **1.04**（幅度学得很准），
   但 cosine 仅 **0.090**、符号 0.552、α 校准后仍 0.00712 > identity 0.00480 ⇒
   **它学到的是"该移动多远"，不是"该往哪移动"**。这解释了为什么 R5 的模型始终打不过 identity。
4. **已验证（历史信息天花板）**：在 25,393 训练帧上拟合 kNN(k=1..25)、二次 ridge、14×14 联合线性，
   在 2,000 验证帧上评估，**没有任何模型优于"直接复制 V_t"（0.001855）**：kNN10 0.002170、
   二次 ridge 0.001989、联合线性 0.002020、逐维线性 0.001855（=1.000×）。即
   **历史状态能解释的专家增量上限就是恒速外推本身**，再多也榨不出来。
5. **已验证**：三次重复后，step 150 两臂区间**重叠**（改善仅 5.5%），step 300 区间仍分离
   （改善 15.5%）。但结合第 1 条，这个分离是幅度效应，不是历史信息效应。
6. **结论**：**不具备**进入 3000 步正式训练的证据。P4（超 identity）、P5（超恒速）、P6（方向）、
   P7（多样性）全部 FAIL，新增的事后闸门 P8（历史效应经幅度校准后仍存在）也 FAIL。

---

## 1. Protocol actually executed

| item | value |
|---|---|
| base model | OpenVLA-7B pristine, runtime SHA verified (`b5431a07…`) |
| LoRA | r=32, alpha=16, all-linear, 110.8M trainable |
| proprio interface | `ProprioProjector(proprio_dim=28, llm_dim=4096)` → 16,900,096 params, identical in both arms |
| action head | autoregressive 256-bin discrete tokens, chunk = 1, R2-fixed codec |
| target | `residual` (`A_t − S_t`), train-only q01/q99 |
| observation | 3 cameras (head + both wrists) + fixed push_T instruction |
| train frames | 200 (diverse selection over 80/80 train episodes, stride 20, TRAIN only) |
| val frames | 60 (spread selection, 20/20 val episodes, 3 per episode, disjoint) |
| updates | **300**, enforced by `MAX_STEPS = 300` in code |
| optimizer | AdamW, lr 5e-4, wd 0, no scheduler, batch 1, grad-clip 1.0 |
| seed | 0 (torch / numpy / random) — every run is a *repetition* of the same configuration |
| precision | bf16 autocast, gradient checkpointing ON (another user's job holds ~40 GB of the A6000) |
| runs | `duplicate` ×3, `true` ×3, plus R5's unmodified `overfit_pr1.py` (14D) ×1 |

The repetitions exist because the single-pair A/B gap turned out to be of the same order as the
run-to-run spread. They are a measurement of noise, not a budget increase: **every run is exactly
300 updates** on the same 200/60 frames with the same seed.

## 2. Fairness evidence

* identical `samples` / `val_samples` / `proprio_dim=28` / `steps` / `lr` / `seed` / `target_mode`
  in every `_snapshot/config.json` (asserted by gate P2);
* identical projector parameter count (16,900,096) and identical LoRA configuration;
* **slot-gap guard inside the trainer**: duplicate asserts `|slot1 − slot0| == 0` over all 200
  samples (`max = 0.000000`), treatment asserts a non-zero gap (`mean 0.024667, max 0.076768`);
* the duplicate arm's projector gradient blocks are numerically **equal** at every logged step
  (`cur = hist = 0.001566`, …), direct evidence both slots carry the same values; the treatment
  arm's blocks differ (`0.009858 vs 0.009865`, …);
* step-0 validation L1: 0.02086 (duplicate, all 3 runs) vs 0.02083 (true, all 3 runs);
* both arms passed the smoke gate before `--execute` was allowed (§3).

## 3. Smoke gate (before any training; both arms PASS, 14/14 checks)

| check | measurement |
|---|---|
| forward loss finite | 16.456 |
| projector gradients | fc1 1.329 / fc1.b 0.416 / fc2 16.359 / fc2.b 1.286 |
| LoRA gradients | 852 tensors, max norm 36.0 |
| **both proprio slots receive gradient** | current 0.952, history 0.928 |
| **history slot reaches the action logits** | max \|Δlogit\| 0.188 / 0.219 / 0.281 at amplitude 0.1 / 0.35 / 1.0 |
| current slot reaches the action logits | max \|Δlogit\| 0.184 / 0.234 / 0.258 |
| no future leak | corrupting `S_{t+1}`/`A_{t+1}` leaves frame-`t` input **and** target bit-identical |
| causal autoregressive decoding | KV-cache argmax == full-sequence argmax, **14/14 dims** |
| determinism / finite logits / shapes | PASS (logits `[1, 820, 32064]`, proprio `[1, 28]`) |

Informational references from the same gate: the 14D interface under the same sweep gives max
\|Δlogit\| 0.188/0.250/0.353 with 0/6/5 token flips; a 0.05 image perturbation gives 5.94 (10/14
flips) and zeroing the images 10.05 (11/14 flips). At initialisation the proprio channel is 20–30×
weaker than the visual channel in logit terms; per-dimension top-1 confidence reaches 0.9996.

## 4. Results

### 4.1 Validation curve (60 frames / 20 episodes, physical 14D L1)

| run | step 0 | step 150 | step 300 |
|---|---:|---:|---:|
| duplicate #1 / #2 / #3 | 0.02086 | 0.01961 / 0.01403 / 0.01182 | 0.02605 / 0.02312 / 0.02348 |
| true #1 / #2 / #3 | 0.02083 | 0.01244 / 0.01256 / 0.01794 | 0.01846 / 0.02183 / 0.02111 |
| 14D reference (R5 script) | 0.02054 | 0.01749 | 0.02253 |
| identity | 0.00480 | 0.00480 | 0.00480 |
| constant velocity | 0.00102 | 0.00102 | 0.00102 |

`replication_summary.json`:

| step | duplicate mean [min, max], spread | true mean [min, max], spread | ranges separated | mean improvement |
|---|---:|---:|---|---:|
| 0 | 0.02086 [0.02086, 0.02086], 0.0% | 0.02083 [0.02083, 0.02083], 0.0% | (init only) | 0.2% |
| 150 | 0.01515 [0.01182, 0.01961], **51.4%** | 0.01431 [0.01244, 0.01794], **38.4%** | **No (overlap)** | 5.5% |
| 300 | 0.02422 [0.02312, 0.02605], **12.1%** | 0.02047 [0.01846, 0.02183], **16.4%** | **Yes** | 15.5% |

With n = 2 per arm the arms looked "completely separated" at both points; the third repetition
removed that at step 150 completely. **This is the clearest illustration that the L1 ranking at this
budget is a noise-dominated quantity.**

### 4.2 Final evaluation detail (val, step 300, mean of 3 runs)

| metric | duplicate | true |
|---|---:|---:|
| macro L1 over episodes | 0.02422 | **0.02047** |
| \|pred delta\| / \|expert delta\| | 4.86 | 4.04 |
| **cosine(Δ̂, Δ)** mean per sample | **−0.080** | **−0.033** |
| corr(Δ̂, Δ) pooled | 0.063 | 0.018 |
| sign agreement at \|Δ\|≥1e-3 | 0.502 | 0.475 |
| episodes beating cv / identity | 0/20, 0/20 | 0/20, 0/20 |
| unique generated token rows | 22 / 3 / 6 (per run) | 14 / 5 / 8 (per run) |
| teacher-forced token accuracy | 0.490 / 0.507 / ~0.50 | 0.562 / 0.545 / ~0.55 |

Per-dimension L1 (val, step 300, true #1 vs duplicate #1), order
`left_arm_0…5, left_gripper, right_arm_0…5, right_gripper`:

| dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | .0081 | .0100 | .0082 | .0074 | .0047 | .0053 | .0013 | .0042 | .0056 | .0046 | .0040 | .0015 | .0024 | .0000 |
| cv | .0015 | .0020 | .0015 | .0016 | .0010 | .0010 | .0000 | .0013 | .0010 | .0010 | .0010 | .0007 | .0007 | .0000 |
| duplicate #1 | .0289 | .0550 | .0409 | .0300 | .0137 | .0190 | .0178 | .0212 | .0341 | .0205 | .0239 | .0109 | .0153 | .0337 |
| true #1 | .0176 | .0421 | .0313 | .0267 | .0110 | .0141 | .0054 | .0156 | .0306 | .0260 | .0178 | .0082 | .0118 | .0004 |

The treatment arm is better on 12/14 dimensions but still above identity on every moving dimension.
Dimension 13 (right gripper) has a **zero** expert delta on these 60 frames: its correlation is
undefined and excluded, and duplicate #1 emits a constant non-zero gripper command (0.0337).

### 4.3 Token-level calibration — the L1 gap is a displacement-magnitude artefact

Every prediction was rebuilt from the stored generated-token rows (`token_calibration.json`); the
reconstruction is exact (recomputed val L1 matches the logged value to < 1e-6). Each run was
re-scored with **one scalar α fitted on the 60 TRAIN frames** (never on validation):

| run | L1 (α=1) | α fitted on train | L1 at α | \|Δ̂\|/\|Δ\| | cosine | sign@1e-3 |
|---|---:|---:|---:|---:|---:|---:|
| duplicate #1 | 0.02605 | 0.38 | 0.01145 | 5.27 | −0.083 | 0.519 |
| duplicate #2 | 0.02312 | 0.00 | 0.00480 | 4.51 | −0.030 | 0.484 |
| duplicate #3 | 0.02348 | 0.26 | 0.00825 | 4.80 | −0.128 | 0.504 |
| true #1 | 0.01846 | 0.26 | 0.00705 | 3.63 | −0.006 | 0.469 |
| true #2 | 0.02183 | 0.00 | 0.00480 | 4.25 | −0.047 | 0.463 |
| true #3 | 0.02111 | 0.00 | 0.00480 | 4.25 | −0.045 | 0.493 |
| 14D reference @300 | 0.02253 | 0.06 | 0.00534 | 4.82 | +0.003 | 0.528 |
| **R5 residual @800** | 0.00749 | 0.90 | 0.00712 | **1.04** | +0.090 | 0.552 |

Readings:

1. **No run carries direction information**: cosine ≤ 0 in every 300-update run, sign ≈ 0.5. R5's
   800-update model has cosine 0.090 — still essentially chance.
2. **α = 0 is optimal for four of the six 300-update runs**: the best available action is to ignore
   the network output entirely and hold the state (L1 0.00480).
3. With α fitted on train, the control mean becomes **0.00817** and the treatment **0.00555**; the
   A/B gap (0.00375 at α=1) is not merely closed but **reversed** (−0.01230, i.e. 428% "explained").
   **The history arm's L1 advantage was a smaller overshoot, not better prediction.**
4. Even with perfect magnitude (expert norm on the predicted direction), L1 stays ≈ 0.0067 for both
   arms — above identity 0.00480 — because the direction is random.

### 4.4 What history can explain at most (verified information ceiling)

`history_ceiling.json`: models fitted on **25,393 TRAIN frames**, evaluated on a deterministic
2,000-frame VAL subsample (fitted models also scored on all 6,711):

| predictor | val L1 | × copy-V |
|---|---:|---:|
| copy `V_t` (no fitting) | **0.001855** | 1.000 |
| per-dimension linear | 0.001855 | 1.000 |
| quadratic-feature ridge | 0.001989 | 1.072 |
| joint 14×14 linear | 0.002020 | 1.089 |
| kNN k=10 | 0.002170 | 1.170 |
| kNN k=5 / 25 / 1 | 0.002206 / 0.002229 / 0.002795 | 1.19 / 1.20 / 1.51 |

On the motion-rich 39.5% of validation frames: hold 0.013958, copy-V 0.003282, kNN10 0.003883.
**No fitted history-only model beats copying the past velocity.** The history ceiling is therefore
the constant-velocity rule (0.001828 on the full val set, 0.002119 through the 256-bin codec), and
the models in this pilot are 4–14× above it.

## 5. Why R5's published numbers must be read with care

The 14D reference run doubles as a byte-level reproduction check of R5's protocol:

| quantity | R5 original | R6 reproduction |
|---|---:|---:|
| step-1 loss | 13.7153 | **13.7153** |
| step-1 grad norm | 49.58 | 49.51 |
| step-0 train eval | 0.03395 | **0.03395** |
| step-0 val eval | 0.02054 | **0.02054** |
| val @ ~200 steps | 0.00753 | 0.01749 (150) / 0.02253 (300) |
| val @ 800 steps | 0.00749 | — (not re-run) |

Initialisation, data order and the first evaluation reproduce exactly; the trajectories separate from
about step 25 (loss 2.6346 vs 2.7333 on the same sample). That is GPU kernel non-determinism (no
`torch.use_deterministic_algorithms`, bf16 GEMMs) amplified by chaotic short-horizon optimisation.

So R5's `0.00749 vs identity 0.00480` gap should not be treated as a stable capability statement;
the same script at the same budget gives 0.0175–0.0225. More importantly, §4.3 shows that even R5's
800-update model is at chance direction: its 0.00749 is a **magnitude-calibrated, direction-blind**
prediction, which is why it cannot beat holding still.

## 6. Pre-registered gates

| gate | criterion | measured | verdict |
|---|---|---|---|
| P1 smoke gate | PASS before training | 14/14 | **PASS** |
| P2 identical interface & budget | same dim/steps/lr/seed/samples; duplicate gap 0; true gap > 0 | all true | **PASS** |
| P3 history beats control by ≥10% | `L1_true < 0.9·L1_dup` | 0.02047 vs 0.02422 (15.5%, 3 runs/arm; 5.5% at step 150) | **PASS (L1 only — see P8)** |
| P4 beats hold-state | `ratio < 1` | 4.27 | **FAIL** |
| P5 beats constant velocity | `L1_true < cv` | 20.1× | **FAIL** |
| P6 direction corr > 0.3 and sign > 0.65 | measured | corr 0.018, cosine −0.033, sign 0.475 | **FAIL** |
| P7 token diversity not collapsed | no dim ≥ 0.95, ≥50% unique rows | gripper dims collapse; 5–14/60 rows | **FAIL** |
| **P8 (post-hoc) history effect survives magnitude calibration** | control at its train-fitted α still worse than treatment at α=1 | control 0.00817 < treatment 0.02047 | **FAIL** |

Phase-1 gates H1–H6 all PASS (see `HISTORY_PROPRIO_DIAGNOSTIC.md`). P8 is explicitly post-hoc and
is reported as such; it is the gate that matters, because it removes the confound P3 cannot.

Checkpoint contract (`checkpoint_contract.json`): both 28D arms declare `proprio_dim=28` and their
mode, the projector on disk is 28-wide with 16,900,096 parameters, SHA-256 binds adapter and
projector, and an R5 14D checkpoint resolves to `mode=none` with a 14-wide projector — the legacy
deployment path is unchanged. **PASS**.

## 7. The six questions

**Q1. Can R5's failure be explained mainly by missing history information?**
**No — not mainly.** R5's residual model (0.00749) was worse than *identity* (0.00480), and identity
needs no history. §4.3 shows R5's model is direction-blind (cosine 0.090) with near-perfect magnitude
(ratio 1.04): it learned the marginal size of the next motion, not where to move. History can at most
explain the identity→cv segment, which phase 1 shows is exactly the copy-velocity rule (§4.4).

**Q2. Does history really explain the expert's next action?**
**已验证 0.9432 correlation** (val pooled; left 0.9400, right 0.9305; per-dim 0.871–0.961; 20/20
episodes positive; sign 95.6% at `|Δ| ≥ 1e-3`), **and the relationship is exactly velocity
persistence**: `A_t ≡ S_{t+1}` bit-exactly, so `Δ_t = V_{t+1}`, and `corr(V_t, V_{t+1}) = 0.9432`.
Nothing beyond copying `V_t` can be extracted (§4.4: kNN/quadratic/joint linear all ≥ copy-V).

**Q3. 256-bin quantization oracle physical error?**
**已验证 0.000584** for residual targets (rounding 0.000272 + clipping 0.000312, 2.09% of entries
clipped, mean step 0.000531) and **0.003444** for absolute targets (mean step 0.005352, bias −0.0001…
−0.0060). Residual is not a bottleneck (0.32× cv); absolute is a hard floor 1.9× worse than cv.

**Q4. Did the 28D history proprio work, and did it help?**
**Implementation: yes** (built, unit-tested, contract-verified, deploy-ready, legacy path unchanged).
**Model benefit: no verified benefit.** The L1 difference (15.5% at step 300 with 3 runs per arm,
5.5% at step 150 with overlapping ranges) is fully explained by a smaller displacement overshoot:
with one train-fitted scalar the control reaches 0.00817 and beats the treatment's 0.02047, and the
direction metrics are at chance in all six runs (cosine −0.08 / −0.03, sign 0.50 / 0.48).

**Q5. Does it beat identity / constant velocity?**
**No**, in every one of the six runs: 4.27× worse than identity, 20.1× worse than velocity
extrapolation, 0 of 20 validation episodes won against either.

**Q6. Is there evidence for the formal 3000-step training?**
**No.** P4/P5/P6/P7 and the post-hoc P8 all FAIL. No run has converged (overshoot 4.0–4.9×,
5–22 of 60 unique token rows, teacher-forced accuracy ≈0.5), the metric still has a 12–51% noise
floor, and the direction channel shows no learning at 300 steps *or* in R5's 800-step model.
A 3000-step run would cost ~10× more and would still be measuring an overshoot artefact.

## 8. Failure and defect log

See `evidence/history_proprio_pilot_20261019T161033Z/FAILURES.md`: the failed first smoke gate
(token-level sensitivity too coarse), a comparison-script KeyError, an over-strict legacy-metadata
check, four wrong unit-test expectations, the non-reproducible 300-update variance, a non-monotone
control curve, a CUDA OOM on a repetition attempt, a kNN chunk that never finished, and the third
repetition that removed the step-150 separation. Nothing is deleted or hidden: the failed OOM run
directory and its log are kept.

## 9. Deployment contract (implemented, unit-tested, never run live)

`code/xpolicylab/policy/OpenVLA_Pure/model.py` resolves the proprio interface from checkpoint
metadata: legacy checkpoints (no metadata) keep the 14D path with byte-identical arithmetic; a 28D
checkpoint must declare `proprio_dim` and `proprio_history_mode`; the projector width is verified
against the checkpoint **before** `load_state_dict`; the previous state comes from a per-env buffer
of **observed** states cleared by `reset()`; the first-observation rule (slot 1 = current state) is
identical in training and deployment. 41 CPU tests total (24 proprio-contract + 17 diagnostic-maths),
including a source-level test that refuses a regression to a hard-coded 14.

**尚未验证**: the 28D path has never been started inside a live policy server; the live server was
not touched in this round.

## 10. Recommendation — what to change next (not "train longer")

The measurements above point at a specific defect, not at a budget shortfall: **the policy predicts a
direction-free displacement of roughly the right size**. Concretely:

1. **Stop using absolute physical L1 as the primary metric.** It is dominated by overshoot and has a
   12–51% noise floor at this budget. Use `cosine(Δ̂, Δ_gt)`, sign agreement at `|Δ| ≥ 1e-3`, and
   report the magnitude ratio next to them; always include "predict zero" and "copy V_t" in the table.
2. **Supervise direction directly.** Add an auxiliary loss on the expected decoded displacement
   (softmax-weighted mean over the 256 bins per dimension) with a cosine/L1 term against the expert
   delta. A token-CE-only objective on 200 frames demonstrably learns the marginal magnitude and
   ignores the input.
3. **Feed the difference, not the pair.** `[S_t, S_t − S_{t−1}]` carries the same information in a
   form an MLP can use without learning a subtraction, and zero-initialising the history half makes
   both arms start from the 14D function (removes the confound that the control arm is a degenerate
   doubling).
4. **Raise the number of *distinct* frames, not the step count.** The current loop revisits 200
   frames (1.5 epochs at 300 steps). Stream the 25,393 train frames instead of a fixed 200-frame
   cache; a step-count increase on 200 frames only overfits (the train-split L1 in these runs stays
   at ~0.03 while the token accuracy saturates at ~0.55).
5. **Keep the residual target** (the absolute codec floor 0.003444 makes P5 unreachable), and keep the
   256-bin codec (oracle 0.000584 is far below every model error measured so far).
6. **Before any 3000-step run**, require: `cosine(Δ̂, Δ) > 0.3` on the 60-frame val set, sign
   agreement > 0.65 at `|Δ| ≥ 1e-3`, and an L1 below the train-fitted-α control (i.e. P8 PASS).
7. **Re-check R5's headline numbers** with repetitions before citing 0.00749; and note that R5's model
   is direction-blind, so its failure is not a history-input problem.

## 11. Reproduction

```bash
bash scripts/run_r6_pilot.sh --check     # preflight + phase-1 CPU diagnostics
bash scripts/run_r6_pilot.sh --smoke     # 28D interface smoke gate (GPU, ~6 min)
bash scripts/run_r6_pilot.sh --execute   # duplicate + true, 300 updates each, then gates (~35 min)

# repetitions and reference runs (same arguments, fresh directories)
python training/r6_history_pilot.py --history-mode {duplicate,true} --target-mode residual \
    --n 200 --val-n 60 --steps 300 --lr 5e-4 --eval-every 150 --out <fresh dir>

# analysis (CPU, no GPU)
python diagnostics/r6_replication.py --run dup1=D1 --run dup2=D2 --run dup3=D3 \
    --run true1=T1 --run true2=T2 --run true3=T3 --out replication_summary.json
python diagnostics/r6_token_calibration.py --run dup1=D1 ... --run r5_residual_800=R5 \
    --out token_calibration.json
python diagnostics/r6_history_ceiling.py --out history_ceiling.json
```

The execute path refuses to overwrite an existing run directory, enforces the 300-update cap in code,
and refuses to start without a PASSing smoke gate, on a dirty tree, or on the wrong branch.
