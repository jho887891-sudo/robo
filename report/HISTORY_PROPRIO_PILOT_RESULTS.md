# HISTORY_PROPRIO_PILOT_RESULTS — R6 phase 2 (28D history-proprio A/B, 300 updates)

Branch `audit/r6-history-proprio-pilot` (base `f974aeb`, R5 audit tip). `main` untouched,
no R2–R5 checkpoint or dataset modified, no closed-loop run, no 3000/6000-step training,
no foreign GPU process touched.

Evidence: `evidence/history_proprio_pilot_20261019T161033Z/` —
`phase1_*.json`, `smoke_true.json`, `smoke_duplicate.json`, `arm_comparison.json`,
`replication_summary.json`, `checkpoint_contract.json`, `config_*.json` (sample manifests),
`metrics_*.jsonl` (per-step loss and every generated token row), `train_*.log`,
`gradient_trace_*.json`, `environment.json`, `FAILURES.md`, `commit_sha.txt`.
Checkpoints and HDF5 stay on the machine and are never committed.

---

## 0. 结论摘要（中文）

**实验**：两组模型接口完全相同（28 维 ProprioProjector，16,900,096 参数），相同基础模型、
LoRA 配置、残差动作目标、200 帧训练 / 60 帧验证、种子、优化器、学习率与 **300 次参数更新**，
唯一差别是第 2 个 slot 的内容：`duplicate` = `[S_t, S_t]`（无历史），`true` = `[S_t, S_{t-1}]`（真实历史）。
每组各跑 **两次**，用于估计同一配置的随机波动（噪声地板）。

| 验证集（60 帧 / 20 回合，物理 L1） | step 0 | step 150 | step 300 |
|---|---:|---:|---:|
| duplicate #1（无历史） | 0.02086 | 0.01961 | 0.02605 |
| duplicate #2（无历史，同配置） | 0.02086 | 0.01403 | 0.02312 |
| true #1（有历史） | 0.02083 | 0.01244 | **0.01846** |
| true #2（有历史，同配置） | 0.02083 | 0.01256 | 0.02183 |
| 参考：R5 原脚本 14 维接口 | 0.02054 | 0.01749 | 0.02253 |
| 保持当前状态 identity | 0.00480 | 0.00480 | 0.00480 |
| 恒速外推 constant velocity | 0.00102 | 0.00102 | 0.00102 |

1. **初步支持（有重复，但样本量仍小）**：在**两个评测点上两组完全分离**（true 的最大值 < duplicate
   的最小值）：150 步 0.01250 vs 0.01682（**−25.7%**），300 步 0.02015 vs 0.02459（**−18.1%**）。
   同配置重复的波动为 0.9% / 33.2%（150 步）与 16.7% / 11.9%（300 步），所以这个方向性差异
   **大于噪声**，但每组只有 2 次运行，尚不足以做统计断言。
2. **已验证（关键否定部分）**：这个 L1 改善**没有**伴随方向能力改善。四组运行的方向相关分别是
   +0.084 / −0.022（无历史）与 +0.097 / −0.054（有历史），符号一致率 0.46–0.52（≈随机）。
   唯一系统性差别是**位移幅度**：`|Δ_pred|/|Δ_gt|` 从 5.27 / 4.51 降到 3.63 / 4.25。
   因此现有证据更支持"历史 slot 让模型少放大位移"，而**不是**"模型学会了预测运动方向"。
3. **已验证**：无论哪一组都远未收敛、都远不如平凡基线：最好的运行（true #1）仍比 identity 差
   3.85 倍、比恒速外推差 18.1 倍，20 个验证回合 0 胜；预测位移是专家的 3.6–5.3 倍。
4. **已验证**：28 维接口本身不是退化原因。用 R5 原封不动的脚本跑 14 维接口、同样 300 步，得
   0.02253，落在两次 28 维 duplicate 运行之间 —— "28 维学得更慢"的顾虑不成立。
5. **已验证（对 R5 数字的复核）**：R5 已发布的早期曲线点（step 200 val = 0.00753）**无法复现**。
   同一脚本、同种子、同数据的新运行在 150 步得 0.01749、300 步得 0.02253，而初始化与第 1 步
   loss（13.7153）、step-0 评测（train 0.03395 / val 0.02054）**逐位复现**。训练在 GPU 上非确定，
   300–800 步尺度的单次验证 L1 有 2–3 倍运行间差异 ⇒ **R5 的 0.00749 只能当作单次运行数字引用**。
6. **结论**：**不具备**进入 3000 步正式训练的证据。闸门 P4（超过 identity）、P5（超过恒速）、
   P6（方向相关）、P7（token 多样性）全部 FAIL；且方向指标仍未显示出任何历史收益。
   下一轮的瓶颈不是"再多训一点"，而是"能否在可负担预算内做出方向性收益"。

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
| seed | 0 (torch / numpy / random) |
| precision | bf16 autocast; gradient checkpointing ON (another user's job holds ~40 GB of the A6000, so free VRAM at start was 9.1 GB) |
| runs | `duplicate` ×2, `true` ×2, plus R5's unmodified `overfit_pr1.py` with the 14D interface ×1 |

The repetitions were added **after** the first pair, because that pair's A/B gap turned out to be of
the same order as the run-to-run spread. They are a measurement of noise, not a budget increase:
every run is exactly 300 updates on the same 200/60 frames with the same seed.

## 2. Fairness evidence

* identical `samples` / `val_samples` / `proprio_dim=28` / `steps` / `lr` / `seed` / `target_mode`
  in both `_snapshot/config.json` (asserted by gate P2);
* identical projector parameter count (16,900,096) and identical LoRA configuration;
* **slot-gap guard inside the trainer**: duplicate asserts `|slot1 − slot0| == 0` over all 200
  samples (measured `max = 0.000000`), treatment asserts a non-zero gap
  (measured `mean 0.024667, max 0.076768`, normalized units);
* the duplicate arm's projector gradient blocks are numerically **equal** at every logged step
  (`cur = hist = 0.001566`, …) — direct evidence that both slots carried the same values;
  the treatment arm's blocks differ (`0.009858 vs 0.009865`, …);
* step-0 validation L1: 0.02086 (duplicate) vs 0.02083 (true), 0.14% apart, and both repetitions
  reproduce their arm's step-0 value **exactly** (0.02086 / 0.02083) — initialisation is deterministic;
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
flips) and zeroing the images gives 10.05 (11/14 flips). At initialisation the proprio channel is
20–30× weaker than the visual channel in logit terms, and per-dimension top-1 confidence reaches
0.9996 — recorded as context, not as a defect.

## 4. Results

### 4.1 Validation curve (60 frames / 20 episodes, physical 14D L1)

| run | step 0 | step 150 | step 300 |
|---|---:|---:|---:|
| duplicate #1 | 0.02086 | 0.01961 | 0.02605 |
| duplicate #2 | 0.02086 | 0.01403 | 0.02312 |
| true #1 | 0.02083 | 0.01244 | **0.01846** |
| true #2 | 0.02083 | 0.01256 | 0.02183 |
| 14D reference (R5 script, unchanged) | 0.02054 | 0.01749 | 0.02253 |
| identity | 0.00480 | 0.00480 | 0.00480 |
| constant velocity | 0.00102 | 0.00102 | 0.00102 |

From `replication_summary.json`:

| step | duplicate mean [min, max], spread | true mean [min, max], spread | separation | improvement |
|---|---:|---:|---|---:|
| 0 | 0.02086 [0.02086, 0.02086], 0.0% | 0.02083 [0.02083, 0.02083], 0.0% | (initialisation only) | 0.2% |
| 150 | 0.01682 [0.01403, 0.01961], **33.2%** | 0.01250 [0.01244, 0.01256], **0.9%** | **true_max < duplicate_min** | **25.7%** |
| 300 | 0.02459 [0.02312, 0.02605], **11.9%** | 0.02015 [0.01846, 0.02183], **16.7%** | **true_max < duplicate_min** | **18.1%** |

No curve is monotone: duplicate #1 rises by 33%, duplicate #2 rises by 65%, true #1 rises by 48%,
true #2 rises by 74% between steps 150 and 300. The A/B ordering nevertheless holds at both points.

### 4.2 Final evaluation detail (val, step 300)

| metric | duplicate #1 | duplicate #2 | true #1 | true #2 |
|---|---:|---:|---:|---:|
| macro L1 over episodes | 0.02605 | 0.02312 | **0.01846** | 0.02183 |
| \|pred delta\| / \|expert delta\| | 5.27 | 4.51 | 3.63 | 4.25 |
| pred delta std (expert 0.01243) | 0.0415 | 0.0386 | 0.0341 | 0.0372 |
| **corr(Δ_pred, Δ_gt)** pooled | +0.084 | −0.022 | **+0.097** | −0.054 |
| sign agreement at \|Δ\|≥1e-3 (n=335) | 0.519 | 0.484 | 0.469 | 0.463 |
| unique generated token rows | 22/60 | 3/60 | 14/60 | 5/60 |
| teacher-forced token accuracy | 0.490 | 0.507 | 0.562 | 0.545 |
| episodes beating cv / identity | 0/20, 0/20 | 0/20, 0/20 | 0/20, 0/20 | 0/20, 0/20 |

**The decisive caveat**: the direction metrics show **no** treatment effect (corr within ±0.1 and of
opposite signs between repetitions; sign agreement ≈ chance in all four runs), while the L1 metrics
do separate. The only systematic difference is the displacement magnitude (ratio 4.89 → 3.94 on
average). A 300-update model that over-predicts motion is punished by L1 in proportion to that
over-prediction, so **a lower L1 with unchanged direction correlation is not evidence of history
utilisation** — it is evidence of better magnitude calibration. This is why P6 (direction) is the
gate that matters and why it FAILs.

Per-dimension L1 (val, step 300, true #1 vs duplicate #1), order
`left_arm_0…5, left_gripper, right_arm_0…5, right_gripper`:

| dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | .0081 | .0100 | .0082 | .0074 | .0047 | .0053 | .0013 | .0042 | .0056 | .0046 | .0040 | .0015 | .0024 | .0000 |
| cv | .0015 | .0020 | .0015 | .0016 | .0010 | .0010 | .0000 | .0013 | .0010 | .0010 | .0010 | .0007 | .0007 | .0000 |
| duplicate #1 | .0289 | .0550 | .0409 | .0300 | .0137 | .0190 | .0178 | .0212 | .0341 | .0205 | .0239 | .0109 | .0153 | .0337 |
| true #1 | .0176 | .0421 | .0313 | .0267 | .0110 | .0141 | .0054 | .0156 | .0306 | .0260 | .0178 | .0082 | .0118 | .0004 |

The treatment arm is better on 12/14 dimensions (worse on `right_arm_2`, `right_arm_3`) but still
above identity on every dimension that moves. Dimension 13 (right gripper) has a **zero** expert
delta on these 60 frames, so its correlation is undefined and excluded; duplicate #1 emits a
constant non-zero gripper command (L1 0.0337), pure underfitting that true #1 avoids.

### 4.3 Token diversity

No run reaches R5's 800-update level (60/60 unique rows): 22/60, 3/60, 14/60, 5/60. Two collapsed
dimensions appear in true #1 (`right_gripper` 1 unique token / mode 1.00, `left_gripper` 3 / 0.95).
This is a budget effect, but it fails gate P7 regardless.

## 5. The 300-update metric is close to its noise floor

The 14D reference run doubles as a **byte-level reproduction check** of R5's protocol:

| quantity | R5 original | R6 reproduction |
|---|---:|---:|
| step-1 loss | 13.7153 | **13.7153** |
| step-1 grad norm | 49.58 | 49.51 |
| step-0 train eval | 0.03395 | **0.03395** |
| step-0 val eval | 0.02054 | **0.02054** |
| val @ ~200 steps | 0.00753 | 0.01749 (150) / 0.02253 (300) |
| val @ 800 steps | 0.00749 | — (not re-run) |

Initialisation, data order and the first evaluation reproduce exactly; the trajectories separate
from about step 25 (loss 2.6346 vs 2.7333 on the same sample). That is GPU kernel non-determinism
(no `torch.use_deterministic_algorithms`, bf16 GEMMs) amplified by a chaotic short-horizon
optimisation. Consequences:

* identical-configuration spread is 33.2% (step 150) and 11.9% (step 300) for the control arm, and
  0.9% / 16.7% for the treatment arm — i.e. **one pair of runs cannot resolve an effect smaller
  than ~30%**;
* the A/B ordering survived replication at both points (2 runs per arm, 4 cross-comparisons), which
  is why P3 is reported as 初步支持 rather than as a single-pair coincidence;
* R5's published early curve point is **not** reproducible, so cross-round comparisons should quote
  per-seed values and direction metrics, not one endpoint.

## 6. Why R5's numbers must be read with care

R5's published residual curve (0.02054 → 0.00753 → 0.00773 → 0.00581 → 0.00749) is not reproducible
at its early points: the same script, seed, data and budget produces 0.01749 at 150 updates and
0.02253 at 300. Either that evaluation caught a transient the trajectory does not normally pass
through, or the run followed a lucky path. The 800-step endpoint (0.00749) was **not** re-run
(a 800-step run of the new interface is outside R6's authorization), so this report does **not**
claim R5's endpoint is wrong. It claims that **single-run validation L1 at this scale is not a
stable quantity** (2–3× spread) and that the "0.00749 vs 0.00480" conclusion in R5 should be
re-checked with repetitions before being used as a capability baseline.

## 7. Pre-registered gates

| gate | criterion | measured | verdict |
|---|---|---|---|
| P1 smoke gate | PASS before training | 14/14 | **PASS** |
| P2 identical interface & budget | same dim/steps/lr/seed/samples; duplicate gap 0; true gap > 0 | all true | **PASS** |
| P3 history beats control by ≥10% | `L1_true < 0.9·L1_dup` | 0.01846 vs 0.02605 (29.1%); replication −18.1% | **PASS (初步支持)** |
| P4 beats hold-state | `ratio < 1` | 3.846 | **FAIL** |
| P5 beats constant velocity | `L1_true < cv` | 18.07× | **FAIL** |
| P6 direction corr > 0.3 and sign > 0.65 | measured | corr 0.097 / −0.054, sign 0.469 / 0.463 | **FAIL** |
| P7 token diversity not collapsed | no dim ≥ 0.95, ≥50% unique rows | dims 6, 13 collapsed; 14/60 rows | **FAIL** |

Phase-1 gates H1–H6 all PASS (see `HISTORY_PROPRIO_DIAGNOSTIC.md`).

Checkpoint contract (`checkpoint_contract.json`): both 28D arms declare `proprio_dim=28` and their
mode, the projector on disk is 28-wide with 16,900,096 parameters, SHA-256 binds adapter and
projector, and an R5 14D checkpoint resolves to `mode=none` with a 14-wide projector — the legacy
deployment path is unchanged. **PASS**.

## 8. The six questions

**Q1. Can R5's failure be explained mainly by missing history information?**
**No — not mainly.** R5's residual model (0.00749) was worse than *identity* (0.00480), and identity
needs no history at all. Phase 1 shows the history-dependent segment is identity→cv
(0.00480→0.00102, 4.7×) and that it is genuinely reachable (corr 0.9432, codec floor 0.000584).
Missing history can at most explain the distance to cv, **not** the distance below identity.
**尚未验证**: whether a history-aware VLA closes that segment — this pilot did not.

**Q2. Correlation between past velocity and the true expert delta?**
**已验证 0.9432** (val pooled; left 0.9400, right 0.9305; per-dim 0.871–0.961; 20/20 episodes
positive; sign agreement 95.6% at `|Δ| ≥ 1e-3`). Detail in `HISTORY_PROPRIO_DIAGNOSTIC.md` §3.

**Q3. 256-bin quantization oracle L1 — is it a main bottleneck?**
**已验证 0.000584** for residual targets (rounding 0.000272 + clipping 0.000312, 2.09% of entries
clipped) and **0.003444** for absolute targets. Residual: **not** a bottleneck (0.32× cv, 12.8×
below R5's residual error). Absolute: a hard floor 1.9× worse than cv.

**Q4. Did adding history proprio improve validation performance?**
**初步支持, but not in the way the question assumes.** L1 improved by 25.7% (step 150) and 18.1%
(step 300) on means, and the two arms are completely separated in both repetitions. However
`corr(Δ_pred, Δ_gt)` stayed at chance (+0.097 / −0.054 versus +0.084 / −0.022 for the control) and
sign agreement stayed at ≈0.47 in every run; the only systematic change is a smaller displacement
overshoot (5.27/4.51 → 3.63/4.25). So the measured gain is **magnitude calibration, not learned
motion direction**, and it is far too small to matter (3.85× worse than identity).
**尚未验证**: whether history can improve direction with a budget that lets the model converge.

**Q5. Does the history model beat identity / true constant velocity?**
**Verified: no.** 3.85× worse than identity, 18.1× worse than velocity extrapolation, 0 of 20
validation episodes won against either, in all four runs.

**Q6. Is there evidence for the formal 3000-step training?**
**Verified: no.** P4/P5/P6/P7 all FAIL; no run has converged (predicted displacement 3.6–5.3× the
expert, 3–22 of 60 unique token rows, teacher-forced accuracy 0.49–0.56); the only signal that
survives replication is a magnitude effect, not a direction effect; and the primary metric still has
a 12–33% noise floor. A 3000-step run would cost ~10× more and would still be judged against a
baseline (cv) that the absolute codec cannot beat and that the residual codec only approaches.

## 9. Failure and defect log

See `evidence/history_proprio_pilot_20261019T161033Z/FAILURES.md`: the failed first smoke gate
(token-level sensitivity was too coarse), the comparison-script KeyError, the over-strict legacy
metadata check, four wrong unit-test expectations, the non-reproducible 300-update variance, the
non-monotone control curve, and one CUDA OOM on a repetition attempt (started while the 14D
reference still held 18 GB — it was re-run alone; no foreign process was touched).

## 10. Deployment contract (implemented, unit-tested, never run live)

`code/xpolicylab/policy/OpenVLA_Pure/model.py` now resolves the proprio interface from checkpoint
metadata: legacy checkpoints (no metadata) keep the 14D path with byte-identical arithmetic; a 28D
checkpoint must declare `proprio_dim` and `proprio_history_mode`; the projector width is verified
against the checkpoint **before** `load_state_dict`; the previous state comes from a per-env buffer
of **observed** states that is cleared by `reset()`; the first-observation rule (slot 1 = current
state) is identical in training and deployment. 24 CPU tests cover the contract, including a
source-level test that refuses a regression to a hard-coded 14.

**尚未验证**: the 28D path has never been started inside a live policy server; the live server was
not touched in this round.

## 11. Recommendation for the next round (not executed here)

1. **Fix the metric before the compute.** L1 at 300 updates has a 12–33% noise floor and mixes
   magnitude error with direction error. Use `corr(Δ_pred, Δ_gt)` and sign agreement as the primary
   gates, report both magnitude ratio and direction separately, evaluate on the full 20-episode
   validation frames (6,711) or a fixed 300-frame subset, and run ≥3 seeds per arm.
2. **Test the magnitude hypothesis directly**: the treatment gain is a smaller overshoot. If that is
   all history buys, a simple output-scale calibration or a magnitude-regularised loss would reach
   the same L1 without any interface change — that comparison is the cheapest decisive experiment.
3. Only then scale to ≥2000 train frames / ≥2000 updates, same A/B, same gates.
4. Keep the residual target; the absolute codec floor (0.003444) makes gate P5 unreachable by
   construction.
5. Re-run R5's headline endpoint with repetitions before continuing to cite 0.00749 as the baseline.

## 12. Reproduction

```bash
bash scripts/run_r6_pilot.sh --check     # preflight + phase-1 CPU diagnostics
bash scripts/run_r6_pilot.sh --smoke     # 28D interface smoke gate (GPU, ~6 min)
bash scripts/run_r6_pilot.sh --execute   # duplicate + true, 300 updates each, then gates (~35 min)
```

Repetitions and the 14D reference were launched with the same arguments into fresh directories
(`duplicate_residual_rep2b`, `true_residual_rep2`, `legacy14d_residual`), and
`diagnostics/r6_replication.py` recomputes the noise-floor table from any set of run directories:

```bash
python diagnostics/r6_replication.py --run duplicate=R1 --run duplicate2=R2 \
    --run true=T1 --run true2=T2 --out replication_summary.json
```

The execute path refuses to overwrite an existing run directory, enforces the 300-update cap in
code, and refuses to start without a PASSing smoke gate, on a dirty tree, or on the wrong branch.
