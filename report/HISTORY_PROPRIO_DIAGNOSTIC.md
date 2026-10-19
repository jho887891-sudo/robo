# HISTORY_PROPRIO_DIAGNOSTIC — R6 phase 1 (real HDF5, CPU only)

Branch: `audit/r6-history-proprio-pilot`
Base: `f974aeb` (R5 audit branch tip), `main` untouched.
Data: `/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data` (100 episodes),
split `split80.json` = 80 train / 20 val episodes, strictly disjoint.
Frames analysed: **25,393 train** and **6,711 val** legal frames
(`t` such that both `t-1` and `t+1` exist, i.e. 2 frames per episode excluded).
No model was loaded, no GPU was used, no training was started in this phase.

Evidence (regenerate with `bash scripts/run_r6_pilot.sh --check`):

* `evidence/history_proprio_pilot_20261019T161033Z/phase1_alignment.json`
* `evidence/history_proprio_pilot_20261019T161033Z/phase1_history_value.json`
* `evidence/history_proprio_pilot_20261019T161033Z/phase1_quant_oracle.json`

Scripts: `diagnostics/r6_data.py`, `r6_align.py`, `r6_history_value.py`,
`r6_quant_oracle.py`; tests `tests/test_r6_history_diagnostics.py`.

---

## 0. 结论摘要（中文）

1. **已验证**：`A_t == S_{t+1}` 在全部 32,104 帧上**逐位相等**（最大误差 0.0），索引偏移扫描最小值落在 `shift=+1`；25 Hz 恒定（dt = 40 ms），无重复帧、无重复状态、末帧有效。
2. **已验证**：历史速度 `V_t = S_t - S_{t-1}` 与专家增量 `Δ_t = A_t - S_t` 的相关性 **0.9432**（验证集池化；左 0.9400 / 右 0.9305；逐维 0.871–0.961；20/20 回合均为正），符号一致率在 `|Δ| ≥ 1e-3` 时为 **95.63%**。
3. **已验证**：恒速外推在 **20/20 验证回合、80/80 训练回合**中都优于保持状态（误差比 0.24–0.29）。但注意：恒速误差 `E3 = E|V_t − V_{t+1}|` 本质是**加速度**，保持状态误差 `E2 = E|V_{t+1}|` 是**速度**；两者之比 0.26 是轨迹平滑性的属性，不是某个隐藏信息通道。
4. **已验证**：256-bin 残差量化 Oracle 的物理 L1 = **0.000584**（rounding 0.000272 + clipping 0.000312），只有恒速基线 0.001828 的 **0.32 倍** ⇒ **残差模式下量化不是主瓶颈**。恒速策略经同一残差编解码后仍为 0.002119（仅比连续值差 15.9%）。
5. **已验证**：**绝对动作**模式的 256-bin 量化 Oracle 是 **0.003444**（步长均值 0.005352，逐维最大 0.008961），比恒速 0.001828 差 1.9 倍 ⇒ **绝对动作 + 256 bins 存在数学硬地板，任何完美模型都无法超过恒速基线**。这为 R2–R5 的绝对动作分支提供了新的、独立于模型的解释。
6. **尚未验证**：VLA 在 28 维（当前+历史）输入下能否在授权预算（≤300 次更新）内学会 `Δ ≈ V`。第一阶段只证明**信息存在**，不证明**模型可学到**。见 `HISTORY_PROPRIO_PILOT_RESULTS.md`。

---

## 1. Scope, and what is deliberately not claimed

This phase answers information-theoretic and data-integrity questions only:

* it does **not** train, fine-tune or evaluate any model;
* it does **not** change the action codec, the target representation or any checkpoint;
* it reads only `state/*` and `action/*` arrays, never `vision/*`;
* `S_{t+1}` is **never** used as a model input anywhere in this code base. It appears only
  in `E1` (an alignment check) and in the anti-leak control.

Error definitions (physical units, 14D, identical concatenation order to
`training/train_pushT_pure2.py::c14`: left arm 6, left gripper 1, right arm 6, right gripper 1):

| symbol | definition | meaning |
|---|---|---|
| `E1` | `\|A_t − S_{t+1}\|\|_1 / 14` | is the label the next observation? |
| `E2` | `\|A_t − S_t\|\|_1 / 14` | hold-current-state error |
| `E3` | `\|A_t − (2S_t − S_{t-1})\|\|_1 / 14` | past-only constant-velocity error |
| `E0` | `\|A_t − S_{t-1}\|\|_1 / 14` | previous-state error (upper reference) |

---

## 2. Task A — action/state temporal alignment

### 2.1 Headline: the label IS the next observation

| split | frames | E1 | E2 | E3 | E0 |
|---|---:|---:|---:|---:|---:|
| train | 25,393 | **0.000000** | 0.006893 | 0.001852 | 0.013713 |
| val | 6,711 | **0.000000** | 0.007043 | 0.001828 | 0.014015 |

E1 is not merely small: `max|A_t − S_{t+1}| = 0.0` and
`fraction of frames with max_abs < 1e-9 = 1.000` on **both** splits.
So `A_t ≡ S_{t+1}` bit-exactly for every frame of every episode, and therefore

`Δ_t = A_t − S_t = S_{t+1} − S_t = V_{t+1}` (the *next* velocity, never an input).

### 2.2 Index-shift scan (proves the offset is +1, not 0 or ±2)

Mean `|A_t − S_{t+shift}|` over all legal frames (val):

| shift | −3 | −2 | −1 | 0 | **+1** | +2 | +3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAE | 0.027419 | 0.020791 | 0.013963 | 0.007005 | **0.000000** | 0.007038 | 0.013988 |

Train is identical (best shift +1, MAE 0.0). No ambiguity, no half-frame offset.

### 2.3 Per arm and per dimension (val)

| error | left (dims 0–6) | right (dims 7–13) |
|---|---:|---:|
| E1 | 0.000000 | 0.000000 |
| E2 | **0.009344** | 0.004742 |
| E3 | 0.002385 | 0.001272 |

Per-dim E2: `[0.009726, 0.015034, 0.011198, 0.009739, 0.006042, 0.007177, 0.006491,
0.005128, 0.008041, 0.006037, 0.005123, 0.002496, 0.003232, 0.003139]`
(→ `left_arm_0 … left_arm_5, left_gripper, right_arm_0 … right_arm_5, right_gripper`).
The left arm moves ~2× as much as the right arm; **the data is left-arm dominant**,
so any per-arm conclusion must be read per dimension, never as "all 14 dims in radians".
Dims 6 and 13 are gripper positions with their own scale (their residual q-range saturates
at exactly ±0.1 in section 4), so they are **not** radians.

### 2.4 Motion stratification (val) — the cv advantage is not universal

| stratum (mean `\|Δ\|`) | n | share | E1 | E2 | E3 | better baseline |
|---|---:|---:|---:|---:|---:|---|
| static < 1e-4 | 28 | 0.4% | 0.000000 | **0.000023** | 0.000794 | **hold state** (cv adds spurious motion) |
| small 1e-4…1e-3 | 530 | 7.9% | 0.000000 | 0.000635 | **0.000427** | cv |
| medium 1e-3…5e-3 | 3,505 | 52.2% | 0.000000 | 0.002844 | **0.000951** | cv |
| large ≥ 5e-3 | 2,648 | 39.5% | 0.000000 | 0.013958 | **0.003282** | cv |

Reported honestly: on the 0.4% of frames where the robot is essentially静止, the
constant-velocity rule is ~35× worse than holding still. On 99.6% of frames it wins,
and it wins in 20/20 val and 80/80 train episode aggregates (ratio E3/E2 between
0.2395 and 0.2891). **No episode anywhere reverses the ordering.**

### 2.5 Structural checks (all 100 episodes)

| check | result |
|---|---|
| `additional_info/frequency` | **25 Hz for every episode** → dt = 40 ms, constant |
| repeated/zero-delta frames | 0 |
| identical consecutive state rows | 0 |
| duplicate state rows (rounded 1e-9) | 0 |
| last frame finite | yes, all episodes |
| median frame step | 0.0039 (max 0.067 — real motion bursts, no gaps) |
| frames with no `S_{t+1}` | 1 per episode (`t = T-1`), excluded everywhere |

There is no timestamp array in the HDF5; `dt` is therefore *declared* by the dataset
frequency, not measured. The absence of zero-step frames and the smooth step distribution
are consistent with a constant-rate recording, but this is **初步支持**, not proof.

### 2.6 Anti-leak test (E3 must be computable from the past only)

For every episode, at every 5th legal frame `t`, the next state `S_{t+1}` was replaced by
garbage and E3 recomputed:

* `e3_unchanged_when_future_scrambled = true` for **all** episodes, `max|ΔE3| = 0.0`;
* control (the test can fail): `E1` **does** change under the same corruption;
* `corr(future motion, expert delta) = 1.0` (0.9999999999999998–1.0) — an independent
  confirmation of `A_t = S_{t+1}`, **not** a usable input.

---

## 3. Task B — does past motion predict the expert delta?

`V_t = S_t − S_{t-1}`, `Δ_t = A_t − S_t`. Correlation excludes zero-variance
dimensions (reported as `null`, never as 0); no dimension was excluded in this data.

### 3.1 Correlation (val)

| statistic | value |
|---|---:|
| pooled `corr(V, Δ)` over all 6,711×14 entries | **0.9432** |
| left arm / right arm | 0.9400 / 0.9305 |
| mean of the 14 per-dim correlations | 0.9352 |
| mean per-dim `R²` | 0.875 |
| per-episode mean (left / right) | 0.9387 / 0.9319 |
| worst episode (left / right) | 0.9105 / 0.9102 |
| episodes with negative arm correlation | **0 / 20** |
| same for train (pooled) | 0.9393 |

Per-dim (val): `[0.9579, 0.9610, 0.9571, 0.9390, 0.9051, 0.9452, 0.9145, 0.9466, 0.9601,
0.9484, 0.9326, 0.8713, 0.9340, 0.9201]`.

`corr(V_t, V_{t+1})` (speed autocorrelation) is numerically identical to
`corr(V_t, Δ_t)` — as it must be, since `Δ_t = V_{t+1}`. This single number, 0.9432,
is the whole story: **the expert action increment is 94%-correlated with the motion the
robot was already performing one frame earlier.**

### 3.2 Sign agreement vs motion threshold (val, basis `|Δ| ≥ thr`)

| threshold | n (entries) | share | sign agreement |
|---:|---:|---:|---:|
| 0 | 93,954 | 100% | 0.8124 |
| 1e-5 | 43,255 | 46.0% | 0.9312 |
| 1e-4 | 42,753 | 45.5% | 0.9360 |
| 5e-4 | 40,893 | 43.5% | 0.9472 |
| **1e-3** | **38,667** | **41.2%** | **0.9563** |
| 2e-3 | 34,638 | 36.9% | 0.9673 |
| 5e-3 | 25,859 | 27.5% | 0.9809 |
| 1e-2 | 17,343 | 18.5% | 0.9875 |

The full-sample (threshold 0) figure is depressed only by entries whose delta is
indistinguishable from zero; every real-motion threshold is ≥ 0.93. Train matches.

### 3.3 Magnitude

| statistic | val |
|---|---:|
| `mean\|Δ\|` (left / right) | 0.009344 / 0.004742 |
| `mean\|V\|` (left / right) | 0.009360 / 0.004750 |
| ratio `mean\|V\| / mean\|Δ\|` | **1.0016** |
| mean signed Δ per dim | all `\|·\| ≤ 2.2e-5` (no systematic drift) |

Past motion and the expert delta have **the same magnitude** to within 0.2%:
plain "keep doing what you were doing" is dimensionally correct, not an
over-/under-shoot.

### 3.4 A TRAIN-fitted linear history model cannot beat plain velocity

Per-dimension OLS `Δ ≈ a·V + b` fitted on **train only**:

* slope `a` = `[0.950, 0.959, 0.955, 0.937, 0.888, 0.942, 0.911, 0.936, 0.957, 0.948, 0.927, 0.860, 0.920, 0.918]`
* intercept `b`: all `|b| ≤ 3.3e-5`

| predictor (val, physical L1) | value |
|---|---:|
| hold state (`Δ = 0`) | 0.007043 |
| constant velocity (`Δ = V`) | **0.001828** |
| train-fitted linear history | 0.001855 (+1.4%) |

So the affine map is essentially `0.94·V` with zero bias and is **slightly worse** than
the trivial rule. **Verified:** there is no extra history signal to extract beyond
"copy the last velocity"; the interesting question is only whether a VLA can learn that.

---

## 4. Task C — 256-bin quantization oracle

Codec and statistics are exactly R5's: R2-fixed `idx = clip(digitize(·)−1, 0, 255)`,
`token = 31744 + idx`, `LEVELS = linspace(−1, 1, 256)`, `q01/q99` from **train episodes
only**, computed on the *target representation* of each mode
(`absolute`: `A`; `residual`: `A − S`).
The oracle encodes the **true continuous expert target** and decodes it back; nothing can
be below this floor. Error is split into a grid part (rounding, codec floors onto the
grid) and a range part (clipping at `q01/q99`), never lumped together.

### 4.1 Residual target (the R5 configuration)

| quantity | val |
|---|---:|
| per-dim target range `q99−q01` | 0.0564 … 0.2143 (mean 0.1232; grippers exactly 0.2) |
| per-dim physical step `(q99−q01)/255` | 0.000221 … 0.000840 (mean **0.000531**) |
| **oracle physical L1** | **0.000584** (left 0.000668 / right 0.000500) |
| — rounding component | 0.000272 |
| — clipping component | 0.000312 (**53% of the total**) |
| entries clipped | 2.09% (all 14 dims clip somewhere) |
| mean signed error per dim (grid floor bias) | −5.2e-5 … −7.5e-4 (≈ −step/2) |
| direction flips, `\|Δ\| ≥ 1e-4` | 0.61% |
| direction flips, `\|Δ\| ≥ 5e-4` | 0.06% |
| direction flips, `\|Δ\| ≥ 1e-3` | **0.00%** |

Same pipeline applied to the baselines (val):

| predictor | continuous L1 | through the 256-bin codec | penalty |
|---|---:|---:|---:|
| oracle true delta | 0 | **0.000584** | — |
| hold state | 0.007043 | 0.007216 | +0.000173 |
| constant velocity | 0.001828 | **0.002119** | +0.000290 (+15.9%) |
| train linear history | 0.001855 | 0.002144 | +0.000289 |

**Verified:** in residual mode the codec costs ~16% on top of a cv-level policy, and the
absolute floor (0.000584) is 3.1× below the cv baseline. It cannot explain R5's residual
error of 0.00749 (12.8× above the oracle) — **量化不是残差模式的主瓶颈**.

### 4.2 Absolute target (the R2–R5 default, and the 6000-step legacy policy)

| quantity | val |
|---|---:|
| per-dim target range | 0.5021 … 2.2851 (mean 1.3974) |
| per-dim physical step | 0.001969 … 0.008961 (mean **0.005352**) |
| **oracle physical L1** | **0.003444** (left 0.004250 / right 0.002637) |
| — rounding | 0.002141 |
| — clipping | 0.001302 (1.51% of entries) |
| mean signed error (floor bias) | −0.0001 … −0.0060 |
| constant velocity through this codec | 0.004474 (2.4× its continuous 0.001828) |

**Verified:** an absolute-action policy with 256 bins over the *absolute joint range* has a
hard floor of ~0.00344 physical L1 — **1.9× worse than the constant-velocity baseline**.
No amount of training can make it pass the cv gate. This is a codec property, independent
of the model, and it explains a part of why the absolute-target arm (R5: 0.13351) was
never going to compete, even though its own error is far above the floor.

### 4.3 Not changed in this round

The codec is left exactly as R2 fixed it. Two follow-ups are now *justified by numbers*
but deliberately **not implemented** here: (i) round-to-nearest instead of floor (removes
the −step/2 bias and up to halves the rounding term), (ii) wider quantiles for the
residual target (clipping is 53% of the residual oracle error). Neither is needed to
explain the current results.

---

## 5. The five phase-1 questions

**Q1. Is past velocity highly correlated with the true action increment?**
**已验证 — yes.** `corr = 0.9432` (val, pooled), 0.9400/0.9305 per arm, 0.871–0.961 per
dimension, positive in 20/20 validation episodes (min 0.9102), sign agreement 95.6% at
`|Δ| ≥ 1e-3`. Magnitude ratio `mean|V|/mean|Δ| = 1.0016`.

**Q2. Does the constant-velocity rule keep its advantage in every validation episode?**
**已验证 — yes, with one honest exception by *motion regime*.** cv < hold-state in
**20/20** val and **80/80** train episodes (per-episode ratios 0.2395–0.2891). But on the
0.4% of frames that are essentially static, holding still is ~35× better than cv. The
aggregate advantage is therefore a property of *this* trajectory distribution
(velocity is 94% persistent), not a universal control law.

**Q3. What is the 256-bin quantization oracle error, and is quantization the main bottleneck?**
**已验证 — 0.000584 (residual) / 0.003444 (absolute).** In the residual configuration the
floor is 0.32× the cv baseline and 12.8× below R5's measured residual error, so
**quantization is NOT the main bottleneck**. In the absolute configuration the floor is
1.9× *worse* than cv, so there **it is a hard bottleneck**.

**Q4. Can the current 256-bin encoding limit achievable precision?**
**已验证 — mode-dependent.** Residual: yes but marginally (+15.9% on a cv-level policy;
53% of the remaining oracle error is clipping, i.e. removable without changing bin count).
Absolute: yes, decisively — mean step 0.005352 physical units, larger than the entire
cv gap (0.001828).

**Q5. Is training with history state worth starting?**
**已验证 for the information question, 尚未验证 for the learnability question.**
The information is present and near-optimally captured by a one-line rule (H1–H5 gates
below all pass), so the pilot is justified. What phase 1 cannot and does not claim is that
OpenVLA will learn to use slot 1 within the authorized budget — and the strongest
counter-evidence available (R5's residual model predicting `Δ` with correlation −0.18 while
a linear rule achieves +0.94) shows the failure to date is in the **mapping**, not in the
**information**.

---

## 6. Pre-registered gates (evaluated in `training/compare_history_arms.py`)

| gate | criterion | measured | verdict |
|---|---|---|---|
| H1 alignment | `E1 max == 0` and best shift == +1 | 0.0, +1 | **PASS** |
| H2 no future leak in E3 | E3 invariant to `S_{t+1}`, and the test can fail | ΔE3 = 0.0, E1 responds | **PASS** |
| H3 history carries Δ information | `corr ≥ 0.5` and `sign@1e-3 ≥ 0.75` | 0.9432, 0.9563 | **PASS** |
| H4 cv wins in almost all episodes | cv < hold in ≥ 17/20 val episodes | 20/20 | **PASS** |
| H5 residual codec not the bottleneck | `oracle ≤ 0.5 × cv_continuous` | 0.000584 ≤ 0.000914 | **PASS** |
| H6 absolute codec floor documented | reported | 0.003444 > 0.001828 | **PASS (informational)** |

Because H1–H5 pass, phase 2 (the 28D A/B pilot) is authorized and was pre-registered with
its own gates — see `HISTORY_PROPRIO_PILOT_RESULTS.md`.

---

## 7. What this means for the pilot design

* The treatment arm must receive `S_{t-1}`; the control arm keeps the **same 28D interface
  and the same parameter count** and fills both slots with `S_t`, so capacity is identical
  and only the information differs.
* `q01/q99` and the codec stay untouched: the residual representation is not the bottleneck.
* Because `Δ_t = V_{t+1}` exactly, a policy that copies `V_t` scores 0.001828 on the full
  validation set (0.00102 on R5's 60 spread frames). Anything above that is *not* a
  history-utilisation success; it is fitting noise.
* The most informative secondary metric is therefore **`corr(Δ_pred, Δ_gt)`**, not the
  absolute L1: R5's residual arm had the right magnitude and the wrong direction (−0.18).

---

## 8. Reproduction

```bash
git switch audit/r6-history-proprio-pilot
bash scripts/run_r6_pilot.sh --check      # preflight + phase-1 diagnostics + CPU tests
```

Every number above is written to the three JSON files listed at the top of this report;
the diagnostic scripts import `diagnostics/r6_data.py`, whose `c14()` is byte-compatible
with the training reader.
