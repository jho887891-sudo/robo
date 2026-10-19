# R5 diagnostic correction: which direction is actually being predicted?

**Scope:** source-based audit of already committed R5 GPU evidence at
`audit/r5-distribution-and-baseline-audit@f974aeb8aba0afb4ed1186d24b57b8d6f6beca12`.
No GPU, retraining, checkpoint modification, online deployment or data replacement.
Reproducible command:
`python diagnostics/r5_direction_audit.py --threshold 0.002`.

## Confirmed R5 A/B result (unchanged)

Physical 14D position L1, averaged across 60 held-out frames from 20 distinct
validation episodes:

| Policy | L1 |
| --- | ---: |
| Absolute VLA | 0.13350554 |
| Residual VLA | 0.00749008 |
| Hold current state | 0.00480059 |
| `S[t] + (S[t] - S[t-1])` historical state extrapolation | **0.00102205** |

Thus residual beats absolute by ~17.8x but loses to a state-only
temporal baseline by ~7.3x. Neither VLA meets the previously registered
acceptance criteria. Historical R5 GPU result files must not be rewritten.

## Three important measurement distinctions

### 1. +0.696 is **absolute position correlation**, not motion direction

In `prediction_direction_corr.json`, the absolute model's
mean 14D Pearson correlation `+0.6962647` is calculated from
**decoded action POSITION tokens vs decoded target POSITION tokens**.
This may capture a relationship with the current joint posture; it does NOT
establish that the model predicts the **sign or direction of
`A[t] - S[t]`**. Previous language suggesting that the absolute
model predicts movement direction correctly is unsupported by this value.

For residual, the targets are increments. Independent recomputation from
the exact committed predicted and GT action Token arrays gives a negative
Pearson correlation across most variable dimensions. The result is not
inconsistent with a 1.04x global average motion magnitude ratio: correct
average magnitude does not imply correct per-frame movement directions.

### 2. A constant dimension does not have correlation zero

The GT `right_gripper` target tokens are **identical** in the R5
60-frame validation evidence. Consequently Pearson correlation on this
column is mathematically undefined. The archived direction JSON stores
`0` for that dimension, producing 14-dimension mean `-0.1804884`.
Excluding this undefined dimension and recomputing from decoded tokens
produces approximately **-0.1943721 over the other 13 dimensions**.

This is a measurement-label correction, not a changed model result and not
a claim that the true physical delta has exactly that correlation; the
original raw HDF5 physical arrays were not committed.

### 3. Naive 79.5% sign agreement is inflated by near-zero quantized values

The original sign agreement counts 840/840 dimension-frame comparisons,
including GT residual components near zero. With **256** reconstruction
levels `linspace(-1,+1,256)`, there is *no exact zero bin center*: even
a physical target of exactly zero is mapped to a small signed number.
Those near-zero signs are not reliable evidence of movement direction.

Using only GT quantized residuals with `|delta| > 0.002` **in each
dimension's physical units**, the independently recomputed sign agreement
is **160/292 = 54.79%**. For a stricter threshold `0.005`,
the diagnostic agreement is **117/210 = 55.71%**.

These are comparisons using **decoded archived tokens**, not unquantized
expert action values. The 0.002 / 0.005 thresholds are exploratory, not
post-hoc success gates. Gripper and rotational-joint channels may have
different physical units, so do not call a 14D threshold “radians”.
Report counts and thresholds whenever giving a sign accuracy.

## Quantization-resolution hypothesis requires another experiment

Quantization bin spacing in the committed R5 residual **q01/q99** statistics
is `(q99 - q01)/255`, across 14 dimensions approximately
**0.000221 to 0.000840 physical units per bin**. This is narrower than
the report's suggested `0.001-0.002` spacing, though units and q01/q99
clipping matter. The model's current poor physical L1 **cannot yet be
attributed to quantization** without measuring the oracle error
`decode(encode(true_delta)) - true_delta` on raw unquantized HDF5 targets.

Changing to a continuous action regression head or adding direction loss
now would introduce a new mechanism before confirming whether the
representational bottleneck is at fault.

## Causal/input-information caveat and next evidence gate

The constant-velocity benchmark uses **both** `S[t]` and `S[t-1]`,
whereas the present OpenVLA receives `S[t]` as its proprio input,
plus current RGB and instruction. Thus the baseline is highly informative
about the dataset dynamics but is **not input-matched** with the existing
proprio-only model. A deployable history buffer could make two-frame
proprio available, but that must be implemented and tested, not assumed.

Before spending another multi-hour A6000 budget:

1. On the remote HDF5, audit the lag-1/lag-2 relationship between
   `A[t]-S[t]` and `S[t]-S[t-1]`, including sign and
   non-static subsets. Compare per-arm, per-episode and displacement
   direction, not only a 14D mixed-units mean.
2. Compute the **quantization oracle** for all TRAIN/VAL targets using
   training-only q01/q99, and separate rounding vs clipping error.
3. Probe whether **two-frame state history** gives a meaningful
   advantage beyond the single-frame model, and whether the current
   image stream has conditional predictive value. Do not infer
   causal visual understanding from token diversity or high
   teacher-forced accuracy.
4. Keep the no-3000-step and no-closed-loop deployment gates until
   an input-matched policy can outperform strong state-only controls
   on independent trajectories.

Scripts and tests in this branch are **offline diagnostics only**.
The old R5 report, evidence and weights remain unchanged.
