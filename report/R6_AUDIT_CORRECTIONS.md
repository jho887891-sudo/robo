# R6 audit correction: conclusions supported by the actual evidence

**Status:** targeted re-interpretation of R6, based on the unmodified original
`audit/r6-history-proprio-pilot@62088f76a36e4847f3c910f95f01a77f9c8de313`.
This audit branch changes neither original experiment numbers nor weights.

## 1. What R6 actually establishes

- On all 32,104 frames checked, expert `A[t]` was bit-identical to `S[t+1]`.
  This verifies the dataset's *one-step next-state label*, not task completion.
- Across the full 6,711-frame heldout pool, historical `V[t]=S[t]-S[t-1]`
  correlated 0.9432 with `A[t]-S[t]`; the copy-V baseline had L1 0.001828.
  On the selected 60-frame R5/R6 pilot validation set its L1 was 0.001022.
  **Do not mix full-pool and sparse-frame baselines.**
- With 200 TRAIN frames and 300 optimizer updates, two matched 28D variants
  gave mean VAL L1 0.024217 for `[S,S]` and 0.020466 for `[S,S_prev]`
  across 3 runs each, both substantially worse than hold-position 0.004801
  and copy-V 0.001022. Six runs had weak/negative direction cosines.
- The R6 action codec floor-encoder/decoder reconstruction error was 0.000584
  for the residual target and 0.003444 for absolute position targets.
  These are **errors of this specific bin-selection rule**.

## 2. Two over-strong claims to withdraw

### A. "Absolute 256-bin has an unbreakable 0.003444 L1 floor"

`diagnostics/r6_quant_oracle.py` runs
`decode(encode_fixed(true_action))`, where `encode_fixed` uses
`digitize - 1`, i.e. a FLOOR-grid convention. A VLA free-generation
head is not required to select that exact floor bin. It can choose a nearby
bin whose decoded physical value is CLOSER to the true action.

**Therefore the 0.003444 result is an encoder-reconstruction error,
not the mathematical lower bound over all 256-bin predictions.**
It is not correct to claim that no 256-bin absolute-action model can
ever beat the 0.001828 copy-V baseline from that number alone.

We have added `diagnostics/r6_representable_oracle.py` to calculate the
**nearest representable bin** for every target component, given the exact
decoder grid and TRAIN-only q01/q99. Because physical L1 decomposes over
components, this is a genuine lower bound for any model that outputs one of
these independent bin-center values with NO downstream postprocessing.

Remote, read-only command (requires R6 HDF5 paths configured):

```bash
python diagnostics/r6_representable_oracle.py --out /tmp/r6_representable_oracle.json
```

The new minimum-oracle numerical result **has not been run on HDF5 here**.
Do not assert whether it beats copy-V until the script has actually run.

### B. "Historical velocity information has a proven ceiling at copy-V"

`r6_history_ceiling.py` tested specific regressions: per-dimension linear,
joint linear, quadratic ridge and kNN1/5/10/25. None beat direct copy-V
in the recorded experiments. This is useful evidence for a strong,
computationally cheap baseline **but is not proof that no other function
of history can do better**.

Another evaluation issue: the original script chooses the best fitted model
from a dictionary including `per_dim_linear_all_val` on 6,711 frames,
and compares it with the copy-V score on a **different 2,000-frame subset**.
The amended script excludes the full-val score from the matched-subset
ranking, compares models on the 2,000-frame subset, and explicitly writes
`information_ceiling_proven: false`. Original evidence is preserved.

## 3. A/B amplitude calibration requires symmetrical comparison

R6 reported:

| Group (3 GPU runs) | VAL L1 raw | VAL L1 after each arm's TRAIN-fitted alpha |
| --- | ---: | ---: |
| duplicate (no history) | 0.024217 | 0.008168 |
| true (history) | 0.020466 | 0.005552 |
| hold current state | 0.004801 | 0.004801 |
| copy-V state baseline | 0.001022 | 0.001022 |

Comparing `duplicate` **after calibration** (0.008168) to
`true` **before calibration** (0.020466) does NOT establish that history
is irrelevant. When BOTH undergo the same TRAIN-only scalar-fitting
procedure, history remains numerically better by 0.002616.
But two of three true runs selected alpha=0 (i.e. **predict no motion**),
the number of repetitions is only 3 and both calibrated variants still
lose to holding state. A convincing history benefit is therefore
**not demonstrated**. The residual historical effect after adequate
training remains unknown; claims of "completely explained by amplitude"
must be treated as premature.

The reported post-hoc P8 criterion compares differently calibrated policies.
It is a useful extra diagnostic, not a valid causal test of whether
history adds predictive information. A symmetric, preregistered analysis
should compare both input arms after TRAIN-only calibrated alpha,
and compare direction on meaningful GT motions.

## 4. Task design and correct next experiment

The strongest result is not the 28D pilot: it is the dataset contract
`A[t]=S[t+1]`. In this task, the one-step offline action-matching objective
is effectively next-observation prediction on smooth 25-Hz trajectories.
History extrapolation is expected to be unusually strong even without images.
**It cannot validate multi-step push_T task reasoning or robotic success.**

Before any larger 7B fine-tuning:

1. Run `r6_representable_oracle.py` on full held-out HDF5 to establish
   whether **best-bin** error (not floor-bin error) beats copy-V.
2. Build a simple **streaming, state-only baseline** with `[S,V]`,
   trained on all available TRAIN frames; it is cheap and tests whether
   learnable regression is possible. Keep the 20 held-out episodes fixed.
   Include copy-V and identity baselines and motion-only direction metrics.
3. Then test history conditioning `[S,V]` with true vs zero velocity,
   initialized from exactly the same 14D representation, trained over more
   **distinct frames** before considering >300-step 7B runs.
   Do not change codec, loss, sample count and input representation
   simultaneously.
4. Report physical L1, cosine/sign on meaningfully moving dimensions, and
   multi-step offline error / closed-loop task success in a later,
   separately authorized experiment.

**Decisions:** No 3000/6000-step promotion; preserve R6 and R5 artifacts,
no online policy deployment; no claim that history is unhelpful or that
copy-V is a mathematical information ceiling.
