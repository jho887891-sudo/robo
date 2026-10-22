# M2 shared 2-task model: physical capability audit and promotion gate (workline B)

Status: **VERIFIED** — and the gate result is **FAIL**, which supersedes the P6 probability PASS for any
promotion decision.

## 1. Two source fixes first (they change the numbers)

* **no-op baseline** (`tools/m2_gate_check.py::baseline_bins`): it computed
  `mean|(S-S)-A| = mean|A|`, i.e. the error of commanding ZERO JOINT POSITIONS, not of holding the current
  pose. Corrected to `mean|S-A|`; the legacy quantity is kept as `l1_zero_position_LEGACY` so the old
  reports stay interpretable. Tests: `tests/test_m2_noop_baseline.py` (3 tests on a real DEV frame).
* **official-result provenance** (`tools/r15_loop_dev.py`): a live run now accepts an official
  `_result.json` only if its mtime falls inside that launch's window, stat-ed on the filesystem that owns
  it (the container, where `eval_result` is a symlink); `--reparse-only` records are labelled
  `PROVENANCE_PENDING` instead of claiming a fresh verdict. Tests:
  `tests/test_r15_loop_provenance.py` (5 tests).

## 2. Physical action error vs baselines (same 8 DEV frames/task, same codec, same SAFE clip)

| task | **model SAFE L1** | no-op (A=S) | CV (A=S+V) | zero-position (legacy) |
|---|---|---|---|---|
| push_T | **0.0014** | 0.0045 | **0.0009** | 0.3806 |
| stack_bowls | **0.0070** | 0.0092 | **0.0022** | 0.4629 |

Paired per-frame deltas, 95% bootstrap CI:
* push_T model - no-op = **-0.0031** [-0.0053, -0.0005] → genuinely better than holding still;
* push_T model - CV = **+0.0005** [-0.0001, +0.0016] → not better, point estimate worse;
* stack_bowls model - no-op = -0.0022 [-0.0090, +0.0052] → not significant;
* stack_bowls model - CV = **+0.0048** [-0.0005, +0.0121] → 3.2x worse in the point estimate.

Per group (mean L1): push_T arm model 0.0017 / no-op 0.0053 / CV 0.0011; gripper model 2.5e-5.
Motion strata (slow vs fast frames): the model is only competitive with CV on SLOW frames, and loses on
fast ones — i.e. exactly where a decision matters.

Residual quality: |R_hat|/|R_true| = **0.47** (push_T) and 1.34 (stack_bowls); R² = **-0.04** and
**-1.28**. The predicted residual is roughly uncorrelated with the demonstration's residual.
Clip rate 12% (push_T) / 4% (stack_bowls).

Task conditioning is real but physically shallow: the wrong prompt changes the predicted bins on **100%**
of frames, yet the resulting SAFE L1 is 0.0016 (push_T) / 0.0072 (stack_bowls) versus 0.0014 / 0.0070 with
the correct prompt — a ~0.0002 difference in a quantity whose total error is ~0.0014.

## 3. Forgetting curve, in PHYSICAL terms

`mission/m2_physical_curve.json`, all ten sealed checkpoints (probability CE in brackets for contrast):

| step | push_T model / no-op / CV | stack_bowls model / no-op / CV |
|---|---|---|
| 200 | 0.0017 / 0.0045 / 0.0009 | 0.0074 / 0.0092 / 0.0022 |
| 600 | 0.0016 / 0.0045 / 0.0009 | 0.0078 / 0.0092 / 0.0022 |
| 1000 | 0.0015 / 0.0045 / 0.0009 | 0.0074 / 0.0092 / 0.0022 |
| 1400 | 0.0015 / 0.0045 / 0.0009 | 0.0072 / 0.0092 / 0.0022 |
| 2000 | 0.0014 / 0.0045 / 0.0009 | 0.0070 / 0.0092 / 0.0022 |

**No catastrophic forgetting** — neither task degrades; both improve marginally and monotonically in the
last 1000 steps. But **no capability is acquired either**: 2000 steps move push_T by 0.0003 and stack_bowls
by 0.0004, both still worse than a zero-parameter constant-velocity extrapolator.

## 4. Promotion gate (2 -> 4 tasks): **FAIL**

| condition | result |
|---|---|
| train/eval/deploy action contract consistent | PASS (one codec, one normalizer, per-task dispatch) |
| P6 metric legal and independently recomputable | PASS (corrected archive, 20/20, reproducible) |
| physical SAFE action beats simple baselines, or closed loop shows extra capability | **FAIL** (loses to CV on both tasks; closed loop 0/1 twice) |
| no unacceptable forgetting | PASS (flat, not degrading) |
| task condition actually affects the action | PASS (100% of bins change under the wrong prompt) |
| eval data not leaked | PASS (DEV excluded from fitting, verified) |

Per the pre-registered rule the multi-task expansion **stops here** and turns into mechanism diagnosis.
The 2000-step checkpoint is preserved; no valid optimisation step is retrained.

## 5. The honest interpretation

CE on the 256-bin window improved (P6 PASS: 2.1214 vs no-op 2.6241 and CV 2.6736) while the physical action
error did not move and stayed above the CV baseline. The probability metric is dominated by which bin is
*slightly* more likely among 256 candidates whose spread is ~7 effective levels of the true residual; being
right about the distribution there does not mean issuing the right joint command. **A P6 PASS must never be
quoted as evidence of physical control**, and this report exists so that it cannot be.
