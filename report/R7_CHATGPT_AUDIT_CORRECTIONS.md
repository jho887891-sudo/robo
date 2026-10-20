# R7 independent code audit by ChatGPT (2026-10-20)

Source: `jho887891-sudo/robo`, `audit/r7-state-dynamics-baseline` at
`77c470494f833252679fef8c686c799123743b69`.
Review: `review/r7-state-dynamics-audit-20261020`.

**Status: CODE CORRECTIONS COMMITTED; CPU RE-EXECUTION REQUIRED.
Do not treat this document as new empirical results.**

## What remains supported by submitted evidence

- On the reported 20 held-out Push-T episodes and matching t>=2 frames,
  `[S,V,Acc] -> R` produced mean physical L1 0.001204 across 3 seeds versus
  constant velocity 0.001795. The reported 20/20 episode wins and good
  direction consistency are credible as measurements, **not yet repeated on
  this corrected source**.
- The corrected nearest-representable 256-bin oracle (0.000462 residual,
  0.002326 absolute, versus 0.001828 constant velocity) is supported by the
  reported independent exhaustive bin enumeration, under fixed independent
  per-dimension quantiles and no postprocessing.
- 469 STEP outliers and Morph One CAD are a **different repository/project**;
  this R7 is about Push-T action and proprio prediction, not chassis simulation.

## Defects fixed

1. **Wrong correction ground truth for direct-delta target:** old code
   computed `R_hat=Delta_hat-V` but called
   `evaluate_correction(R_hat, Rva)` where `Rva` was actually **Delta**.
   Hence R², correlation and sign in the old direct-delta rows were computed
   against the wrong target. The *physical command L1* for delta was
   reconstructed correctly and is not directly changed by this correction.
   Both target modes now use the invariant `R_gt=(A-S)-V` for diagnostics;
   dev command reconstruction also uses the proper target semantics.

2. **Stale loop-variable bug corrupting pilot200:** the 200-frame ablation
   always trained `R`, but evaluated via the final value of outer variable
   `tgt`, which was `delta` in `scripts/run_r7.sh --mlp`.
   This subtracted V **a second time** from the predicted correction.
   The reported 200-frame L1 of 0.0248–0.0274 and the 16–20x inference
   **must not be cited as an unbiased data-coverage estimate** until rerun.

3. **Fixed-epoch pilot wasn't actually last-epoch:** old pilot used
   `train_mlp(Xp,Rp,Xp,Rp,patience=10**9)`, but `train_mlp` restored the
   lowest **train-set** loss checkpoint, rather than evaluating the last
   checkpoint after the allocated budget. The new `select_best=False` path
   enforces last-epoch for this ablation. All pilot predictions are now saved
   as `pred_pilot200_R_<seed>.npz` for verification.

4. **Holdout-selected channel gating:** the reported 0.00120795 hybrid was
   constructed after inspecting which dimensions had worse **holdout** L1.
   This is a descriptive, hindsight-conditioned analysis, not an unbiased
   deployment/architecture selection. The new code selects corrected
   dimensions on DEV only, freezes indices and then computes holdout L1.

5. **Verification gaps:** the old verifier permitted missing seed files by
   skipping them, verified action-next-state identity on only 3 of 20 heldout
   episodes, and could not verify the pilot output target. New checks require
   every seed and pilot sidecar, verify correct pilot R reconstruction and
   fixed budget, and check label alignment across all 20 held-out episodes.

6. **Historical-run preservation:** the audit runner now defaults to a
   separate output directory and refuses to overwrite an existing
   `verification.json`.

## Experimental interpretation after these fixes

- **Do not revise the original 15/15 verification JSON in place.**
  Keep the original as historical evidence and produce entirely new artifacts
  from the review SHA. A previous reproducible arithmetic result can still be
  accompanied by an invalid semantic interpretation.
- The head-to-head **L1** difference `SV->R = 0.001727` versus
  `SV->Delta = 0.001787` is an observed, fixed-split result. The old
  *Delta correction R² = -0.063* is invalid because of the wrong target
  comparator. It must be recalculated, not merely described as bad learning.
- The corrected pilot will give new L1 numbers. Additionally, the pilot200
  samples came from **80 train+dev episodes**, whereas the full model uses
  **64 train episodes** and early stopping on 16 dev episodes. Therefore
  "only frame coverage caused the improvement" is **not a causal
  isolation**, even after code repair. For causal coverage ablation,
  DeepSeek should author a matched 200-frame sample **from the same 64
  training episodes**, with the same DEV selection and comparable update/
  epoch budgets, then evaluate the untouched 20 holdout episodes.
- The old hybrid holds the reported holdout labels fixed while deciding
  which grippers to bypass. The new dev-selected hybrid is the first
  credible train/dev-driven gate, with numerical result pending rerun.
- Comparing 3 seed means across one train/dev/holdout split is an
  encouraging offline result. It is not a robot closed-loop success rate,
  cross-task generalization, or proof that proprio is sufficient.
- Adapting OpenVLA to 42-dim input and R targets is **not literally a
  one-line drop-in**: target quantiles/action normalization, proprio projector
  dimensions, dataset temporal alignment, decoding and inference reconstruction
  must all be audited. No 7B finetuning authorized by this report.

## CPU rerun contract (DeepSeek executor, ChatGPT author)

1. Use exact review SHA in isolated worktree; no GPU/7B tasks.
2. Run `python tests/test_r7_state_dynamics.py` and
   `python tests/test_r6_representable_oracle.py`.
3. Use fresh `R7_RUNS` and `R7_TS` values and run
   `bash scripts/run_r7.sh --all` (may take longer than the original
   minimal tests). Do not overwrite the previous `r7_state_dynamics_20261020T0050Z`
   evidence or alter checkpoints/HDF5.
4. Export **all 3 seeds** of `pred_SVAcc_R_<seed>.npz`,
   `pred_pilot200_R_<seed>.npz`, corrected main/acc JSON, probes and
   verification.json, run logs, provenance and SHA in new evidence branch.
5. Report direct-delta correction R², corrected pilot200 L1, all three
   DEV-selected hybrid dim masks/L1, and comparisons to earlier numbers.
6. If errors, return raw traceback; do not edit audit code or thresholds.
7. Only after this audit passes should DeepSeek build the *new* 64-episode
   matched coverage ablation or a controlled visual-vs-proprio comparison.

Original R7 15/15 does not override these new reproducibility conditions.
