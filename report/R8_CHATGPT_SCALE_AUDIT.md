# R8 independent audit: 42D proprio normalization gate

Source reviewed: `audit/r8-pretraining-readiness@a2b71805d8abb9314bcca5025aa9e449c525a08e`.
This branch is a **non-invasive audit**: R7/R8 source, data, weights, and original
evidence are unchanged. New GPU training is **not authorized** by this report.

## Evidence confirmed

1. `evidence/r8_pretraining_readiness_20261020/ablation_verification.json`
   reports all 19/19 checks passed. It reconstructs L1 from archived prediction
   sidecars and confirms 200 ⊂ 2000 ⊂ 19127, 64/16/20 disjoint episodes.
2. On the existing, repeatedly consulted holdout (6,691 t>=2 frames),
   the 42D `[S,V,Acc] -> R` MLP averaged **0.00118589** at DEV-selected
   checkpoints versus constant velocity **0.00179496**, and all 3 seeds beat
   copy-V in 20/20 original holdout episodes.
3. `report/R7_CORRECTED_REPRODUCTION.md` correctly retracts the old
   direct-Δ R² error: `SV -> Δ` corrected R² is ~0.301, close to
   `SV -> R` ~0.311; the R target has a modest **3.4%** L1 gain.
4. R8 CPU tests check a standalone numpy-only 42D feature/target
   contract. They do **not** instantiate an actual 42D OpenVLA
   ProprioProjector, train the VLA, or verify KV-cache equivalence.

## Critical new preflight risk: inappropriate scale for V and Acc

In `code/xpolicylab/policy/OpenVLA_Pure/r8_contract.py`:

```python
now_n = normalize_block(now, pmin, pmax)
v_n = normalize_block(v, pmin, pmax)
acc_n = normalize_block(acc, pmin, pmax)
```

Here `pmin/pmax` are **absolute joint-position bounds**, but V and Acc are
**signed differences of joint positions**. Using the same affine offset and
clipping rule can discard information. For a position dimension bounded
`[0, 1]`, normalized velocity -0.01 and 0.00 both map to -1; 0.01 maps
to -0.98. Thus negative motion becomes indistinguishable from zero
motion, even though their action implications can differ. The
synthetic counterexample is decisive for this class of bounds; its
**actual frequency in Push-T HDF5 has not yet been measured**.

Also, the successful R7/R8 small MLP uses train-fitted `(x - mu) / std`
over all 42 features, **not** the above state-bounds normalization.
The strong MLP result therefore does not validate the information
quality of the proposed OpenVLA input transform.

### Safe measurement (CPU only; no model edits)

```bash
python diagnostics/r8_scale_audit.py \
    --bounds /home/dgut/pusht_runtime/push_T_bounds.npz \
    --out /tmp/r8_scale_audit.json
```

This checks, separately for 64/16/20 or existing 80/20 according to the
split, each dimension's share of normalized V/Acc at -1/+1 and counts
negative-valued velocities clipped to -1. Candidate symmetric scales
are calculated using **TRAIN episodes only**. Do not fit on the holdout.

**Do not replace production normalization without first checking the
real-data clipping rates and running a train/dev-only protocol comparison**:
(a) existing position-bounds normalization vs (b) signed, zero-centered
train-only V/Acc normalization. Freeze the choice using TRAIN/DEV data,
store its statistics in checkpoint metadata, and evaluate on a **new,
previously unused independent trajectory set**. Keep the old path as
an explicit legacy mode.

### GPU smoke may be developed independently

It is safe to implement the `diagnostics/r8_smoke.py` shell/test harness
without running long training. The actual 42D VLA forward/backward
and causal KV check should use the *resolved, frozen* normalization
contract. This smoke must not be called complete on the basis of 27
numpy contract tests.

## Other qualifications

- All 100 original episodes have been reused through R2–R8 for
  training/development or evaluation. The original 20 holdout episodes
  are no longer untouched for final generalization claims. A **new**
  independent held-out set must be frozen *before* more model comparison
  or large-scale 7B promotion.
- The 19/19 verification validates arithmetic and sample nesting on
  archived predictions; the verifier imports the same sampling
  functions as training. It is **not a wholly independent evaluation
  implementation**. Its numerical evidence is still useful.
- A dataset where `A[t] == S[t+1]` exactly and time steps are 25Hz
  rewards near-constant-velocity joint prediction. It does not by
  itself demonstrate task planning, multimodal grounding, or
  closed-loop Push-T success.

## Next gated pipeline

1. **DeepSeek, CPU verification lane:** execute r8_scale_audit.py
   on real data, archive raw JSON, actual bounds and SHA. Report
   per-dimension sign/clipping and the new train-only candidate scales.
2. **DeepSeek, independent implementation lane:** prepare 42D
   ProprioProjector, loss/decode adapters and separate r8_smoke.py,
   without touching R5–R7 and without long GPU fine-tuning.
3. **ChatGPT, audit lane:** review raw result, test for model-signal
   annihilation or data leaks; submit narrow regression-tested fixes.
4. **Research leader:** approve final new-test-set acquisition,
   normalization contract and small GPU experiment budget.
