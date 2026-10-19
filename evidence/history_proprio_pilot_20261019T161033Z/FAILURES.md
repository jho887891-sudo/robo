# R6 failure and defect log

Every failure encountered while producing R6, including ones that were fixed, with
the evidence that the fix worked.

## F1 — smoke gate v1 failed on token-level sensitivity (kept as a real observation)

First smoke run: `S2_history_slot_changes_output` and `S3_current_slot_changes_output`
were FAIL — perturbing either proprio slot by +0.35 changed **0 of 14** decoded action
tokens, even though the projector received gradient 0.93.

Diagnosis: the check itself was too coarse, not the interface. Measured per-dimension
top-1 softmax confidence up to 0.9996, and a max |Δlogit| of only 0.19–0.28 from the
proprio channel versus 5.94 from a 0.05 image perturbation and 10.05 from zeroing the
images. Token argmax cannot flip at that scale.

Fix: the gate now measures **logit-level** sensitivity over an amplitude sweep and
reports token flips as information only. Both arms then passed. The measurement
(the proprio channel is 20–30× weaker than the visual channel in logit terms at
random init) is retained in `smoke_*.json` and quoted in the report; it is not
hidden by the fix.

## F2 — `compare_history_arms.py` KeyError: 'model'

`r5_metrics.summarize_prediction` returns `per_arm_physical_L1[arm][method]`, and the
comparison script read it as `[method][arm]`. Fixed and re-run; the gate output in
`arm_comparison.json` is from the fixed version.

## F3 — checkpoint-contract check rejected the legacy 14D checkpoint

The first version required `pmin`/`pmax` inside `run_meta.json`. R5's checkpoints do
not carry them (the bounds live in `push_T_bounds.npz`), so the legacy arm failed for
a metadata convention rather than for a real incompatibility.

Fix: the bounds are only required **if present**, and are then required to be 14-long.
The legacy arm now resolves to `mode=none` with a 14-wide projector and PASSes; the
deployment path for old checkpoints is unchanged.

## F4 — four unit-test expectations were wrong on first run

`test_first_frame_policy`, `test_roundtrip_error_bounded_by_step`,
`test_clipping_is_reported_separately`, `test_hold_state_predictor_is_reference_free`
asserted behaviour the code does not promise (e.g. that value 0.0 is exactly on the
256-bin grid — it is not, the codec floors, so the error is one half step 1.96e-4).
The tests were corrected to assert the real contract; the codec was **not** changed.

## F5 — the pilot's validation L1 is not reproducible to better than ~2–3×

`training/overfit_pr1.py` was re-run unchanged (R5's own script, 14D interface, same
seed/data/budget). Step 1 loss (13.7153), grad norm (49.58 vs 49.51) and the step-0
evaluations (train 0.03395, val 0.02054) reproduce R5 **exactly**, but the trajectories
diverge from about step 25 (loss 2.6346 vs 2.7333) and R5's published step-200 val
(0.00753) is nowhere near this reproduction's step-150 (0.01749) or step-300 (0.02253).

Interpretation: identical initialization and data order, non-deterministic GPU kernels,
chaotic short-horizon optimization. Consequence: the 29% A/B gap at 300 updates is
**not** interpretable as an effect of history information. A second identical
`duplicate` run was executed to size that noise directly (see the report).

## F6 — the control arm's validation curve is not monotone

`duplicate`: 0.02086 (step 0) → 0.01961 (150) → 0.02605 (300). Any 300-update A/B
conclusion drawn from a single seed and a single evaluation point would be fitting
noise; gates P4–P7 are therefore reported independently of P3.

## F7 — no closed-loop, no policy-server run

Not attempted (not authorized, and the live policy server must not be touched).
The 28D deployment path is implemented and contract-tested on disk only; running it
in a live server remains 尚未验证.

## F5b — the noise floor was measured, and the A/B gap is only ~1.5–2× it

Two repetitions per arm (identical configuration, 300 updates each) give:

* duplicate step 150: 0.01961 / 0.01403 (33.2% spread); step 300: 0.02605 / 0.02312 (11.9 %);
* true step 150: 0.01244 / 0.01256 (0.9% spread); step 300: 0.01846 / 0.02183 (16.7%);
* the A/B ordering is the same in both repetitions at both steps (true max < duplicate min), mean
  improvement 25.7% (150) and 18.1% (300) — see `replication_summary.json`.

The first single-pair conclusion therefore survives replication, but the effect is only 1.5–2× the
same-configuration spread with n = 2 per arm, so it is reported as 初步支持, never as proven.
Critically, the direction metrics do **not** separate (corr +0.084/−0.022 vs +0.097/−0.054; sign
≈0.46–0.52 in all four runs), so the L1 gain is a magnitude-calibration effect, not learned motion
direction. This is the main reason gate P6 is treated as the decisive gate.

## F8 — CUDA out of memory on the first repetition attempt

The first `duplicate` repetition was launched while the 14D reference run still held 18.0 GB; the
GPU had 66 MiB free and allocation failed at model load
(`status.json: FAILED_RUNTIME`, log kept as `train_duplicate_rep2_FAILED_oom.log`).

Fix: wait for exclusive GPU use and run the repetition alone in a fresh directory
(`duplicate_residual_rep2b`). No foreign process was killed or touched, and the failed directory is
kept as evidence rather than deleted.

## F9 — kNN history-ceiling analysis: chunk too large, first run never finished

The first `r6_history_ceiling.py` used a 2,048-frame kNN chunk, which materialises a
2048 x 25,393 x 14 temporary (2.9 GB) inside the distance computation. The process made
no progress for 18 minutes, wrote an empty log, and was killed with the terminal session
(it also produced a follow-up "1 frame" that never appeared).

Fix: `CHUNK = 32` and kNN evaluated on a deterministic 2,000-frame subsample of the
6,711 validation frames (the fitted models are still scored on all 6,711). The analysis
then finished in about 40 s. Log: `history_ceiling.log`. No result from the failed run
was ever reported.

## F10 — the third repetition removed the step-150 separation

With two runs per arm the treatment was below the control at both evaluation points
("complete separation"). The third repetition of each arm (identical configuration)
gives control 0.01961 / 0.01403 / 0.01182 and treatment 0.01244 / 0.01256 / 0.01794 at
step 150: the ranges now overlap and the mean difference is only 5.5%. At step 300 the
ranges stay separated with a 15.5% mean difference.

Consequence recorded in the report: the earlier "complete separation" was an artefact of
n = 2, and any A/B claim at 300 updates must be based on the token-level analysis (which
shows the L1 difference is a displacement-magnitude artefact), not on the L1 ranking.
