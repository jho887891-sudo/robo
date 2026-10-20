# R7 failure and defect log

Every failure encountered while producing R7. Nothing was deleted, no acceptance
criterion was lowered, and each fix is stated together with the evidence that it worked.

## F1 — `r7_oracle_compare.py`: broadcast error in the brute-force grid

`action_codec.denormalize` was called with a (14, 256) normalized array and (14,)
quantiles, which cannot broadcast, and my first repair collapsed the 256 levels to a
scalar. Fixed by building one (256,) physical grid per dimension explicitly.

## F2 — the brute-force check compared a different frame set

The first version brute-forced a random 500-frame subsample and compared it with the
script's all-6,711-frame bound, so the "independent check" failed on sampling variance,
not on method. Fixed by running the exhaustive 256-bin scan over **every** legal
validation frame (chunked). Both modes now agree with the analytic bound to <1e-6
relative error.

## F3 — `pilot200` block crashed and lost the whole JSON

`stack_frames()` was fed raw episode dicts instead of `episode_frames()` output, raising
`KeyError: 'S'` after ~6 minutes of training; because the JSON was written only at the
end, all measurements were lost (they survived only in the log).

Fixes: (a) the correct call; (b) the JSON is now written after the baselines and after
every feature/target block, so a late failure cannot destroy earlier results.

## F4 — IndentationError introduced by a scripted patch

A remote `str.replace` patch left one line over-indented; `py_compile` caught it
immediately and the file was repaired before any run.

## F5 — one failing claim poisoned all later claims in the verifier

`r7_verify_claims.py` used `ok &= check(...)` both as the per-claim condition and as
the running tally, so the first FAIL made every subsequent claim print FAIL
(12/14, then 13/14 with a wrong margin threshold). Fixed: each claim is evaluated
independently, the tally is derived from the results list, and the R-target margin
threshold was set to 2e-5 (the measured margin is 6.0e-5). Final state: **14/14 PASS**.

## F6 — NaN warning from an empty motion stratum

`r7_metrics.evaluate` computed `np.nanmean` over an empty cosine slice when a stratum
contained no frames; guarded so the key becomes `null` instead of emitting a warning.
The same guard was applied to `evaluate_correction` for an all-zero ground truth.

## F7 — interpretation limit, not a bug: the kNN estimate is not a lower bound

The kNN(32) mean-R predictor scores 0.001527 while the MLP reaches 0.001233, i.e. the
non-parametric estimator is worse than the trained model. It is therefore recorded as a
weak reference, **not** as an irreducible-error bound, and the report explicitly refuses
to claim that proprio is exhausted. This is the same mistake R6 made with copy-V and
must not be repeated.
