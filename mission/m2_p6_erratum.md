# M2 P6 erratum — action-window alignment and the CE definition (2026-10-22)

Status: supersedes the P6 numbers in `mission/m2_p6_two_report.json` (legacy). The legacy report is
kept, unmodified, as evidence; it is **permanently invalid for a P6 verdict**.

## E1. Action windows were read from vision-token rows

The model prepends the projected multimodal **vision** embeddings to the text embeddings: 6 fused image
copies over 3 cameras give a **769-token prefix**. `token_ce_from_logits()` always applied that offset
(`pad = logits_len - len(labels)`); `action_window_probs()` did **not**, so the 14 rows it read for
`ce_window`, `pred_bins` and every counterfactual task-routing distribution were vision rows, not the
legal action rows.

Consequences, all measured: the legacy `ce_window` (4.8883 / 5.3775) overstated the loss by 2.77 nats
per task, the own-prompt win rate read 0.25 instead of 1.0, the wrong-prompt penalty read ~0 instead of
0.65-0.77 nats, and the model looked like it emitted a single constant bin vector per task (1 distinct
bin) instead of 8.

Fix: `action_window_probs(logits, n_prompt, ..., n_text_tokens=None)` applies the per-prompt offset;
`evaluate_frame()` passes **each** prompt's own label length (the counterfactual prompt has its own
length and therefore its own pad); negative pads fail closed. Regression:
`tests/test_m2_action_window_alignment.py` (RED on the old code: 4 failures).

## E2. Archive marker and fail-closed loading

New archives carry `action_window_alignment = "vision-prefix-aware-v2"` and `schema =
m2-eval-archive-v2`. `load_eval_archive()` refuses anything else with `LEGACY_INVALID_FOR_P6`, so the
old file cannot silently re-enter a verdict. Old archives are never overwritten or deleted.

Corrected archive (regenerated from the sealed step2000 checkpoint, **no retraining**):
`evidence/dev_eval_v2_aligned.npz`, sha256 `17f8559b95cc411327915bb2c914a0c486dc88b3c3ef037a246b1843c888e0f1`.
Same frames, same true bins — the push_T `true_bins_sha256` `b45fcbaa88dd096ec8c8993672990068b7171a0dd987081ffda22d10016d56c3`
is byte-identical to the legacy run, so the comparison is like-for-like.

## E3. The CE used for a point baseline was not a likelihood

The legacy score `(1 - hit_rate) * ln(256)` charges **0 nats on a hit** (eps -> 0) and **ln(256) on a
miss** (eps -> 1). Two different smoothing limits inside one number: it is a heuristic bin-error score,
not a negative log-likelihood, and comparing a distributional model against it is not a like-for-like
SOTA comparison.

Pre-registered replacement (this round, before looking at any model number): eps-smoothed NLL with
**eps = 1e-3**, `p = (1-eps) * p_model + eps/256` on the model side and the same eps applied to the point
baseline's one-hot bin. The legacy score is still computed and reported as
`P6.ce_*_legacy_hybrid_score`, for continuity only, and is never decisive.

## E4. Corrected verdict

`mission/m2_p6_two_corrected_report.json` — **PASS, exit 0, 20/20 checks**, alignment
`vision-prefix-aware-v2`:

| task | corrected ce_window | no-op | CV | smoothed NLL | no-op NLL | CV NLL |
|---|---|---|---|---|---|---|
| push_T | 2.1214 | 2.6241 | 2.6736 | 2.1221 | 5.8934 | 6.0046 |
| stack_bowls | 3.0963 | 3.3667 | 3.3172 | 3.0964 | 7.5611 | 7.4499 |

Routing: own-prompt win rate 1.0 (required 0.9), min wrong-prompt penalty 0.653 nats (required 0.05),
8 distinct bins per task. Forgetting: both tasks improved against their own step-0 anchor.

## E5. Known limitation that is NOT fixed by this erratum

The per-save-point CE curve stored in `stage_evidence.json` was measured with the legacy alignment, so
its absolute values are not comparable to the corrected final point. The forgetting *check* remains
internally consistent because every point of that curve was measured the same wrong way, and it passes
either way. Regenerating all 10 save points would cost ~20 min of GPU and is queued as a bounded
follow-up, not claimed as done.

The sample is also tiny: **8 DEV frames per task**. Every number above is exploratory and the gate says
so; an episode-level cohort with pre-registered confidence intervals is the next step before any claim
about task-level capability.
