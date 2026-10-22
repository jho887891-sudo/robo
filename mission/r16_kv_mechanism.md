# R16 KV-cache mechanism note (sidecar) — 2026-10-22

Status: the LLM-prefix cache is **NOT deployable**. Evidence and mechanism, in the order it was obtained.

## 1. What was built

`_TorchBackend.prefill_cached()` runs one multimodal forward with `use_cache=True` (vision + 42D +
prompt) and `step_cached()` feeds only the new action token to `self.model.language_model(...)` with the
returned `past_key_values`. `R42Policy._decode_tokens()` uses it when `kv_cache_enabled` is set; the cache
is created and dropped inside one call, so cross-step/frame/env reuse is impossible by construction.

## 2. What was measured

32 frozen DEV inputs (8 frames x 2 tasks x 2 prompts), real 7B step2000 checkpoint:

| quantity | result |
|---|---|
| inputs that diverge | **32 / 32** |
| k=0 max abs logit difference | **0.0** (bit-identical prefill) |
| first diverging k | **1** (2.4 to 29.6 logit units) |
| speedup, measured | **4.05x** (5906.9 -> 1459.8 ms p50, 64 actions after 20 warmups) |
| stale-cache negative control | detected (2.94) |
| VRAM | 17.1 GB |

## 3. Attribution (three-way, reference = the TRAINING forward)

`tools/r16_three_way.py`, 8 inputs:

| comparison | agreement |
|---|---|
| reference == **full deploy** path | **8 / 8** |
| reference == **cached** path | **0 / 8** |
| k=0 max abs vs reference | 0.0 for both |
| window-boundary bins (0/255) | reference **42**, full **42**, cached **0**, of 112 |

Two things follow. First, the deployed full path is a faithful reproduction of training — so the earlier
closed-loop attempts were driven by a correct decoder, and the 0/255 saturation seen in the first
equivalence run is real model behaviour rather than corruption. Second, the defect is in the first
incremental step.

## 4. What the mechanism is NOT

* **Not the vision tower.** `tools/r16_vision_cache_check.py` shares the backbone output across the 14
  tokens and gets **bit-exact** agreement (32/32, worst |delta| 0.0, stale cache detected at 0.0625). So
  the three-camera encode is deterministic and reusable; it simply is not the bottleneck.
* **Not FiLM conditioning.** `_process_vision_features(pixel_values, language_embeddings, use_film)`
  would make the vision embeddings text-dependent (which WOULD invalidate any frozen prefix), but the
  deploy path never passes `use_film`, so it is `False` and the vision features are image-only.
* **Not a mask-length mismatch in the prefill.** k=0 is bit-identical, so the multimodal attention mask,
  position ids and the vision/text concatenation are all correct for the first call.

## 5. Where it stands

The remaining suspect is the cached-attention path itself (bf16 K/V stored and re-read by a different
attention implementation than the fused prefill, over an ~800-token sequence) — but 29 logit units is far
outside normal bf16 variance, so the honest statement is: **the divergence is localised to the first
incremental step and not explained**.

Two grounded attempts were made (the incremental implementation, then this attribution). The mission rule
for this gate is explicit — a cache whose action outputs are not equivalent must not be deployed, and a
sidecar must be left instead of silently replacing the full backend. That is what happened:

* `kv_cache_enabled` defaults off, the full path remains the only serving path;
* the measured 4.05x is real but **unusable**, and is recorded as such rather than reported as a win;
* a fourth attempt is deliberately NOT made in this round: the fault budget for this root cause is spent,
  and the next hypotheses (attention implementation, cache dtype) are cheap to test but must be planned,
  not improvised.

## 6. Consequence for 25 Hz

Neither cache moves the needle on its own terms: the exact one is 1.0002x, the 4.05x one is not
equivalent. Per-action latency stays ~3.06 s p50 (clean loop) / 5.9-6.0 s (live server). That is 75-150x
the 40 ms budget, so the control path is **OFFLINE_SIM_ONLY**, and reaching 25 Hz requires an
architectural change (action chunking, distillation, shorter context) that needs a lead decision.

## SUPERSEDED (2026-10-22, commit 7c24c13 -> f249cf2)

**Everything above about "the incremental step is the defect" is wrong and is kept only as a record.**
The divergence came from the diagnostic harness feeding bin indices 0..255 instead of legal window tokens
31744..31999 into the FULL path. After that single missing offset was fixed, the cache is equivalent to
bf16 rounding (teacher-forced per-k worst 0.5, k=0 exactly 0.0), only 5/448 free-running tokens differ by
one bin, the physical impact is <= 0.16 mm, and the speedup is 4.42x. See
`mission/r16_kv_root_cause_report.md`.

The general lesson is recorded deliberately: a diagnostic whose REFERENCE path is illegal will produce a
convincing mechanism story (k=0 matches, k=1 diverges, "INCREMENTAL_DEFECT") out of its own bug, and the
cross-check that would have caught it immediately is "are the tokens I am feeding inside the legal
window?" - which is now enforced by `needs_legal_token_check` and four regression tests.
