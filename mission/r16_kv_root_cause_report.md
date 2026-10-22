# R16 KV cache root-cause report (2026-10-22)

Status: **VERIFIED** on real 7B weights + real DEV frames. The verdict is unusual and worth stating
first, because it changes what the whole R16 thread means.

## 1. Root cause, stated exactly

**The cache was never wrong. The diagnostic harness was.**

Location: `tools/r16_kv_equivalence.py`, function `decode_full()` (before commit `7c24c13`):

    ids.append(int(np.argmax(w)))                        # appends a BIN, 0..255
    lg = backend.next_token_logits(prompt + ids, pv, feat)   # ... and feeds it back as a TOKEN id

The action codec lives in the last 256 ids of a 32000-token vocabulary: legal action token =
`31744 + bin`. `decode_cached()` did that conversion correctly (`R.TOKEN_BEGIN_IDX + 1 + ids[-1]`), the
full-recompute path did not. So from the second step on, the "reference" was decoding from ordinary text
tokens 0..255 instead of the legal window:

* its logits saturated on the window edges (the 0/255 pattern that looked like corruption),
* its argmax wandered off, so the two paths differed by up to **31.4** logit units,
* and every comparison concluded the cached path diverged.

The first wrong operation is therefore **one missing `+ 31744` in a test harness**, not an attention
mask, not a RoPE position, not a cache contract, not Prismatic's multimodal path.

## 2. How it was pinned (each step falsified a different hypothesis)

| experiment | result | what it ruled out |
|---|---|---|
| PR#28 single-frame probe: prefill vs full, then k=1 with naive / explicit-mask / wrapper paths | k=0 **bit-identical**, every k=1 path within **0.03-0.13** | attention mask, `position_ids`, `cache_position`, Prismatic wrapper |
| multi-frame probe (5 cases, both task prompts), same three methods | A/B/C all **0.03-0.13**, including frames my parity run called divergent (t=155: 17.7 -> 0.125) | frame-dependence, prompt-length dependence |
| per-layer audit of the last position (layer 0..35) | layer 0 already at bf16 scale (0.14) | "a specific layer computes something else" |
| PR#28 text-only LM control | **0.031** | the language model's own cache implementation |
| cache length read from the real tensors | **805** = 36 prompt + 769 multimodal, as expected | the `n_past=text-only` suspicion (the variable is unused anyway) |
| after fixing bin->token: same 32-input run | worst **31.39 -> 1.31**, physical residual **1.05e-2 -> 1.1e-5** | the harness |

The probes never showed a mechanism error. That was the signal: a real cache bug does not vanish when you
change the *harness*.

## 3. Fix and regression

`7c24c13`: both decode loops are now module-level, backend-injected (`decode_full`, `decode_cached`),
and every token they append is checked against the legal window `31744..31999`
(`needs_legal_token_check`). `tests/test_r16_kv_decode_tokens.py` (4 tests) uses a stub backend that
records every call and asserts: only legal tokens are ever appended; both loops agree on the same stub;
the pre-fix loop is *detectably* wrong; the guard bounds are exact.

## 4. Numerical equivalence (32 frozen inputs x 14 tokens)

Teacher-forced per-k parity (both paths consume the SAME token sequence, so no cascade):

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| worst abs logit delta over 32 inputs | **0.0** | 0.375 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.125 | 0.125 | 0.125 | 0.25 | 0.125 | 0.25 | 0.5 |

**Verdict: NOISE_LEVEL_PARITY** (`mission/r16_kv_forced_parity.json`, tolerance 0.5). Logits in this model
are O(10-30) inside the window, so 0.125-0.5 is bf16 rounding, not a semantic difference.

Free-running (each path chooses its own tokens) the residual effects are small and bounded:

* **5 of 448** action tokens differ (1.1%), each by one codec bin,
* the implied physical 14D residual differs by at most **1.57e-4** (0.16 mm),
* replay over 10 dynamic sequences: worst 0.3125, and 7/10 frames produce different tokens from the
  previous frame (so the cache really is rebuilt per frame).

Honest statement: the KV path is **not bit-identical**. It is equivalent to bf16 rounding, and it changes
one bin on ~1% of tokens. That is a deployable-for-DEV property, not a certificate.

## 5. Speed (same weights, same inputs, 64 timed actions after 20 warmups)

| path | p50 | p95 | p99 |
|---|---|---|---|
| full recompute (14 multimodal forwards) | **5060.1 ms** | 5162.4 | 5211.5 |
| KV cached (1 prefill + 13 incremental) | **1145.1 ms** | 1175.1 | 1210.9 |

**4.42x** measured speedup, VRAM 17.1 GB. In a clean single-path loop the full path also measures
3058.9 ms, so the ratio depends on what shares the process; the 5060/1145 pair is the like-for-like
measurement (same loop, same tensor, adjacent in time).

At 1145 ms per control step a 600-step Push-T episode still takes ~11.5 minutes of policy time, i.e.
**28.6x** the 40 ms budget at 25 Hz. The cache removes most of the redundant compute; it cannot close a
28x gap. **25 Hz remains unreachable without an architectural change** (action chunking, distillation,
shorter context) — that is a research decision, not an inference optimisation.

## 6. Safety and invalidation

* a deliberately stale cache is detected (delta 2.94 vs 0.02 fresh),
* the vision-only cache is bit-exact (32/32, worst 0.0) and correctly invalidated per step,
* the KV cache is created and destroyed inside one `_decode_tokens()` call, so it cannot survive a step,
  a frame, an environment or a reset,
* `R42_KV_CACHE` defaults to 0: the full backend is the default and the cached path must be switched on
  explicitly and recorded.

## 7. The retracted claim

The earlier "three-way attribution" (reference == full 8/8, cached 0/8, "the incremental step is the
defect") was produced by the same broken tool: both of its paths were fed bin indices, so they agreed with
each other for the wrong reason. It is withdrawn in `mission/r16_kv_mechanism.md` and in the commit that
supersedes it. The lesson is recorded rather than quietly deleted: a diagnostic that makes the *reference*
illegal will happily manufacture a convincing mechanism story.

## 8. Reproduction

    # roots cause / parity
    python tools/r16_kv_first_divergence.py --checkpoint runs/m2_two_2000_20261021T123314Z/checkpoints/step2000 --task push_T --out evidence/r16_kv_first_divergence.json
    python tools/r16_kv_probe_multi.py --checkpoint <ckpt> --cases mission/r16_kv_cases_divergent.json --out evidence/r16_kv_multiprobe.json
    python tools/r16_kv_equivalence.py --checkpoint <ckpt> --frames-per-task 8 --out mission/r16_kv_equivalence_v3.json
    python tools/r16_kv_forced_parity.py --checkpoint <ckpt> --frames-per-task 8 --out mission/r16_kv_forced_parity.json
    python -m pytest tests/test_r16_kv_decode_tokens.py -q

`step2000` = `runs/m2_two_2000_20261021T123314Z/checkpoints/step2000`, LoRA
`c221e73932e97c43…`, projector `1635bd0202e43432…`, modeling sha256
`b5431a074c0025a12e46dc954a5e18d1d73477babb5ae42e3a12ab4b907f33a6`.
