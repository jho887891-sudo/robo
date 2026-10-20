# Gate C — failure log (原始失败留证)

Three iterations: two harness defects and one methodology correction. No gate criterion was
relaxed; the CPU contract requirement (same prefix must agree, controls must be detected) is
unchanged from the first attempt.

| # | action | result | diagnosis | fix |
|---|---|---|---|---|
| 1 | first `--cpu-selfcheck` | `same_prefix_fp32` 7/8, median max\|dlogit\| 0.087 | my probe called the cached branch with `past_key_values=None` and only the last prefix token, i.e. it never actually prefilled | prefill now returns last-prefix logits and caches KV; cached steps always thread the cache |
| 2 | after prefill fix | clean 8/8 with noise floor 8.0e-08, but the negative control `shifted_position_ids` also matched 8/8 | a +1 position shift moves tiny-model logits by only 7.7e-04, not enough to flip argmax | controls are now judged on (argmax flip) OR (logit delta > 100x the clean noise floor); shift increased to +3 |
| 3 | review of the audit criterion | - | `margin <= 2*max|dlogit|` is NECESSARILY true for any genuine argmax flip, so it cannot separate numeric from structural causes | replaced with a documented numeric-noise-floor classification (`numeric_plausible` / `structural_suspect`); no root cause is claimed |
| 4 | final run | **PASS**: clean 8/8 (floor 8.0e-08), both controls detected, `tests/test_r9_kv_same_prefix.py` 10/10 | - | - |

Iteration 3 is a methodology correction, not a criterion relaxation: the GPU verdict remains
OPEN and the module still refuses to run without `R9_KV_GPU_AUTHORIZED=1`.
