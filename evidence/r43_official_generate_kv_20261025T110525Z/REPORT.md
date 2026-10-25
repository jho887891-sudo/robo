# R43 — Official Generate + KV Cache: integration, three-path comparison, real 7B A/B

**STATUS: FAIL (diagnostic only)** — the official generate path is wired and runs, but it is **not**
token-identical to the deployed full-recompute path on 1 of 20 requests, so it does not pass the
strict deployment gate. No weights, prompts, decoders or safety constraints were changed.

| field | value |
|---|---|
| evidence dir | `evidence/r43_official_generate_kv_20261025T110525Z/` |
| code baseline | `d99635c4cebff1a5b8ee459ad60aea7695408bc6` (R42 smoke harness) |
| research branch | `research/r43-official-generate-kv-20261025` |
| GPU | NVIDIA RTX A6000, 50.897 GB total, 47.68 GB free before load |
| model load | 29.38 s (weights loaded exactly once for all three paths) |
| exit code | 3 (FAIL) |
| training steps | 0 |
| simulator / robot | never started, never commanded |

## 1. What was actually built

`code/xpolicylab/policy/OpenVLA_R42/r42_official_generate.py` — a minimal adapter around
`model.generate()` that supplies the three R42-specific facts the stock generation loop drops:

1. **`prepare_inputs_for_generation` (instance-level replacement, never the class).**
   The stock implementation (`modeling_prismatic.py:678-713`) forwards only
   `input_ids / attention_mask / pixel_values / past_key_values / use_cache`, so `proprio`,
   `proprio_projector` and `labels` are silently discarded on the prefill and the model
   runs with `proprio=None` — a different computation from training. The replacement keeps them
   on the prefill and forces them to `None` on every incremental step, which is exactly what
   `modeling_prismatic.py:534-550` asserts (batch 1, `past_key_values` present,
   `labels is None`).
2. **`labels` semantics.** `pure_autoreg_mode.py:22-27` raises when `labels` is `None`,
   so the prefill passes `labels=input_ids.clone()` — the same tensor the deployed backend passes
   (`r42_deploy_server.py:525-528`). Nothing about the action mask is relaxed.
3. **Legal-token constraint.** A real `LogitsProcessor` masks every logit outside 31744..31999 to
   `-inf`, so the generation loop cannot emit EOS/PAD/vocabulary tokens. `min_new_tokens=max_new_tokens=14`
   and `eos_token_id` is unreachable; the adapter additionally refuses to return unless it saw
   exactly 1 prefill + 13 cached steps, all 14 tokens inside the window, and the vision backbone ran once.

`diagnostics/r43_official_generate_kv.py` — the three-path experiment (below).
`tests/test_r43_official_generate.py` — 12 CPU tests (all pass), including a miniature VLM that
reproduces the two-phase forward contract *and* the stock `prepare_inputs_for_generation` defect.

Two implementation facts worth keeping (both cost a failed debug run):

* `prismatic.vla.constants` is imported lazily and selects `ACTION_DIM=14` **from `sys.argv`**.
  The ARX marker must stay in `argv` for the whole process (use `parse_known_args()`), otherwise the
  import falls back to the LIBERO constants (`ACTION_DIM=7`) and `assert_runtime_contract()` refuses to serve.
* `generate()` must run inside the same `torch.autocast('cuda', bfloat16)` context as the deployed
  forward (`r42_deploy_server.py:522-529`). Without it the bf16 weights meet float32 activations and the
  first incremental step dies with `q_proj: mat1 float != mat2 bfloat16`.

## 2. The three paths (same weights, same inputs, same decode contract)

| path | vision encodes / request | LLM work / request | measured vision calls |
|---|---|---|---|
| **A** full recompute (deployed default) | 14 | 14 full multimodal forwards | 14.00 |
| **B** full recompute + vision cache | 1 | 14 full multimodal forwards | 1.00 |
| **C** official generate + KV cache | 1 | 1 prefill + 13 incremental steps | 1.00 |

Path A/B go through the unmodified `R42DeployServer.act()`; path C through the adapter. All three
use the same `prompt_ids` (31 tokens), the same `pixel_values` (18 channels = 6 per camera x 3 cameras),
the same 42D feature (fresh `reset_all()` per request, so velocity/acc are identically zero), the same
`decode_with_metadata(prev_cmd=state, dt=0.04)` and therefore the same SAFE clip + rate rule.

Weights were identified, not assumed: 12 base-model files / 15,084,602,956 B and 9 checkpoint files /
1,125,377,222 B, each sha256'd (`base_model_weight_hashes.json`, `checkpoint_hashes.json`), and
`core.forward` was verified to equal `PeftModel.forward` bit-for-bit (max |delta| = 0.0), which proves the
LoRA adapter is still applied when `generate()` runs on the unwrapped core.

## 3. Equivalence

| comparison | tokens identical | requests identical | max absolute logit delta |
|---|---|---|---|
| A vs B | **280 / 280** | 20 / 20 | **0.0** (bit-identical) |
| A vs C | **276 / 280** | **19 / 20** | 3.8125 |

* The three-camera vision cache (path B) is **exactly** equivalent: not a single logit moves.
* Path C is identical at the **prefill step on all 20 requests** (`k=0`, max |delta| = 0), i.e. vision +
  42D projector + prompt + label mask + position offsets are all correct.
* The first divergence is request 15, token index **1**: path A's top-2 logits were **exactly equal**
  (`5.53125` vs `5.53125`, margin `0.0`) and a `0.03125` logit difference — exactly one bf16 ulp at
  that magnitude — selected bin 133 instead of 129. The remaining 3 differences (k=3,4,5 of the same
  request) are the resulting prefix cascade.
* Across all 280 steps: the 4 flips sit at margins 0.0, 0.0, 0.0625, 0.75; of the 3 exact ties in the
  whole matrix, 2 flipped. Action-space impact stays small but is **not** used to excuse the gate:
  max |delta action| over all 20 requests = **9.0e-4**, mean 5e-5.

**Candidate mechanisms (source-localised, NOT proven here).** The full multimodal branch feeds the LLM an
explicit all-ones multimodal attention mask (`modeling_prismatic.py:632-634`) while the cached generation
branch hardcodes `attention_mask=None` (`modeling_prismatic.py:539-550`). Logically identical, but it
selects a different SDPA kernel shape (`q_len=1` with no mask vs `q_len=N` with a mask), and the bf16
recompute of the K/V for the whole prefix uses a different GEMM shape each step. Either alone explains a
sub-ulp difference that only matters at an exact tie. Proving it requires an eager-attention A/B, which
this run did **not** do.

## 4. Performance (20 fixed requests per path, 1 warmup, same process)

| path | p50 | p95 | p99 | mean | token p50 | speedup vs A | multiple of the 40 ms period |
|---|---|---|---|---|---|---|---|
| A | 5505.1 ms | 5848.3 ms | 6001.4 ms | 5560.7 ms | 359.0 ms | 1.00x | 137.6x |
| B | 3481.8 ms | 3563.3 ms | 3588.5 ms | 3456.6 ms | 204.3 ms | **1.58x** | 87.0x |
| C | 1166.6 ms | 1266.1 ms | 1574.2 ms | 1195.4 ms | 59.8 ms | **4.72x** | 29.2x |

Path C decomposition: prefill p50 **283.5 ms** (3-camera vision encode + 42D + 31-token prompt),
incremental decode p50 **59.7 ms/token** (13 steps). Peak memory 16.9 GB allocated / 17.5 GB reserved
(A 16.6 / 17.4; B 16.6 / 17.4).

Remaining bottleneck after the cache: the **single multimodal prefill is still 24% of the request** and the
13 decode steps are 750 ms; 29x the 40 ms control period means this is an offline audit, not a real-time
claim. 20 requests are not a statistically sufficient real-time characterisation and no closed loop was run.

## 5. Answers to the five questions

1. **Did official Generate + KV Cache get wired into R42?** Yes, mechanically: 1 multimodal prefill +
   13 incremental steps per control step, exactly 14 legal tokens, vision encoded once, no cache reuse
   across requests/frames/states. It does **not** pass the equivalence gate.
2. **What does each path reuse?** A: nothing (14 full multimodal forwards). B: the three-camera vision
   encoding, once per control step. C: vision once + the whole multimodal prefix K/V across the 13
   decode steps.
3. **Is the KV cache strictly equivalent?** **No.** 276/280 tokens, 19/20 requests; the prefill is exact
   everywhere, the divergence starts at the first incremental step and only bites at exact ties.
4. **Real speedup, and what is left?** 4.72x end-to-end (1.58x from the vision cache alone, i.e. the
   KV cache contributes ~3x on top). Left: the 283 ms prefill and 13 x 60 ms decode steps.
5. **Does it pass the pre-deployment gate?** **No.** Per the R43 rule, a single differing token makes this
   a diagnostic experiment, not a deployment candidate.

## 6. Reproduction

```bash
export PYTHONPATH=/home/T7/ojh/robot_actual/RoboDojo:/home/T7/ojh/robot_actual/RoboDojo/XPolicyLab:/home/dgut/pusht_runtime/openvla_oft_src:/home/dgut/pusht_runtime/openvla_oft_src/vla-scripts
export PYTHONDONTWRITEBYTECODE=1 R42_POLICY='' R42_KV_CACHE=0 R42_VISION_CACHE=0
unset R42_AUDIT_LOG_PATH R42_AUDIT_RUN_ID
/home/dgut/pusht_runtime/venv_oft/bin/python diagnostics/r43_official_generate_kv.py   --deploy-yml code/xpolicylab/policy/OpenVLA_R42/deploy.yml   --out /tmp/r43-evidence-$(date -u +%Y%m%dT%H%M%SZ) --cases 20
```

CPU contract tests: `python tests/test_r43_official_generate.py` (12 tests, CPU only, no CUDA, ~3 s).

## 7. Artifacts

`report.json`, `console.log`, `environment.txt`, `equivalence_report.json`, `sequences.json`,
`latency_by_token.jsonl` (840 rows), `logits_window_{A,B,C}.npy` (20x14x256 float32),
`base_model_weight_hashes.json`, `checkpoint_hashes.json`, `exit_code.txt`, `SHA256SUMS`.

## 8. What this run does NOT establish

No closed-loop control, no simulator, no robot, no real camera data, no statistical real-time guarantee,
and no root-cause claim about the 20 requested deployment questions. The candidate mechanism in section 3
is a hypothesis with line numbers, not a proven cause.
