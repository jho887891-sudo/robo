# R11 P0 closeout — training-blocking source defects, red first then green

Branch: `feature/r11-p0-closeout` (from `r11/p0-audit@8a444d1`), base spec
`missions/R11_R16_DELEGATED_EXECUTION_SPEC.md@83b5794`, budget
`mission/R11_R16_DELEGATED_BUDGET_AND_GATES.json@83b5794`.
GPU seconds spent in this stage: **0** (CPU only). Optimizer steps: **0**.

Protocol used for every item: write the negative/red regression first, show it failing against
`8a444d1`, then make the minimal fix and show the whole suite green.
Red log: `evidence/r11_p0/red_regression.log` (14 failed / 7 passed / 7 errors).
Green log: `evidence/r11_p0/green_regression.log` (**250 passed, 3 skipped**; the pre-existing
baseline was 208 passed / 3 skipped).

## P0-1 — the 256-bin codec is now a single source of truth

Measured defects in `r42_adapter` at `8a444d1`:

1. `decode_command()` denormalised with `(q99-q01)` while `action_codec.denormalize()` uses
   `(q99-q01+1e-8)`. Encode and decode were therefore not inverses and were not bit-equal to the
   codec that produced the frozen 64-TRAIN q01/q99.
2. `build_labels()` did not apply the trailing `<space>` (29871) rule that
   `train_pushT_pure2.make_ids()` applies, so the prompt was one token short and the HF causal
   shift landed one position early.

Fix: `token_ids_for_residual()` / `residual_from_indices()` now delegate to `action_codec`;
`normalize_prompt_ids()` + `build_labels()` reproduce the legacy construction exactly.
Regression: `tests/test_r11_p0_codec.py` (random 14D bit equality, all 256 bins on four
dimensions, boundary/saturation, STOP=2 vs PAD=32000 separation, label/causal-shift positions).

Measured codec property (documented, not "fixed", because the 64-TRAIN quantiles are frozen):
the map is `2(x-q01)/(q99-q01+1e-8)-1`, so a residual of exactly `q99` sits
`2e-8/(q99-q01) ~ 6.7e-7` below +1.0 and encodes to bin **254**, while the closed top of the
lattice (`q99+1e-8 == denormalize(+1.0)`) encodes to **255**. The test asserts this exact
behaviour so a future silent change of the frozen codec cannot pass.

## P0-2 — prismatic 7D/8D/chunk defaults are fenced off

`prismatic.vla.constants` chooses its table from `sys.argv`: LIBERO (`ACTION_DIM=7`,
`NUM_ACTIONS_CHUNK=8`, `PROPRIO_DIM=8`) unless `arx`/`goai` appears in the command line, in
which case ARX (`14/25/14`) is used. The legacy entrypoint hard-codes `sys.argv` to smuggle
`arx_goai_...` in, and its evaluation loop iterates `range(ACTION_DIM)` — i.e. it would decode
**7** tokens instead of 14 if the argv marker were ever missing. That defect is now pinned by a
regression instead of being relied upon.

Scope note (measured, so it is not overstated): `enable_pure_autoreg()` replaces
`_process_action_masks` with an all-False mask, so on the pure-autoregressive path ACTION_DIM
does **not** decide which embeddings are zeroed. What it does still decide is (a) the legacy
evaluation loop's token count (`range(ACTION_DIM)` -> 7 instead of 14) and (b) the chunked
`_predict_action` placeholder length (`ACTION_DIM*NUM_ACTIONS_CHUNK`). The R42 entry asserts
ACTION_DIM==14, never calls the chunked path, and keeps its own 14-token loop, so the remaining
exposure is exactly the two paths the regression pins.

Fix: `assert_runtime_contract()` fails closed unless the live runtime reports
`ACTION_DIM==14`, `ACTION_TOKEN_BEGIN_IDX==31743`, `STOP_INDEX==2`, `IGNORE_INDEX==-100`;
`assert_projector_width()` rejects 7/8/14/28/80-wide projectors; a static regression forbids the
dangerous patterns (`range(ACTION_DIM)`, `ACTION_DIM *`, `proprio_dim=7/8/`) in every R42 source.
Two subprocess regressions prove both directions (LIBERO rejected, ARX accepted and recorded).

## P0-3 — frozen normalizer, 21-field metadata, bit-identical round trip

`REQUIRED_METADATA` keeps all 19 historical fields and adds `normalizer` and
`normalizer_sha256` (canonical JSON SHA). The payload carries the positional bounds for S, the
signed q99 motion scales for V/Acc, the residual q01/q99, the motion quantile and the
64-TRAIN / 16-DEV-excluded provenance. `build_metadata(normalizer=None)`, legacy
`legacy_position` normalisation, 80-episode quantiles and a tampered `normalizer_sha256` are
all refused. `save_metadata`/`load_metadata`/`decode_with_metadata` reproduce the 14D action
**bit for bit** after the round trip.

## P0-3b — the rate limit and the physical box can no longer disagree silently

`decode_command()` now validates `prev_cmd`, `dt > 1e-6` and `max_rate >= 0`, refuses a
`prev_cmd` outside the box (the rate rule is undefined there), and re-checks the physical box
after rate limiting. With `prev_cmd` inside the box the post-rate command is inside it by
convexity, so any residual violation means the two constraints are inconsistent: the action is
refused (`on_unsafe="raise"`) or replaced by a defined hold (`on_unsafe="hold"`) instead of
being quietly clamped.

## P0-4 — real deployment server skeleton

`code/xpolicylab/policy/OpenVLA_R42/r42_deploy_server.py` loads base 7B + LoRA + the 42D
projector + validated metadata, pre-processes three raw RGB cameras in the frozen order (RGBA and
BGR refused), keeps per-env 42D history with `reset(env_idx)`, decodes exactly 14 tokens with the
argmax restricted to 31744..31999, applies the physical/rate safety through the shared decode
path, and replays a recorded step under `torch.no_grad()` with all weights frozen.
`tests/test_r11_p0_deploy.py` exercises all of it on CPU with an injected deterministic backend —
this is a contract test, **not** a 7B GPU verification (that is R13's job).

## P0-5 — training entry assertions, verified on the real HDF5

`training/r42_train_contract.py` enforces `t >= 2`, `A[t] == S[t+1]`, history-only S/V/Acc,
future-mutation invariance, `[B,42]/[B,14]/[B,14,256]` shapes, gripper dims (6,13) and per-env
history isolation. Measured on the real Push-T HDF5: `max|A[t]-S[t+1]| == 0.0` over 360 frames,
i.e. A really is the next-step absolute joint target.

## Remaining gaps (explicit, not claimed as done)

* `--smoke` (R13) and every real training entry have never been executed on the GPU yet; that is
  the next stage and it is booked against `R13_7B_ZERO_STEP`.
* `num_actions_chunk`/chunked diffusion paths are unused by the R42 path but still exist in the
  pristine modeling code; only `ACTION_DIM` drives the 14-token loop, and it is asserted.
* The R11 collector (G1) has not been started at the time of writing.
