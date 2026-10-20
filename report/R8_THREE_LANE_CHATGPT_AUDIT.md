# R8 three-lane delivery: independent ChatGPT audit

**Reviewed delivered branch tips**
- A `experiment/r8-motion-normalization@22d0a95c9a27f1ef187cbce02edefd599f2f49d8`
- B `feature/r8-vla-42d-smoke@6b1559772f5fd084e6ebde1a624a0cc8468c905a`
- C `research/r8-independent-evaluation@4f28417c18747bc2b208650b690b444f4d7a8678`
- Original audit ancestor `02acabc0130bb7b643ce2414e238d8dbb53107c0`.
- This **review branch** is independent, based only on B. It does NOT
  rewrite the three delivery branches or change `main` or any original evidence.

## VERIFIED from archived data / logs (not new GPU measurements)

1. Workstream A real HDF5 audit shows 30.7% V and 31.6% Acc entries
   saturated under the old *position-bounds* normalization. A's train
   64 episode q99 signed-motion scheme wins DEV proxy MLP L1 0.001113
   vs legacy 0.001166. Retain A scheme; do not confuse this with
   7B model effectiveness.
2. Workstream B stored GPU attempt3 JSON shows 14/15 reported gates,
   including real 7B forward/backward, 42D projector sensitivity and LoRA
   gradients. The KV comparison **FAILS** at 8/14 identical tokens.
3. Workstream C archives a no-overlap review of the 100 local push_T
   episodes, but there is no independent *new* task holdout. Final
   cross-episode generalization remains UNVERIFIED.

## Three independent review findings

### F1 (P0): frozen target bins use 80 episodes, not 64-TRAIN-only

B `tools/make_frozen_contract.py` loaded `split["train"]` (all 80)
as the q01/q99 fitting pool without excluding the 16 DEV episodes
used for normalization selection/early stopping.
B's committed GPU JSON reports
`frozen_from.n_train_episodes_for_quantiles=80`.
This violates the declared 64/16/20 protocol and makes target bins
incompatible with the truly train-only V/Acc normalization statistics.

**This review fixes the builder:** it calls
`r7_state_mlp.split_episodes(...,n_dev=16)` and uses only the 64 TRAIN
episodes by default, checks split disjointness and rejects >64
`--episodes`. It preserves the old evidence and requires generating a
**new** frozen contract from the original A `norm_contracts.json`.

### F2 (P0): GPU smoke ignores the frozen target q01/q99

In B `diagnostics/r8_smoke.py:466`, GPU passed
`residual_quantiles(real_frames(1))` to the decoder. This helper
uses `split["train"][:20]`, neither the frozen contract's 80 nor
the actual 64 training episodes. Hence B's real GPU Smoke did **not**
validate the same target bins it claims to have frozen.

**This review fixes the smoke:** it reads q01/q99 from the frozen
metadata, checks 14D finite/nondegenerate ranges and requires
64-TRAIN/16-DEV provenance, then uses the identical q01/q99 for
CPU and GPU checks. The old frozen file correctly FAILS CLOSED until
regenerated.

### F3 (P0): KV diagnosis compares different autoregressive histories after divergence

In B `--kv-diagnose` the cached path fed `token_kv` and the
full path appended `token_full`. Once the models disagree, the
subsequent logits come from DIFFERENT contexts. Subsequent logit
distance, argmax margin and numeric-vs-structural classification
therefore cannot identify the root cause.

**This review fixes the probe:** each step's full-sequence greedy token
is teacher-forced into the cached branch. Thus logits are
compared with exactly the SAME token history at every step. Even
when the logit differences are compatible with a low argmax
margin, the verdict remains **OPEN**: such compatibility is necessary
for a flip, not proof that bf16 numerical error rather than masking,
positional or cache mismatch caused it. Diagnose/fp32 A/B before making
that claim. Non-matching tokens cause nonzero exit status.

## Scope caveat: adaptation vs train-path integration

B's smoke explicitly uses `T.build(3, False)` and manually passes
`proprio=pr,proprio_projector=proj`. This is real GPU **adapter/forward**
evidence, but is **not** proof that the standard dataset-to-trainer
`train_pushT_pure2` entrypoint is connected to the 42D adapter,
signed normalization and target-R codec. Before any 7B optimization
step, add a no-update, real HDF5 trace test from the actual R8 training
loader through the loss and decode path with serialized metadata.

## Gates / next permitted work

**CPU gate, can run now without authorization:** execute
`python tests/test_r8_frozen_metadata_audit.py`, legacy R8 adapter
and normalizer regressions, and regenerate target q01/q99 with the
fixed builder using the exact A DEV-selected normalization contract.
Archive SHA + hashes + split identities; do not overwrite the
previous 80-episode artifact.

**GPU diagnostic gate, research-leader approval required:** after
CPU gate succeeds and GPU is free, at most one bounded
`diagnostics/r8_smoke.py --kv-diagnose` session against the new frozen
contract. Record common-prefix per-step full 256-bin logits,
top1/top2 margin and differences. A bf16-compatible margin is **not**
a pass without a precision-controlled check. Stop if OOM, interferes
with another process, or exceeds agreed resource budget. No
long-training authorization.

**Independent evaluation gate:** workstream C requires permission to
download or collect new push_T episodes. Freeze content hash and
protocol before evaluation; the original 20 held-out episodes are
repeatedly used and only support exploratory comparisons.

## Reviewer status

- **A:** evidence-backed, but production integration still gated.
- **B:** GPU adapter Smoke partial VERIFIED, KV **BLOCKED**, target
  frozen-contract consistency previously incorrect, with review fix
  implemented but not GPU-validated.
- **C:** inventory/tooling VERIFIED; new data BLOCKED on authorization.
- **7B training:** NOT AUTHORIZED, regardless of these CPU code changes.
- All review patches require remote HDF5/CPU re-run and separate GPU
  gating; GitHub CI covers synthetic contracts only.
