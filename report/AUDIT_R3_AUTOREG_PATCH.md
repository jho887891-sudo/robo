# R3: Pure OpenVLA causal autoregressive input fix

Base: `audit/r2-action-codec-fix@d63ad8a74fb16df9f2f61f0b9b6d1b707f37caba`.

## Source-confirmed issue

The actual dynamic module `modeling_prismatic.py`, multimodal `forward()`, masks action input embeddings with
`input_embeddings = input_embeddings * ~all_actions_mask`. In existing teacher-forced pure OpenVLA training,
all fourteen action positions are recognized by `_process_action_masks(labels)` and therefore zeroed.
Cached inference feeds actual previously generated action token ids through `language_model` without this
masking. This is a source-confirmed mismatch, **not yet a GPU-quantified cause of failed rollouts**.

## Minimal implementation

- New `code/xpolicylab/policy/OpenVLA_Pure/pure_autoreg_mode.py` selectively binds a **zero action-input mask**
  to a *single* pure OpenVLA core instance. Causal LM's normal attention and shifted cross-entropy
  remain intact; no global change to OFT source code.
- Training: call helper after constructing base core, **before PEFT wrapping**, so earlier action ids remain visible.
- Deployment: install on core before PEFT loading and again after LoRA merge. Cached action generation remains unchanged.
- R2 fixed codec still applies; do **not** resume from legacy 6000-step weights when assessing clean training.

## Local CPU smoke tests

`python tests/test_pure_autoreg_mode.py`

The tests cover instance isolation, action embeddings retained, nonzero action-input gradient,
missing labels guard, and incompatible core rejection. They **do not** prove runtime LoRA/transformers
compatibility, correct multimodal alignment, model logits equivalence, or real-world success.

## On the remote machine

The training script imports the helper from the repository checkout path
`code/xpolicylab/policy/OpenVLA_Pure/` or, if running from a standalone copy,
from `PUSHT_PURE_POLICY_DIR` (defaulting to RoboDojo's deployed OpenVLA_Pure directory).
**Copy/sync the helper before running any training**. The script fails loudly when it is missing.

Verify in the remote GPU runtime, without launching long training:
1. `python tests/test_pure_autoreg_mode.py` plus R2 codec tests.
2. Instantiate a fresh core; record mask and action embeddings for one real sample with
   fourteen labeled action tokens, and assert their input embeddings are retained.
3. For fixed image/state/prompt and action tokens, compare full teacher-forced logits
   (with causal shift: logit at previous position predicts current token) with
   prefix + cached sequential logits over all fourteen action positions.
   Compare per-token top-1, cosine/relative error, and differences with tolerances for BF16.
4. Verify expected per-position loss contributions and gradient to trainable LoRA and
   proprio projector. Reproduce on PEFT-wrapped training model.
5. Only after these pass: train from the same *unmodified base model* on a separate
   directory, fixed 32-sample overfit <=300 steps, and report action-token accuracy,
   autoregressive decoded L1, expert-vs-predicted action delta, and identity baseline.
6. Do not merge, overwrite checkpoints, claim rollout success, or start 6000-step training.

## Not done in this branch

No remote GPU executions, no simulator rollouts, no fresh model checkpoint, no benchmarking.
This is a source-level minimal patch with local testable contracts, pending full GPU validation.
