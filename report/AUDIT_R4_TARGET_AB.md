# R4: unify pure-OpenVLA training/deployment and prepare absolute vs residual A/B

## 0. Status and boundaries

- Branch `audit/r4-unified-action-target-ab`, based on
  `audit/r3-experiment-integrity-fix@40d143a99bd6ba3696db224483b24f811f475e39`.
- **GitHub code committed**; no pretrained weights, checkpoint files, simulator results,
  real GPU trainings, or online policy services changed.
- New `.github/workflows/r4-cpu-contracts.yml` compiles new Python source and runs
  dependency-light CPU contracts. Workflow completion/status must be read from GitHub,
  not assumed from the existence of the YAML.
- R3 evidence (32-example keep 0.02052 vs zero 0.08808) is not R4 A/B performance.
  **No empirical claim that residual targets outperform absolute targets.**

## 1. One canonical causal self-conditioning path

This branch retains only `pure_autoreg_mode.enable_pure_autoreg(core)`, applied
**before PEFT** for training and **before PEFT and after merge** for deployment.
Do not independently patch `modeling_prismatic.py` or pass an extra
`mask_action_inputs` kwarg to the pristine module.

The other R3 branch modified the HF modeling source via a masking gate. To avoid
silently mixing forks, the R4 training `build()` checks the **actual imported dynamic
model class's Python source file hash** against
`code/openvla_dynamic_modules/modeling_prismatic.py`. The default base model path
is the previously GPU-verified pristine clone:
`/home/dgut/pusht_runtime/model_cache/openvla-7b-pristine`.
The hash must match, otherwise training aborts. The deployed model checks the
`modeling_sha256` written to checkpoint metadata.

`diagnostics/check_r4_sources.py` compares the checkout's five policy source files
against the deployed RoboDojo policy tree and the base-model dynamic module.
**It is read-only.** The previously reported live policy tree is from a DIFFERENT
R3 implementation and is not automatically overwritten here. Its preflight
will initially FAIL until source synchronization is explicitly handled by an
authorized remote operator.

## 2. Target contract (single source for training and deployment)

Source: `code/xpolicylab/policy/OpenVLA_Pure/action_target.py`.

| Mode | Training target | Physical absolute command sent to RoboDojo |
| --- | --- | --- |
| `absolute` | `A[t]` | `decoded_target` |
| `residual` | `A[t] - S[t]` | `S[t] + decoded_target` |

The robot's action interface remains an **absolute 14D joint-position command**.
The `q01/q99` codec statistics are computed on the selected representation,
**only from the 80 training episodes**. The 256-bin action codec stays `MODE_FIXED`
with IDs 31744–31999. Both models are evaluated on reconstructed **absolute**
commands against `A[t]` in physical units, never in incomparable target scales.

Safety contracts:

- New residual deployment requires an explicit `action_target_mode: residual` in
  both `deploy.yml` and the checkpoint's `run_meta.json`.
- Residual deployment refuses old/legacy codec mode, missing adapter/projector
  SHA256 identities, and missing or mismatched runtime modeling SHA256 identity.
- New absolute and residual training runs must start from the same pristine base,
  not a legacy 6000-step adapter, and must write to fresh empty output directories.
- Unchanged deployed `deploy.yml` is **still configured for the old absolute
  checkpoint**. Do not simply switch it to residual while keeping the old adapter
  or stats file.
- This source code does not establish physical robot safe joint limits; do not
  deploy a residual checkpoint to hardware without separate limit and closed-loop
  validation.

## 3. The fair 32-example experiment

`training/overfit_pr1.py` supports `--target-mode absolute|residual`,
`--mechanism keep`, `--n 32`, `--val-n 20`, `--steps 300`.

Both arms use: pristine OpenVLA-7B + LoRA r32 / alpha16 / ProprioProjector,
same seed=0, 3 cameras, identical 32 TRAIN samples, disjoint 20 VAL examples,
same optimizer and 300 updates, identical preprocessors and mask patch.
Only the predicted target representation and its train-only quantiles differ.
Separate output directories and snapshots ensure reproducibility.

**Each evaluation** generates 14 action tokens using the real KV-cache
autoregressive path and its own previous predicted tokens, reconstructs a 14D
absolute action and reports physical `gen_L1`, `identity_L1`,
`repeat_previous_L1`, `pred_delta`, `gt_delta`, per-dimension errors,
free-generation tokens, and teacher-forced token accuracy. Repeating the previous
expert action at frame 0 falls back to state, and other frames use `A[t-1]`.
Neither teacher-forced accuracy nor an overfitting train score is a task success rate.

`training/compare_target_runs.py` fails if the two run manifests differ in
sample selection, validation selection, mechanism, step count, LR, available
source hashes, train-only source, or baseline. It compares the **held-out** values
and whether either beats identity and repeat-previous baselines.

## 4. CPU validation

From a clone of this branch:

```bash
python tests/test_action_codec.py
python tests/test_overfit_protocol.py
python tests/test_action_target_contract.py
python tests/test_target_ab_compare.py
python -m py_compile training/train_pushT_pure2.py training/overfit_pr1.py \
  training/compare_target_runs.py diagnostics/check_r4_sources.py
```

GitHub Actions should run the CPU subset on branch pushes. Confirm actual
workflow result. No remote HDF5 or A6000 test was performed by the assistant.

## 5. Remote GPU experiment commands (NOT executed)

Within a fresh worktree checked out at this R4 branch, from repository root:

```bash
export PUSHT_RUNTIME=/home/dgut/pusht_runtime
export PUSHT_MODEL=/home/dgut/pusht_runtime/model_cache/openvla-7b-pristine
export PUSHT_HDF5=/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data
export PUSHT_OFT=/home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_OFT/openvla_oft
export PUSHT_SPLIT=/home/dgut/pusht_runtime/split80.json
export PUSHT_SAMPLES=/home/dgut/pusht_runtime/fixed_samples.json
export PUSHT_BOUNDS=/home/dgut/pusht_runtime/push_T_bounds.npz

python tests/test_action_target_contract.py
python tests/test_target_ab_compare.py

python training/overfit_pr1.py --mechanism keep --target-mode absolute \
  --n 32 --val-n 20 --steps 300 --out /home/dgut/pusht_runtime/r4_ab_absolute
python training/overfit_pr1.py --mechanism keep --target-mode residual \
  --n 32 --val-n 20 --steps 300 --out /home/dgut/pusht_runtime/r4_ab_residual

python training/compare_target_runs.py \
  --absolute /home/dgut/pusht_runtime/r4_ab_absolute \
  --residual /home/dgut/pusht_runtime/r4_ab_residual \
  --out /home/dgut/pusht_runtime/r4_target_comparison.json
```

**Do not run** these commands if those output directories are already occupied.
The script refuses nonempty directories and source-fork mismatches.

For a read-only deployment-source check (expected to FAIL on the unsynchronized
online tree):

```bash
python diagnostics/check_r4_sources.py \
  --policy-dir /home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_Pure \
  --base-model /home/dgut/pusht_runtime/model_cache/openvla-7b-pristine
```

## 6. Decision rule

Only if the independent validation result is from identical splits/metrics and
correct action reconstruction should it be compared:

- If residual wins on the *held-out* physical 14D L1 **and** makes adequate
  actual joint progress without excessive `mean|A_pred-S|`, consider larger
  training/rollouts.
- If residual wins only on the 32 memorized training samples but loses on
  disjoint validation, it is not evidence to adopt residual representation.
- If both lag `repeat_previous_L1`, neither earns a full 3000-/6000-step budget.
- Even a held-out improvement is only evidence about offline next-action
  prediction, **not** RoboDojo task success.

No 300-step A/B numbers or closed-loop improvements are currently available.


## 7. One-command remote preflight / opt-in run

A separate, fail-closed wrapper now exists at `scripts/run_r4_ab.sh`.

```bash
# In a clean worktree on audit/r4-unified-action-target-ab:
bash scripts/run_r4_ab.sh --check
# Only after examining the preflight and with explicit authorization:
bash scripts/run_r4_ab.sh --execute
```

The default `--check` **never launches training**. It validates branch, clean
tracked files, dataset/split paths, pristine modeling source SHA-256, 20 held-out
examples and training episode availability, plus the CPU contracts. Using
`--execute` additionally requires at least 20 GB free VRAM and creates a brand-new
timestamped output directory under `$PUSHT_RUNTIME/r4_ab_jobs/` by default.
It then runs each 300-step A/B experiment and the strict comparator. A failure
stops the script immediately while preserving whichever output/logs were written.

The wrapper does **not** copy or overwrite the live RoboDojo policy files,
merge a PR, restart services, or initiate a simulator rollout. It does not
bypass the separately required deployment-source reconciliation gate.
