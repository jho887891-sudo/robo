# R3 overfit experiment-integrity patch

Base commit: `d945331128d936f867a105d9056963a4059bf1f7`.
This is a **separate review branch**; no trained weights, runtime environments, or existing evidence files are changed.

## Evidence from the original committed experiment

`training/overfit_pr1.py` previously selected examples using `sorted(split["val"])`. The
committed keep experiment config confirms 32 samples came from **four validation episodes**:
`episode_0000001.hdf5` (13), `episode_0000002.hdf5` (10),
`episode_0000007.hdf5` (6), `episode_0000008.hdf5` (3).

These 32 samples were deliberately *fitted*, so their reported result is an **in-sample
memorization test**, not independent validation. The keep-vs-zero controlled comparison
remains meaningful for the question of whether the action-conditioning mechanism learns.
The original numerical results are preserved in `evidence/r3_pr1/`; do not relabel
those historical metrics as untouched held-out performance.

The prior JSON key `gen_tokens_per_sample` also stored
`gt_flat.reshape(len(P),14).tolist()`, i.e. **ground truth labels**, not generated
tokens. Thus any downstream per-token prediction diagnosis from that historical key
would be incorrect. Numeric `gen_L1` was computed from decoded generated actions and
is not affected by this reporting bug.

## Modifications

- `training/overfit_protocol.py`: deterministic train-only selection; guard against
  overlapping train/val episodes, repeated training episodes, insufficient data, and
  invalid frame ranges; validate the actual 14 generated indices.
- `training/overfit_pr1.py`: use `split["train"]` via the shared helper, tag
  `sample_source=train` in config/logs, and retain generated indices from `gen()`.
  `gen_tokens_per_sample` now records **model outputs** and
  `gt_tokens_per_sample` separately records reference labels. The per-step eval JSONL
  includes both, making analysis auditable.
- `tests/test_overfit_protocol.py`: 9 CPU test cases for selection, isolation, and logs.

## Verification status

- **Verified**: the submitted source files were fetched again from GitHub; all intended
  sampling/logging substitutions are present.
- **CPU verification**: a local standalone copy of the submitted dependency-free helper
  and 9-test file passed `python -m unittest discover -s tests -p test_overfit_protocol.py -v`
  (9 tests, 0 failures) and `py_compile`. The original GitHub checkout was not directly
  downloaded into the local runtime, so this is *not* a test run inside the remote GPU machine.
- **Not executed**: real HDF5 loading, 300-step training with new train-only sampling,
  any GPU test, any RoboDojo rollout, and any branch merge.

## Remote reproduction

From a checked-out copy of this branch:

```bash
python tests/test_overfit_protocol.py

# GPU/HDF5 experiment only when resources and methodology are approved:
# export PUSHT_MODEL=/home/dgut/pusht_runtime/model_cache/openvla-7b
# export PUSHT_HDF5=/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data
# python training/overfit_pr1.py --mechanism keep --steps 300 --out <new-output-directory>
```

Re-run comparisons only with the same newly selected *training* examples across keep/zero,
and assess generalization on disjoint validation episodes. Historical `r3_pr1`
results must remain unchanged for auditability.

**Do not merge to main or interpret fixed logging as recovered closed-loop skill.**
