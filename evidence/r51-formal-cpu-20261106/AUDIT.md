# R51 formal two-task CPU Gate and worker path

Execution source SHA: `3fe7723d85295d968277dd8bb92d1ae399612c2c`. Branch: `experiment/r50-image-pipeline-20261105`. Standard OpenVLA-7B remains the model base; this run did **not** load its 7B weights or initialize CUDA.

## Verified on the Linux server

- `final-preflight/`: actual `training/r51_train.py --preflight-only --r51-workers 2 --r51-prefetch 2` exit **0**. It invoked the R44 strict full Manifest validator, rehashed all 300 HDF5 files, then derived the R51 two-task plan. Report: `CPU_CONTRACTS_PASS`, TRAIN **72,922**, DEV **19,552**, `cuda_context_created=false`. The full split and historical task-specific normalizers remained intact. The preflight report pins split, normalizer, Manifest, train and DEV sample hashes.
- `final-tests/`: original stderr/stdout/exit codes for R44 **49/49**, R45 **9/9**, R46 **36/36**, R50 **7/7**, R51 **14/14** CPU tests; total **115/115**, all command exits 0. The R44 default no-replacement and R46 authorization refusal tests remain passing.
- `worker-path/`: a bounded standalone CPU probe calls the exact `make_r51_image_loader` function used in the formal loop, starting at committed draw cursor **731**. Five real two-task TRAIN frames were delivered in order by two separate spawned workers and matched serial OpenVLA CPU Processor tensors by SHA256; exit **0**. The probe source is included. This does not execute 7B forward/backward.
- `initial-preflight/` preserves the earlier successful full data attestation at source SHA `9f1f996c3b1551ab2f1f4782f61bffd4cb758193`; it must not be substituted for the final-SHA run.

## Reproduction shape

The execution used the isolated CPU Python 3.10 / NumPy 1.26.4 environment, with `CUDA_VISIBLE_DEVICES=` and `PYTHONDONTWRITEBYTECODE=1`. Supply the server's *existing verified* roots through `M2_HDF5_ROOT`, `M2_NORMALIZER_JSON`, `R44_DELTA_NORMALIZER`, `R44_DATASET_MANIFEST`, `R44_BASE_MODEL`, and `R44_OFT_ROOT`; no path is guessed or published here.

```bash
python training/r51_train.py --preflight-only \
  --splits data/m2_splits.json --hdf5-root "$M2_HDF5_ROOT" \
  --feature-normalizer "$M2_NORMALIZER_JSON" \
  --delta-normalizer "$R44_DELTA_NORMALIZER" \
  --dataset-manifest "$R44_DATASET_MANIFEST" \
  --base-model "$R44_BASE_MODEL" --oft-root "$R44_OFT_ROOT" \
  --epochs 5 --r51-workers 2 --r51-prefetch 2 --out "$NEW_OUTPUT"
```

R51 defaults to replacement sampling. With effective Batch 1, each Epoch has 72,922 draws and 72,922 optimizer updates; five Epochs have 364,610 updates. A repeated draw is still only one existing TRAIN sample identity. The CPU report's **426.48 GPU-hour** budget is calculated from historical per-sample estimates; it is **not** a measured R51 wall-time forecast.

## Limits and next Gate

**VERIFIED:** strict data/normalizer CPU contract, fixed two-task split, CPU regression, ordered worker image path and SHA256 checks. **IMPLEMENTED:** opt-in B Worker prefetch/cache and deterministic R51 formal draw/checkpoint wiring. **BLOCKED:** separately approved, resource-available standard OpenVLA-7B GPU Smoke and later C/F end-to-end comparison. **UNKNOWN:** CUDA parity of C hybrid preprocessing, F DALI decode (dependency absent), 7B training throughput, peak VRAM, DEV loss/success and stop/resume under a live optimizer. The preflight does not enter the GPU admission path or attest model weight shards at this SHA. No formal training or deployment was started.
