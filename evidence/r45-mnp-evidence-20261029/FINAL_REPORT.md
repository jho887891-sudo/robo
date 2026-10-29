# R45-MNP latest evidence status

Snapshot UTC: 2026-10-29T13:06:30Z
Reviewed code SHA: d5be4de94ea37f1d5a715d5d47b7ebb16e07b097
Evidence branch: evidence/r45-mnp-cpu-20261029

> Snapshot note: the earlier FINAL_REPORT.md and stage_status.json published at commit d37da9023319a887a8f0cba4a5ccb1fa242dcbcc were intermediate snapshots created before the unified-environment rerun. They recorded P01 as blocked/exit 130. Git history preserves those interim records. This report and the current stage_status.json supersede them.

## Current status

- M01 VERIFIED PASS: all 300 HDF5 files rehashed; TRAIN=108780, DEV=28268; frozen and recomputed joint_delta_v1 payload SHA256 match.
- S00 VERIFIED PASS: one Python 3.10.12 interpreter used NumPy1.26.4, Torch2.5.1+cu124, Transformers4.46.3, PEFT0.13.2 and h5py3.16.0. Torch/Transformers/PEFT were inherited from system site packages; NumPy1.26.4 and missing Prismatic import dependencies were installed in the dedicated venv.
- Standard OpenVLA Prismatic API VERIFIED CPU-only: imports and class registrations passed; local AutoProcessor construction and dynamic OpenVLAForActionPrediction class import passed; active modeling source SHA matched the frozen repository reference. ARX dimensions were 14D action / 14D proprio, chunk 25. No weight tensors were loaded and no CUDA context was initialized.
- OpenCV/NumPy VERIFIED on actual data: with NumPy1.26.4 and OpenCV4.13, all three real TRAIN JPEG streams decoded as contiguous RGB uint8 480x640x3 arrays. OpenCV package metadata declares NumPy>=2; record this metadata mismatch despite successful actual decode.
- P01 VERIFIED PASS_CPU_GATE_ONLY: the existing Gate rehashed all 300 HDF5 files; report status CPU_CONTRACTS_PASS. Package, real-data preflight, runtime tests 9/9, toy resume tests 3/3, Delta tests 15/15, and contract tests 4/4 all exited 0. Gate exit code 0. This is not GPU or training verification.
- N01 REMOVED_BY_ARCHITECTURE_DECISION: no historical OFT provenance/model-selection gate.
- N02 PARTIAL: config/index and three weight shard file identities were checked; upstream model provenance remains UNKNOWN.
- GPU Smoke: plan documented in GPU_SMOKE_PLAN.md; not run and not authorized. The current r45_gpu_bench.py intentionally returns BLOCKED_NOT_IMPLEMENTED before CUDA access.

## Limits

No 7B weights loaded in this CPU verification; no CUDA context, GPU Smoke, stable benchmark, optimizer update, formal training, deployment, or data/checkpoint modification occurred. CPU Gate does not establish GPU memory fit or training throughput.

The public Manifest redacts 300 server-local absolute paths. redaction_manifest.json now pairs the source and published redacted file SHA256 values, records preservation of all 300 embedded HDF5 SHA values, and preserves the original source SHA256SUMS. The published redacted Manifest has a separately recalculated SHA256SUMS which verifies successfully.

Detailed logs and hashes are under cpu-gate-rerun-20261029/ and manifest/manifest-reattest/.
