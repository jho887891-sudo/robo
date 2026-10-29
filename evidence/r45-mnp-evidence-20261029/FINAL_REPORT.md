# R45-MNP CPU evidence report

Fixed code SHA `d5be4de94ea37f1d5a715d5d47b7ebb16e07b097`; independent clean worktree.

- **M01 VERIFIED PASS:** 300 HDF5 full hashes, TRAIN=108780, DEV=28268; frozen and recomputed `joint_delta_v1` payload SHA both `9f7c4f00fc2945825cc892493c21127b5e38d4f83a212bf91f7cf410776ba459`. Manifest checksum passed.
- **N02 VERIFIED static files:** OpenVLA directory config/index and all three shards hashed; index matched the exact shard set. Status remains `FILE_IDENTITY_VERIFIED_PROVENANCE_UNCHECKED`; upstream model provenance is UNKNOWN.
- **N01 removed per architecture clarification:** no historical OFT source selection gate. Runtime compatibility remains unresolved: current trainer and CPU Gate require an OFT root and import Prismatic modules.
- **P01 BLOCKED:** formal CPU Gate started with the system Python package set (NumPy2.2.6, Torch2.5.1+cu124, Transformers4.46.3, PEFT0.13.2), not the isolated NumPy1.26.4 environment. Its preflight began a second full HDF5 hashing pass and was interrupted; exit 130, stderr records KeyboardInterrupt in `sha256_file`. Only `packages=0` completed; later CPU checks did not run.
- No CUDA context, GPU Smoke, benchmark, training, weight loading or deployment was performed.

The result is not CPU-ready and does not authorize GPU work.
