# Standard OpenVLA runtime compatibility finding

The requested training target remains standard OpenVLA-7B + LoRA + the existing 42D projector. No alternative model was loaded.

At fixed source SHA `d5be4de94ea37f1d5a715d5d47b7ebb16e07b097`, `training/r44_train.py` requires `--oft-root`, validates that directory, adds it and `vla-scripts` to `sys.path`, then imports `prismatic` modules in `build_model()` for standard OpenVLA config/processor registration. `scripts/r45_server_cpu_gate.sh` requires `R44_OFT_ROOT` and passes `--oft-root`. The model directory contains three indexed safetensors and its dynamic config, but the separate offline source checkout is not a viable `prismatic` package root. The repository has three frozen dynamic module files; the current loader still imports the broader `prismatic` runtime.

Thus N01 historical OFT provenance can be removed as a model-selection gate, but an OFT-compatible Prismatic runtime package remains a current code dependency. The fixed SHA formal CPU Gate was interrupted while rehashing HDF5 before this compatibility path could be exercised. No source change was made in this evidence run.
