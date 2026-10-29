# R45-MNP CPU Gate rerun

- Reviewed source SHA: d5be4de94ea37f1d5a715d5d47b7ebb16e07b097
- Gate: PASS_CPU_GATE_ONLY; exit code 0.
- Formal preflight: CPU_CONTRACTS_PASS; the existing Gate path rehashed all 300 frozen HDF5 files.
- TRAIN 108780; DEV 28268; five max epochs / 543900 TRAIN sample visits. No training started.
- Unified interpreter: Python 3.10.12, NumPy 1.26.4, Torch 2.5.1+cu124, Transformers 4.46.3, PEFT 0.13.2, h5py 3.16.0.
- Each Gate step exit code and raw stdout/stderr are retained under gate/.
- Runtime compatibility: standard OpenVLA config model_type=openvla; AutoProcessor construction and dynamic OpenVLAForActionPrediction class import passed without loading model weights. The active dynamic modeling SHA matches the frozen repository reference. ARX dimensions were 14/14/25. CUDA was not initialized.
- R44_OFT_ROOT was supplied only as the existing CLI's Prismatic runtime source path. No OFT model/checkpoint or historical OFT provenance gate was used.
- The venv reused system Torch/Transformers/PEFT and overlaid NumPy 1.26.4; draccus, timm and jsonlines were added for Prismatic imports. OpenCV 4.13 imported under NumPy 1.26.4, though package metadata requires NumPy >=2; this remains a metadata limitation.
- No model weights loaded, CUDA/GPU Smoke/benchmark, optimizer update, or formal training.
- Local server paths are redacted. SHA256SUMS covers this evidence package.
