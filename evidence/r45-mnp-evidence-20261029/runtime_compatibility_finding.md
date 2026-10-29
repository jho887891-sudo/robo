# Standard OpenVLA runtime compatibility

## Current verification

At code SHA d5be4de94ea37f1d5a715d5d47b7ebb16e07b097, the unified CPU interpreter imported the Prismatic symbols used by the trainer, registered the standard OpenVLA config/image-processor/processor classes, loaded the local config as model_type=openvla, constructed AutoProcessor from the model directory, and dynamically imported OpenVLAForActionPrediction.

The active dynamic modeling_prismatic.py SHA256 exactly matched the repository frozen reference. ARX runtime constants were ACTION_DIM=14 and PROPRIO_DIM=14. torch.cuda.is_initialized() was false. The 7B weight tensors were not loaded.

The existing CLI variable R44_OFT_ROOT was supplied only to expose the Prismatic runtime package required by the current code. This is a runtime API compatibility check; no OFT model/checkpoint or historical OFT provenance gate was used.

The complete CPU Gate subsequently passed, including the mandatory rehash of all 300 HDF5 files. Logs are in cpu-gate-rerun-20261029/.

## Limits

The earlier compatibility note was an interim finding before this rerun and is superseded here. No GPU weight load, forward/backward pass, optimizer step, CUDA compatibility, memory fit, or GPU throughput has been verified.
