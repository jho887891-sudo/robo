# R51 C/F CPU-only candidate audit

Two-task fixed TRAIN input and the frozen standard OpenVLA-7B `PrismaticImageProcessor` were used. CUDA was hidden; neither the 7B model nor DALI was loaded.

## C: GPU image preprocessing boundary

The first candidate moved bicubic resize into `torch.nn.functional.interpolate`. On 10 real samples it matched the production PIL processor exactly on **0/10**, with BF16 maximum absolute error **0.078125**. This path is rejected for training. The initial raw report, stdout, stderr and exit code are preserved in `direct-cpu/`.

The revised candidate preserves PIL resize and crop on CPU, then uses batched tensor conversion, tower normalization and assembly on the selected device. The CPU execution matched the production OpenVLA processor **10/10** samples exactly, maximum absolute error **0**. This is only CPU numerical parity. CUDA kernel parity, transfer overhead, GPU memory and end-to-end training throughput remain UNKNOWN. It is an opt-in candidate module, not wired into the formal trainer. The second report and per-sample differences are in `hybrid-cpu/`.

## F: DALI / nvJPEG

The optional adapter uses a fixed-size three-camera batch, DALI `external_source`, and `fn.decoders.image(device="mixed", output_type=RGB)` as documented by [NVIDIA](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/external_input.html). It returns RGB to CPU solely for future parity diagnosis, including that transfer in any later timing. The current isolated CPU environment has **no `nvidia.dali` package**; a CPU test verifies `BLOCKED_DALI_DEPENDENCY` before Pipeline construction or CUDA initialization. Actual DALI decode, OpenCV pixel parity and GPU performance remain UNKNOWN. No dependency was installed.

Both candidates require a separately authorized bounded GPU window before evaluation. Neither is selected for production. The OpenVLA-7B base, LoRA configuration, state projector, action target and historical data remain unchanged.

The authoritative rerun is `final-cpu/`, executed on committed source SHA `65046e39d77cf032940ac805d2581e8308e1232f`: parity exit 0, 10/10 hybrid exact, 0/10 direct exact; seven R51 candidate/sampling/cache CPU tests passed, test command exit 0. `direct-cpu/` and `hybrid-cpu/` preserve the preliminary diagnostic outputs from before this commit. The root `SHA256SUMS` covers all published raw files.
