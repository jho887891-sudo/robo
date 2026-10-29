# R46 source and gap audit

Reviewed source base: `ba563d0f8c638b089f588027dd0af41ff290e1f4` (PR #51 head). Final CPU-tested implementation: `75b6a1f84d4af513472f20df5548117a5e908066`. See `r46_audit.json` for function-level findings and frozen interface.

## Existing paths

- `diagnostics/r44_gpu_smoke.py` checks its legacy authorization flags before runtime imports, but implements one disposable optimizer step only. It lacks a parent-owned hard process-group timeout and the frozen five-sample 1+4 protocol.
- `training/r44_train.py` is the formal full-data trainer. It validates the three-task split, both frozen normalizers and HDF5 manifest, constructs the standard OpenVLA model, keeps the base frozen, and configures LoRA `r=32, alpha=16, q_proj/v_proj` with the 42D projector and `joint_delta_v1` targets. R46 leaves this trainer unchanged.
- `training/r45_data.py::RuntimeReader.load_images` is the production camera path. The three JPEGs decode as RGB 480x640x3 in frozen order. The actual processor output on the server is `[1,18,224,224]`; the 18-channel dimension comes from the processor's transform and is not a camera count.
- `diagnostics/r45_gpu_bench.py` explicitly returns `BLOCKED_NOT_IMPLEMENTED` before any GPU work. No historical placeholder timing is accepted as a benchmark.

## R46 CPU implementation

The new supervisor follows the exact supplied approval template and refuses the untouched `approved=false` template. It binds approval to source SHA, GPU UUID, UTC window, maximum wall duration, the full static OpenVLA identity file list (including config, processor/tokenizer metadata, index, and three weight shards), original manifest SHA and frozen sample-plan file SHA. Both parent and child hash the actual model identity files and verify the shard index before GPU execution; GPU admission uses `nvidia-smi` read-only queries and refuses unknown resources or compute processes.

The sample plan contains five real, unique `push_T` TRAIN samples with one warm-up and four measured roles. CPU verification confirmed HDF5 SHA against the original strict manifest, RGB data, actual CPU processor output, 42D features and encoded targets. The exact sample file SHA is recorded in `stage_status.json` and the file itself.

## Limits

No base-model weight was loaded and no CUDA context or GPU benchmark was started. R46 runner correctness on real GPU, 7B model load, BF16 forward/backward, optimizer update, VRAM peak and throughput remain `UNKNOWN_GPU_NOT_RUN`. R47 is `STOP_WAIT_GPU_APPROVAL`; R48 remains `NOT_RUN` until real R47 evidence exists.
