# R46/R51 GPU Smoke input attestation — CPU only

Source checkout SHA `4f1cc95525236fdb63f93115258b704d0d77263d` (Draft PR #57). The separate evidence branch records this read-only CPU run without changing the PR head.

The run validated the existing R46 five-sample `push_T` TRAIN plan against the frozen three-task split and strict 300-file Manifest, rehashed the five selected HDF5 episodes, rehashed all files in the frozen standard OpenVLA-7B model identity including its **three safetensors shards**, and checked the model index's shard set. `r46_smoke_contract.validate_sample_plan(..., verify_files=True)` returned `CPU_SAMPLE_CONTRACT_PASS`, five distinct TRAIN episodes with one warm-up and four measured steps. Process exit **0**; `report.json` status `PASS_INPUT_ATTESTATION_CPU`.

Key SHA256 values:

- Strict original Manifest: `a220bd45e49c623c3a9f77dcf7460c95f8305909c05122e2d2a0556bff522ead`.
- R46 sample-plan file: `8bcf9fac0a420fbca09a577e2965222fd2c7911baa8fc59f7c8f3e5b20214ef1`.
- Sample-plan canonical payload: `67458ab3c6b23518c8616ec229ff62ed9af51d3d6bc817d54b21cdd109937d4f`.
- Model-identity file: `101b0fc57ee7012dd8d8fe5a1fdf862a6e3699037c88837264979110403749ae`.

`report.json` contains the complete 11-file model SHA list. `probe.py`, `stdout.log`, `stderr.log`, `exit_code`, `source_sha.txt`, and `SHA256SUMS` make this CPU attestation reproducible and auditable. The model and HDF5 bytes are not published.

**VERIFIED:** CPU input hashes, index consistency and fixed sample contract. **BLOCKED:** actual GPU Smoke until a human-filled R47 approval document has an approval ID/approver, exact UTC window, GPU UUID, memory thresholds and these fixed hashes, and `probe_gpu` sees an exclusive resource window. At the last read-only check the RTX A6000 had two other compute PIDs and about 91% utilization. **UNKNOWN:** model load, forward/backward, peak VRAM and GPU throughput. No CUDA context or training was started by this run.
