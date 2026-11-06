# R51 two-task CPU cache matrix and offline cost prototype

Execution date: 2026-11-06 UTC. Branch: `experiment/r50-image-pipeline-20261105`. Cache code used commit `10b6f154272b4a9791a4b7d4b198906ac3071603`; the final commit adds the regression and report without modifying the measured reader. CUDA was hidden. The OpenVLA processor ran on CPU; no 7B weight loading, optimizer, DEV loss or training throughput was measured.

The fixed input was 120 replacement draws from a 500-sample two-task TRAIN pool, repeated unchanged for two passes in each of three independent runs. All ten cache configurations exited 0 and matched the serial uncached pixel SHA for every draw. `matrix/` contains each run's `environment.json`, `runs.jsonl`, `report.json`, stdout, stderr and exit code. Its nested `SHA256SUMS` covers raw files before this report was assembled.

| Cache / total cap | First pass steady draws/s | Repeated pass steady draws/s | Repeated pass hit rate | Peak parent + workers RSS MiB |
|---|---:|---:|---:|---:|
| none | 41.96 | 45.59 | 0% | 2072.5 |
| JPEG / 128 MiB | 40.78 | 47.46 | 100% | 2071.9 |
| JPEG / 256 MiB | 41.18 | 51.89 | 100% | 2076.4 |
| JPEG / 512 MiB | 40.89 | 49.61 | 100% | 2065.0 |
| RGB / 128 MiB | 42.12 | 42.66 | 4.2% | 2200.6 |
| RGB / 256 MiB | 37.27 | 43.55 | 9.2% | 2381.3 |
| RGB / 512 MiB | 40.58 | 69.37 | 100% | 2434.3 |
| processed / 128 MiB | 44.75 | 45.16 | 4.2% | 2214.4 |
| processed / 256 MiB | 44.21 | 46.36 | 7.5% | 2368.4 |
| processed / 512 MiB | 44.14 | 157.63 | 100% | 2469.2 |

The 100% repeated-pass hit rate is caused by replaying the same short window and is not a full-epoch estimate. First-pass throughput did not show a reliable improvement. The 512 MiB processed cache adds roughly 397 MiB of measured RSS over no cache. No cache policy is selected for production from these data.

`offline-prototype/` contains the report, 100 per-sample digests, stdout, stderr and exit code. The CPU-only BF16 tensor serialization was byte-identical on mmap readback; 1,806,336 bytes per sample projects to 167,039,115,264 bytes for 72,922 TRAIN and 19,552 DEV samples. The 180,633,600-byte derived image shard stays on the server and is deliberately absent from this public evidence branch. Its SHA256 is recorded in the report; it is not included in the public `SHA256SUMS`. This is a bounded cost prototype, not full offline preprocessing or GPU speedup evidence.

Classification: **VERIFIED** CPU reader parity, matrix outputs and prototype readback; **IMPLEMENTED** bounded cache modes; **BLOCKED** C/F GPU measurement pending a separate approved window; **UNKNOWN** 7B end-to-end throughput and DEV behavior.
