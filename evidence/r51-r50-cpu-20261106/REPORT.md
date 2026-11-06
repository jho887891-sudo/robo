# R51 two-task R50 CPU scenario audit

Status: CPU data pipeline only. Model forward/backward, GPU transfer, GPU image processing, DEV learning curves and success rates remain UNKNOWN.

Code SHAs: sample selection and B/E CPU profile `b67ca4082b74da275830899a14a98a7abbad2178`; cache comparison `25378f014946484fb5057c0913aebb95220b49bb`. All runs used Python 3.10.12, NumPy 1.26.4, Torch 2.5.1+cu124, Transformers 4.46.3, PEFT 0.13.2 and disabled CUDA visibility.

The frozen two-task TRAIN index contains 72,922 samples. Each complete replacement Epoch draws 72,922 times. With seed 51, the full pool draw order had 46,051 unique samples (36.85% duplicate draws); the 500-item active pool had 500 unique samples (99.31% duplicate draws). The fixed 120-draw CPU benchmark window had 120 unique identities from the full pool, and 106 from the 500-item pool. These are visits, not new data.

## B/E on the 500-item pool window

All three commands exited 0 and verified exact OpenVLA processor tensor equality against the old path. P00 old reader mean 53.39 ms/draw, R45 Reader mean 39.70 ms/draw. P01 DataLoader second-pass steady CPU preparation throughput, three independent repeats: 1 worker 23.77, 2 workers 44.44, 4 workers 80.11 draws/s. P02 assembly mean: `torch.cat` 0.770 ms, preallocated BF16 0.437 ms. These are CPU preparation metrics; they do not imply 7B training speedup.

## G cache screen on the same fixed windows

Two workers, prefetch 2, three independent repeats; 128 MiB is the **total** cap, divided into 64 MiB per worker. The candidate is the existing R45 postprocessed tile cache; RGB uint8 and JPEG byte caches are not implemented or evaluated here.

| Pool | Cache | Epoch 1 steady draws/s | Epoch 2 steady draws/s | Hit rate each epoch | Peak process RSS |
|---|---:|---:|---:|---:|---:|
| Full | 0 | 43.30 | 45.84 | 0% | 2.00 GiB |
| Full | 128 MiB | 42.88 | 45.67 | 0% | 2.17 GiB |
| 500 | 0 | 43.17 | 45.86 | 0% | 2.00 GiB |
| 500 | 128 MiB | 44.88 | 45.88 | 4.2% | 2.16 GiB |

The 500-item pool did not produce a stable end-to-end CPU data-preparation gain from this cache. Keep the production cache disabled pending a broader G matrix, DEV evaluation and real GPU training throughput. The active pool is experimental only.

All sample selection and benchmark outputs use the same seed and frozen TRAIN split. No stack_bowls sample was selected. Logs, JSON records, exit codes and SHA256 manifests are stored beside this report. Original HDF5, model weights and checkpoints were not copied into GitHub.
