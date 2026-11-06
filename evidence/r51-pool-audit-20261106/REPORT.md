# R51 two-task sampling pool CPU audit

Fixed source SHA `c00277ea2c3a986ab23ec41fa35041dee44e177e`; seed 51, epoch 0, frozen two-task TRAIN 72,922, 1,000-draw comparison window. Every command exited 0. No GPU, model, DEV evaluation or training was run.

| Pool | Unique in 1,000 draws | Window duplicate ratio | Unique in full 72,922 draws | Full Epoch duplicate ratio |
|---|---:|---:|---:|---:|
| 500 | 421 | 57.9% | 500 | 99.31% |
| 1,000 | 625 | 37.5% | 1,000 | 98.63% |
| 2,000 | 772 | 22.8% | 2,000 | 97.26% |
| 5,000 | 903 | 9.7% | 5,000 | 93.14% |
| Full | 995 | 0.5% | 46,051 | 36.85% |
| Full, no replacement | 1,000 | 0% | 72,922 | 0% |

Proportional replacement sampled 353 push_T and 647 sweep_blocks draws in the 1,000-draw window. Balanced replacement produced 36,386 push_T and 36,536 sweep_blocks draws across the complete Epoch. Every variant makes 72,922 draws, so Batch Size 1 would retain 72,922 optimizer updates per Epoch. Draws are visits, not distinct training data. Small Active Pools remain experimental until DEV loss and per-task success are measured.

Each subdirectory retains the generated `selected_samples.json`, `report.json`, raw stdout/stderr and SHA256SUMS. These are CPU-only sample plans, not throughput tests.
