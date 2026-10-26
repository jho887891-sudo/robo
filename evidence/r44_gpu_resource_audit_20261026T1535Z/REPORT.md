# R44-GPU-RESOURCE-AUDIT — A6000 half-resource training feasibility

**STATUS: PASS** for the GPU inventory and the isolation-capability questions.
**TRAINING-VRAM ITEM: UNKNOWN** — `training/r44_train.py` does not exist at `fd0a7b87`, so the R44
memory requirement was never measured. No CUDA context was created, no training was started, no
process was stopped, no GPU/driver/MPS configuration was changed.

| field | value |
|---|---|
| UTC | 2026-10-26T07:35:39Z (sampling 07:36:10Z → 07:37:08Z) |
| host | `jxxy` (Linux 5.15.0-191-generic) |
| GPU | NVIDIA RTX A6000, UUID `GPU-73599ded-b934-3bb7-53fe-4fb3121db213`, 49,140 MiB |
| driver / CUDA | 580.178.04 / 13.0 |
| compute mode | **Default** (concurrent contexts allowed → driver time-slicing) |
| MIG | **not supported** (`nvidia-smi mig -lgip` → "No MIG-supported devices found", exit 6) |
| MPS | **not enabled** (no daemon, no `/tmp/nvidia-mps`, no systemd unit, no `CUDA_MPS_*` env) |
| evidence dir | `evidence/r44_gpu_resource_audit_20261026T1535Z/` |

## 1. Memory at the time of the audit

```
memory.total 49140 MiB | memory.used 2803 MiB | memory.free 45737 MiB   (5.7% used)
power 89.18 W / 300 W | temperature 52 C | SM clock 1800 / 2100 MHz | ECC disabled
```

## 2. 60-second observation (29 samples, 2 s interval, read-only)

| metric | avg | min | max | note |
|---|---|---|---|---|
| GPU utilisation | 14.86% | 14% | 16% | flat, one idle-ish workload |
| memory used | 2803 MiB | 2803 | 2803 | **constant**, no allocation growth |
| memory free | 45737 MiB | — | — | constant |
| power | 88.87 W | 88.83 | 89.01 | far below the 300 W cap |
| compute processes | 2 | 2 | 2 | stable |

No process was signalled, no test kernel was launched. The series is in `raw/09_timeseries.csv`.

## 3. GPU processes and ownership (verified, not guessed)

| PID | owner | how verified | what it actually is | GPU mem | uptime |
|---|---|---|---|---|---|
| 3111650 | uid **1234** (container user) | `/proc/3111650/status` + PPid chain | `/isaac-sim/kit/kit` inside docker container `rd_setup` (image `nvcr.io/nvidia/isaac-sim:5.1.0`), launched by `/isaac-sim/runheadless.sh sleep infinity`; chain 3111650 → 3111578 (`containerd-shim-runc-v2`) → 3111549 (`systemd`) | 2314 MiB | 8 d 03:38 |
| 1608287 | **dgut** (uid 1000) | `/proc/…/status` + cwd | `python3 /home/dgut/yuwang/open_world_with_ptogressive_simulator/C4_proposed_ablation_network_loss_5seeds_resume.py`, in tmux session `OW` (PPid chain → `tmux new -s OW` → systemd) | 408 MiB | 17:23 |
| 2461 | root (Xorg) | from nvidia-smi | display server, graphics context | 20 MiB | — |

Both compute processes were verified through `/proc/<pid>/cmdline`, `cwd` and the parent chain, not
by name. The IsaacSim instance is **containerised** and survives independently of this session; the
other GPU user is a live personal experiment of user `dgut` in tmux `OW` that must not be touched.

## 4. Isolation capability — three mechanisms kept strictly apart

**4a. Capping training memory (per-process).** No hardware mechanism exists on this GPU. The only
available lever is the self-imposed `torch.cuda.set_per_process_memory_fraction()` inside the
training process; it stops *that* process from allocating more and does nothing to, or for, any other
process. It is a convention, not an isolation boundary. Untested here (starting the trainer is
forbidden in this audit).

**4b. Capping compute share (SM time).** **Not possible.** MIG is unavailable on the A6000
(`mig -lgip` exit 6, MIG mode `N/A`), and MPS is neither running nor permitted to be started. The
device is in `Default` compute mode, which means the driver **time-slices between context switches** —
that is best-effort concurrency, not a partition: there is no way to say "training gets 50% of the
SMs". Even with MPS (a shared daemon), clients still share SMs; MPS changes scheduling granularity,
it does not partition the device.

**4c. Capping training time.** Possible only as a policy, not as a mechanism: there is **no scheduler**
on this host — `slurm/sinfo/scontrol/sbatch` absent, no Kubernetes, no DCGM, `crontab` holds only
comments. Docker is present (one container, the IsaacSim one) but nothing schedules GPU work.

**4d. Privileges.** `id` = uid 1000 `dgut`, groups include `sudo`, `docker`, `nvidia`; but
`sudo -n true` fails ("a password is required"), so no passwordless root. Nothing was reconfigured.

## 5. Is a strict 50% compute split achievable?

**No.** With MIG unavailable and MPS not permitted, there is no mechanism on this box that can hand
one process a deterministic half of the GPU. What is achievable is one of:

1. **Whole-device time windows** (recommended): the training process owns the whole card during an
   agreed window; the other workloads keep their 2.7 GB resident and simply wait. No isolation
   mechanism needed, no OOM risk from sharing, fully reversible.
2. **Voluntary concurrency**: training runs with a self-imposed memory fraction (~0.45 ≈ 22 GB) while
   the other processes keep running. Compute is shared by the driver's time-slicing; neither side can
   guarantee latency. Risk: an OOM is possible from either side, and both workloads slow down
   unpredictably.
3. **Memory-headroom first**: make the trainer cheaper (gradient checkpointing) and *then* run
   concurrently. This changes training speed and has not been measured for R44.

## 6. Does 24 GB of VRAM suffice for R44 training?

**UNKNOWN for R44, with a measured sibling as the only evidence.**

* `training/r44_train.py` is **absent** at `fd0a7b87813dd55bb330a0f722a8ba6f69cbf5e3` (`test -e` →
  ABSENT; `git ls-tree` shows only historical train logs). The entry point that would define batch
  size, sequence length, gradient-checkpointing setting and optimizer grouping does not exist, so its
  memory footprint cannot be inspected, let alone measured.
* The closest measured evidence from **this exact stack** is the last real training run:
  `m2_two_2000_20261021T123314Z/stage_evidence.json` records `"peak_mem_gb": 23.043` — a bf16 LoRA
  (r=32, all-linear) 7B training step at batch 1 / 14 action tokens already peaked at **23.0 GB**.
* R44's LoRA is *narrower* (`q_proj`, `v_proj` only), so its optimizer state is smaller, but the
  activations that dominate the 23 GB are essentially unchanged (same 7B, same 3-camera vision
  encoder, same 14-token target).
* Therefore a 24 GB budget leaves ≈ **1 GB (4%) headroom** against the sibling measurement: plausible
  but **not verified**; one longer sequence, one logging spike or one extra copy would OOM.
* Note the existing trainer (`training/r11_train_pushT_r42.py`) does **not** enable gradient
  checkpointing — only `overfit_pr1.py` and `r6_history_pilot.py` do. Enabling it for R44 would cut
  activation memory substantially but costs roughly 30% wall clock and has not been measured here.

Conclusion: **24 GB is not proven sufficient**; the honest statement is "≈23 GB measured for a
sibling configuration, 24 GB target, ~1 GB unverified headroom".

## 7. Recommendation

**Recommended: whole-GPU time windows (option 1).** Current load is 2.8 GB and 15% utilisation, so
the card is idle enough that a negotiated window costs the other users little, and it needs no
mechanism this GPU cannot provide. Risks: the window must be honoured by convention (no scheduler),
and the 8-day-old IsaacSim container still holds 2.3 GB inside the window.

If concurrency is mandatory: cap the trainer at ~22 GB, expect both jobs to slow down, and treat any
OOM as a normal outcome rather than a bug — while remembering that this is **not** a 50% compute
split, only time-slicing.

## 8. Raw evidence

`raw/00_host.txt` … `raw/14_historical_memory.txt` (every command, stdout, stderr and exit code),
`raw/09_timeseries.csv` (29 samples), `SHA256SUMS`, `exit_code.txt`.

Not established by this audit: R44 training memory, R44 step time under checkpointing, and the
behaviour of the card under a concurrent 7B job (deliberately not measured — that would have
required starting a CUDA context).
