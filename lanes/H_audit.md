# Lane H — independent audit of the "current R15 evidence chain"

Author: Lane H (independent auditor, delegated subagent). Date 2026-10-21, 07:48–08:12 UTC.
Host: remote GPU box 172.31.68.251 (user `dgut`), repo worktree `/home/T7/ojh/robot_actual/r11`
(branch `feature/r15-true-continuation`, HEAD `25989628c56ef776c9bc8e086e62a97d624dc36f`).
All work read-only on the remote repo; every artefact I wrote went to `/home/dgut/r15_lanes/H/`.
Nothing here is taken on trust: every number below was recomputed by my own code.

## Verdict summary

| # | Item | Verdict |
|---|------|---------|
| 1 | Append-only WAL: hash chain, seq monotonicity, totals, event count, lifecycle, projection cross-check | **PASS** (chain/totals/projection reproduce exactly) — with two accounting defects, see 1e/1f |
| 2 | `runs/r15_stage_300_20261021T030423Z/checkpoints/step300` MANIFEST + `trainer_state.json` | **PASS** (9/9 file hashes reproduce; global_step 300) |
| 3 | "invalid segment" at `runs/r15_stage_1000_*/checkpoints/step963` | **FALSIFIED AS STATED** — no `step963` checkpoint exists anywhere (last = step950; run verdict PARTIAL). **The underlying "resume consumed one sample" claim is TRUE** and I proved it three independent ways. |
| 4 | Pre-registered gate numbers (frozen 32-frame hash; CV macro L1 = 0.002110443303536158) | **PASS on both numbers exactly** — but the CV number is the *RAW/v1* variant, the current gate口径 is *SAFE* (0.002047480782493949, −3.0%), and the 7B model loses to both. Pre-registration file was **edited during this audit**. |
| 5 | Unbilled DEV 7B inference / double-booked GPU time | **CONFIRMED, both.** 323 s of DEV 7B inference between 03:17:03Z and 06:15:59Z is billed nowhere; GPU time and formal steps are double-booked (gstep 301 booked 3×; 50 gsteps booked >1×; 159.4 s re-spent on already-computed steps as of my last snapshot). |

Two blunt headline contradictions:

* **The step-963 checkpoint the chain is supposed to resume from does not exist.** The stage-2 run
  stopped at 963 because of its own GPU-estimate guard and never wrote a checkpoint after step950.
  The last 13 formal steps are unrecoverable.
* **The VLA@300 model is worse than a constant-velocity baseline on its own DEV gate, under both
  metric definitions** (−1.13 % RAW, −0.99 % SAFE). The `R15_GATE_PREREGISTRATION.json` field
  `cv_macro_l1 = 0.002110443303536158` is exactly reproducible, but it is the RAW number, and the
  model is graded on SAFE (CV = 0.0020475) — quoting one and grading the other inverts the sign of
  the comparison (a mixed comparison reads "+2.02 % better", the correct comparison is −0.99 %).

## Snapshot identity (the evidence chain is live and moving)

The WAL is being written by a trainer while I audit, so I took three snapshots and verified each.

| snap | utc | bytes | events | events.jsonl sha256 | projection `n_events`/`wal_sha256` |
|------|-----|-------|--------|---------------------|------------------------------------------|
| A | 07:49:30Z | 494941 | 986 | `1b7357cdf062f5c7b5ad8b56d06f86dbfd07b080243a6a711670a6606ccaca8e` | 986, same sha — exact match |
| B | 08:05:35Z | 522698 | 1040 | `0455def470e8a3cc7d709578f04a68ec7d3099c64965318269742c8b982c6170` | 1042 (proj newer than the pull — race, not a defect) |
| C | 08:09Z | 525700 | 1046 | `8e4023a228d1537756154148d2db966780c4a7823c8d51791609954d4812d40c` | 1046, **same sha — exact match** |

Snapshot C verification is race-free by construction: because the WAL is append-only,
`projection.wal_sha256` must equal sha256 of the byte-prefix containing `projection.n_events`
lines. It does.

---

## Item 1 — append-only WAL

### Method (independent reimplementation)

I re-implemented the documented canonicalisation from scratch rather than importing
`training/r15_ledger.py`: for each line, `body = event minus "hash"`,
`hash == sha256(json.dumps(body, sort_keys=True))`, plus `prev_hash` and `seq == line index`.
Script: `lanes/H/h_wal_audit.py` (snapshot A), `lanes/H/h_wal2.py` (B), `lanes/H/h_projcheck.py` (C).
Ledger contract read from `training/r15_ledger.py` lines 197-225 (`read_events`) and 280-296 (`totals`).

```
python sftp.py get 'E:\具身智能\_r11\lanes\H\events.jsonl' '/home/T7/ojh/robot_actual/r15_ledger/events.jsonl'
python h_wal_audit.py
```

### Raw results — snapshot A (sha `1b7357cd…`, 986 events)

```
wal_bytes=494941 sha256_file=1b7357cdf062f5c7b5ad8b56d06f86dbfd07b080243a6a711670a6606ccaca8e
physical_lines=986 nonempty=986 trailing_newline=True
CHAIN_BREAKS=0 SEQ_VIOLATIONS=0 HASH_MISMATCHES=0 OTHER_SCHEME_MATCHES=0
RECOMPUTED gpu_seconds_used=3268.381 formal_steps_used=964 n_events=986
WAL last_seq=986 last_hash=399d106796bc7d18508c9a8d2a681400c56428d42e3f16a1e34b87c8b270674f
KINDS: {'step_commit': 964, 'job_begin': 9, 'job_end': 8, 'import_history_bucket': 2,
        'import_history': 1, 'abort': 1, 'job_blocked': 1}          # 964+9+8+2+1+1+1 = 986
```

* Hash chain: **0 breaks** over all 986 events; every `prev_hash` equals the previous event's
  `hash`; seq is 1..986 with no gaps or reordering; no event verifies under an alternative
  canonicalisation (so the scheme is unambiguous).
* Totals recomputed from events: `gpu_seconds_used = 1027.5 + 576.967 + 1605.273 + 58.641 = 3268.381`;
  `formal_steps_used = 300 + 663 + 1 = 964`. Both equal the WAL's own projection to the last bit.
* Event count 986.

### Every job_begin / job_end / abort (`seq, kind, run_id, utc, booked s, booked steps`)

Batch runs carry `gpu_seconds=0` on their lifecycle events — the seconds are booked on the 964
`step_commit` events instead — so I also give per-run booked totals.

| seq | kind | run_id | utc | gpu_s on event | steps on event |
|-----|------|--------|-----|----------------|----------------|
| 4 | job_begin | `r15-s300-20261021T030423Z` | 2026-10-21T03:04:25Z | 0.000 | 0 |
| 305 | job_end | `r15-s300-20261021T030423Z` | 2026-10-21T03:17:03Z | 0.000 | 0 (payload `formal_steps_booked_here=300`, `global_step=300`) |
| 306 | job_begin | `r15-s1000-20261021T061554Z` | 2026-10-21T06:15:59Z | 0.000 | 0 |
| 322 | job_begin | `r15-s1000-20261021T061711Z` | 2026-10-21T06:17:16Z | 0.000 | 0 |
| 364 | **abort** | `r15-s1000-20261021T061711Z` | 2026-10-21T06:19:10Z | **55.000** | 0 (`reason=cuda_oom_duplicate_launch`, `incident=true`) |
| 973 | job_end | `r15-s1000-20261021T061554Z` | 2026-10-21T06:51:02Z | 0.000 | 0 (payload `formal_steps_booked_here=663`, `global_step=963`) |
| 974 | job_begin | `sup:startup` | 2026-10-21T07:48:35Z | 0.000 | 0 |
| 975/976/977 | job_begin | `sup:s1`/`sup:s2`/`sup:s3` | 07:48:35–36 | 0.000 | 0 |
| 978/981/982 | job_end | `sup:s3`/`sup:s1`/`sup:s2` | 07:48:36–39 | 0.000 | 0 |
| 980 | job_begin | `sup:s6` | 07:48:38 | 0.000 | 0 |
| 983 | job_begin | `sup:s5` | 07:48:40 | 0.000 | 0 |
| 984 | job_end | `sup:s5` | 07:48:40 | 0.000 | 0 |
| 985 | job_end | `sup:drained` | 07:48:41 | 0.000 | 0 |
| 986 | job_end | `sup:s6` | 07:48:57 | 0.000 | 0 (TIMEOUT) |

All `sup:*` events carry `counts_toward_total=false` — they are supervisor smoke tests, correctly
non-counting. Per-run booked totals (all events, snapshot A):

```
history-import                     gpu=1027.500 steps=0    02:58:42Z
r15-s300-20261021T030423Z          gpu= 576.967 steps=300  03:04:25Z..03:17:03Z
r15-s1000-20261021T061554Z         gpu=1605.273 steps=663  06:15:59Z..06:51:02Z
r15-s1000-20261021T061711Z         gpu=  58.641 steps=1    06:17:16Z..06:19:10Z
sup:* (12 run_ids)                 gpu=   0.000 steps=0
```

### Projection vs WAL (`verify_projection` contract: last_seq, last_hash, sha256, totals, buckets)

Snapshot A — every field reproduced by my own recomputation:

```
gpu_seconds_used        proj=3268.381      wal=3268.381      OK
formal_steps_used       proj=964           wal=964           OK
gpu_seconds_remaining   proj=111931.619    wal=111931.619    OK
formal_steps_remaining  proj=2036          wal=2036          OK
n_events                proj=986           wal=986           OK
wal_last_seq            proj=986           wal=986           OK
wal_last_hash           proj=399d1067…     wal=399d1067…     OK
wal_sha256              proj=1b7357cd…     wal=1b7357cd…     OK
bucket R15_TRAIN_FORMAL used_s=2240.8809999999976 steps=964 cap=50400.0   OK
all other 7 buckets     used_s=0.0 steps=0  caps OK                     OK
PROJECTION_MATCHES_WAL=True
```

Snapshot C (race-free prefix check, 1046 events):

```
PREFIX(1046) sha256       = 8e4023a228d1537756154148d2db966780c4a7823c8d51791609954d4812d40c
projection.wal_sha256     = 8e4023a228d1537756154148d2db966780c4a7823c8d51791609954d4812d40c   MATCH=True
wal_last_seq 1046 vs prefix last seq 1046  MATCH=True
wal_last_hash MATCH=True
gpu_seconds_used  proj=3427.8240000000005 wal=3427.8240000000005 MATCH=True
formal_steps_used proj=1014 wal=1014 MATCH=True
bucket mismatches: NONE
FULL SNAPSHOT: events=1046 chain_violations=0
```

### 1e. Defect (reporting, not chain): `sum(buckets) != gpu_seconds_used`

```
INTERNAL: sum(bucket used_s)=2240.881000 vs gpu_seconds_used=3268.381000 delta=1027.500000
INTERNAL: sum(bucket used_s)=2400.324000 vs gpu_seconds_used=3427.824000 delta=1027.500000
```

The 1027.5 s carried over from history is booked on `seq 1` with `bucket=null`, and the two
`import_history_bucket` markers carry `gpu_seconds=0` + `counts_toward_total=false`
(`training/r15_ledger.py` lines 359-367, deliberate). Consequence: **any per-bucket report
understates R13/R14 usage by 169.7 + 857.8 s.** `bucket_used_reported()` exists to correct this but
the projection file does not use it.

### 1f. Defect (accounting): the duplicate-step guard is per-run only

`training/r15_ledger.py` lines 254-259 keys the duplicate check on
`(run_id, kind, payload["global_step"])`. Two *different* run_ids may therefore book the same
`global_step` without error. This is exactly what happened (Item 5). The WAL is internally
consistent, but its `formal_steps_used` is not the number of distinct optimizer steps.

**Item 1 verdict: PASS** for chain integrity, seq monotonicity, totals, event count and projection
consistency — recomputed independently three times. Items 1e/1f are accounting/reporting defects,
not tampering.

---

## Item 2 — `runs/r15_stage_300_20261021T030423Z/checkpoints/step300`

### Method

SHA256 of all 9 manifested files recomputed on the box, then cross-checked locally: the small files
were downloaded and hashed on Windows with an independent implementation
(`Get-FileHash`), so the manifest is compared against two independent computations.

```
cd .../checkpoints/step300
sha256sum lora_adapter/README.md lora_adapter/adapter_config.json \
          lora_adapter/adapter_model.safetensors optimizer_state.pt proprio_projector_42.pt \
          rng.pt sampler.json scheduler_state.pt trainer_state.json
python sftp.py get '<local>' '.../step300/sampler.json'      # then Get-FileHash locally
```

### Raw results — all 9 entries reproduce

| file | MANIFEST sha256 | recomputed | local re-hash |
|------|-----------------|-----------|---------------|
| lora_adapter/README.md | `bde1852e27f8a15c486c9cbc87f4f15fb3ff90ce0810946609e42de8310552cf` | same | – |
| lora_adapter/adapter_config.json | `bc842ecc420a81cd11ec7df2e394de2fea7f5d04f87aeef72cf133ebc19c7d23` | same | – |
| **lora_adapter/adapter_model.safetensors** | `f0316a91a7e50d117c9665736c8c171c2450aa06bb650cacb158ed04d9014779` | same | – |
| optimizer_state.pt (572,904,354 B) | `821da65f870fa4c25174f490fdfcabd2d2a92d8523840205dd96ade37c12aedd` | same | – |
| proprio_projector_42.pt (67,831,992 B) | `b71ec79f5dad5670cc70d761e1e3bde122aa9feaad55b430ce3a0f47ef2bf830` | same | – |
| rng.pt | `af0a9acc7b9590205f4021075ac72ea8c47b23e5d6afc4b22bda7f496668b356` | same | – |
| sampler.json | `c1fdcd211a819b136325eb96ccc7e3148661bc61b364ebd69a6721f481173605` | same | `c1fdcd21…` OK |
| scheduler_state.pt | `35807bf06ea1d97f3f4c4e7afea6f80900e7074b6c52366833548e56bb8524d2` | same | – |
| trainer_state.json | `b2c9fda33882314e9ee8b0e28db61ecf2d50e9c44f41a59788ffccebbda7fca5` | same | `b2c9fda3…` OK |

```
files_listed=9 recomputed=9 unmanifested=none missing=none  ALL_MATCH=True
MANIFEST: global_step=300 run_id=r15-s300-20261021T030423Z created_utc=2026-10-21T03:16:58Z schema=r15-full-state-v1
directory contains exactly 10 files = 9 manifested + MANIFEST.json (no unmanifested payload)
```

`trainer_state.json` (read in full): `global_step = 300`, `target = 300`,
`run_id = r15-s300-20261021T030423Z`, `created_utc = 2026-10-21T03:16:58Z`,
`checkpoint_schema = r15-full-state-v1`, `model_kind = peft`, torch 2.4.1+cu121 / py 3.10.12,
`base_model._dir = /home/dgut/pusht_runtime/model_cache/openvla-7b-pristine`,
`modeling_sha256 = b5431a074c…`, `split_sha256 = 2654e31912…` (= sha256 of `data/split80.json`,
which I verified independently).

**Item 2 verdict: PASS.** I also recomputed `step350` and `step950` of the stage-2 run: 9/9 each,
`EVERY_MANIFEST_ENTRY_REPRODUCED = True` (27/27 entries across three checkpoints).

---

## Item 3 — the "invalid segment" claim

### 3a. `checkpoints/step963` does not exist — the claim as written is FALSIFIED

```
find /home/T7/ojh/robot_actual -maxdepth 7 -name '*963*'   # only .git object blobs, no checkpoint
ls runs/r15_stage_1000_20261021T061554Z/checkpoints
  step350 step400 step450 step500 step550 step600 step650 step700 step750 step800
  step850 step900 step950                     <-- 13 dirs, save_every=50, NOTHING at 963
```

The run's own evidence says why (`runs/r15_stage_1000_20261021T061554Z/stage_evidence.json`):

```
global_step_reached 963    target 1000    start_global_step 300    steps 700
wall_clock_s 2102.5        verdict "PARTIAL"     utc_start 06:16:47Z   utc_end 06:51:02Z
```

and `logs/r15_stage_1000_20261021T061554Z.log`:

```
[06:50:21] saved full-state checkpoint step=950 sha=3a18fb422d0e971d
[06:50:52] step=960 loss=2.2705 ...
[06:51:02] stopping at global_step=963: bucket/time guard (2103s used of 2220s)
[06:51:03] R15 PARTIAL reached step 963 in 2102.5s (2.418s/step, peak 23.05 GB)
```

So the stage-2 segment terminates 37 steps short of its target, 13 steps past its last checkpoint.
**Formal steps 951-963 have no persisted state: they can never be resumed, only recomputed.** Any
chain that claims to resume "from step963" is claiming a resume from a file that does not exist.

### 3b. The "resume consumed one sample" claim is TRUE — three independent proofs

**(i) sampler position vs global_step** (arithmetic; `sampler.json` has only `position`/`perm`/`order_hash`):

| checkpoint | trainer_state `global_step` | sampler.json `position` | offset |
|---|---|---|---|
| `r15_stage_300.../step300` (end of run 1) | 300 | **300** | 0 |
| `r15_stage_1000_061554Z/step350` | 350 | **351** | **+1** |
| `r15_stage_1000_061554Z/step950` | 950 | **951** | **+1** |

The offset is zero at the end of the fresh run and +1 at the first checkpoint after the resume, and
it stays +1 for 650 steps. Exactly one frame was consumed at the seam without a gradient step.

**(ii) the resume log vs the first committed step.** `logs/r15_stage_1000_20261021T061554Z.log`:

```
[06:16:47] resumed from runs/r15_stage_300_20261021T030423Z/checkpoints/step300 at global_step=300
           (next frame ('episode_0000004.hdf5', 202))
```

but the WAL's first `step_commit` for that run (seq 307, 06:16:50Z, `global_step=301`) is:

```
{"episode": "episode_0000081.hdf5", "t": 7, "global_step": 301, "loss": 2.303494}
```

i.e. the frame the resume announced as next (`episode_0000004`, t=202) was **never trained on**; the
step went to the following frame instead.

**(iii) perm arithmetic ties (i) and (ii) together.** `step300/sampler.json`:
`perm[298:304] = [14066, 17389, 431, 15645, 16870, 8306]`, `position = 300`.
A correct continue must use `perm[300] = 431` for step 301. The 061554Z run used `perm[301] = 15645`.
`perm[299] = 17389` was step 300's frame — consistent with "300 steps consumed positions 0..299".

**(iv) independent confirmation from a *different* run.** The stage-2 restart launched during this
audit, `r15-s2-1000-20261021T0803Z`, resumed from the *same* step300 checkpoint and booked
(WAL seq 997, 08:03:13Z):

```
{"episode": "episode_0000004.hdf5", "t": 202, "global_step": 301, "loss": 2.254457}
```

That is exactly `perm[300]` — the frame 061554Z skipped. Two runs from one checkpoint, one frame
apart, on the same step: the offset is a property of the 061554Z resume, not of the data.

**(v) the fix is visible in the restart's own log**, which names the mechanism:

```
logs/r15_s2_1000_20261021T0803Z.log
[08:03:04] resumed from runs/r15_stage_300_20261021T030423Z/checkpoints/step300 at global_step=300
           position=300  # peek only, never sampler.next()
```

"peek only, never `sampler.next()`" is the patch for exactly the defect described above: the old
resume consumed one sample from the sampler, the new one only reads the position. Comparing the two
resume log lines is the cleanest single-piece proof of the off-by-one.
(`[06:16:47] resumed ... (next frame ('episode_0000004.hdf5', 202))` — the old code materialised the
next frame, and that frame was never trained on.)

### 3c. Bonus finding: `rng.pt` is a constant, so "true continuation" cannot be bit-exact

```
rng.pt sha256 identical at step300, step350 and step950: af0a9acc7b9590205f4021075ac72ea8c47b23e5d6afc4b22bda7f496668b356
torch.load(...) -> dict keys ['python','numpy','torch_cpu','torch_cuda']
  python    : identical tuple tuple(3, (...)) at all three checkpoints
  numpy     : identical MT19937 state at all three checkpoints
  torch_cpu : uint8[5056], sum=315221.0 at all three checkpoints
  torch_cuda: [tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.uint8)]   <-- all zeros
```

A generator state that does not move across 650 optimizer steps, and a CUDA RNG slot that is all
zeros, cannot restore the RNG at a step boundary. The only real continuity mechanism is
`sampler.json:position` + optimizer/model/scheduler state. (For the 061554Z run even that was wrong
by one.) Note this also means `rng.pt` being "present and manifested" proves nothing about
reproducibility.

**Item 3 verdict: FALSIFIED as stated (no `step963` checkpoint; the chain cannot resume from 963);
the sub-claim it was meant to support — "the resume consumed one sample" — is CONFIRMED beyond
doubt.**

---

## Item 4 — pre-registered gate numbers

### 4a. Frozen 32-frame DEV list hash — PASS

I recomputed the frame list from first principles (`lanes/H/H_frames_recompute.py`, run at
`/home/dgut/r15_lanes/H/`): DEV cohort = `set(split80.json["train"]) - set(frozen_contract_64train
episodes)`, then per episode `rs = np.random.RandomState(0); pool = arange(2, max(3, T-1));
np.sort(rs.choice(pool, 2, replace=False))`, hash = `sha256(json.dumps([[ep,t],...], sort_keys=True))`.

```
/home/dgut/pusht_runtime/venv_oft/bin/python /home/dgut/r15_lanes/H/H_frames_recompute.py
split80.train=80  contract_train=64  derived DEV=16
n_frames=32
RECOMPUTED frames_sha256 = 67b4320420249aef7cd560f33a66e2ae454960238c695e05a3910ebad3e50dff
PREREG     frames_sha256 = 67b4320420249aef7cd560f33a66e2ae454960238c695e05a3910ebad3e50dff
MATCH=True
```

It also equals `dev_frames_sha256` in **both** recorded evaluation artefacts
(`runs/r15_dev_gate_baseline/dev_eval.json`, `runs/r15_dev_step300/dev_eval.json`) and the value
stored in the archived prediction file `preds_dev300.npz` (whose `frames` array I compared
element-by-element with my own list — identical).

Frame table (episode, t) — 16 episodes × 2, all history-safe `t ∈ [2, T-2]`:

```
(0,8) (0,142) | (9,144) (9,214) | (17,185) (17,382) | (22,231) (22,258) | (28,156) (28,382)
(33,244) (33,250) | (38,92) (38,178) | (44,153) (44,164) | (51,15) (51,221) | (58,30) (58,137)
(64,132) (64,143) | (70,210) (70,294) | (75,92) (75,215) | (82,326) (82,343) | (88,232) (88,237)
(95,104) (95,218)
```

Cohort check: the 16 "DEV" episodes are exactly `split80.train − the 64 training episodes`
(disjoint from training, verified empty intersection). Note for the record: they are **not** drawn
from `split80.json["val"]` (the 20-episode old holdout, which the prereg forbids using); no leakage
found, but "DEV" here means "a carved-out slice of the train partition".

### 4b. CV baseline macro L1 = 0.002110443303536158 — reproduced, with a caveat that matters

Command as specified (CPU only, no GPU):

```
cd /home/T7/ojh/robot_actual/r11 && CUDA_VISIBLE_DEVICES= \
  /home/dgut/pusht_runtime/venv_oft/bin/python diagnostics/r15_dev_eval.py \
  --per-episode 2 --seed 0 --skip-mlp --out /home/dgut/r15_lanes/H/dev_gate
EXIT=0  elapsed_s=17
```

Console:

```
identity_A=S               SAFE macroL1=0.007738 | RAW macroL1=0.007734 | arm=0.008482 grip=0.003274 R2=-7.5517
constant_velocity_A=S+V    SAFE macroL1=0.002047 | RAW macroL1=0.002110 | arm=0.002389 grip=0.000000 R2=+0.0284
train_only_affine          SAFE macroL1=0.003264 | RAW macroL1=0.003526 | arm=0.003516 grip=0.001749 R2=-0.1804
train_only_mlp             NOT_AVAILABLE (skipped by flag)
gate: {"cv_macro_l1": 0.002047480782493949, "cv_macro_l1_raw": 0.002110443303536158, ...,
       "vla_macro_l1": null, "gated_on": "SAFE", "improvement_vs_cv": null}
```

| quantity | prereg | recomputed (evaluator) | my own recomputation | verdict |
|---|---|---|---|---|
| CV macro L1 **RAW** | `0.002110443303536158` | `0.002110443303536158` (`gate.cv_macro_l1_raw`) | `0.0021104433108121157` | **exact** (Δ 7e-12, float32 reduction order) |
| CV macro L1 **SAFE** | not registered | `0.002047480782493949` (`gate.cv_macro_l1`) | `0.002047480782493949` | exact, **3.0 % lower** |
| identity_A=S | `0.007733860024018213` | 0.007734 RAW | `0.007737873` SAFE / RAW matches | exact (RAW) |
| train_only_affine | `0.0035257016206742264` | 0.003526 RAW | – | exact (RAW) |

**Verdict on the number: PASS — 0.002110443303536158 is exactly reproducible.** But it is the
**RAW / v1** quantity. `R15_GATE_PREREGISTRATION.json` calls it `cv_macro_l1` and all four
`same_sample_baselines` are RAW numbers, while the evaluator now grades on **SAFE**
(`gated_on: "SAFE"`), where the same baseline is 0.002047480782493949. The two are not
interchangeable and the direction of the G5 comparison flips depending on which one is quoted.

### 4c. The verdict those numbers actually support

Recomputing both sides myself from the archived `preds_dev300.npz` (32 frames, target/state read
straight from the HDF5s, joint box from `data/push_T_bounds.npz`), `lanes/H/H_vla_metrics.py`:

```
CV raw (unclamped)     macro=0.0021104433108121157   arm=0.00238405866548419  grip=0.00046875106636434793
CV SAFE (clamped)      macro=0.002047480782493949    raw_oob_coord=0.006696428571428571
VLA@300 raw            macro=0.0021343568805605173   raw_oob_coord=0.060267857142857144
VLA@300 SAFE           macro=0.0020677782595157623

G5 test on SAFE (apples to apples): vla=0.002067778 cv=0.002047481 improvement=-0.991339%  -> 5% required: FAIL
G5 test on RAW  (v1 metric)      : vla=0.002134357 cv=0.002110443 improvement=-1.133106%  -> FAIL
MIXED (vla SAFE vs cv RAW)       : improvement=+2.021616%   <-- the flattering, invalid comparison
```

The archived v1 artefact agrees: `runs/r15_dev_step300/dev_eval.json` `gate` =
`{"cv_macro_l1": 0.002110443303536158, "vla_macro_l1": 0.002134356982423924,
"improvement_vs_cv": -0.011331116475717322}`. **The pre-registered G5 condition (≥5 % better than
constant velocity) fails at step 300 by ~1 %, in both metric variants.** What does pass is the much
weaker G4 condition "strictly better than identity A=S" (0.00213 vs 0.00773) — that is a bar any
smooth predictor clears on Push-T.

`model_sha256` inside `preds_dev300.npz` = `f0316a91a7e50d117c9665736c8c171c2450aa06bb650cacb158ed04d9014779`
= the `adapter_model.safetensors` of the step300 MANIFEST I verified in Item 2, so the DEV numbers
are pinned to that checkpoint.

### 4d. The pre-registration is not a fixed artefact — it was edited during this audit

```
HEAD blob  mission/R15_GATE_PREREGISTRATION.json = e78c861ee2904b89f563e00e75ba2cbf028eaf93e9955fe912fe8e1f0fc13362  (3874 B)
working tree /home/T7/ojh/robot_actual/r11/mission/R15_GATE_PREREGISTRATION.json
                   = 03cec65a22b439fb6bcb93f57cde66c1264cb254597d2c04168a96a459239777  (9361 B)  mtime 2026-10-21T08:00:51Z
git status --porcelain  ->  " M mission/R15_GATE_PREREGISTRATION.json"
```

The pristine 3874-byte version (mtime 02:59, commit `7fcbc32` 02:59:53Z) is the one I pulled at
07:52 and the one quoted above; it is preserved at `lanes/H/mission/R15_GATE_PREREGISTRATION.json`.
At 08:00:51Z — five hours after the step-300 DEV predictions existed and *during* this audit — an
`errata` block was appended by the (untracked) `tools/prereg_append_errata.py`. That tool asserts
"nothing above the errata block changed" and pins `pre_sha256 = e78c861e…`; I confirm
`e78c861e…` is indeed the HEAD blob, and I confirm **no threshold, cohort, seed, frame list or gate
action changed** — the errata only redefines metrics (sign agreement / magnitude ratio on
ΔA instead of absolute position; RAW vs SAFE clamping; G4 re-adjudicated on SAFE). So the numbers
above are not post-hoc-adjusted thresholds. But two things must be said bluntly:

* The file's own `written_utc` is `2026-10-21T11:05:00Z` — **8 h 05 m after its own mtime (02:59:28Z)
  and still 3 hours in the future** at the time of writing. The `"written_before_any_vla_dev_prediction": true`
  claim therefore rests on a timestamp that cannot be right; the only trustworthy anchor is the git
  commit time `7fcbc32 = 2026-10-21T02:59:53+00:00`, which *is* before the 03:40Z DEV prediction
  (and after the 02:57Z CPU baseline run).
* The metric definitions under which G4/G5 are judged changed **after** DEV scores existed. The
  errata discloses this honestly (it even states that under SAFE the model is worse than CV), but it
  means "the pre-registered gate" is not a single frozen object.

### 4e. Same problem for the evaluator itself

```
HEAD  diagnostics/r15_dev_eval.py  blob sha256 8e5695fb6bd21135753fb3cd0b4ec1c6e79c6011f934dc71f9f4ceca4cb6119a  (14539 B)
tree  diagnostics/r15_dev_eval.py       sha256 1e93e668d4238369ce0a225e3e1c83878f86ab6a6bd4070788fc2e64242a11f5  (22033 B)
      mtime 2026-10-21T07:53:11Z ; git diff --stat = 158 insertions(+), 32 deletions(-) ; M (uncommitted)
```

The evaluator that produced `0.002110443303536158` at 02:57-02:59 no longer exists in the working
tree; `METRIC_VERSION` is now `r15-dev-metrics-v2` and the gate quantity changed to "SAFE (clamped)
commands". My 4b run used the current (mutated) 22033-byte file — I recorded its sha before and
after the run and it did not change during the run. `diagnostics/r15_dev_eval.py` is **not pinned by
any hash inside the gate artefacts**, so the DEV numbers cannot be assumed reproducible from the
current evaluator. (The HEAD 14539-byte version is available as `git cat-file -p HEAD:diagnostics/r15_dev_eval.py`
and a mirror is at `E:\具身智能\_r11\wt\diagnostics\r15_dev_eval.py`.)

**Item 4 verdict: PASS on both required numbers (frame hash exact; CV macro L1 exact).** The
interpretation the chain wants from them is FALSIFIED: the model does not beat constant velocity at
step 300 under either metric, and the pre-registered file + evaluator were both mutated after those
scores existed.

---

## Item 5 — evidence gaps: unbilled DEV inference, double-booked GPU time

### 5a. YES — 7B DEV inference whose GPU seconds are unbilled

`runs/r15_stage_300_20261021T030423Z/devpreds.log` (7B inference on the frozen 32 frames):

```
[03:34:07] process start (tensorflow/torch init)
[03:34:15] runtime contract {...}
[03:34:59] pristine modeling sha256=b5431a074c…
[03:35:35] LoRA trainable=110.83M | 42D projector=16.957M | llm_dim=4096
[03:39:30] DEV predictions (32 frames, global_step=300) -> runs/r15_stage_300_20261021T030423Z/preds_dev300.npz
```

That is **~323 s of 7B model load + greedy decoding on the A6000**. The WAL has **no event at all**
between `seq 305` (`job_end`, 03:17:03Z) and `seq 306` (`job_begin`, 06:15:59Z) — a 2 h 58 m hole
that contains the whole inference. `runs/r15_dev_step300/dev_eval.json` (03:40Z) is the CPU-only
metric evaluator (17 s in my reproduction), so no GPU time is due there.

The executor already knows: the new, still-untracked `scripts/r15_devpreds_1000.sh` says in its own
header comment —

> "The 7B inference wall clock is booked into the WAL: **the audit found this GPU time unbilled before**."

— an explicit admission. That script has not run yet: `sup:m1_dev_preds_step1000` was booked
`job_blocked` at WAL seq 989 (`"dependency m1_700_step300_to_1000 is FAILED"`), and no
`runs/r15_dev_gate_1000*` artefact exists.

### 5b. YES — GPU time and formal steps are double-booked

**(i) concurrent duplicate launch (already booked in the WAL).**
`r15-s1000-061554Z` ran 06:15:59Z→06:51:02Z; `r15-s1000-061711Z` ran 06:17:16Z→06:19:10Z — a
**114 s overlap on the single RTX A6000**. The second one booked 55.000 s (`abort`,
`reason=cuda_oom_duplicate_launch`) + 3.641 s (`step_commit`) = 58.641 s. Worse, its single step was:

| seq | run | utc | global_step | episode/t | loss | gpu_s |
|---|---|---|---|---|---|---|
| 307 | 061554Z | 06:16:50Z | 301 | episode_0000081 t=7 | 2.303494 | 2.860 |
| 353 | 061711Z | 06:18:23Z | 301 | episode_0000081 t=7 | 2.303494 | 3.641 |

**The same optimizer step on the same sample was computed twice and booked twice** (93 s apart).

**(ii) systematic re-booking — the ledger cannot detect it.** `training/r15_ledger.py` keys the
duplicate guard on `(run_id, kind, global_step)` (lines 254-259), so two run_ids may book the same
`global_step` freely. My snapshot-C analysis of all 1014 `step_commit` events:

```
step_commit events: 1014
distinct global_step values booked: 963
global_step values booked MORE THAN ONCE: 50
total extra bookings (sum(c-1)): 51
max multiplicity: 3 at global_step 301
range of duplicated global_steps: 301 .. 350
   gs=301   x3 [061554Z 06:16:50 2.86s, 061711Z 06:18:23 3.641s, 0803Z 08:03:13 8.697s]
   gs=302   x2 [061554Z 06:16:52 1.805s, 0803Z 08:03:18 3.401s]
   ...
   gs=350   x2 [061554Z 06:18:32 2.134s, 0803Z 08:06:08 2.608s]
per-run: s300 n=300 gs=1..300 | 061554Z n=663 gs=301..963 | 061711Z n=1 gs=301 | 0803Z n=50 gs=301..350
unique global_step count = 963 ; sum(formal_steps) = 1014 ; inflation = 51
```

The cause is live: the stage-2 restart `r15-s2-1000-20261021T0803Z` resumed **from `step300`, not
from `step950`** (WAL seq 994: `"start_global_step": 300`, `"resume_from": ".../step300"`,
`steps_requested 700`). It is therefore recomputing 301→1000 from scratch while 301→963 stay booked.
As of snapshot C the extra GPU already spent on duplicated steps is **159.443 s**, and the formal
step count is inflated by **51**. If it reaches step 1000 the inflation reaches **~663** — at which
point G6 ("cumulative formal steps == 3000") is unverifiable by this ledger, since neither
`formal_steps_used` nor `gpu_seconds_used` measures distinct work.

### 5c. Also unbilled: GPU occupancy inside the training runs (not double-booking — under-billing)

The ledger books only the sum of per-step times; model load, compile and 13 checkpoint writes
(572 MB each) hold the card without being booked:

| run | wall (lifecycle events) | booked `gpu_seconds` | unbilled |
|---|---|---|---|
| `r15-s300-…030423Z` | 03:04:25→03:17:03 = 758 s | 576.967 | 181 s |
| `r15-s1000-…061554Z` | 06:15:59→06:51:02 = 2103 s | 1605.273 | 498 s |
| `r15-s1000-…061711Z` | 06:17:16→06:19:10 = 114 s | 58.641 | 55 s |
| **total** | | | **≈ 734 s** |

So `gpu_seconds_used = 3268.381` (snap A) understates real GPU occupancy by ≥ 734 s, plus the 323 s
of DEV inference = **≥ 1057 s unbilled (~26 % of the reported figure)**.

### 5d. Dead ends I checked and cleared

* `logs/r15_stage_300_20261021T025958Z.log` is an argparse failure
  (`error: unrecognized arguments: --save-every 50 --time-margin 120`) — the process died before
  loading the model, so no hidden GPU time there.
* `logs/r15_s2_1000_20261021T0758Z.log` — same, `error: unrecognized arguments: arx_x5_push_T`.
* The CPU-only evaluator runs (`r15_dev_baseline_cpu` 02:57, `r15_dev_gate_baseline` 02:59,
  `r15_dev_step300` 03:40) need no GPU booking; my own reproduction took 17 s of CPU.
* `sup:*` supervisor events all carry `counts_toward_total=false` and `gpu_seconds=0`; correct.

**Item 5 verdict: CONFIRMED on both counts.** Unbilled DEV inference ~323 s; unbilled in-run GPU
occupancy ~734 s; double-booked GPU time 58.6 s (concurrent duplicate) + 159.4 s and growing
(recomputed steps); formal steps double-counted by 51 already and heading for ~663.

---

## What is UNKNOWN (I did not measure it)

1. The **documentary** confirmation of the new run's sampler offset. Two independent indicators
   already say it is fixed — WAL seq 997 put step 301 on `perm[300]` (the frame 061554Z skipped),
   and the log says `# peek only, never sampler.next()` — but the artefact that would pin it,
   `runs/r15_s2_1000_20261021T0803Z/checkpoints/step350/sampler.json:position`, did not exist yet at
   08:09:43Z: the checkpoint was mid-write (`.tmp-2691034-5q4siuqg` holding 451 MB of the ~640 MB
   payload while the trainer kept stepping). **Expected value: 350.**
2. Whether `r15-s2-…0803Z` will reach step 1000 at all (07:54:39 start, 4.0 s/step estimate,
   `time_margin 180`; the previous attempt's guard stopped it 37 steps short).
3. The exact GPU-seconds semantics intended by the mission ("per-step compute" vs "card occupancy").
   My 5c under-billing claim depends on that definition; 5a and 5b(i)/(ii) do not.
4. Why `rng.pt`'s CUDA slot is all zeros — whether any consumer reads it (I only proved it is a
   constant and therefore cannot restore a boundary state).
5. Whether the `import_history` 1027.5 s in the WAL equals the historical ledger's real spend. I
   verified the *internal* consistency and the recorded `source_sha256`, but I did not re-audit the
   R13/R14 upstream evidence that produced it (out of Lane H scope).
6. Anything about the shared IsaacSim kit PID 3111650 — never touched, per the hard rule.

## Highest-value next action

**Check one number on the currently running stage-2 restart before it burns 700 steps:**

```
/home/dgut/pusht_runtime/venv_oft/bin/python -c "import json;print(json.load(open('/home/T7/ojh/robot_actual/r11/runs/r15_s2_1000_20261021T0803Z/checkpoints/step350/sampler.json'))['position'])"
```

It must print `350`. `351` means the off-by-one survived into the restarted chain and the whole
301→1000 segment is invalid again; `350` means the seam is finally correct and only the *step
accounting* (Item 5b) remains to be fixed. As of 08:09:43Z the `step350` directory did not exist yet
(the write was still staged in `.tmp-2691034-5q4siuqg`, 451 MB), so this remains the one open
verification — everything else about that restart already looks right.
Second priority: add a cross-run
`(global_step)` duplicate guard to `training/r15_ledger.py` and de-book `r15-s1000-…061554Z`'s
663 steps (or re-label them `superseded`) so the 3000-step cap counts distinct lineage.

## Local evidence bundle

`E:\具身智能\_r11\lanes\H\` — `events.jsonl`, `events_snap2.jsonl`, `events_snap3.jsonl`,
`projection.json`, `projection_snap3.json`, `ck/step{300,350,950}/{MANIFEST,trainer_state,sampler}.json`,
`mission/R15_GATE_PREREGISTRATION.json` (pristine 3874 B), `data/split80.json`,
`dev/{r15_dev_eval.py, dev_eval_step300.json, dev_eval_gate_baseline.json, dev_gate_H.json, preds_dev300.npz, devpreds.log}`,
`H_frames_recompute.py`, `H_vla_metrics.py`, `H_rng_sampler.py`, `frames_recomputed.json`.
Remote (written by me, read-only repo otherwise): `/home/dgut/r15_lanes/H/`.
