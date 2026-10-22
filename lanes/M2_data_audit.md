# M2 — multi-task data feasibility audit (push_T / stack_bowls / sweep_blocks)

Lane: M2 (one shared 7B LoRA VLA on three ARX-X5 RoboDojo tasks).
Host: 172.31.68.251 (user dgut). Repo: /home/T7/ojh/robot_actual/r11 @ feature/r15-true-continuation.
Python: /home/dgut/pusht_runtime/venv_oft/bin/python (h5py 3.16.0, numpy 1.26.4, torch 2.4.1+cu121, pytest 9.1.1).
Constraints honoured: read-only on the dataset tree, CPU only, no GPU/IsaacSim, no process touched, additive files only.
Evidence copies: /home/dgut/r15_lanes/M2/ (scripts + raw stdout + JSON).

---

## 0. Bottom line

| question | answer | confidence |
|---|---|---|
| Is the HDF5 schema identical across the three tasks? | **YES — identical dataset set (29 datasets + 7 groups = 36 paths), identical dtypes and trailing shapes, 100 % of episodes, 3/3 tasks.** Zero schema differences. | proven |
| 14D layout / cameras / encoding / action semantics identical to the push_T contract? | **YES.** Same 14D order, same 3 JPEG cameras in the same order, same 480x640x3, same 25 Hz, and **A[t] == S[t+1] with max abs discrepancy exactly 0.0 over all 300 episodes.** | proven |
| Gripper normalised [0,1] or metres? | **[0,1] normalised, all three tasks**, both state and action, dims 6 and 13. | proven |
| Is there real, task-distinguishing language? | **YES — exactly one instruction per task, 100/100 episodes each, all three strings different.** | proven |
| Can ONE shared 7B LoRA be trained on the existing files? | **YES for push_T + stack_bowls, and yes for all three.** 240 TRAIN episodes / 109,500 TRAIN frames / 108,780 usable rows. | proven |
| Single blocking issue? | **Not the data.** The single blocker is the **single-task normalizer/metadata contract**: the R15 gate (training/r9_trainer_contract.py:30-53) admits exactly one 14D pmin/pmax/v_scale/a_scale/q01/q99 payload with 64-TRAIN/16-DEV provenance, and has no field for a task id or for three per-task normalizer sets. The three tasks genuinely need different scales (residual q99 abs max = 0.0213 / 0.0520 / 0.0415), so one global normalizer would clip or quantise away most of stack_bowls/sweep_blocks target range. | proven (scale spread) + design consequence |
| Honest UNKNOWN | **Scene/layout provenance of the 100 training episodes per task.** The HDF5 records no layout id, scene seed or episode index, so scene-level disjointness between TRAIN and DEV, and between TRAIN and the 165 eval layout files per task, cannot be established from these files. | UNKNOWN |

**Correction to the brief:** push_T is *not* "T ~= 360 frames". Measured: min 156, median 313, max 773, 86 distinct lengths over 100 episodes (section 1). Any batching/curriculum code that assumes a fixed 360 will be wrong.

---

## 1. Frames per episode (measured over all 100 episodes of each task)

Command: `/home/dgut/pusht_runtime/venv_oft/bin/python m2_audit2.py` (evidence: /home/dgut/r15_lanes/M2/audit2_out.txt, audit2.json).
Frame count = `state/left_arm_joint_states.shape[0]`, cross-checked against every other time-indexed dataset (T_consistency_violations = 0 in all three tasks, audit3_out.txt).

| task | files | min | median | max | total frames | distinct lengths | disk |
|---|---|---|---|---|---|---|---|
| push_T | 100 | **156** | **313** | **773** | 32,304 | 86 | 7.3 G |
| stack_bowls | 100 | 204 | 444.5 | 780 | 44,874 | 88 | 12 G |
| sweep_blocks | 100 | 344 | 591 | 1064 | 60,770 | 96 | 15 G |

Real output (audit2_out.txt lines 5, 26, 47):

```
TASK push_T files 100
 frames min/median/max/sum: 156 313.0 773 32304 unique_lengths: 86 T-mismatch-files: 200
TASK stack_bowls files 100
 frames min/median/max/sum: 204 444.5 780 44874 unique_lengths: 88 ...
TASK sweep_blocks files 100
 frames min/median/max/sum: 344 591.0 1064 60770 unique_lengths: 96 ...
```

The "T-mismatch-files: 200" line is a bug in my *first* pass (it compared full shapes instead of shape[0]); it was re-run correctly in m2_audit3.py: push_T files=100 T_violations=0 sig_diff_files=0 for all three tasks (audit3_out.txt lines 3-5). Every dataset in an episode agrees on T.

---

## 2. Full HDF5 schema — and the proof that all three tasks are identical

### 2.1 One episode = 36 paths (29 datasets, 7 groups), no attributes

Command: m2_schema_audit.py (schema_dump.json / schema_out.txt) + m2_audit3.py.

| path | kind | dtype | shape per episode |
|---|---|---|---|
| /action/left_arm_joint_states | DATASET | float64 | (T, 6) |
| /action/left_ee_joint_states | DATASET | float64 | (T, 1) |
| /action/left_ee_poses | DATASET | float64 | (T, 7) |
| /action/right_arm_joint_states | DATASET | float64 | (T, 6) |
| /action/right_ee_joint_states | DATASET | float64 | (T, 1) |
| /action/right_ee_poses | DATASET | float64 | (T, 7) |
| /additional_info/frequency | DATASET | int64 | () = 25 |
| /data_format_version | DATASET | object (bytes) | () = "v1.0" |
| /instruction | DATASET | object (bytes) | () = task string |
| /state/left_arm_joint_states | DATASET | float64 | (T, 6) |
| /state/left_delta_ee_poses | DATASET | float64 | (T, 7) |
| /state/left_ee_joint_states | DATASET | float64 | (T, 1) |
| /state/left_ee_poses | DATASET | float64 | (T, 7) |
| /state/right_arm_joint_states | DATASET | float64 | (T, 6) |
| /state/right_delta_ee_poses | DATASET | float64 | (T, 7) |
| /state/right_ee_joint_states | DATASET | float64 | (T, 1) |
| /state/right_ee_poses | DATASET | float64 | (T, 7) |
| /vision/cam_head/colors | DATASET | S<N> byte buffer | (T,) JPEG per frame |
| /vision/cam_head/extrinsic_matrix | DATASET | float64 | (T, 4, 4) |
| /vision/cam_head/intrinsic_matrix | DATASET | float64 | (3, 3) |
| /vision/cam_head/shape | DATASET | int64 | (3,) = [480, 640, 3] |
| /vision/cam_left_wrist/{colors,extrinsic_matrix,intrinsic_matrix,shape} | DATASET | as above | (T,), (T,4,4), (3,3), (3,) |
| /vision/cam_right_wrist/{colors,extrinsic_matrix,intrinsic_matrix,shape} | DATASET | as above | (T,), (T,4,4), (3,3), (3,) |

Groups: /action, /additional_info, /state, /vision, /vision/cam_head, /vision/cam_left_wrist, /vision/cam_right_wrist.
No HDF5 attributes anywhere (file_attrs: {}, every group/dataset attr set empty). No compression, no chunking.

Note on the colours dtype: the byte-string item size is **per file** (e.g. S91799, S78426, ...) because it is the *largest* JPEG in that episode. That is expected for a fixed-size byte-string column and is not a schema difference; 100/100 push_T episodes therefore look like 100 "distinct signatures" if you compare the raw dtype string (schema_out.txt line 41). Comparing shape-and-kind with the S<N> abstracted, it is identical across all 100 episodes of all 3 tasks (sig_diff_files=0).

### 2.2 Cross-task structural diff: EMPTY

Real output (audit3_out.txt lines 8-9):

```
push_T vs stack_bowls: paths_only_pushT=[] paths_only_stack_bowls=[] changed=[]
push_T vs sweep_blocks: paths_only_pushT=[] paths_only_sweep_blocks=[] changed=[]
```

So: **there is no per-task schema difference at all.** The differences that do exist are *statistical* (frames per episode, action ranges, residual scales, gripper occupancy) and are reported below — they are the finding that matters for a shared policy, and they are why the prototype fits a **per-task** normalizer.

---

## 3. Action semantics and the 14D layout — verified numerically per task

### 3.1 A[t] == S[t+1], all 300 episodes

Command: m2_audit2.py (reads the four 14D blocks, concatenates, compares a[:-1] against s[1:]).

| task | max abs(A[t] - S[t+1]) | episodes with >1e-6 | episodes with >1e-5 |
|---|---|---|---|
| push_T | **0.000e+00** | 0 | 0 |
| stack_bowls | **0.000e+00** | 0 | 0 |
| sweep_blocks | **0.000e+00** | 0 | 0 |

Per-dimension maxima are 0.00e+00 on all 14 dims of all three tasks (audit2_out.txt lines 7, 28, 49) — the identity is **bit-exact**, not merely within tolerance. Action is the next-step absolute joint target, i.e. exactly the semantics the R15 contract requires (training/r42_train_contract.py:85-98).

### 3.2 14D layout

Widths are [6, 1, 6, 1] in both /state and /action, in the same order, in all three tasks. This is asserted fail-closed on every episode by m2_multitask_data._inspect_episode ("14D layout widths %s != [6, 1, 6, 1]") and read through the shared helper r42_train_contract.read14 (training/r42_train_contract.py:72-82).

### 3.3 Gripper channels: normalised [0,1], not metres

| task | action dim 6 (min,max) | action dim 13 (min,max) | state dim 6 | state dim 13 |
|---|---|---|---|---|
| push_T | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |
| stack_bowls | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |
| sweep_blocks | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |

Metres would have shown ~0.044 (the URDF full-open travel); nothing near that appears. This matches the simulator-side mapping in lanes/C_taskspec_audit.md:243-245 (action 0 -> -0.01 m closed, action 1 -> 0.044 m open, i.e. the recorded channel is the normalised command, not the metre value).

**Per-task statistical difference that is a first-class finding:** the global [0,1] range is identical, but occupancy is not. TRAIN-fitted 1 %/99 % of the left gripper (dim 6) is 0.0/1.0 for push_T and stack_bowls but **0.146/1.0 for sweep_blocks** (dim 13: 0.0766/1.0) — in sweep_blocks the left gripper essentially never fully closes (m2_normalizers.json). A shared normalizer fitted on the pooled gripper column would compress that difference away.

### 3.4 Action ranges differ materially per task (why one global normalizer is wrong)

| task | action dim 0 min/max | dim 3 min/max | dim 10 min/max | residual R=(A-S)-V, max abs q99 |
|---|---|---|---|---|
| push_T | -1.2448 / 0.9009 | -1.5545 / 0.0565 | -1.4038 / 0.0820 | **0.0213** |
| stack_bowls | -1.2836 / 1.5064 | -1.8973 / 0.5567 | -2.0262 / 0.4759 | **0.0520** |
| sweep_blocks | -1.1025 / 0.5698 | -2.1864 / 1.3323 | -2.3617 / 1.5541 | **0.0415** |

(full 14D min/max in audit2_out.txt lines 11-12, 32-33, 53-54; residual q99 from the TRAIN-only fits in m2_normalizers.json.)

The residual target range differs by 2.4x between push_T and stack_bowls. With the 256-bin codec (training/r42_train_contract.py:20) a push_T-fitted q01/q99 would push most stack_bowls residuals into the outer bins. **Per-task normalizer is mandatory, not a nicety.**

### 3.5 Cameras

| property | push_T | stack_bowls | sweep_blocks |
|---|---|---|---|
| key set and ORDER | cam_head, cam_left_wrist, cam_right_wrist (100/100) | identical | identical |
| /shape per camera | 480x640x3 (300/300) | identical | identical |
| encoding | per-frame JPEG byte buffers, SOI FFD8 / EOI FFD9 (300/300) | identical | identical |
| decoded frame 0 (PIL) | JPEG, 640x480, RGB | JPEG, 640x480, RGB | JPEG, 640x480, RGB |
| intrinsic cam_head | [[288.132541,0,320],[0,288.132541,240],[0,0,1]] | identical | identical |
| extrinsic jitter over T | cam_head 0.0; wrists move (expected) | cam_head 0.0 | cam_head 0.0 |
| Hz | 25 | 25 | 25 |

Real output (audit2_out.txt lines 16-22):

```
 camera key order: {'cam_head|cam_left_wrist|cam_right_wrist': 100}
 camera shape dataset: {'480x640x3': 300}
 jpeg magic: {'ffd8/ffd9': 300}
 decoded ep0: {"cam_head": {"format": "JPEG", "size": [640, 480], "mode": "RGB", ...}, ...}
 intrinsic ep0 cam_head: [[288.132541, 0.0, 320.0], [0.0, 288.132541, 240.0], [0.0, 0.0, 1.0]]
```

The camera order also matches the frozen deployment order CAMERA_ORDER in code/xpolicylab/policy/OpenVLA_R42/r42_adapter.py:46.

---

## 4. Language / instruction availability

Every one of the 300 episodes carries a non-empty /instruction (single-element bytes dataset) and there is **exactly one distinct string per task**, so a shared policy gets a clean, unambiguous task signal:

| task | instruction (quoted verbatim from the HDF5) | distinct | count |
|---|---|---|---|
| push_T | "Push the T-shaped block to align it precisely with the gray T-shaped pad." | 1 | 100/100 |
| stack_bowls | "Stack the three bowls together." | 1 | 100/100 |
| sweep_blocks | "Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the blocks." | 1 | 100/100 |

Evidence: audit2_out.txt lines 13-14, 34-35, 55-56. The strings are task-distinguishing (no two are equal, and each names its task objects/verbs); nothing needs to be invented, and the prototype re-verifies them at index time (prompt_matches_constant=True for all three, m2_p10_out.txt lines 6-11).
Cross-check independent of the data: the same strings are the task modules instruction templates — lanes/C_taskspec_audit.md:285-286 (stack_bowls.py:94, push_T.py:32).

**No task is missing a language signal.** No episode carries more than one instruction (so there is no per-episode language variation to model yet).

---

## 5. Leakage and split design

### 5.1 What the existing split file actually covers

data/split80.json (md5 73312597aea0ef25aee1349eea523264) has keys split_seed=0, train(80), val(20), no overlap, no task field. Its names are **bare file names** ("episode_0000000.hdf5", ...).

Command + real output (m2_p6_out.txt lines 2-6):

```
split80 keys: ['split_seed', 'train', 'val'] split_seed = 0
split80 n_train=80 n_val=20 overlap=0
push_T        files=100  split80_names_covered=100  train_hits=80 val_hits=20  missing_from_split=0
stack_bowls   files=100  split80_names_covered=100  train_hits=80 val_hits=20  missing_from_split=0
sweep_blocks  files=100  split80_names_covered=100  train_hits=80 val_hits=20  missing_from_split=0
```

So: split80.json **covers exactly the 100 push_T episodes** and nothing else — but only by convention, because **every task has the same 100 file names**. Its lineage is push_T-only: training/r11_train_pushT_r42.py:40-41 (HDF5_DIR defaults to the push_T tree) and :44 (SPLIT defaults to data/split80.json), documented in data/README.md:4-5,35.

**Leakage hazard (real, discovered):** because episode names collide across tasks, any 2-task/3-task code that keys a single split dict by bare name would silently alias push_T/episode_0000003.hdf5 with stack_bowls/episode_0000003.hdf5 and could put the same *name* in TRAIN for one task and DEV for another. The prototype therefore keys every cross-task structure by <task>/<episode> and fails closed on foreign keys (section 6).

### 5.2 Proposed per-task TRAIN/DEV (80/20, episode-isolated)

Written to the new file data/m2_splits.json by: python training/m2_multitask_data.py --out-splits data/m2_splits.json (26,813 B, schema m2_splits/v1):

| task | TRAIN eps | TRAIN frames | usable TRAIN rows (t in [2, T-2]) | DEV eps | DEV frames | provenance |
|---|---|---|---|---|---|---|
| push_T | 80 | 25,553 | 25,313 | 20 | 6,751 | **reused frozen data/split80.json (train/val)** |
| stack_bowls | 80 | 36,098 | 35,858 | 20 | 8,776 | seeded split_train_dev(seed=1) |
| sweep_blocks | 80 | 47,849 | 47,609 | 20 | 12,921 | seeded split_train_dev(seed=2) |
| **3-task total** | **240** | **109,500** | **108,780** | **60** | **28,448** | |

push_T deliberately re-uses the frozen split so its numbers stay comparable with R15 (which further slices the 80 into 64 fit / 16 dev, training/r9_trainer_contract.py:56-65). The other two tasks get a seeded shuffle with seed = seed + task_id so their DEV sets are decorrelated from each other and from push_T frozen assignment.

Real output (m2_p12_out.txt lines 18-24):

```
push_T        train=80 dev=20 train_frames=25553 dev_frames=6751 rows=25313 reused frozen data/split80.json (train/val)
   first train key: push_T/episode_0000000.hdf5 | first dev key: push_T/episode_0000001.hdf5
stack_bowls   train=80 dev=20 train_frames=36098 dev_frames=8776 rows=35858 seeded split_train_dev(seed=1)
   first train key: stack_bowls/episode_0000000.hdf5 | first dev key: stack_bowls/episode_0000004.hdf5
sweep_blocks  train=80 dev=20 train_frames=47849 dev_frames=12921 rows=47609 seeded split_train_dev(seed=2)
   first train key: sweep_blocks/episode_0000003.hdf5 | first dev key: sweep_blocks/episode_0000000.hdf5
qualified keys are task-prefixed: True
```

Isolation guarantees enforced in code (all tested): TRAIN and DEV are disjoint per task; splits of different tasks never share a qualified key; the union of a task split equals that task index exactly.

### 5.3 What a 2-task and a 3-task curriculum consume

| curriculum | tasks | TRAIN episodes | TRAIN frames | usable rows | DEV episodes | DEV frames | normalizer sets |
|---|---|---|---|---|---|---|---|
| 2-task | push_T + stack_bowls | 160 | 61,651 | 61,171 | 40 | 15,527 | 2 (14D each) |
| 3-task | + sweep_blocks | 240 | 109,500 | 108,780 | 60 | 28,448 | 3 (14D each) |

Each task contributes its own pmin/pmax (state q01/q99), v_scale/a_scale (q99 of |V| and |Acc| over TRAIN), action_q01/q99 and residual_q01/q99 — i.e. 6 x 14D per task, stored per task id. Batch composition: sample by task, apply that task normalizer to build the 42D feature and 14D residual, and prepend that task language string (optionally behind the stable task token <task:NAME#ID>).

### 5.4 Eval layouts vs training episodes — what is established, what stays UNKNOWN

Facts (audit5_out.txt, audit6_out.txt, m2_p5_out.txt):

* /home/T7/ojh/robot_actual/Assets/Eval_Layout/RoboDojo/arx_x5/<seed>/<task>_<n>.json, seeds 0/1/2, **55 layouts per task per seed** (index 0..54, contiguous) → **165 layout configs per task**;
* the seed dirs hold 965 files / 24 task prefixes; the three seed dirs are **different content** (md5 of push_T_0.json differs per seed: 4fa4cf72e307 / 080fa85fa3ec / f0b923d07265);
* a layout file is a **scene descriptor of ranges, not a fixed scene**: keys are Rigid / Geometry / Room / Table / Ground / Background, e.g. {"category": "t", "xlim": [-0.45,0.45], "ylim": [-0.25,-0.1], "zlim": null, "rotate_deg": 360, "rotate_rand": true, "default_pos": [...], ...} — position and heading are *sampled*, so two episodes from the same layout file are different scenes;
* a recursive key scan of a layout file found **no field referencing an episode, demo, trajectory or training seed** (only coincidental names such as category_idx and place_tag);
* the training HDF5 records **no layout id / scene seed / episode index** — all 29 datasets are listed in section 2, and there are no HDF5 attributes.

How disjointness is therefore established, and its limit:

1. **File-level disjointness — proven.** The 3x55 layout JSONs live under Assets/Eval_Layout/, the episodes under data/data/hdf5/<task>/arx_x5/data/; no layout file is inside a dataset directory and no episode file is inside a layout directory (directory listings, m2_p5_out.txt lines 91-120).
2. **Instance-level disjointness — UNKNOWN, and unprovable from these files.** Because layouts are sampling ranges and the episodes carry no provenance, I cannot rule out that some of the 100 training episodes were generated from one of the 55 eval layouts. Two further complications: (a) the same layout file drawn twice is not the same scene, so even a name match would not imply scene reuse; (b) eval itself has an intra-seed duplicate for sweep_blocks — seed 0 files 10, 13 and 23 are byte-identical (53 distinct of 55).
3. **What a defensible M2 claim can say:** "TRAIN/DEV are episode-disjoint; the three eval seeds form three disjoint scene sets; the 165 eval layouts per task are a separate asset tree." It must **not** say "TRAIN scenes are disjoint from eval scenes" — that is UNKNOWN unless the collection script layout provenance is recovered (a mission-level artifact, not present in this repo).

Practical mitigation if a scene-disjointness claim is required: re-collect with the layout file id written into the HDF5 (e.g. /provenance/layout_path + /provenance/layout_md5), an additive schema change for future data; today 300 episodes cannot be retro-annotated.

---

## 6. CPU-only prototype shipped in the repo

New, additive files (nothing existing was modified, no commit, no push):

| file | lines | md5 | purpose |
|---|---|---|---|
| training/m2_multitask_data.py | 709 | cf997527b60fc91851dcdd188929c123 | index / split / per-task normalizer / row / prompt |
| tests/test_m2_multitask_data.py | 372 | eb2eef587eaf2671d93a83b75c1ef546 | 22 CPU tests |
| data/m2_splits.json | 26,813 B | — | generated 80/20 per-task splits (new file) |

git status --porcelain shows them only as untracked additions (?? tests/test_m2_multitask_data.py, ?? training/m2_multitask_data.py), alongside files that belong to other lanes, not to me.

### 6.1 API (exactly the requested surface)

* **load_task_index(task, hdf5_dir)** — accepts the dataset root **or** the task directory; walks all episodes and fails closed on: missing dataset, non-floating 14D block, 14D widths != [6,1,6,1], T disagreement between datasets, wrong camera set **or order**, non-(480,640,3) shape, non-JPEG buffer (no FFD8/FFD9), empty instruction, frequency != 25, and (optionally, verify_targets=True) A[t] != S[t+1].
* **split_train_dev(episodes, dev_frac, seed)** — deterministic, order-independent, disjoint, complete; rejects duplicates, empty input, dev_frac outside (0,1), degenerate splits.
* **TaskNormalizer.fit_from_task_index(index, train_episodes, dev_episodes)** — **TRAIN-only**; refuses any overlap with DEV or any episode not in the index; fits pmin/pmax (state q01/q99), action_q01/q99, v_scale/a_scale (q99 of |V| and |Acc| via r8_normalization.fit_scales), and residual_q01/q99 for R = (A-S)-V. to_contract() returns the exact R8/R15 NormalizationContract(SIGNED_MOTION, ...).
* **build_row(state, action, t, contract, task)** — **delegates to r42_train_contract.build_training_row** (training/r42_train_contract.py:114-135), so the 42D [S,V,Acc] feature + 14D velocity-residual target are the R15 contract by construction, then attaches and re-asserts the task id/token/prompt. assert_row_no_future_leakage delegates to training/r42_train_contract.py:138-158.
* **build_prompt / task_prompt / task_token / task_id** — one language string per task; stable ids {push_T:0, stack_bowls:1, sweep_blocks:2}; token <task:NAME#ID>; supplying a drifted instruction raises instead of silently changing the prompt.
* **build_m2_splits(...)** + CLI: python training/m2_multitask_data.py --verify-targets --out-splits data/m2_splits.json — writes the split artefact and prints the per-task audit and the TRAIN-only normalizer summaries.

Hard assertions carried: no future leakage (row depends only on t, t-1, t-2 plus the asserted target), TRAIN/DEV disjoint, task id preserved through the row, exact shapes (42,)/(14,), gripper dims (6,13).

### 6.2 Test command and real output tail

```
cd /home/T7/ojh/robot_actual/r11
/home/dgut/pusht_runtime/venv_oft/bin/python -m pytest tests/test_m2_multitask_data.py -q
```

```
......................                                                   [100%]
22 passed in 204.01s (0:03:24)
```

(saved as /home/dgut/r15_lanes/M2/pytest_m2.log; an earlier identical run reported "22 passed in 76.96s")

Coverage (all passing): per-task index + split; cross-task isolation (a stack_bowls episode can never enter push_T index/split — proven both on live data and with a doctored index); wrong-dim fail-closed (14D width 5, bad frequency 30, a second/different instruction); TRAIN-only normalizer provenance (refuses DEV as a fit source, refuses unknown episodes, is footed on strictly fewer rows than an all-episode fit, metadata round-trip); prompt/task-token consistency (three distinct strings, stable idempotent token, drift rejection); row shapes/task preservation/no-future-leakage; gripper [0,1] on live data.

---

## 7. Honest answer to "can one shared 7B LoRA be trained on these files?"

**Yes, data-wise.** All three tasks share one schema, one 14D action space with identical absolute-next-step semantics, one camera rig and one instruction each — everything a single multi-task policy needs is present, and the prototype makes the shared contract executable on CPU:

* 2-task (push_T + stack_bowls): **160 TRAIN episodes / 61,651 frames / 61,171 rows**, 40 DEV episodes / 15,527 frames.
* 3-task (+ sweep_blocks): **240 TRAIN episodes / 109,500 frames / 108,780 rows**, 60 DEV episodes / 28,448 frames.

**What is missing (the single blocker) — the multi-task normalizer/metadata contract, not the data.**
load_frozen_contract() (training/r9_trainer_contract.py:30-53) accepts exactly one 14D pmin/pmax/v_scale/a_scale/q01/q99 payload, demands 64-TRAIN/16-DEV provenance, and has no field for a task id; r42_adapter.normalizer_payload() (code/xpolicylab/policy/OpenVLA_R42/r42_adapter.py:370-392) serialises that same single payload. A shared model must carry **three** normalizer sets plus a task id, select them at train and deploy time, and the deploy gate must accept a payload it currently rejects. My module supplies the per-task payloads and the mapping; wiring them through the trainer/deployer gate is M3 work and cannot be done without touching files this lane is forbidden to modify.

Secondary gaps, in order of importance (none blocking training):

1. **Scene provenance UNKNOWN** (section 5.4) — no layout id in the HDF5, so scene-level TRAIN/DEV and TRAIN/eval disjointness is an assumption; future data should record /provenance/layout_path.
2. data/push_T_bounds.npz and the frozen contract are push_T-only; they must not be reused for the other tasks (the prototype never reads them).
3. Episode length varies 156..1064 frames with 86-96 distinct lengths per task — the sampler must be length-aware; the row contract already defines the valid frame pool t in [2, T-2].
4. sweep_blocks left gripper almost never closes (1st percentile 0.146) — expect the shared policy to need per-task conditioning there, not a global gripper prior.

---

## 8. Evidence index (/home/dgut/r15_lanes/M2/)

| file | content |
|---|---|
| schema_dump.json / schema_out.txt | full per-task structure dump, all 100 episodes each |
| audit2.json / audit2_out.txt | frames, A==S+1, gripper ranges, cameras, instructions |
| audit3.json / audit3_out.txt | T consistency, per-episode structure equality, cross-task diff (empty) |
| audit4_out.txt | split80.json vs the three tasks file-name sets |
| audit5_out.txt | eval layout counts per task/seed |
| audit6_out.txt | layout content keys, cross-seed distinctness, intra-seed duplicates |
| m2_normalizers.json | TRAIN-only fitted normalizer payloads per task (full 14D vectors) |
| m2_splits.json | the generated split artefact (copy of data/m2_splits.json) |
| m2_multitask_data.py / test_m2_multitask_data.py | copies of the shipped files |
| pytest_m2.log | pytest tail |

Reproduce: python rsh.py <cmd.sh> <out.txt> <timeout> with the scripts m2_schema_audit.py, m2_audit2.py, m2_audit3.py, m2_audit4.py, m2_audit5.py, m2_audit6.py, m2_norm.py (all in that directory).
