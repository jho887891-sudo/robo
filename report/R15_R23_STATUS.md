# R15–R23 status — what is true right now

Generated 2026-10-21T08:50Z by the executor. **Every number below was produced by a command on the
server, quoted in `mission/R15_R23_EXECUTOR_LOG.md` or in `lanes/*`.** Nothing here is a plan.

## 1. Where the GPU is right now

| item | value |
|---|---|
| running | stage 2b: `500 -> 1000 (500 steps)`, run `r15-s2b-1000-20261021T0803Z`, `--save_every 200` |
| launched by | `tools/mission_supervisor.py` (server-side, PPID 1, own session, survives disconnect) |
| next automatically | DEV predictions at step1000 -> `tools/r15_gate_check.py --gate G5_1000` -> stage 3 (+2000) **only if the gate PASSES**; a FAIL leaves stage 3 BLOCKED and keeps the checkpoint |
| why the split | `/home/T7` is ntfs-3g at **1.5–3 MB/s**; one 1.1 GB full-state save costs 4–8 min, so a single job cannot reach step1000 before its wall guard |

## 2. M1 Push-T: real state

| quantity | value | how |
|---|---|---|
| verified parent | `runs/r15_stage_300_*/checkpoints/step300` | 9/9 MANIFEST sha256 reproduce (audit lane H) |
| stage 2a | +200 steps, `PARTIAL` at step500 (wall guard), last save step500 | log + WAL seq 1201 |
| resume seam | `position=300` at `global_step=300` — peek only, never `sampler.next()` | `logs/r15_s2_1000_*/…` |
| sample order | `sampler_order_hash = f1024210…` **identical to stage 1** | stage_evidence.json |
| GPU booked | 6169.8 s total WAL (incl. the historical 1027.5 s import) | `tools/r15_ledger_lineage.py` |
| formal steps booked | 1164 executed, of which **963 distinct lineage steps** (audit H) | lineage tool |
| overhead billing | fixed: stage 2a `job_end` books 2269.1 s of load/save/IO that the old code dropped | WAL seq 1201 |
| offline DEV @300 | SAFE macro L1 **0.0020678** vs CV **0.0020475** -> **1.0 % worse than constant velocity** | v2 evaluator on the archived 32-frame gate set |
| G4_300 | **PASS** under the v2 definitions (better than identity 0.0077, no collapse, SAFE inside box, RAW-OOB reported) | `/home/dgut/r15_lanes/gate_G4_300_recomputed.json` |
| G5_1000 | **not decided yet** (needs step1000) | pending |

### Why G5 may legitimately fail, stated before the run
`A[t] == S[t+1]` means CV's error *is* `|R_t|`, the residual magnitude, so CV is the "predict R = 0"
model. On the frozen DEV frames a TRAIN-only linear map from the 42D proprio feature to R has
**DEV R² = −0.141** (and the TRAIN-mean residual has R² = −0.035): the proprio history does not
predict the residual, so beating CV requires the 3 RGB frames to carry information the history does
not. A negative G5 is therefore a residual/vision finding, not a training bug — and the pre-registered
action is to stop at 1000, keep the checkpoint and diagnose, not to train blindly to 3000.

## 3. Engineering state (all pushed)

| commit | content |
|---|---|
| `97cd5e8` | PR#21 findings 3–4: ΔA−S and residual metrics, RAW vs SAFE split, machine gate, lineage accounting, resumable supervisor, prereg errata E1–E4 |
| `c6c0059` | multi-task data contract for the 3 real datasets (22 tests) |

Branch `feature/r15-true-continuation`, pushed to `origin`. Tests: **45 passed** (dev_eval + gate +
ledger + resume) and **22 passed** (multi-task data). Draft PR still needs the research lead: the box
has no `gh` and no token, so a PR cannot be opened from here.

## 4. The 12-task programme: what the audits actually found

| lane | result |
|---|---|
| C TaskSpec | **12/12 installed and traceable, 0 BLOCKED_INTERFACE.** `generalization` is exactly the 12 ids in `scripts/internal/task_inventory.py:25-39` (executed: `--list-dimensions` -> `generalization | 12 | 24`). Robot `dual_x5` = 2×ARX X5, 14D action = our contract, 3 RGB cameras, **25 Hz** control, timeout = `step_lim` |
| D expert | **6 realistic, 4 hard-but-rigid, 2 BLOCKED_DATA** (`fold_clothes` cloth mesh vertices, `pour_liquid_into_cup` PhysX particles — the predicates need geometry→action inverses that do not exist). The `k1_*` scripts in the RoboDojo root are **not** legal experts (they set `env.success[0]=True` by hand) |
| B R16 | transport is WebSocket+msgpack; obs/action are dicts (verified shapes); our 14D/gripper/camera semantics match. **The official FINAL is 25 episodes, not 30** — `_task.yml:105-107` sets `eval_nums: 25` and `main.py:325-329` clamps `EVAL_NUM` downward only. Pre-registered equivalent contract written to `mission/R15_R23_FINAL_EVAL_CONTRACT.json` |
| M2 data | 3 of the 12 tasks have real data and they are **structurally identical** (schema diff empty over 300 episodes, `A[t]==S[t+1]` bit-exact, language present and task-distinguishing). 2-task curriculum = 160 TRAIN episodes / 61,171 rows. Blocker: the normalizer payload has **no task id** and the per-task residual scale spread is 2.4× |

## 5. Honest gaps (do not let anyone quote past these)

1. **No closed-loop number yet.** Offline L1 is not success; the RoboDojo wire is built (serving
   wrapper + converters + metadata bridge, in progress) but not yet driven against the simulator.
2. **9 of 12 tasks have no data.** Collection needs scripted experts (6 realistic candidates) and GPU
   simulator time; 2 are BLOCKED_DATA by physics.
3. **Scene-level TRAIN/eval disjointness is unprovable from the data**: the HDF5 files carry no layout
   id. File-level disjointness is proven. This must be stated in any FINAL report.
4. **The licence is contradictory** in the installed stack (`LICENSE:1` MIT vs `README.md:129`
   non-commercial research). Not an execution blocker; it is a publication decision for the lead.
5. **The historical WAL cannot be retro-fixed**: ~734 s of load/save time and 323 s of DEV inference
   from before this session are unbilled there. They are quantified, not hidden
   (`tools/r15_ledger_lineage.py`), and every job started in this session books its own overhead.
