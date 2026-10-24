# R27 — M2 step2000 只读证据（0 optimizer steps）

来源：GitHub Issue #38，工具固定 SHA `d61239e351b444b656058c7f4ace778a76fbeedd`（PR #37 head）。
执行机：原 M2 GPU 服务器 `jxxy`（用户 `dgut`），只读 CPU 采集，未启动 CUDA / 7B 前向 / Isaac Sim / 训练。

## 交付格式（runbook §5）

```text
STATUS: VERIFIED
GIT SHA: d61239e351b444b656058c7f4ace778a76fbeedd
CHECKPOINT MANIFEST SHA256: df62650015abf279204418a73410f6ea0c6eae7c2814cc5ed20577fa5630a68d
STAGE EVIDENCE SHA256: 9cfb9fffa7256395ebb48c5ceb88a7e9a462389a467e74364a4d5f276c29a1e6
SAMPLER POS: 2000
DISTINCT SCHEDULE SLOTS: 2000
ACTUAL DISTINCT TRAIN-SEEN FRAMES: 2000
R24 136 TRAIN SELECTED -> SEEN COUNT: 3
DEV CONTAMINATION COUNT: 0
OUTPUT LOG DIRECTORY: /tmp/r27-m2-check-JfI3qsxZ
EVIDENCE SHA256SUMS: mission/r27/evidence/m2_step2000_20261021T123314Z/SHA256SUMS.txt
COMMAND EXIT CODES: inventory=0, seen_exposure=0
OPTIMIZER STEPS ADDED: 0
```

## 唯一一次运行（未覆盖、未重跑）

```bash
CKPT=/home/T7/ojh/robot_actual/runs/m2_two_2000_20261021T123314Z/checkpoints/step2000
STAGE=/home/T7/ojh/robot_actual/runs/m2_two_2000_20261021T123314Z/stage_evidence.json
HDF5=/home/T7/ojh/robot_actual/data/data/hdf5
WT=/tmp/r27-m2-wt-d61239e          # 干净 detached worktree @ d61239e
OUTDIR=$(mktemp -d -t r27-m2-check-XXXXXXXX)   # -> /tmp/r27-m2-check-JfI3qsxZ

# A: 全状态身份核验（stdlib-only, CPU）
python tools/r27_m2_checkpoint_inventory.py --checkpoint "$CKPT" \
  --stage-evidence "$STAGE" --out "$OUTDIR/m2_inventory.json" \
  >"$OUTDIR/m2_inventory.stdout.log" 2>"$OUTDIR/m2_inventory.stderr.log"   # rc=0

# B: 真实训练帧曝光重建（只读 HDF5 索引，无 7B 前向）
python tools/r24_seen_exposure_audit.py --checkpoint "$CKPT" --hdf5-root "$HDF5" \
  --out "$OUTDIR/m2_seen_exposure.json" \
  >"$OUTDIR/m2_seen_exposure.stdout.log" 2>"$OUTDIR/m2_seen_exposure.stderr.log"   # rc=0
```

`--hdf5-root` 取值依据：`training/m2_multitask_data.py::task_dir()` 接受 DATASET ROOT
`.../data/data/hdf5`（含每任务子目录），内部下探 `<root>/<task>/arx_x5/data`。
`data/m2_splits.json`、`data/m2_task_normalizer_2task.json`、`mission/r24/frame_manifest.json`
均来自同一 SHA 的工作树，未做任何修改。

## 两个证据文件的关键字段

`m2_inventory.json`

- `status = VERIFIED_FULL_STATE_CPU_INVENTORY`
- `checkpoint.files_verified = 9`，`manifest_sha256 = df6265...a68d`（9 个成员文件逐一 SHA256 全部匹配）
- `checkpoint.run_id = m2-2task-2000-20261021T123322Z`，`global_step = 2000`，`model_kind = peft`
- `checkpoint.tasks = [push_T, stack_bowls]`，`task_ids = {push_T:0, stack_bowls:1}`
- `normalizer_sha256 = 43a166c30f7b8eac9a8bbd64d4e2e8224cc2c6e634379356136807c326f1f714`
- `split_sha256 = 0c0c65eb1aeae17a177ea5cef9c868de25442d55649c3b87f45b0583ac515218`
- `sampler.position = 2000`，`frozen_sampler_slot_pool_size = 61171`，`distinct_consumed_slot_indices = 2000`
- `sampler.actual_unique_gradient_exposed_frames = UNKNOWN_REQUIRES_FROZEN_FRAME_MAPPING`（未改写为 2000）
- `stage.status = SOURCE_FIELDS_MATCHED`，`stage_crosscheck_verified = true`
- `optimizer_steps_executed_by_this_tool = 0`

`m2_seen_exposure.json`

- `schema = r24-actual-exposure-v1`，`no_optimizer_steps = true`
- `sampler_position = 2000`，`total_training_slots = 61171`，`unique_gradient_exposed_frames = 2000`
- `frame_manifest_sha256 = 71811453c184efde635a02c07e201f05cae773f2d85ad06f7a1ad373b1db2c36`（= R24 冻结清单）
- `checkpoint_manifest_sha256 = df626500...`（与 A 独立重算一致）
- `sampler_order_hash = 7e452eb0d98e71296136fcdca57160667932e33d1fc14ca3ea8ea06de5e22181`（= trainer_state.sampler_order_hash）

| task | selected_train | seen | unseen | dev selected | dev seen（必须 0） |
|---|---|---|---|---|---|
| push_T | 64 | 2 | 62 | 68 | 0 |
| stack_bowls | 72 | 1 | 71 | 72 | 0 |
| 合计 | 136 | 3 | 133 | 140 | 0 |

被真实梯度覆盖的 3 个 R24 TRAIN 帧：

- push_T：`episode_0000010.hdf5@t=10`、`episode_0000012.hdf5@t=367`
- stack_bowls：`episode_0000001.hdf5@t=297`

## 边界与限制（不由本证据推断）

- 本证据**只证明**：checkpoint 文件身份与其自述一致、stage 字段与 checkpoint 自述一致、
  sampler 前缀在冻结池上可复算、R24 选中 TRAIN 帧中仅 3 帧被梯度覆盖、DEV 零污染。
- 本证据**不证明**：训练确实在该 GPU 上按 WAL 执行过（无不可变 WAL 交叉验证），
  也不对 Loss 收敛、H1/H2/H3/H4、是否追加训练、closed-loop success 作任何判断。
- `unique_gradient_exposed_frames = 2000` 是「被消费的 2000 个 slot 对应 2000 个互不相同的帧」，
  不等于「训练集里 2000 个不同帧各被看过一次」；epoch 暴露量 = 2000/61171 ≈ 0.0327。
- 原始 OUTDIR `/tmp/r27-m2-check-JfI3qsxZ` 保留在服务器上（未覆盖、未清理），
  本目录是同一次运行的完整拷贝（2 个 JSON + 4 份日志 + SHA256SUMS.txt）。

