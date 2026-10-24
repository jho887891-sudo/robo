# R29 STAGE_TRACE — 真实执行阶段记录（并更正 R28 的一处表述错误）

## 1. 更正：R28 并非「未读取 HDF5」

R28 的 `DIAGNOSIS.md` 写的是「失败发生在读取 HDF5/RGB **之前**」。这句话是错的，本轮更正为：

> R28 **已经完成** Push-T 全部 80 个回合的 HDF5 索引与结构检查（含每回合首帧三路 JPEG 的 SOI/EOI 校验），
> 之后才在 `run_audit()` 加载 splits/normalizer 契约时因模块绑定错误退出。

依据（可复核，非推断）：

1. 原始 traceback 的抛错点是 `tools/pusht_startup_data_audit.py:378`（`M2.load_m2_artifacts`），
   而 `M2.load_task_index("push_T", ...)` 在**第 371 行**，先于 378 行执行；异常出现在 378 行即证明 371 行已成功返回。
   原始 traceback 仍在 `mission/pusht/evidence/startup_audit_20261024T131007Z/stderr.log`（sha256 `12f5fa68…79174`，未改动）。
2. `training/m2_multitask_data.py:234 load_task_index()` 对每个 episode 调用 `_inspect_episode()`，
   后者在 `training/m2_multitask_data.py:209` 校验相机首帧是否为 JPEG（SOI/EOI）。
   因此 R28 实际读过全部 Push-T HDF5 文件的结构与首帧 JPEG 标记。

R28 当时 stdout 为空，是**因为该工具在成功路径上不打印中间进度**，不代表没有读取数据。

## 2. R28 实际到达的阶段（据原始 traceback 与代码顺序判定）

| 阶段 | R28 状态 | 依据 |
|---|---|---|
| splits JSON 读取与 TRAIN/DEV 回合集合校验 | 完成 | 第 360–369 行先于 378 行，未抛错 |
| HDF5 回合索引 + 结构 + 首帧 JPEG SOI/EOI | 完成 | 第 371 行先于 378 行，未抛错 |
| splits/normalizer 契约加载（`load_m2_artifacts`） | **失败** | 第 378 行 AttributeError |
| checkpoint manifest / trainer_state 验证 | 未执行 | 在第 378 行之后 |
| sampler 前缀重建 | 未执行 | 同上 |
| A[t] vs S[t+1] 对齐统计 | 未执行 | 同上 |
| TRAIN-SEEN / TRAIN-UNSEEN / DEV 统计 | 未执行 | 同上 |
| RGB 解码与帧间像素差分析 | 未执行 | 同上 |

## 3. R29 本轮实际到达的阶段

| 阶段 | R29 状态 | 证据 |
|---|---|---|
| 固定版本核验（evidence 7054b7e / fix 9b94823 / main 329614f） | 完成 | `ENV.txt` |
| 独立 worktree + `--no-ff` 合并修复分支，无冲突 | 完成 | `ENV.txt` 的 `worktree_head` |
| CPU 回归测试（7 tests） | 完成，rc=0 | `unit_test.log` |
| splits/normalizer 契约加载 | 完成 | `report.json` → `scope.split_file_sha256` / `normalizer_file_sha256` |
| HDF5 回合索引 + 结构 + 首帧 JPEG 校验 | 完成 | `report.json` → `episodes[100]`、`integrity_counts` |
| checkpoint manifest / trainer_state 验证 | 完成 | `scope.checkpoint_manifest_sha256 = df6265…` |
| sampler 前缀重建 | 完成 | `scope.sampler_position = 2000`、`sampler_pool_size = 61171`、`unique_consumed_frame_keys_all_tasks = 2000` |
| A[t] vs S[t+1] 对齐统计 | 完成 | `integrity_counts.action_alignment_episodes_over_atol = []`（atol 1e-6） |
| TRAIN-SEEN / TRAIN-UNSEEN / DEV 统计 | 完成 | `frame_group_metrics`（6 组） |
| RGB 解码与帧间像素差分析 | 完成 | `image_integrity_by_split_and_camera` |
| 报告生成 | 完成 | `report.json`，`status = DATA_INTEGRITY_PASS` |

## 4. 边界

- 本轮未修改任何源码、splits、normalizer、HDF5 或 checkpoint；修复仅来自 `origin/audit/pusht-startup-import-fix-20261024` 的既有提交。
- 未加载模型权重、未初始化 CUDA（`no_cuda_initialized = true`、`no_model_loaded = true`）、`no_optimizer_steps = 0`。
- 本文件只陈述执行事实与可达阶段，不解释数据含义，不做训练建议。

