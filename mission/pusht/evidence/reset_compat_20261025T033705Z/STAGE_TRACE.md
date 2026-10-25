# R30 STAGE_TRACE

## 阶段状态

| 阶段 | 状态 | 证据 |
|---|---|---|
| 1 固定基线核验（证据分支 / R29 报告 / 工具 / normalizer SHA256） | VERIFIED | `ENV.txt`、`reset_state_compat_report.json:baseline` |
| 1 14D 维度顺序确认（不靠命名推断） | VERIFIED | 数据契约 docstring；`r42_train_contract.py` 常量；本轮 A[t]==S[t+1] 复算 |
| 2 演示起始状态统计（80 TRAIN + 20 DEV） | VERIFIED | `dataset_initial_state_summary.csv`、报告 `splits.*` |
| 3 定位 R15 部署证据 | PARTIAL | 见下「已定位」与「缺失」 |
| 3 复位后 14D proprio | **NOT FOUND / UNKNOWN** | 见 `deployment_side.reset_state_14d.searched` |
| 3 第一条预测动作 | **NOT FOUND / UNKNOWN** | `deployment_side.first_predicted_action` |
| 3 第一条下发命令 | **NOT FOUND / UNKNOWN** | `deployment_side.first_commanded_action` |
| 3 控制链路源码审计 | VERIFIED | `controller_path_audit.md` |
| 4A 复位状态 vs 演示 S[0] | **BLOCKED** | 无复位状态样本 |
| 4B 首条动作语义 | **BLOCKED** | 无预测/下发记录 |
| 4C 时间对齐 | **UNKNOWN** | 无逐帧时间戳 |

## 已定位的 R15 部署证据（只读，未改动）

- `r11/runs/r15_loop_dev/loop_dev_ledger.jsonl`（13 条，episode 级）
- `r11/runs/r15_loop_dev/push_T_stage1_*.json`（10 个，episode 级：`verdict/exit/success_rate/stdout_tail`）
- `r11/logs/r15_policyserver_19000.log` / `19001` / `19002`（启动横幅、握手失败 traceback、keras 横幅）
- `r11/evidence/r15_loop_attempt2/`、`r15_loop_attempt6/`（同类日志 + `real_latency.txt` 聚合延迟）
- `r11/evidence/ep1_motion.json`（600 个标量运动幅值，**无关节向量**）

## 缺失的字段与原始位置

| 缺失字段 | 应当存在的原始位置 | 实际情况 |
|---|---|---|
| 复位后 14D proprio | RoboDojo 客户端每 episode 目录，或 policy server 请求日志 | 每 episode 目录在 `/tmp/eval_result`（已不存在）；server 不落盘请求 |
| 第一条预测动作（bin 索引 / R_hat / A_hat） | `r42_adapter.decode_command` 返回值 | 仅内存，未持久化 |
| 第一条下发命令 | 同上 `cmd` | 仅内存，未持久化 |
| 逐帧时间戳 | HDF5 与部署日志 | HDF5 无（R29 `scope.timestamps_found = []`）；日志只有 episode 级 `utc_start/utc_end` |

## 未生成的交付物

`deployment_state_action_comparison.csv` **未生成**：工单第 8 节要求「仅在原始数据足够时生成」。
复位状态、首条预测、首条命令三者均无记录，任何一行数值都只能靠补造，因此不生成、不填零。

## 边界声明

- 未训练、未推理、未加载模型权重、未初始化 CUDA、未控制机器人。`optimizer_steps = 0`。
- 未修改 HDF5 / checkpoint / normalizer / main / 任何历史证据目录。
- 未因证据缺失而启动新的闭环实验。
- R29 证据（`startup_audit_20261024T171403Z/`）原样保留，本轮只新增目录。

