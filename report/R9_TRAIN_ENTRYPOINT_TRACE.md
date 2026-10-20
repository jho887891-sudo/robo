# R9 Gate B — 标准训练入口的 42D/[S,V,Acc] + R 链路 trace（CPU，零优化步）

**工作包** B ｜ **分支** `feature/r9-vla-trainer-contract` ｜ **基线** `review/r8-three-lane-contract-audit@5242237`
**结论**：Gate B **PASS_CPU**（11/11），其中真实 7B 前向/反向/loss 属于 **PENDING_GPU**，本轮**没有**运行、也不声称通过。
**关键约束**：`training/train_pushT_pure2.py` **未改动一行**（SHA256 `e8813558…`）；新增的是显式 42D 入口与契约层。

---

## 1. 标准入口现在做什么（可核对，非推测）

| 环节 | 现状（`train_pushT_pure2.py`） | R9 要求 |
|---|---|---|
| proprio | `obs()` 只取 `S_t`，14 维，位置边界归一化 | 42 维 `[S,V,Acc]`，V/Acc 用 signed_motion |
| 目标 | `AT.ABSOLUTE` / `AT.RESIDUAL`（A 或 A−S） | `R=(A−S)−V` |
| 分位数 | `action_stats(split["train"])` = 80 集 | 64 TRAIN 冻结 q01/q99 |
| 帧采样 | `t = randrange(0, T-1)`（t=0,1 无历史） | t ≥ 2，离线 t ∈ [2, n−2] |
| 投影层 | `ProprioProjector(proprio_dim=14)` | 42 → llm_dim |
| checkpoint 元数据 | variant/cams/proprio/q01/q99 | 另加归一化模式、codec、split/contract 哈希 |

以上 6 条由 `r9_trainer_contract.standard_entrypoint_gaps()` 以机器可读形式随 trace 一起存档。

## 2. 新增代码

| 文件 | 作用 | 状态 |
|---|---|---|
| `training/r9_trainer_contract.py` | 42D/R 契约层：64/16/20 划分、history-safe 帧池、42D 特征、R 目标、冻结 bins、解码、**fail-closed 守卫** | VERIFIED_CPU |
| `training/r9_train_pushT_svacc.py` | 新 42D 入口（不改旧入口）：`--selfcheck` 走完整数据链路；训练需 `R9_TRAINING_AUTHORIZED=1` | IMPLEMENTED（自检 PASS；**从未训练**） |
| `tools/r9_trainer_trace.py` | Gate B trace harness（复用真实入口模块的 `c14`/`make_ids`/`action_stats`/`norm_pr`） | VERIFIED_CPU |

## 3. Trace 结果（11/11 PASS，`evidence/r9_trainer_trace_20261020/r9_trainer_trace.json`）

| 检查 | 标签 | 关键数字 |
|---|---|---|
| B1 真实入口被导入 | VERIFIED_CPU | 模块 SHA256 `e8813558…`，`CAMS3` 三路相机 |
| B2 旧 14D 路径未变 | VERIFIED_CPU | q01/q99 形状 (14,)、`norm_pr` (14,)；2 集小样本下 `left_gripper` 的 q99==q01（小样本退化） |
| B3 真实 prompt/label 构造 | VERIFIED_CPU | 监督位 **15**（14 动作 + stop=2），prompt **36** 位全部 IGNORE，token 落在 **[31859, 31892]** 合法窗口 |
| B4 42D 特征（真实 HDF5） | VERIFIED_CPU | 12 行、形状 (42,)、与手工重算 **最大偏差 0.0** |
| B5 目标恒等式 | VERIFIED_CPU | `R == AT.RESIDUAL − V`，偏差 0.0 |
| B6 解码到物理命令 | VERIFIED_CPU | 范围内误差 **1.5926e-04** ≤ 一个 bin（1.7186e-04）；夹爪 [6,13] 取 `S+V`；命令在关节界内；该帧物理 L1 4.64e-05 |
| B7 零更新 loss trace（MOCK） | MOCK_FORWARD | loss 5.4939（≈ln256=5.545），14 个监督位，梯度绝对值合计 39.51，**优化步 0**，CPU，CUDA 未初始化；真实 42D 投影层参数量（llm 4096）**16,957,440** |
| B8 特征不吃未来帧 | VERIFIED_CPU | 把 t+1 以后置噪，特征偏差 **0.0** |
| B9 fail-closed | VERIFIED_CPU | 旧 80 集契约/14 维投影/t=1/ABSOLUTE 目标 4 项全部被拒 |
| B10 CUDA 专属范围声明 | PENDING_GPU | `obs()` 第 123/126 行、`build()`、`evaluate()` 需 CUDA；真实 7B 前向/反向/loss **未运行** |
| B11 入口常量审计 | **HYPOTHESIS** | 见 §4 |

## 4. 新发现的开放风险（HYPOTHESIS，不是结论）

`prismatic/vla/constants.py` 用 **`sys.argv` 子串**判断平台，而入口把 `sys.argv` 设成 `["pure2","arx_goai_12task_r0"]`，
因此本机实际取的是 **LIBERO 常量**：`ACTION_DIM=7`、`NUM_ACTIONS_CHUNK=8`、`PROPRIO_DIM=8`，
而 codec/数据是 **14 维**。

* 旧入口 `evaluate()` 里存在 `for k in range(ACTION_DIM):` —— 在该环境下只解码 7 个 token，
  与 14 维 codec 不一致（是**风险假设**：历史 eval 日志是否因此受影响，需单独核对，本轮不下结论）。
* 新的 42D 入口**完全不引用**该常量（源码扫描确认），固定解码 14 个 token，因此不受此风险影响。
* 在任何 GPU 训练/评测前，必须显式断言这三元组与本任务一致；否则 `physical_L1` 之类指标不可比。

## 5. 入口自检与授权守卫

* `r9_train_pushT_svacc.py --selfcheck`：8 帧 × 2 个 TRAIN 回合，42D 特征/合法 token/解码误差/夹爪策略/关节界全部 PASS
  （`evidence/r9_trainer_trace_20261020/entrypoint_selfcheck.json`）。
* 无授权尝试训练：`--steps 1` → 退出码 1，`refusing to train: R9_TRAINING_AUTHORIZED is not set by the research lead`，
  且**未创建任何输出目录**（`authorization_guard.log`）。

## 6. 回归（16/16 PASS，`tests/test_r9_trainer_contract.py`）

覆盖：64/16/20 划分、拒绝旧 80 集契约、拒绝 legacy 归一化、帧池从 2 开始、拒绝 t<2、
42D 特征与手工重算一致、R 恒等、token 合法、解码往返（合成数据严格落在关节界内，误差 0.0）、
夹爪策略、拒绝 14 维投影、拒绝 ABSOLUTE 目标、集成元数据完整性、入口差异清单。

## 7. 边界

1. **Mock forward**：B7 的 forward 是声明的 MOCK（`MOCK_FORWARD`），只验证 loss/label/梯度可达性，
   不代表 7B 上可训练；真实 7B 42D 前向/反向仍 PENDING_GPU。
2. **未训练**：新入口从未执行任何优化步；训练需研究负责人授权 + 独立 GPU 窗口。
3. **旧 20 holdout** 已反复使用，任何指标只作探索性证据。
4. 常量三元组（§4）在 GPU 训练前必须显式断言，否则指标不可比。
5. 本 Gate 通过 **不等于** 允许训练：仍需 Gate C（KV）判定与预算授权。
