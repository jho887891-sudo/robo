# R9 Gate A — 64 TRAIN-only 冻结契约（真实 HDF5 重拟合 + 独立复算）

**工作包** A ｜ **分支** `experiment/r9-freeze-64train` ｜ **基线** `review/r8-three-lane-contract-audit@5242237`
**结论**：Gate A **PASS_CPU_HDF5**（未使用 GPU，未跑任何优化步）。
**核心变化**：审计 F1（目标分位数用了 80 集、含 16 个 DEV）已修复并可复算 —— 新契约的 q01/q99 只来自 64 个 TRAIN 回合，
旧 80 集契约在新路径上被 **fail-closed 拒绝**，原始证据未被覆盖。

---

## 1. 环境与数据身份（`evidence/r9_gate_20261020/environment.json`）

| 项 | 值 |
|---|---|
| 代码 SHA | `52422372e5eacdcd0ee841ef2b91bdc5a71066c7`（未修改任何既有文件） |
| Python / 执行器 | 3.10.12 / `/home/dgut/pusht_runtime/venv_oft/bin/python` |
| torch / transformers / peft / h5py / numpy | 2.4.1+cu121 / 4.40.2 / 0.11.1 / 3.16.0 / 1.26.4 |
| GPU 状态（开始前） | A6000 49,140 MiB 中已用 **24,414 MiB**，三个外部进程 3111650 / 3451404 / 3469786 未受影响 |
| Push-T 回合 | 100 个，文件名清单 SHA256 `eb00a40d20e31c0bd1af9a307ebc8e40b2cc3f6ea6c8a0d19b8d354ada4b7207` |
| `push_T_bounds.npz` | SHA256 `5c567abcd5e6afa9e579038bb2d3cd1e9919e8e439ab4cb2ddadbcd8aa9f7855` |
| `split80.json` | SHA256 `2654e31912fd04ca403a3c03e19e2f8ad527d7d8eec3aa99305bcc2661d685f2` |
| 工作契约来源 | `evidence/r8_vla_42d_smoke_20261020/frozen_norm_contracts.json` SHA256 `3fc0c9ead108b801be797b3eecd1342f456bcdacafa1d5f854550c3a9946c40c` |

## 2. 划分（严格回合隔离）

| 集合 | 回合数 | 用途 |
|---|---|---|
| TRAIN | **64** | 唯一允许拟合 S 边界（沿用）、V/Acc `signed_motion` q99、R 的 q01/q99 |
| DEV | **16** | 只用于 A 已完成的归一化方案选择；**不参与**本次任何拟合 |
| 旧 holdout | **20** | 仅回顾性报告，不参与拟合与选择 |

* 64 TRAIN 名单 SHA256：`9216cb80e13b909e924977a43f79529ee435fa6bbe3f180c2fbce0bab619f2d7`
* 独立重算规则（`sorted(train)[::len//16][:16]` 取 DEV）与契约内 `train_episode_names` **集合完全一致**；
  三集两两无交集，DEV 回合在拟合集中出现次数 **0**。
* 拟合行数：64 TRAIN → **19,127 行**；对照 80 集 → 25,313 行；即排除了 **6,186 行（24.4%）** 的 DEV 帧。
* 帧池规则与 R7 相同：`t ∈ [2, n−2]`（`valid_frames` 去掉 t=0/1 与末帧），保证 V 与 Acc 都已定义。

## 3. 冻结量（`evidence/r9_gate_20261020/frozen_contract_64train.json`）

契约文件 SHA256 `2ab68fd34ad2bb5f8075450758f46f37b2df6d1fa5ce75d849b6fd09ff5b9f55`（4,519 字节）。

**目标分位数 q01 / q99（R = (A−S)−V，仅 64 TRAIN）**

| 维度 | q01 | q99 |
|---|---|---|
| left_arm_0 | -0.016690 | +0.013265 |
| left_arm_1 | -0.021862 | +0.021962 |
| left_arm_2 | -0.016695 | +0.017308 |
| left_arm_3 | -0.016494 | +0.016630 |
| left_arm_4 | -0.014417 | +0.012956 |
| left_arm_5 | -0.012509 | +0.010504 |
| left_gripper | -0.019672 | +0.014142 |
| right_arm_0 | -0.014066 | +0.017511 |
| right_arm_1 | -0.022911 | +0.020828 |
| right_arm_2 | -0.022729 | +0.017957 |
| right_arm_3 | -0.020034 | +0.018843 |
| right_arm_4 | -0.012456 | +0.016514 |
| right_arm_5 | -0.012892 | +0.014218 |
| right_gripper | -0.010000 | +0.010000 |

**材料性**：与旧 80 集拟合相比，单维最大相对变化 **29.29%** —— 审计 F1 不是名义问题，旧 bins 与 64-TRAIN-only bins 实质不同。

**运动缩放（沿用 A 的 DEV 决定，未重新调参）**：`normalization_mode=signed_motion`、`motion_quantile=0.99`、
`v_scale`/`a_scale` 与 `frozen_norm_contracts.json:B99_signed` **逐元素相同（差值 0.0）**；`pmin/pmax` 与 `push_T_bounds.npz` 一致。

## 4. 独立复算（`tools/r9_contract_gate.py`，不调用生成器）

直接读原始 HDF5（`state/*`、`action/*` 四组关节序列拼接成 14D）重算，9/9 PASS：

| 检查 | 结果 |
|---|---|
| 划分身份（64/16/20、无重叠、DEV 未入拟合） | PASS |
| q01/q99 由 64 TRAIN 原始数据重算 | PASS，最大偏差 q01 3.02e-09 / q99 9.54e-09 |
| 分位数对 dtype 不敏感（float32 管线 6.52e-09） | PASS |
| 归一化尺度等于 A 契约、边界等于 npz、模式 signed/q99 | PASS |
| 有符号运动尺度正定、零运动映射零 | PASS |
| 编解码往返（20 旧 holdout 帧） | PASS，见下 |
| 特征只用 S_t/S_(t−1)/S_(t−2)，对 t+1 之后置噪不敏感 | PASS（截断/置噪最大偏差 0.0） |
| 元数据完整性（mode/dim/target/codec/夹爪 6,13/边界策略/TRAIN-only） | PASS |
| 旧 80 集契约被拒绝 | PASS（退出码 1，`ValueError: R8 frozen target quantiles require 64 TRAIN / 16 DEV`） |

**编解码**（生产 codec `action_codec`，固定 256 bin）：单 bin 步长最大 **1.71859e-04**，范围内往返最大误差 **1.71646e-04**（≤ 1 步）；
越界截断比例 **1.80%**（q01/q99 设计上各切 1% 尾部，截断误差最大 0.2648，属**设计内**，不是编码缺陷）；
**可分辨运动（|R| > 1 bin）与零 bin 碰撞数 = 0**。

## 5. CPU 契约链路（与未来训练/推理同一份分位数）

`diagnostics/r8_smoke.py --check-contract` 读取**同一份**冻结契约，10/10 PASS：
特征 42D 有限、零运动槽位为 0、在线 buffer == 离线特征、`A=S+V+R̂` 重建误差 1.79e-09、
解码残差在一个 bin 内、夹爪 [6,13] 恒速规则、关节边界裁剪、元数据往返、维度不符 fail-closed。

## 6. 证据与可复算命令

```
evidence/r9_gate_20261020/
  environment.json              # 代码/包/GPU/数据哈希
  commands.tsv                  # 每条命令 + 退出码
  frozen_contract_64train.json  # 新契约（本文档所有数字的来源）
  r8_smoke_cpu.json             # CPU 契约 10 项
  gate_a_verification.json      # 独立复算 9 项 + 统计
  test_r9_contract_gate.json    # 回归 15/15（含 7 项负对照）
  logs/                         # 每步原始日志（含旧 80 集被拒的完整堆栈）
```

复算：
```bash
export R9_OUT=/home/dgut/pusht_runtime/r9_gate_<UTC>
bash scripts/r9_gate_a.sh                                   # 全流程 + 原始日志
python tools/r9_contract_gate.py \
  --contract evidence/r9_gate_20261020/frozen_contract_64train.json \
  --repo . --out evidence/r9_gate_20261020                  # 独立复算
python tests/test_r9_contract_gate.py                       # 回归 + 负对照
```

## 7. 回归与负对照（`tests/test_r9_contract_gate.py`，15/15 PASS）

除 8 项正向检查外，7 项负对照证明检查器不会假 PASS：把 DEV 回合塞进拟合名单、把 `n_train` 改回 80、
把 q01 放大 1.15 倍、把 q99 压成 q01+1e-9、删掉 `target_representation`、把夹爪维改成 [6,12]、把归一化改回 `legacy_position` —— 全部被检出。

## 8. 边界与未决

1. 本 Gate 是 **CPU/HDF5** 级别；GPU 侧消费同一份契约（`--kv-diagnose`）仍待研究负责人授权后一次运行。
2. 冻结契约 v1 与 `split80.json`、A 的 `B99_signed` **绑定**：任一方变更都必须重新生成并重跑本 Gate。
3. 旧 80 集 artifact（SHA256 `b680d702ca95ce95c8a7aba62c29c1525c424b4ab7a81cd2151ea02d5e1d8862`）保持只读；
   新路径对它 fail-closed，历史失败记录（KV 8/14）同样保留。
4. 旧 20 holdout 已被 R2–R9 反复使用：其上的任何指标只作探索性证据，不能作为最终泛化结论。
5. 本 Gate 通过 **不等于** 允许训练：训练仍需 Gate B/C 通过与现实预算授权。
