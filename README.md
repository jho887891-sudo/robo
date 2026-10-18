# 纯 OpenVLA 在 RoboDojo 中的真实执行验证（外部审查包）

分支：
- `audit/pure-openvla-collapse` — 第一轮审查包（诊断：为什么闭环里机器人几乎不动）
- `audit/r2-action-codec-fix` — 第二轮：针对第一轮意见的**最小修复 + 可复现性整理**
  （修复动作离散化边界、统一离线评测、部署指令契约、本体消融、SHA 核对、动态模块收录、学习率取证）
- `audit/r3-autoregressive-mask` — 第三轮：**自回归动作输入掩码**
- `audit/r3-autoreg-training-fix` — 第三轮（正式交付分支）：自回归动作输入掩码的根因取证 + 最小修复 + A/B/C/D 四阶段验证
  （B 训练/推理一致性、C 梯度检查、D 32 样本受限过拟合；报告 report/AUDIT_R3_AUTOREG_RESULTS.md 与 report/MORNING_SUMMARY.md）
  （训练时 14 个动作输入位置被置零 → 训练/部署机制不一致；已加开关并验证，详见 report/AUDIT_R3_AUTOREGRESSIVE_MASK.md）

目标：让审查者能从**训练数据 → 动作目标 → 损失 → 解码 → 仿真控制接口**完整追踪，
并复现"闭环中机器人几乎不动"这一现象的诊断。**本包不修改算法、不重训、不含权重与数据集。**

第二轮追加的硬约束同样成立：**没有重训、没有改旧检查点、没有把动作改成残差、没有修改主分支，
也没有任何"已恢复任务执行能力"的结论**（详见 `report/AUDIT_R2_FIXES.md` 第 6 节）。

## 结论（来自实测，不含推测）

| 任务 | 回合 | 官方结果 | 运动证据 |
|---|---|---|---|
| push_T | 1（600 步） | 失败 score 0.0 | 末端总位移 1.5 mm；XY 误差恒 0.3157 m；角度恒 119.6° |
| stack_bowls | 1（800 步） | 失败 score 0.0 | 左臂 0.00 mm / 右臂 0.01 mm |
| hang_mugs | 1（800 步） | 失败 score 0.0 | 左臂 0.000 mm / 右臂 0.010 mm；mug0 位姿不变 |

三类核心证据：
1. 闭环动作链日志中 `mean|action - state| -> 0.0`（模型输出恒等于当前状态）；
2. 离线同一 20 留出样本，"完全不动"（action = state）L1 = **0.0121**，优于 6000 步模型的 0.0586；
3. proprio 受控消融（第二轮修正）：把本体状态置零后动作**塌缩到 ≈0**，14 步里只有 6 个 token 与真实状态
   相同；即模型学到的是 **A ≈ S（复制当前状态）**，而不是“忽略本体状态”。
   （第一轮“proprio 未起作用”的表述**复现不出**，详见 report/AUDIT_R2_FIXES.md 第 4 节。）

> 注意：stack_bowls / hang_mugs 的失败**不是跨任务泛化证据**——该检查点只见过 push_T 数据，
> 且部署时的任务指令是固定的 push_T 指令（`code/xpolicylab/policy/OpenVLA_Pure/deploy_contract.py`）。

## 目录

```
README.md                     本文件
CHECKPOINT_IDENTITY.md        检查点路径/大小/SHA-256/步数/超参 + 部署侧加载核对（不含权重）
ENVIRONMENT.md                驱动、容器、Python 依赖版本与自检命令
MISSING.md                    未纳入或缺失项（如实标注）
report/                       完整诊断报告 + 第二轮 AUDIT_R2_FIXES.md + 第三轮 AUDIT_R3_AUTOREGRESSIVE_MASK.md
training/                     真实训练脚本 + run_meta + README（命令重建说明、学习率调度更正）
logs/                         两个阶段的原始训练日志
data/                         数据与动作表示说明 + 归一化统计量（npz、split.json；不含数据集）
tests/                        动作编解码 / 部署指令契约 / 动作输入掩码契约的单元测试（纯 CPU）
diagnostics/                  离线评测 / 单元基线 / 本体消融 / 解码校验 / 学习率取证 / 身份核对
deploy/                       start_ps_pure.sh（策略服务启动）
code/robodojo/                push_T_progress.py、_run_client.sh
code/xpolicylab/              policy/OpenVLA_Pure/{model.py,action_codec.py,deploy_contract.py,deploy.py,deploy.yml}
code/openvla_dynamic_modules/ 实际被 import 的 HF 动态模块（modeling/configuration/processing + SHA-256）
patches/                      对 RoboDojo 上游的改动 + 第二轮代码改动的完整 diff
evidence/                     结果 JSON、逐步过程量、动作链日志、头部相机录像
```

## 关键复现命令

```bash
# (A) 动作编解码边界单元测试（纯 CPU，无需 GPU / 数据集）
python -m unittest tests.test_action_codec -v        # 20 项
python -m unittest tests.test_deploy_contract -v     # 8 项

# (B) 真实标签合法性（需要数据集；缺数据时自动 skip，skip 不算通过）
PUSHT_HDF5=<push_T hdf5 目录> python -m unittest tests.test_label_dataset -v

# (C) 统一离线评测（需 GPU；三路相机 + 14 维状态，模型 vs 恒等策略）
python diagnostics/eval_unified.py --n 20 --out /tmp/r2_eval.json

# (D) 本体状态消融（需 GPU；固定图像与指令，14 维全量 + token 概率 + 投影特征）
python diagnostics/ablate_proprio.py --out /tmp/r2_ablate.json

# (E) 学习率调度取证（纯 CPU，读日志）
python diagnostics/lr_schedule_forensics.py --log logs/pure3c_cont.log --stage 2 --claimed-milestone 1000

# (F) 部署权重与 CHECKPOINT_IDENTITY.md 的 SHA-256 核对
python diagnostics/check_checkpoint_identity.py

# (G) 恒等策略基线（第一轮，纯 CPU）
python diagnostics/identity_baseline.py
```

## 下一轮最值得改的一项

把动作目标改为**相对当前 proprio 的残差 `Δ = A[t] - S[t]`**，按 Δ 逐维分位数归一化并分箱，
使"不动"不再是最低损失解；离线评估同时报告 `mean|Δ_pred|` 与 `mean|Δ_gt|`。
（本轮**未**做此改动，这是刻意的：本轮只修正确性缺陷。）
