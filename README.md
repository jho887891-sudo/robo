# 纯 OpenVLA 在 RoboDojo 中的真实执行验证（外部审查包）

分支：`audit/pure-openvla-collapse`。目标：让审查者能从**训练数据 → 动作目标 → 损失 → 解码 → 仿真控制接口**完整追踪，
并复现"闭环中机器人几乎不动"这一现象的诊断。**本包不修改算法、不重训、不含权重与数据集。**

## 结论（来自实测，不含推测）

| 任务 | 回合 | 官方结果 | 运动证据 |
|---|---|---|---|
| push_T | 1（600 步） | 失败 score 0.0 | 末端总位移 1.5 mm；XY 误差恒 0.3157 m；角度恒 119.6° |
| stack_bowls | 1（800 步） | 失败 score 0.0 | 左臂 0.00 mm / 右臂 0.01 mm |
| hang_mugs | 1（800 步） | 失败 score 0.0 | 左臂 0.000 mm / 右臂 0.010 mm；mug0 位姿不变 |

三类核心证据：
1. 闭环动作链日志中 `mean|action − state| → 0.0`（模型输出恒等于当前状态）；
2. 离线同一 20 留出样本，"完全不动"（action = state）L1 = **0.0121**，优于 6000 步模型的 0.0586；
3. proprio 消融：真实状态与全零状态输出**完全相同**（proprio 未起作用）。

## 目录

```
README.md                     本文件
CHECKPOINT_IDENTITY.md        检查点路径/大小/SHA-256/步数/超参（不含权重）
ENVIRONMENT.md                驱动、容器、Python 依赖版本与自检命令
MISSING.md                    未纳入或缺失项（如实标注）
report/                       完整诊断报告（260 行）
training/                     产生 6000 步检查点的真实训练脚本 + run_meta + README（命令重建说明）
logs/                         两个阶段的原始训练日志
data/                         数据与动作表示说明 + 归一化统计量（npz、split.json；不含数据集）
diagnostics/                  离线 L1 / 恒等基线 / 本体消融 / 输出方差 / 解码校验 / 过程量汇总 脚本
deploy/                       start_ps_pure.sh（策略服务启动）
code/robodojo/                push_T_progress.py、_run_client.sh
code/xpolicylab/              policy/OpenVLA_Pure/{model.py,deploy.py,deploy.yml}
patches/robodojo_changes.patch 对 RoboDojo 上游文件的 4 处改动（git diff）
evidence/                     结果 JSON、逐步过程量、动作链日志、头部相机录像
```

## 关键复现命令

```bash
# (A) 恒等策略基线 vs 模型：离线 L1（审查重点，几分钟可跑完）
/home/dgut/pusht_runtime/venv_oft/bin/python diagnostics/identity_baseline.py

# (B) 解码正确性（需 GPU，加载 6000 步检查点）
R=/home/T7/ojh/robot_actual/RoboDojo; OFT=$R/XPolicyLab/policy/OpenVLA_OFT/openvla_oft
PYTHONPATH=$R/XPolicyLab:$OFT:$R /home/dgut/pusht_runtime/venv_oft/bin/python diagnostics/verify_decode.py

# (C) 外观 × proprio 消融
PYTHONPATH=$R/XPolicyLab:$OFT:$R /home/dgut/pusht_runtime/venv_oft/bin/python diagnostics/ablate.py

# (D) 闭环一个回合
PURE_ACTION_LOG=/tmp/pure_actions.jsonl nohup bash deploy/start_ps_pure.sh > /tmp/ps_pure.log 2>&1 &
sudo docker exec -e ROBODOJO_PROGRESS_LABELS=t,target_t rd_setup \
  bash /workspace/RoboDojo/_run_client.sh push_T OpenVLA_Pure 6020 1
```

## 下一轮最值得改的一项

把动作目标改为**相对当前 proprio 的残差 Δ = A[t] − S[t]**，按 Δ 逐维分位数归一化并分箱，使"不动"不再是最低损失解；
离线评估同时报告 `mean|Δ_pred|` 与 `mean|Δ_gt|`。
