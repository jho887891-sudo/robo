> **当前项目状态与新会话交接入口**：请先阅读 [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) 和 [AGENTS.md](AGENTS.md)。
> 本 README 记录的是 **2026-10-18 左右的历史闭环验证快照**，不代表 R44/R45 的最新研发进度。
> 当前 R44/R45 的代码与 CPU/GPU Gate 详见状态文档；不要在新对话中从这里重头开始。

---

# 纯 OpenVLA 在 RoboDojo 中的真实执行验证（提交包）

本包内容：把「纯 OpenVLA（3 相机 + 14 维 proprio，离散动作 token）接入 RoboDojo 官方评测闭环」的
代码改动、评测结果与诊断证据整理成可提交形式。

## 一句话结论

闭环链路（相机/状态/动作/归一化/解码/评测判定）已逐项验证正确；但**现有 6000 步检查点输出"保持当前姿态"
（action ≈ state），三个任务机器人都几乎不动**，因此官方成功率 0/3。问题在模型侧，不在接口侧。

## 三个任务的闭环结果（RoboDojo 原生任务判定，未改任何奖励/阈值）

| 任务 | run_id | 回合 | 结果 | 运动证据 |
|---|---|---|---|---|
| push_T | 2026-10-18_04-07-08 | 1（600 步） | 失败 score 0.0 | 末端总位移 1.5 mm；XY 误差恒 0.3157 m；角度恒 119.6° |
| stack_bowls | 2026-10-18_12-10-46 | 1（800 步） | 失败 score 0.0 | 左臂 0.00 mm / 右臂 0.01 mm；EE→bowl0 恒 0.4245 m |
| hang_mugs | 2026-10-18_12-36-41 | 1（800 步） | 见 evidence/ | 45 步内 ≤0.010 mm（其余由 hang_mugs 归档补齐） |

## 关键证据（数字均可复算）

1. **动作语义**：HDF5 `mean|A[t] − S[t+1]| = 0.0000` → 动作是"下一步绝对关节目标"；环境侧
   `robot_manager.py:399 set_joint_position_target` 一致。
2. **状态顺序**：`[left_arm(6), left_ee(1), right_arm(6), right_ee(1)]`，部署 pack 与训练 c14 **maxdiff = 0.0**。
3. **相机**：3 路、RGB、480×640、顺序一致；部署原先多一步 480×640→256×256 BICUBIC（训练没有），
   修复后模型输入张量 **maxdiff = 0.0**。
4. **离散解码**：14 token 全落在 `[BEGIN+1, BEGIN+256]` 窗口、维度齐全，真实帧反归一化 L1 = 0.0126。
5. **离线 L1 指标失效**：同一 20 留出样本，"完全不动"（action = state）L1 = **0.0121**，
   而 6000 步模型为 0.0586、3000 步 0.0779、OFT 0.1793 —— 不动优于所有模型。
6. **闭环自我锁定**：`evidence/push_T/action_chain.jsonl` 显示 delta 0.0087→0.001→0.0001 衰减，
   第 74 次调用起 `mean|action−state| = 0.0`。
7. **proprio 无效**：消融（训练图+训练状态）与（训练图+零状态）输出完全相同。

## 目录

```
report/PURE_OPENVLA_CLOSED_LOOP_REPORT.md   完整报告（260 行，含所有数值与源码位置）
patches/robodojo_changes.patch              对 RoboDojo 的 4 个文件改动（git diff）
code/robodojo/push_T_progress.py            逐步过程量模块（物体位姿/末端位姿/EE→目标距离）
code/robodojo/_run_client.sh                参数化评测入口（task/policy/port/eval_num）
code/xpolicylab/policy/OpenVLA_Pure/        纯 OpenVLA 策略服务（model.py/deploy.yml/deploy.py）
evidence/push_T/                            结果 JSON、121 行过程量、动作链日志、头部相机录像
evidence/stack_bowls/                       同上
```

## 复现（关键命令）

```bash
# 1) 策略服务（ws://localhost:6020，KV-cache 解码，实测 0.93 s/次推理）
PURE_ACTION_LOG=/tmp/pure_actions.jsonl \
  nohup bash /home/dgut/pusht_runtime/start_ps_pure.sh > /tmp/ps_pure.log 2>&1 &

# 2) 单回合闭环
sudo docker exec -e ROBODOJO_PROGRESS_LABELS=t,target_t rd_setup \
  bash /workspace/RoboDojo/_run_client.sh push_T OpenVLA_Pure 6020 1

# 3) 结果与过程量
sudo docker cp rd_setup:/tmp/eval_result /tmp/er         # result.json / 3 路 mp4
sudo docker cp rd_setup:/tmp/eval_result/_progress /tmp/erp   # progress_<run_id>.jsonl
```

## 下一轮最值得改的一项

把动作目标改为**相对当前 proprio 的残差 Δ = A[t] − S[t]**，按 Δ 的逐维分位数归一化并分箱，
使"不动"不再是最低损失解；评估同时看闭环成功率与 `mean|Δ_pred|`（应与 `mean|Δ_gt|` 同量级）。

## 环境

jxxy：RTX A6000 48GB / 驱动 580.178.04 / Isaac Sim 5.1 容器 `rd_setup` / PyTorch 2.4.1 / transformers 4.40.2 / peft 0.11.1。
