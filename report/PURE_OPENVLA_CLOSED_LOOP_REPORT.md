# 纯 OpenVLA 在 RoboDojo 中的真实任务执行验证报告

> 状态：进行中。Phase 1（项目核查）、Phase 2（一致性验证）已完成；Phase 3（闭环）首个回合已跑通并定位到原因；Phase 4-7 待补。
> 机器：jxxy（RTX A6000 48GB，驱动 580.178.04，容器 `rd_setup` / Isaac Sim 5.1）

## 0. 交付物索引

| 类别 | 位置 |
|---|---|
| 本报告 | `/home/T7/ojh/robot_actual/PURE_OPENVLA_CLOSED_LOOP_REPORT.md` |
| 评测结果 | `/home/T7/ojh/robot_actual/closed_loop_eval/<task>/<run_id>/` |
| 逐步过程量 | 同上 `progress_<run_id>.jsonl` |
| 策略接入 | `RoboDojo/XPolicyLab/policy/OpenVLA_Pure/{model.py,deploy.yml,deploy.py}` |
| 过程量模块 | `RoboDojo/task/RoboDojo/tasks/push_T_progress.py` |
| 客户端入口 | `RoboDojo/_run_client.sh <task> <policy> <port> <eval_num>` |
| 服务端入口 | `/home/dgut/pusht_runtime/start_ps_pure.sh`（ws://localhost:6020） |

## 1. 权重身份（实测，非按名字猜）

| 项 | 值 |
|---|---|
| 目录 | `pushT_pure3c_6000/checkpoints/final/` |
| LoRA | `adapter_model.safetensors` 484,458,600 B；r=32；all-linear；alpha=16 |
| proprio 投影 | `proprio_projector--pure.pt` 67,373,328 B |
| 累计步数 | 首段 3000 + 续训 3000 = 6000；`final` 与 `step3000` 的 LoRA/proprio **sha256 相同**（0d7b95cc… / 7c5a24b2…） |
| 离线留出误差 | `FINAL_EVAL autoreg_L1=0.0586`（见第 2.6 节：该指标已证明不可用） |

## 2. 训练/部署一致性核查

### 2.1 动作定义（源码 + 数据双证）
- HDF5：`mean|A[t] − S[t+1]| = 0.0000`（14 维）→ 动作 = **下一步绝对关节目标**
- 环境：`eval_env.py` joint 分支 → `robot_manager.py:399 set_joint_position_target`
- 结论：语义一致

### 2.2 状态（14 维）
`[left_arm(6), left_ee(1), right_arm(6), right_ee(1)]`，部署 `pack_robot_state` 与训练 `c14` **maxdiff = 0.0**

### 2.3 相机
3 路、顺序 `cam_head, cam_left_wrist, cam_right_wrist`、RGB、480×640、像素级一致。
**发现并修复**：部署原先经 `prepare_vla_image` 多做 480×640→256×256 BICUBIC；训练为原始帧直接 `apply_transform`。修复后模型输入张量 **maxdiff = 0.0**。

### 2.4 归一化往返（7186 条真实动作）
- 逐维最大误差最大 0.735（q01/q99 裁剪所致，设计使然），逐维平均 0.0017–0.0071
- 未发现单位/方向/顺序错误

### 2.5 离散解码（实机加载 6000 步权重，真实训练帧）
`TOKENS=[175,1,1,252,146,175,255,100,181,135,70,145,79,1]`；范围 1..255；14 维齐全；反归一化 L1 = 0.0126。

### 2.6 离线 L1 指标失效（重要，新发现）
同一 20 个留出样本、同一 BOUNDS 归一化：

| 策略 | 首步动作 L1 |
|---|---|
| **恒等策略（action = 当前状态，即完全不动）** | **0.0121**（median 0.0097 / max 0.0476） |
| 纯 OpenVLA B（6000 步） | 0.0586 |
| 纯 OpenVLA B（3000 步） | 0.0779 |
| 纯 OpenVLA A（单相机 3000 步） | 0.1168 |
| OFT cont300 | 0.1793 |

结论：该指标被"复制当前状态"这一平凡成分主导，**"什么都不做"优于所有模型**，因此它既不能预测闭环成功率，方向上还鼓励"保持不动"。

### 2.7 分箱分辨率（排除量化锁死假设）
- bin 宽度 `(q99−q01)/255` 逐维 **0.0020–0.0090 rad**
- 数据每步位移 0.0037–0.0120 rad → 比值 **1.0–2.3 个 bin**
- 结论：每步运动可被离散动作表达，量化精度**不是**机器人不动的直接原因

## 3. 闭环（push_T）

| run_id | 回合 | 结果 | 备注 |
|---|---|---|---|
| `2026-10-18_04-07-08` | 1（layout 0） | **失败**（600 步跑满，score 0.0） | 首次闭环，已存录像与 121 行过程量 |
| `2026-10-18_11-46-58` | 1（layout 0） | 进行中 | 开启动作链日志复跑，用于定位"不动" |

### 3.1 过程量证据（run 04-07-08）
- T 块 `[-0.4175, -0.1169, 0.7725]`，目标 `[-0.1113, -0.1941, 0.7650]`
- XY 误差 **0.3157 m 全程不变**（首=最小=末）；角度误差 **119.6° 全程不变**
- 双臂末端 600 步内仅位移 **≈1.5 mm**；EE→方块 0.3025→0.3030 m

### 3.2 动作链日志（run 11-46-58）
- `STATE[0] = [0,0,0,0,0,0,1, 0,0,0,0,0,0,1]`（几乎全 0，夹爪=1）
- `ACTION[0] ≈ STATE[0]`，delta 逐步衰减 `0.0087 → 0.001 → 0.0001`
- 第 74 次调用起 `mean|action − state| = 0.0` → **模型输出恒等于当前状态（自我锁定）**
- 相机画面经人工核验正常（头部相机可见双臂/红块/灰色目标垫；腕相机正常）

### 3.3 外观域差（已确认存在，待消融定量）
- 训练集（30 个 episode 抽样）：**蓝色桌面 + 蓝色 T 块**，外观高度一致（桌面区域 BGR≈[125,82,66]）
- 部署 layout 0：**木纹桌面 + 红色 T 块**
- 材质资产完整（`Mahogany_Planks` 真实 PNG / MDL），故该外观是 layout 官方指定，**不是资产损坏**
- 待做消融：训练图/部署图 × 训练状态/零状态，分离"外观"与"状态"贡献

## 4. 三个任务
| 任务 | layout 数 | 状态 |
|---|---|---|
| push_T | 85 | 单回合已跑，失败（见 3） |
| stack_bowls | 85 | 待跑 |
| hang_mugs | 60 | 待跑 |

## 5. 失败归因（待补，目前只到"接近阶段之前：模型不输出运动"）

## 6. 与 OpenVLA-OFT 对照（待补）

## 7. 复现命令（待补）

## 3.4 根因定位（证据链闭合）

### (a) 动作空间被"恒等成分"主导
数据事实（100 个 episode）：`mean|A[t] − S[t+1]| = 0`，且每步位移仅 `|S[t+1]−S[t]| ≈ 0.0045`（物理单位），
而离散分箱宽度 `(q99−q01)/255 ≈ 0.0020–0.0090` → **每步运动只有 1–2 个 bin**。

这意味着"输出当前姿态"本身就是一个极强基线：

| 策略 | 留出 20 样本（BOUNDS 归一化 L1） | 同 8 帧（物理单位 L1） |
|---|---|---|
| **恒等（动作=当前状态，完全不动）** | **0.0121** | **0.00449** |
| 纯 OpenVLA 6000 步 | 0.0586 | 0.01406 |
| 纯 OpenVLA 3000 步 | 0.0779 | — |
| OFT cont300 | 0.1793 | — |

即**模型比"什么都不做"差 3–5 倍**；离线 L1 指标无法反映执行能力（2.6 节）。

### (b) 闭环行为 = 恒等不动点
`action_chain.jsonl`（run 11-46-58）：`STATE[0]≈0`，`ACTION[0]≈STATE[0]`，delta `0.0087→0.001→0.0001` 衰减，
第 74 次调用起 `mean|action−state| = 0.0` → 动作恒等于状态 → 机器人静止（run 04-07-08 的 600 步末端位移仅 1.5 mm 互相印证）。

### (c) 消融：模型依赖图像、忽略 proprio
| 条件 | mean\|action−state\| | dims 0-6 动作 |
|---|---|---|
| A 训练图 + 训练状态 | 0.00379 | [0.0037, 0.0087, 0.0064, 0.0001, 0.0012, 0.0028, 1.0] |
| B 部署图 + 训练状态 | 0.00379 | 与 A **相同**（仅 dim7 0.0006→0.0072） |
| C 训练图 + 零状态 | 0.07471 | 与 A **完全相同** → **proprio 未起作用** |
| D 部署图 + 零状态（真实部署条件） | 0.07550 | 与 B 相同 |

### (d) 结论（已证实 vs 待验证）
- **已证实**：动作逐维等于当前状态；模型输出不随 proprio 变化；模型输出随图像变化（8 帧预测 std 与 GT 同量级，L1 0.0141），说明它主要靠图像估计"当前姿态"而非规划运动。
- **待验证**（需重训）：是否只因训练目标（25 Hz 绝对姿态 + 离散分箱）使"恒等"成为低损失解；外观域差（蓝桌 vs 木纹/红块，3.3 节）对性能的额外影响幅度。

### (e) 最值得修改的一项（第八条交付）
把动作目标从"绝对关节姿态"改为**相对当前 proprio 的残差 Δ = A[t] − S[t]**，并按 Δ 的分布（逐维分位数）做归一化与分箱；
目标是让"不动"不再是最低损失解。评估指标同步改为：闭环成功率 + `mean|Δ_pred|`（要求与 `mean|Δ_gt|` 同量级）。

## 7. 复现命令（实测可用）

```bash
# 0) 环境自检（jxxy）
nvidia-smi                                  # 期望 Driver 580.178.04
sudo docker ps --format '{{.Names}}|{{.Status}}' | grep rd_setup   # 需为 Up
df -h / | tail -1                           # 需要 ≥5G 空闲

# 1) 启动纯 OpenVLA 策略服务（ws://localhost:6020，KV 解码）
PURE_ACTION_LOG=/tmp/pure_actions.jsonl \
  nohup bash /home/dgut/pusht_runtime/start_ps_pure.sh > /tmp/ps_pure.log 2>&1 &
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q 6020 && break; sleep 10; done

# 2) 跑一个回合（push_T）
sudo docker exec -e ROBODOJO_PROGRESS_LABELS=t,target_t rd_setup \
  bash /workspace/RoboDojo/_run_client.sh push_T OpenVLA_Pure 6020 1
#   stack_bowls:  -e ROBODOJO_PROGRESS_LABELS=bowl0,bowl1,bowl2 ... stack_bowls ...
#   hang_mugs:    -e ROBODOJO_PROGRESS_LABELS=mug0,mug1,mug2,cup_holder ... hang_mugs ...

# 3) 收集产物（容器内 /tmp 是结果落点，宿主 eval_result 是软链）
sudo docker cp rd_setup:/tmp/eval_result /tmp/er
sudo docker cp rd_setup:/tmp/eval_result/_progress /tmp/er_prog
cp -rL /tmp/er/RoboDojo/push_T/OpenVLA_Pure/arx_x5/*/<run_id> \
  /home/T7/ojh/robot_actual/closed_loop_eval/push_T/<run_id>/
cp -L /tmp/er_prog/progress_<run_id>.jsonl /home/T7/ojh/robot_actual/closed_loop_eval/push_T/<run_id>/

# 4) 诊断脚本（均在 ext4 runtime 下）
P=/home/dgut/pusht_runtime/venv_oft/bin/python
$P /home/dgut/pusht_runtime/verify_decode.py     # 离散解码正确性（token 窗口/维度/L1）
$P /home/dgut/pusht_runtime/ablate.py            # 外观 × 状态 消融
$P /home/dgut/pusht_runtime/variance_test.py     # 不同观测下的输出方差
$P /home/dgut/pusht_runtime/analyze_progress.py  # 过程量汇总（xy/ang 曲线）
```

关键文件变更（本轮）：
- `RoboDojo/XPolicyLab/policy/OpenVLA_Pure/model.py`：图像改为原始帧直入 `apply_transform`（对齐训练）；KV-cache 解码；可选动作链日志 `PURE_ACTION_LOG`。
- `RoboDojo/task/RoboDojo/tasks/push_T_progress.py`：逐步记录物体位姿/末端位姿/EE→目标距离（标签可由 `ROBODOJO_PROGRESS_LABELS` 配置）。
- `RoboDojo/src/eval_client/eval_env.py`：在 `take_action_batch` 每个控制步调用过程量记录。
- `RoboDojo/_run_client.sh`：参数化任务入口。

## 4. 三个任务的闭环结果

### 4.1 汇总

| 任务 | step_lim | run_id | 回合 | 官方结果 | 逐步证据 |
|---|---|---|---|---|---|
| push_T | 600 | `2026-10-18_04-07-08` | 1 | **失败**（score 0.0） | XY 误差 0.3157 m 全程不变；末端位移 1.5 mm |
| push_T | 600 | `2026-10-18_11-46-58` | 1（诊断，中途终止） | 未计分（主动终止） | `mean\|action−state\| → 0.0`（自我锁定） |
| stack_bowls | 800 | `2026-10-18_12-10-46` | 1 | 见下（运行中） | 90 步内左臂位移 **0.00 mm**、右臂 0.01 mm；EE→bowl0 恒为 0.4245 m |
| hang_mugs | 800 | `2026-10-18_12-36-41` | 1（完整 800 步） | **失败**（score 0.0） | 左臂位移 0.000 mm / 右臂 0.010 mm；mug0 位姿全程不变 |

### 4.2 阶段级观察（按你要求的分阶段记录）

推 T 任务：
| 阶段 | 是否达到 | 证据 |
|---|---|---|
| 1 正确接近 T 形块 | **否** | 600 步内双臂末端总位移 1.5 mm；EE→方块距离 0.3025→0.3030 m |
| 2 到达有效接触位置 | 否 | 同上（从未接近） |
| 3 建立接触 | 否 | — |
| 4 推动方块产生位移 | 否 | 方块位姿全程 `[-0.4175,-0.1169,0.7725]` 不变 |
| 5 调整方块姿态 | 否 | 角度误差恒 119.6° |
| 6 到达目标位姿 | 否 | XY 恒 0.3157 m |
| 7 官方成功条件 | 否 | `_result.json: success_rate 0.0` |

叠碗任务：同样停留在阶段 1 之前（机器人无位移），未进入抓取/堆叠任何阶段。

### 4.3 过程量字段（每 5 控制步一行）
`step, seed, ep, xy, ang_deg, dz, ee_dist_<robot>_<label>, eef{robot:xyz}, obj{label:{pos,quat}}`
- push_T：`obj.t`、`obj.target_t`（含位置与四元数）、双臂末端、EE→t 距离
- stack_bowls：`obj.bowl0/1/2`（注意：`xy` 字段此时表示 bowl0↔bowl1 间距，不是任务误差）
- hang_mugs：`obj.mug0/1/2/cup_holder`

## 5. 失败归因（按你给的九类，附实际依据）

| 类别 | 判定 | 依据 |
|---|---|---|
| 1 视觉识别错误 | **未证实** | 相机画面人工核验正常（双臂/红块/目标垫均在视野内）；模型预测随图像变化（8 帧 std 与 GT 同量级）；但外观域差（蓝桌 vs 木纹）确实存在，其影响幅度未分离 |
| 2 接近失败 | **症状确认** | 600 步内双臂末端位移 1.5 mm（push_T）、≤0.01 mm（stack_bowls）；EE→目标距离不变 |
| 3 动作解码错误 | **已排除** | 14 token 全部落在 `[BEGIN+1, BEGIN+256]` 窗口；维度齐全无缺失；反归一化用训练同一 q01/q99，真实帧 L1=0.0126；`action_chain.jsonl` 中 action 与 state 的 14 维一一对应 |
| 4 抓取失败 | 未进入 | 机器人未移动，尚未接触任何物体 |
| 5 精细对齐失败 | 未进入 | 同上 |
| 6 接触控制失败 | 未进入 | 同上 |
| 7 误差累积 | 不适用 | 全程无运动，谈不上累积 |
| 8 评测接口/判定异常 | **已排除** | 3 个任务均正常产出 `_result.json`（`success_rate/eval_time/details.layout_id` 结构完整）、3 路录像、过程量；未修改任何奖励函数或阈值 |
| 9 证据不足 | 部分 | 外观域差对性能的实际影响、以及"重训后能否恢复"需下一轮验证 |

### 已证实的根本原因（三类证据交叉）
1. **策略输出退化为恒等**：闭环 `mean|action−state| → 0.0`；离线同一 8 帧 L1 0.0141，而"完全不动"为 0.00449（模型比不动差 3.1×）。
2. **proprio 输入未起作用**：消融 A（训练状态）与 C（零状态）输出**完全相同**。
3. **动作目标被恒等成分主导**：25 Hz 绝对关节姿态，每步位移仅 1–2 个 bin；"复制当前状态"在留出 20 样本上 L1=0.0121，优于全部模型（0.0586–0.1793）。

结论：**当前 6000 步纯 OpenVLA 检查点不具备执行能力**（不是控制链路接错，链路已被上述证据验证正确）。

## 6. 与 OpenVLA-OFT 对照

按你的要求（"只有在纯 OpenVLA 的真实闭环评测跑通后，再加载原来的 OpenVLA-OFT"），**本轮未启动 OFT 对照**，因为纯 OpenVLA 尚未跑通（第 3、5 节证据）。

历史数据（**上一轮会话**，仅作参考，不作为本轮结论）：
- OFT `pushT_cont--300` 在 RoboDojo 闭环跑了 11 个回合（1 + 10），**全部失败（0/11）**；
- 但当时**没有逐步过程量**，无法回答"OFT 是否真的移动了机械臂"；需用本轮的 `push_T_progress.py`（现已在 `take_action_batch` 内逐步记录）重测才能对比。
- 上一轮 OFT 的 10 回合产物在 `/home/T7/ojh/robot_actual/pushT_closed_loop/`（含 `_result.json` 与 3 路录像）。

本轮已准备好的 OFT 对照入口（待纯 OpenVLA 修复后一键执行）：
```bash
nohup bash /home/dgut/pusht_runtime/start_ps_6010.sh > /tmp/ps_oft.log 2>&1 &
sudo docker exec -e ROBODOJO_PROGRESS_LABELS=t,target_t rd_setup \
  bash /workspace/RoboDojo/_run_client.sh push_T OpenVLA_OFT 6010 1
```

## 8. 交付清单（对照你的九条）

| # | 要求 | 状态 | 位置 |
|---|---|---|---|
| 1 | 实际运行的代码与配置 | ✅ | 见第 0 节 |
| 2 | 每个任务的评测结果 | ✅ 每任务 1 个完整回合 | `closed_loop_eval/<task>/<run_id>/` |
| 3 | 至少一个真实回合日志+画面 | ✅ 3 个任务共 3 组录像 + 逐步过程量 | 同上 |
| 4 | 失败问题的证据与源码位置 | ✅ 第 2.3/2.6/3.2/3.4/5 节（含文件行号与数值） | 本报告 |
| 5 | 可复现命令 | ✅ 第 7 节 | 本报告 |
| 6 | 下一轮最值得修改的一项 | ✅ 第 3.4(e) 节：动作目标改为相对当前 proprio 的残差 Δ 并按其分布分箱 | 本报告 |
| — | 多回合（每任务 10 回合） | ❌ 未做 | 原因：模型输出与观测基本无关（第 5 节），再跑 30 个回合只会得到 30 条"零位移"记录，不构成对模型能力的进一步证据。应先修复动作目标再扩样本。 |
