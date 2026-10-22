# ROUTE | RoboDojo × OpenVLA 科研历程与决策账本

> **项目范围**：仅记录 `jho887891-sudo/robo` 的 OpenVLA / RoboDojo 具身智能研究，不混入 STEP→USD、CAD、舵轮/底盘、PINN、羽毛球机器人等独立工作。
>
> **首次编档**：2026-10-20（日期表示本档案编写/授权日期；未查明的早期实验日期不反推）。  
> **阶段位置**：R2–R9 已形成代码、日志、独立审计证据；**R10 授权已记录，未执行**。  
> **研究负责人**：用户（关键研究决策、GPU、数据、合并与部署授权）。  
> **执行者**：DeepSeek（工作树、多阶段执行、测试、原始证据、GitHub 交付）。  
> **审计者**：ChatGPT（设计实验、代码审计、独立复算、最小修复、Gate 验收）。  
> **硬约束**：没有固定 SHA + 原始证据，就不能把文字报告标记为实测成功；`main`、原 HDF5、旧 checkpoints 与历史日志未经授权不得修改。

本文件回答四件事：**最初为什么做、每轮究竟证实了什么、哪些结论被纠错撤回、目前被什么阻断**。它既是科研成长时间线，也是防止“在相同错误上反复训练”的路线图。具体实验产物优先以链接到的 commit 为准；Markdown 摘要不是独立证据。

## 0. 当前的一句话结论

**项目已经从“离线动作 L1 很好”转向“完整数据/目标/推理接口必须可复算，并最终用 RoboDojo 闭环成功率裁决”。** 三帧 `[S,V,Acc]` + 速度修正量 `R=(A-S)-V` 在 *CPU MLP 离线* 对照中优于恒速；真实 7B 的 42D projector 曾完成部分 GPU Smoke，但缓存与全序列出现 **8/14** token 不一致。R9 完成 64 TRAIN-only 分位数冻结和 CPU 接线；**真实 7B 新入口训练、KV 根因、新 FINAL 集与任务闭环成功率均尚未验证**。

## 1. 最初的问题：为什么不能把 L1 当作机器人能力

研究目标为 RoboDojo/Push-T 在多相机、语言指令和关节状态条件下输出真实可执行的动作，并以**仿真闭环成功率**判断，不只优化离线 L1。早期探索使用 OpenVLA-7B、LoRA、三路 RGB、14D proprio；还尝试过 Planner/FiLM 与 OFT 多步头。这些路线不应混成同一测量协议。

### 2026-10-18：第一次关键闭环负结果（main）

- 纯 OpenVLA 6000-step 检查点，Push-T 600 步 **score=0**、物体目标误差几乎不变；Stack Bowls 800 步失败。闭环数据与相机/状态/动作链已接通，问题不能简单归咎为 WebSocket/动作顺序坏掉。
- 同一旧 20 **样本**离线首步 L1：单纯不动 `A_hat=S` **0.0121**，旧纯 OpenVLA B6000 **0.0586**、B3000 **0.0779**、单相机 A3000 **0.1168**、OFT cont300 **0.1793**。**这些数值不是后续 R7/R8 的新 20 回合全帧评测，禁止跨表排名。**
- 数据语义确认 `A[t]` 对应下一步绝对关节目标 `S[t+1]`；训练/部署三路相机变换曾不一致，修复后输入张量逐位吻合。
- **机制变化 → 直接结果 → 实际影响**：只比较模型 L1 → 加入“完全不动”策略与闭环录像 → 发现离线排名无法证明会动，更不能证明会完成任务。
- **证据**：[main `329614f` README](https://github.com/jho887891-sudo/robo/blob/329614f54d3090b9b6ab120855ba50fbc93bdd1e/README.md) 与 [闭环报告](https://github.com/jho887891-sudo/robo/blob/329614f54d3090b9b6ab120855ba50fbc93bdd1e/report/PURE_OPENVLA_CLOSED_LOOP_REPORT.md)。

## 2. 研究阶段时间线（R2–R9）

### R2 | 先修动作语言：编码器不等于解码器

**问题**：旧训练把 256-bin 标签映射为 `31745..32000`，部署读 `31744..31999`；`32000` 是 PAD。早期审计记录约 **12.23%** 标签会落 PAD，不能把旧 6000-step 权重的动作行为等同于修复后的模型能力。

**处理**：建立唯一 `action_codec.py`；新训练使用固定合法的 `31744..31999`，旧权重仍用显式 legacy 兼容；边界测试 20 项。没有宣称机器人因此已经成功。

**证据**：`audit/r2-action-codec-fix@d63ad8a74fb16df9f2f61f0b9b6d1b707f37caba`，`report/AUDIT_R2_FIXES.md`。

### R3 | 恢复真正的因果动作 token 输入；纠正实验完整性

**问题**：纯自回归路径中，训练时的 `all_actions_mask` 把先前动作 token 的输入 embedding 置零，推理时 KV 路径却保留历史 token，造成训练/推理的自回归语义不一致。

**处理**：实例级 `enable_pure_autoreg`，不修改 OFT 全局动态模块；做真实 7B 局部过拟合与梯度测试，验证条件通路可以学习。**独立审计又发现**：原 32 个过拟合样本来自旧 `val` 回合；`gen_tokens_per_sample` 某字段实际保存了真值而不是生成 token。纠正抽样与日志后，原实验只能算 in-sample 机制实验，不能冒称新留出泛化；旧证据不覆盖。

**证据**：`audit/r3-pure-autoreg-input-fix@d945331128d936f867a105d9056963a4059bf1f7`、`audit/r3-experiment-integrity-fix@40d143a99bd6ba3696db224483b24f811f475e39`，`report/AUDIT_R3_AUTOREG_RESULTS.md` / `report/AUDIT_R3_EXPERIMENT_INTEGRITY.md`。

### R4 | 目标定义 A/B：绝对动作 vs 位移残差

**问题**：专家动作 `A` 与当前状态 `S` 数值接近；绝对动作标签容易鼓励模型回归到不动。测试 `A` 与 `Δ=A-S` 两种目标，同配置做受控 300-step A/B。

**结果**：该次匹配的 **20 验证样本**：绝对模式物理 L1 **0.563377**，残差模式 **0.014313**；但残差仍没有证明优于简单状态策略，样本太少且泛化明显退化。此结论只适用于本 R4 具体协议。

**证据**：`audit/r4-unified-action-target-ab@808abfa3866f193c71ffddf9338e5c2e8a02aa72`，`report/AUDIT_R4_GPU_RESULTS.md`。

### R5 | 简单基线比复杂模型强：必须正视速度

**问题**：R4 采样口径太窄，“上一动作外推”不是“历史状态恒速外推”；动作幅度/方向必须单独测。

**结果**：在 R5 的 **60 个验证帧、20 个回合**上，恒等 `A_hat=S` 的 L1 **0.00480**，真正恒速 `A_hat=S+(S-S_prev)` 的 L1 **0.00102**，800-step 残差 VLA **0.00749**；相对恒速 **0/20 回合获胜**。说明给模型增加训练步数并没有解决机制问题。

**证据**：`audit/r5-distribution-and-baseline-audit@f974aeb8aba0afb4ed1186d24b57b8d6f6beca12`，`report/AUDIT_R5_GPU_RESULTS.md`、`report/R5_SAMPLING_AND_BASELINES.md`；方向指标修正 `audit/r5-direction-metrics-correction@e68d0a2e86690afcb2913592f84a042d88b0847c`。

### R6 | 历史状态有帮助，但 200 帧远远不够

**问题**：14D 当前状态没有显式历史速度。对照 `[S,S]` 与 `[S,S_prev]` 28D。

**结果**：200 TRAIN 帧、300 更新、3 次重复，旧 60-frame 验证平均 L1 从 **0.024217** 降到 **0.020466**，仍远差于恒速 **0.001022**。同时确认在 32,104 检查帧上 `A[t]==S[t+1]`。R6 把向下取整 bin 的重建误差误说成“理论不可突破量化下界”，后由独立审计撤回，改为**最近可表示 bin 的真正下界**。

**证据**：`audit/r6-history-proprio-pilot@62088f76a36e4847f3c910f95f01a77f9c8de313`、`audit/r6-oracle-claim-correction@781781650164827e670b2d6152a2b95ec99e6c23`，`report/R6_AUDIT_CORRECTIONS.md`。

### R7 | 从“历史帧”转到动力学：三帧、速度、加速度

**机制**：

```text
S_t      : 当前 14D 关节状态
V_t      = S_t - S_(t-1)
Acc_t    = S_t - 2*S_(t-1) + S_(t-2)
P_t      = concat(S_t, V_t, Acc_t)     # 42D
Delta_t  = A_t - S_t
R_t      = Delta_t - V_t
A_hat_t  = S_t + V_t + R_hat_t
```

**匹配全帧结果**：三帧 `[S,V,Acc]→R` MLP 的 L1 **0.001204**，恒速 **0.001795**，同批两帧 MLP **0.001696**；旧 20 回合 **20/20** 改善，但已反复使用不能算独立 FINAL。更重要的是**审计纠错**：原 `[S,V]→Δ` 的修正量 R² 曾错算为 **−0.063**，正确为 **+0.301**；直接 Δ 与 R 目标的收益差距实际只有约 **3.4%**（0.001787 vs 0.001727），撤回“Δ 完全学不到修正量”。同时修正 200 帧重复减速度、训练 epoch 选择和夹爪 DEV/holdout 选择方式。

**证据**：`audit/r7-state-dynamics-baseline@77c470494f833252679fef8c686c799123743b69`、`review/r7-state-dynamics-audit-20261020@3faed920421f02ac23e7a6a41050f4b5e93a762b`，`report/R7_CORRECTED_REPRODUCTION.md`。

### R8 | 数据覆盖比盲目增加步数重要，归一化和 7B Smoke 暴露新缺陷

**证据 1：真实数据覆盖**。64 TRAIN 回合合法 **19,127** 帧 MLP：DEV 选点三种子 mean L1 **0.001186**，恒速 **0.001795**，R² **0.536**，旧 20 回合 20/20 优于恒速；200/2,000 帧即使增加更新次数也不能等价替代全覆盖。它验证的是轻量模型上的状态预测，不是 OpenVLA-7B 的表现。

**证据 2：归一化信息损失**。旧方案把 `V/Acc` 用位置 `pmin/pmax` 归一化，真实数据约 **30.7%/31.6%** 被裁剪；64 TRAIN-only `signed_motion` q99 被 DEV 选择（proxy L1 0.001166→0.001113）。

**证据 3：42D 真实 7B Smoke 部分通过**。7B 加载，42D projector / LoRA 前向及梯度通过，但 KV-cache vs full 仅 **8/14 token** 一致；不构成生产推理通过。

**审计发现三处 P0**：目标 q01/q99 用了 **80** 回合（混入 16 DEV）；Smoke 用了与冻结文件不同的分位数；KV 失配后两路拿不同历史 token 比 logits，不能据此判别根因。均保留原始失败，独立审计修补，尚需重验。

**证据**：`audit/r8-pretraining-readiness@a2b71805d8abb9314bcca5025aa9e449c525a08e`、A `22d0a95c9a27f1ef187cbce02edefd599f2f49d8`、B `6b1559772f5fd084e6ebde1a624a0cc8468c905a`、C `4f28417c18747bc2b208650b690b444f4d7a8678`、审计 `review/r8-three-lane-contract-audit@52422372e5eacdcd0ee841ef2b91bdc5a71066c7`；参见 `report/R8_PRETRAINING_READINESS.md` 和 `report/R8_THREE_LANE_CHATGPT_AUDIT.md`。

### R9 | 修复冻结契约，打通 CPU 链路，等待真实资源 Gate

**执行基线**：`plan/r9-training-gate-atlas@fdb65da13fcf0ae5ae8d927b5864b0aedf2dd4d5`；代码固定基线 `5242237`。四条线无 GPU、无训练优化步、无下载/采集，保留旧 `main@329614f` 和历史证据：

| 工作线 | HEAD | 可用证据 | 仍未完成 |
|---|---|---|---|
| A TRAIN-only 冻结 | `experiment/r9-freeze-64train@f7b8b39e2394772ae6b2d8d929c7a0ba0fb10ed9` | 64 TRAIN / 16 DEV / 20 旧 holdout；19,127 TRAIN 行，独立复算 q01/q99 差约 1e-8；旧 80 合约 fail-closed | 新 GPU 路径消费冻结文件 |
| B 训练入口 | `feature/r9-vla-trainer-contract@1cd50f1ae447a73289fd3a2dfd24eb14e973681d` | 42D→R→256-bin→标签→解码，11/11 CPU（其中 forward=MOCK），训练优化步 0 | 真实 7B forward/backward、checkpoint 恢复、14D 常量断言 |
| C KV 诊断 | `diagnostics/r9-kv-same-prefix@02735bc2f16d60275ece01526f68860feafd5054` | CPU tiny Llama 同前缀 8/8，破坏性对照检出 | 7B GPU 失配根因 OPEN |
| D 新数据准备 | `research/r9-new-test-readiness@f8e43b0ca6ccaa8a5711e338ba12e0f3be69328f` | 旧 100 回合全部用过；新集协议和预算已归档 | 新数据采集、严格内容去重、FINAL 冻结 |

**独立审计补充**（均为 Draft PR，**尚未合并**）：
- [PR #11](https://github.com/jho887891-sudo/robo/pull/11) `review/r9-final-content-disjoint-audit-20261020@93faece89a101b4fc31071781d6297eab121783a`：发现原 `check_disjoint` 仅旧文件名比对，旧轨迹改名可漏过；新增跨旧 100 回合 content/state-action hash 对照及合成回归；真实新 HDF5/协议接线待验证。
- [PR #12](https://github.com/jho887891-sudo/robo/pull/12) `review/r9-kv-authorization-exit-audit-20261020@026d48e0e5efbdee58eb31185a8bb135b0ffba09`：发现 GPU 授权包装器只打印命令却退出 0，可能假报已完成；独立分支修改为未执行时非零退出；远程完整测试待运行。

**特别风险**：`prismatic` 在既有环境下按 `sys.argv` 解析成 LIBERO `ACTION_DIM=7`，与 14D 机器人动作冲突；R9 新入口不引用该常量，但旧 `evaluate()` 可能只解 7 tokens。历史指标是否受影响仍需证据，不得先给定根因。

**证据**：R9 `report/R9_STAGE_READINESS.md`、`mission/R9_MISSION_MANIFEST.json`、各分支 `evidence/r9_*/` 与全部 `FAILURES.md`。

## 3. 关键科研结论的“更正记录”（必须保留旧说法与撤回原因）

| 原说法/风险 | 更正后的可支持结论 | 为什么重要 |
|---|---|---|
| “OFT/VLA 离线 L1 低意味着会做任务” | 旧 6000-step 闭环仍不动；不动基线在旧 20 **样本**中更好 | 把闭环与简单策略变成必报项 |
| “训练/部署动作编码没问题” | R2 发现 1-token 错位与 PAD 标签 | 旧权重不能直接代表修复后模型 |
| “R3 val 过拟合提升了泛化” | 样本来自 val，被用于训练，仅是拟合测试；旧 gen token 字段存错 | 不能伪称未触碰 holdout |
| “向下取整 codec 重建误差是不可突破 floor” | 只等于该编码器的回放误差；真正可表示下界需 nearest-bin oracle | 不能错判 256-bin 方案已死 |
| “直接 Δ 修正量 R² 为负，完全学不到” | R7 重算为 +0.301；R 仅小幅更好 | 避免根据错误指标选择研究方向 |
| “旧 20 回合是全新最终测试” | R2–R9 反复比较，早已受到研究选择污染 | 必须另获新 FINAL |
| “R8 42D 训练路径通过” | 只有 adapter GPU 部分 Smoke，KV 8/14；R9 只有 CPU/MOCK 入口 | 不能未验即启动 7B 训练 |
| “R9 D 内容级去重已经实现” | PR #11 证明旧逻辑未跨数据集比 hash | 防旧数据改名泄漏 |
| “授权变量存在就表示 GPU 诊断成功” | PR #12 修复未执行也返回 0 的风险 | 结果以真实 GPU JSON/退出码为准 |

## 4. 2026-10-20 研究负责人新授权：R10（尚未执行）

研究负责人已明确同意**上一轮提出的两项方案**，本档将授权解释为以下精确范围：

1. **一次 7B KV GPU 同前缀机制诊断**：最多 **1 次**实际 GPU 会话、总 **5 分钟含权重加载**，启动时空闲显存至少 **24 GiB**，不干扰其他进程；仅运行冻结 64 TRAIN 合约下的原始 `r8_smoke.py --kv-diagnose`（经独立守卫复验），输出 14 步每步 256-bin logits 与缓存/全序列差异；失败也必须保留原始日志。**无 optimizer.step；不得自动扩大预算或用多次试验假装一次。**
2. **新采集 Push-T 25 个回合**：同一任务、机器人、3 相机、25Hz，写入全新的数据目录（预计原格式约 **1.94 GB**；只是按旧 77.8 MB/回合估算），**5 个新 DEV + 20 个全新 FINAL**；在模型接触 FINAL 之前按协议内容/状态动作哈希去重，核验三相机/14D/频率/场景覆盖，冻结 manifest、协议 SHA 和 Git SHA。若发现任何旧数据重复，冻结 Gate 必须失败。**没有授权无限采集、把官方增量下载当作已批路线或调整测试门槛。**
3. **仍未授权**：任何 OpenVLA-7B optimizer step、常规训练/续训、进一步重型 GPU 实验、把 PR 合并进 `main`、覆盖原始 HDF5/权重/历史证据、正式部署或修改 FINAL 的预注册规则。技术原因未解决时只准诊断和证据留档。

**执行任务文件**：[`missions/R10_AUTHORIZED_EXECUTION.md`](missions/R10_AUTHORIZED_EXECUTION.md)。本 ROUTE 不代表远程 GPU 或新数据已运行：除非出现真实的新 SHA、日志、数据清单与 Gate manifest，R10 状态一直是 `AUTHORIZED_NOT_EXECUTED`。

## 5. 当前 Gate 与不变量（下个阶段前必读）

- **A 冻结数据**：只允许 64 TRAIN 拟合归一化/目标分位数；16 DEV 做开发选择；旧 20 不能再作为“全新最终测试”。
- **B 训练**：R9 新 42D 训练入口只能标 `VERIFIED_CPU + MOCK_FORWARD + PENDING_GPU`；GPU 无更新 smoke 和正确的 14D 编码/维度检查都必须另验。
- **C KV**：tiny Llama 8/8 只证明探针协议；7B 曾 8/14，不是 bf16 根因证据。固定 `numeric_floor=0.05` 只是一种启发式，不能作为结构 vs 数值的因果判别。
- **D 数据**：旧 100 内容集合全部已进入过去试验；新 FINAL 的最小承诺是 **20 个全新回合**，先冻结后评一次。不能把不同任务数据当作 Push-T FINAL。
- **唯一最终目标**：离线 14D 物理 L1、修正量 R²、方向/动作分层指标提供诊断；**RoboDojo/Push-T 闭环成功率**决定任务是否成功。即使新离线 FINAL 通过也不是闭环通过。

## 6. 未来如何续写（科研账本规则）

**只追加新阶段，不覆盖旧结论，不重写失败。** 新的 R10、R11… 使用统一模板：

```markdown
### Rn | [UTC 日期] | [明确问题]
- 冻结基线：<完整 SHA，分支>
- 假设（HYPOTHESIS）：……
- 最小实验与预算：数据划分/样本/种子/指标/对照/GPU 上限
- VERIFIED：原始日志与独立复算（完整证据路径）
- IMPLEMENTED：有代码但尚未完成实测的内容
- FAILED / BLOCKED：失败现象、边界条件、阻断子树
- 研究结论：哪条假设被支持/反驳/仍未定
- 决策：下一阶段通过什么 Gate，谁授权
- Evidence links：PR、commit SHA、manifest、日志、预测存档
```

**比较准则**：只有**同批样本、同种帧筛选、同尺度指标、同训练预算口径**才比较数值；区分回合宏平均/帧平均、旧 20 样本/旧 20 回合、训练/DEV/FINAL、CPU MLP/真正 7B、离线/闭环。遇到指标纠错时追加 **Correction** 条目指向旧证据，不删除。

## 7. 核心入口索引

- 仓库：<https://github.com/jho887891-sudo/robo>
- 原始闭环与失败：`main@329614f` 的 `README.md` / `report/PURE_OPENVLA_CLOSED_LOOP_REPORT.md`
- R2–R8：上述独立审计分支及 `report/` 证据。
- R9 执行地图：[`R9_COMPLETE_STAGE_EXECUTION_ATLAS.md`](missions/R9_COMPLETE_STAGE_EXECUTION_ATLAS.md)，固定 `plan/r9-training-gate-atlas@fdb65da`
- R9 四线 Gate 汇总：`feature/r9-vla-trainer-contract@1cd50f1` 的 `mission/R9_MISSION_MANIFEST.json`
- R9 独立审计：[Draft PR #11](https://github.com/jho887891-sudo/robo/pull/11) / [Draft PR #12](https://github.com/jho887891-sudo/robo/pull/12)
- R10 授权执行：[`missions/R10_AUTHORIZED_EXECUTION.md`](missions/R10_AUTHORIZED_EXECUTION.md)

_本 ROUTE 是长期索引，审计结论始终以对应的源代码/原始数据/固定 SHA 为最终裁判。_


### R10 | 2026-10-20 | 授权资源后的执行：CPU Gate 完成，GPU 与采集资源不可用

- **冻结基线**：`review/r8-three-lane-contract-audit@5242237`；A `experiment/r9-freeze-64train@f7b8b39`；
  B `feature/r9-vla-trainer-contract@1cd50f1`；C `diagnostics/r9-kv-same-prefix@02735bc`；
  D `research/r9-new-test-readiness@f8e43b0`；独立安全补丁 PR #11 `93faece`、PR #12 `026d48e`；`main@329614f`（只读）。
- **本轮真实发生**：在 `r10/cpu-gates` 上完成 R10 §2 的 P0 CPU 闸门（PU/PR 复验、sidecar 存档补丁、
  缓存 preflight、freeze guard、A/B 复验），并对 GPU 与采集做只读资源探测。
- **VERIFIED（CPU，原始日志在 `evidence/r10/`）**：
  - PR #11 内容去重工具通过 9/9 合成测试；在真实旧 100 清单上把旧回合改名 → 被拒（content overlap=1）；
    同一 state/action 换 instruction → 被拒（state_action duplicate=1）；占位文件/错频率/缺相机/错回合数 → 全部被拒。
  - PR #12 退出语义实测：无授权 7、有授权但未真实执行 **8**（不再假装 PASS）；C 线回归 12/12。
  - 新增 `tools/r10_freeze_guard.py`：verify + 跨旧 100 内容/state-action 去重 + 内部去重 + 真实 HDF5 schema
    （三路 RGB/25 Hz/14D/≥100 帧/指令）+ 回合数（25=5+20）+ 种子化 DEV/FINAL 分配，全部通过才允许 freeze；回归 10/10。
  - `diagnostics/r8_smoke.py::kv_diagnose` 补丁：每步存 `logits_kv/logits_full[14,256]` 到 `.npz` sidecar 并记录
    sha256/shape/dtype；缓存 preflight 显式关闭 gradient checkpointing、强制 `use_cache=True`、cache 为 None 即拒绝；
    回归 16/16；模型数学与窗口常量（31744/256）未变。
  - A 冻结契约复验 9/9（15/15 回归）；B 42D/R/codec 无更新 trace 11/11（16/16 回归）；
    `prismatic` 仍解析为 LIBERO `ACTION_DIM=7`（HYPOTHESIS，任何 GPU 前必须断言 14）。
- **BLOCKED（资源，非失败）**：
  - **GPU KV 诊断**：开机以来三个外部进程占用 24.10 GiB（isaac-sim kit 2486 MiB、isaac-sim eval client 5604 MiB、
    venv python 16200 MiB，后者为本项目端口 6020 的策略服务，起于 10-18），实测空闲 **23.300 GiB < 24 GiB**（差 0.700 GiB）。
    按 R10 §3 判 `BLOCKED_RESOURCE`，未启动、未抢占、未终止任何进程。
  - **新采 25 回合**：`robodojo.sh` 只有 `eval/server/client` 三个子命令，仓库内**没有**数据采集入口；
    `isaacsim` 在宿主 python 与 venv 中均不可导入（模拟只在容器 `rd_setup` 内）；该容器正在跑 10 路 push_T 评测。
    判 `BLOCKED_ENV`，未采集、未下载、未改名复用旧数据。
- **FAILED / 修复轮次**：probe 测试 1 次误判（把主 smoke 的 GC 调用算进 kv_diagnose）+ guard 测试 5 次失败
  （build 异常未捕获、缺 `dev_episodes` 覆盖、verify 在无 manifest 时崩溃）→ 全部保留日志，修复 3 轮，验收标准未放宽。
- **研究结论**：R10 的两个真实动作（GPU 诊断、25 回合采集）都被**资源**阻断，而不是被代码阻断；
  CPU 前置条件已全部可复算，一旦资源释放即可执行，不需要重新设计任务。
- **决策（需研究负责人）**：①是否释放/暂停 10-18 起的长跑评测以腾出 ≥24 GiB；②采集入口缺失，需指定采集程序
  （或授权使用既有 `k1_*` 脚本）并给出独立窗口；③7B 训练仍未授权。
- **Evidence**：`r10/cpu-gates` 的 `mission/R10_MISSION_MANIFEST.json`、`report/R10_*.md`、`evidence/r10/`。

## R11–R16 first execution window (2026-10-20 19:20Z → 20:05Z, branch feature/r11-p0-closeout)

Executed, with real GPU evidence and a book-kept budget (0.285 GPU-hours of 32 h, formal optimizer
steps 0 of 3000):

* **P0 VERIFIED_CPU** — five source-level defects fixed red-first: the R42 hand-written codec was
  not bit-equal to `action_codec` (missing `+1e-8` in denormalisation) and the prompt lacked the
  legacy trailing `<space>` (29871) token, so the causal shift was one position early; the
  prismatic `ACTION_DIM=7` table (chosen from `sys.argv`) now fails closed unless it is ARX 14D,
  and the legacy `range(ACTION_DIM)` 7-token loop is pinned by a regression instead of being
  relied on; the rate limiter and the physical box can no longer disagree silently; the 21-field
  checkpoint metadata now carries the frozen normalizer and its canonical SHA; a real 42D deploy
  server replaces the CPU-only adapter. Suite: 250 passed, 3 skipped (baseline 208/3).
* **R13 VERIFIED_GPU_ZERO_UPDATE** — first real 7B 42D forward/backward in this project:
  loss 14.4278, LoRA grad norm 56.75, projector grad norm 16.17, `optimizer_steps = 0`,
  14 legal action tokens, common-prefix logit delta 0.0, checkpoint→reload→identical tokens.
  99.7 s, peak 22.16 GB. A first attempt crashed on a `torch.Size` bug after loading the 7B and
  its 70 s are also booked.
* **R14 OVERFIT_SIGNAL_PASS** — 100 steps on 32 fixed TRAIN frames from 32 distinct episodes:
  CE 14.814 → 1.908 → 1.305 → 0.541 → 0.380 → 0.125 at steps 0/20/40/60/80/100, training loss
  head mean 5.63 → tail mean 0.40, greedy 14-token decode 9 → 13 distinct bins. 857.8 s.
* **R11 collector BLOCKED_EXPERT** — read-only discovery found no executable Push-T expert in the
  Isaac Sim container (robodojo.sh drives policies under test; the 100 episodes were downloaded;
  the only "expert" file is a pybullet artefact). COLLECTOR bucket untouched, no synthetic data.
* **R15/R16 NOT_STARTED** — the staged trainer, gate machinery and ledger are implemented and both
  upstream gates are green, but the window ended before the 300/1000/3000 runs.
* **Draft PR not created** — the box has no GitHub API credential (no `gh`, no token); the branch
  is pushed and this is recorded rather than papered over with a `/pull/new/` link.

## 2026-10-22 — R16 round: P6 measurement defect fixed and re-verdicted, M2 booking corrected (append-only)

Fixed source for this round: `feature/r15-true-continuation@b57d16a`, audited by PR #25 @`22e6061`.
Server-verified state (not inferred from the previous report):

* `m2_two_2000_r5` = **DONE exit 0**, `REACHED_TARGET`, global_step **2000**, 7742.6 s wall,
  2.939 s/step, peak 23.04 GB, checkpoint `runs/m2_two_2000_20261021T123314Z/checkpoints/step2000`
  with manifest-verified LoRA `c221e739…` and projector `1635bd02…`.
* The WAL confirmed audit item 3 exactly: `job_end` booked `gpu_seconds=0.0` against a 7827 s wall.

### P6 action-window alignment (RED -> GREEN, then re-verdict)

`action_window_probs()` read the 14 action positions from `n_prompt-1` without the 769-token vision
prefix, while `token_ce_from_logits()` applied it. Everything P6 compared - `ce_window`, `pred_bins`,
the counterfactual task-routing distributions - came from vision-token rows. RED: 4 failing tests
(`tests/test_m2_action_window_alignment.py`) on the old code. GREEN after the fix, and an archive
without the `vision-prefix-aware-v2` marker is now refused (`LEGACY_INVALID_FOR_P6`).

Corrected evidence, regenerated from the sealed step2000 checkpoint **without retraining**
(`tools/m2_regen_dev.py`, same 8 DEV frames/task, same true bins - the push_T `true_bins_sha256`
`b45fcbaa…` is byte-identical to the legacy run):

| task | ce_window before | ce_window corrected | no-op | constant velocity |
|---|---|---|---|---|
| push_T | 4.8883 | **2.1214** | 2.6241 | 2.6736 |
| stack_bowls | 5.3775 | **3.0963** | 3.3667 | 3.3172 |

Coherent eps-smoothed NLL (eps = 1e-3 pre-registered, same eps on both sides): 2.122 vs 5.893 (no-op)
and 6.005 (CV) for push_T; 3.096 vs 7.561 and 7.450 for stack_bowls. Routing: own-prompt win rate
0.25 -> **1.0**, wrong-prompt penalty ~0 -> **0.65-0.77 nats**, distinct predicted bins 1 -> **8**.

`mission/m2_p6_two_corrected_report.json`: **PASS, exit 0, 20/20 checks**. The previous FAIL was a
measurement artefact, not the model. The legacy report and archive are untouched and remain permanently
invalid for P6. Erratum: the per-save-point CE curve was recorded with the legacy alignment, so its
absolute values are not comparable to the corrected final point; the forgetting conclusion still holds
because every point of that curve was measured identically.

### M2 GPU booking (append-only correction)

`job_end` now books `wall - sum(step_seconds)`, so the 7B load, the DEV counterfactual CE and every
full-state save are charged. The historical gap was corrected append-only:
`tools/m2_wal_correction.py` booked +9937.7 s, but that summed supervisor wrappers *and* their inner
trainers, so `tools/m2_wal_correction_fix.py` appends a documented reversal (-9937.7) and a correctly
scoped rebooking (+2058.7 = 1933.7 for the completed run + 125 bounded for the crashed one).

Ledger after correction: gpu **26242.6 s**; `TWO_TASK_PILOT` 7952.0 s / **2000** steps;
`R15_TRAIN_FORMAL` 8261.6 s / 1664 steps - the M1 3000-step cap and the M2 20000-step budget are
separate buckets, so P6 evidence can never be produced by borrowing M1 headroom.

### What is still NOT claimed

* M1 G5_1000 = FAIL stands; no further same-hypothesis Push-T steps were taken and none will be.
* The 2->4 task expansion is now *permitted* by the corrected P6, but it is a budget decision for the
  research lead and has not been started.
* The 12-task expert data for the 9 missing tasks, and any official FINAL, are unchanged.

## 2026-10-22 (cont.) — first complete official DEV episode; corrected forgetting curve; third task (append-only)

### R16 official DEV closed loop: 1 episode COMPLETED (job `loop_pushT_stage1_r7`)

600 control steps (the task's `step_lim`), three 601-frame camera videos, driver wall 3842.9 s. Official
`_result.json` from the container: `success_rate 0.0`, `details[0].success = false`, layout 0. Recovered
record: `verdict=OFFICIAL_FAILURE`, n=1, Wilson 95% CI **[0.0, 0.7935]**, and the client's own stdout
tally (`Success nums: 0, Fail nums: 1`) agrees independently.

Live inference wall over that run, 20-action windows: **6749 ms** p50 at 100 actions (169x budget),
5585 at 300, **5529 ms p50 / 7040 p95 / 8166 p99** at 600 (forward 5111 ms) = **138x** the 40 ms budget.
`OFFLINE_SIM_ONLY`: the sim advances in lock-step behind a synchronous policy call, so wall clock inflates
while the simulated timestep and control semantics are unchanged — which is why a bounded DEV episode is
still evidence. It is not a realtime claim.

The failure is consistent with the offline gate (M1 G5_1000 FAIL, SAFE macro L1 0.0025385 vs CV
0.0020475), not a surprise. n=1: a data point, not a rate.

Harness defect fixed: the driver recorded NO_RESULT although the client had written a verdict, because
(a) with `--docker` the client writes into the CONTAINER's eval_result while discovery searched the host
tree, and (b) `eval_result` is a symlink inside the container and `find` needs `-L` to descend into it.
Now: container-side `find -L`, read there, cross-checked against the stdout tally, plus `--reparse-only`
which recovers a verdict from a finished run without launching anything and stamps `reparse_only: true`.

### Corrected DEV CE curve (erratum E5 closed) — `mission/m2_ce_curve_corrected.json`

All ten sealed save points recomputed with the corrected alignment, no retraining,
`alignment_verified = true` at every point:

| step | 200 | 400 | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 |
|---|---|---|---|---|---|---|---|---|---|---|
| push_T | 2.3282 | 2.2882 | 2.2134 | 2.1817 | 2.1738 | 2.1563 | 2.1521 | 2.1436 | 2.1347 | **2.1214** |
| stack_bowls | 3.4586 | 3.6332 | 3.2071 | 3.3341 | 3.3169 | 3.2301 | 3.1571 | 3.1136 | 3.1015 | **3.0963** |

The legacy curve's absolute values were measured on vision-token rows and are not comparable; its
forgetting conclusion stands because every point of it was measured the same way. Two early points
(200, 400) initially failed my tool's exit check: the cause was my own criterion conflating **alignment**
(the window rows must be the supervised positions — holds everywhere) with the **model's** argmax landing
inside the legal window (0.91 and 0.78 at those steps). Separated; no point dropped.

### Third task wired, and a real deploy bug found

`data/m2_task_normalizer_3task.json` (80 TRAIN episodes/task, DEV excluded) plus deploy-consistency tests.
Writing them exposed that `validate_normalizer()` read `fit_source` as a mapping while the fitted payloads
store it as a sentence — so `normalizer_for_task()` raised `AttributeError` on **every real multi-task
payload**, including the 2-task one the stage was trained with. The existing tests passed only because
their fixtures were synthetic. Consequence: the M2 shared model could never have been served. Fixed and
pinned; the 2->4 expansion itself remains a budget decision and was not started.

## 2026-10-22 — R16 KV Cache: root cause found, fixed, verified, driven through the closed loop (append-only)

### The root cause was in the diagnostic harness, not the model

`tools/r16_kv_equivalence.py:decode_full()` appended the argmax **bin** (0..255) as the next token id while
`decode_cached()` correctly appended `31744 + bin`. From step 1 on the "full reference" decoded from
ordinary text tokens, saturated on the window edges (the 0/255 pattern) and manufactured gaps up to **31.4**
logit units. Every R16 conclusion drawn from that tool - including the three-way attribution I reported
earlier (reference == full 8/8, cached 0/8, "the incremental step is the defect") - is **withdrawn**.

Proof that it was never the cache, from independent probes that force the same token into both paths:
k=0 bit-identical; k=1 identical to 0.03-0.13 for the naive step, for an explicit
mask+position+cache_position step, and for a cache built by the multimodal forward itself; the same on
frames the broken tool called divergent (17.7 -> 0.125); the text-only LM control shows correct cache
parity (0.031); the cache really carries 805 = 36 prompt + 769 multimodal entries; per-layer audit is at
bf16 scale from layer 0.

### Fix and verification (real 7B, step2000)

`decode_full`/`decode_cached` are now module-level, backend-injected and refuse to append any token outside
31744..31999; `tests/test_r16_kv_decode_tokens.py` pins it with a stub backend that records every call.

| measurement | result |
|---|---|
| teacher-forced per-k worst logit delta, 32 inputs x 14 tokens | **0.5** (k=0 exactly 0.0) -> NOISE_LEVEL_PARITY |
| free-running tokens differing | **5 / 448** (1.1%), one codec bin each |
| physical residual impact | **<= 1.57e-4** (0.16 mm) |
| speedup, 64 timed actions | **4.42x** (p50 5060.1 -> 1145.1 ms; p95 5162.4 -> 1175.1) |

### Closed loop with the cache ON (Push-T DEV, job `loop_pushT_stage1_cache_r8`)

| | full path (`..._r7`) | KV cached (`..._cache_r8`) |
|---|---|---|
| control steps | 600 | 600 |
| latency p50 / p95 / p99 | 5529 / 7040 / 8166 ms | **1582 / 2021 / 2595 ms** |
| episode wall | 64.0 min | **21.4 min** |
| official `_result.json` | success_rate 0.0, Wilson [0, 0.7935] | success_rate 0.0, Wilson [0, 0.7935] |

The verdict is unchanged, which is exactly what a noise-level cache should produce. The cache stays
opt-in (`R42_KV_CACHE=0` default) because it is equivalent up to bf16 rounding, **not bit-identical**.

25 Hz is still not reached (1582 ms = 40x the 40 ms budget). The cache removes most redundant compute;
closing the remaining 40x needs an architectural decision (action chunking / distillation / shorter
context), which is not mine to take.

### 2026-10-22 R16 KV bin/token诊断工具根因、真实加速及官方SIM provenance交叉审计

- 执行者feature/r15-true-continuation@569677b5b6ec403458178c7036e2ef00e493052f。独立核实旧tools/r16_kv_equivalence.py的FULL路径把bin 0..255当token ID送回模型；正确action token需31744+bin，旧三方reference/full一致是共同喂错bin造成的伪证据，已有修正和SUPERSEDED，不得继续说KV自身严重偏离31.4 logits。
- 修复后真实7B有噪声级近似等价：固定输入32x14=448 token，teacher-forced maxabs0.5，自由5/448个bin变化，非位级/严格14token一致；新strict脚本保留DIVERGED而另一新0.5容差协议为NOISE_LEVEL_PARITY，两者绝不能混称。1.57e-4是本地原生joint/gripper动作数值单位差，不能直接称0.16mm末端位移。
- 真7B同脚本profiling p50 FULL5060ms/cache1145ms，4.42倍；官方DEV服务器缓存p50 1582ms ≈25Hz周期40倍，不是实时。M2 step2000 checkpoint用于缓存parity，但PushT官方1回合的ckpt label是R15 step1000，不能直接把前者对等结果迁移到后者。
- 官方PushT full/cached目录均有layout0 success=false的1回合_ result，不能由n=1推断模型固有成功率=0。缓存目录的push_T_stage1_cache_reparsed.json明确reparse_only=true/exit=null、从04:48:44历史官方结果目录重新读取，非05:10新运行。需固定当时原client stdout/PID/run id、R42_KV_CACHE=1日志、视频/源checkpoint、原始result sha以证明该结果和缓存600步属于同一作业。
- 新P0发现工具的容器 _result.json抓取使用find -L按task+时间窗取最近文件，没有运行开始/结束唯一校验，可能借到上一轮同seed结果。独立review分支增加"新LIVE结果mtime必须在当前任务UTC墙钟区间"的最小fail-closed保护和三条CPU回归；已有reparse分支保留历史恢复声明，绝不覆盖旧报表。这是必要但非充分的关联校验，完整唯一run_id绑定仍需实际原生result路径日志。
- M2历史P6的no-op physical14D L1原算成零关节预测，需从此前独立PR#27合并修补，并新评测真实SAFE物理动作L1/遗忘；不能从8DEV帧概率Gate直接扩成12任务成功。继续并行12TaskSpec和专家数据工作。
- 完整源码审计见report/R16_KV_AND_PUSH_T_DEV_INDEPENDENT_AUDIT_20261022.md。本审计没有新远程GPU作业，保护main/旧WAL/权重/HDF5/FINAL。
