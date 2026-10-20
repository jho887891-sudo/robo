# R8_PARALLEL_STATUS — 归一化 / VLA 集成 / 独立评测三线统一状态

本轮按"无数据依赖则并行"的方式推进三条工作线，全部只做 CPU 实验与一次受控 GPU Smoke；
未启动 7B 训练、未合并 PR、未修改 main、未改旧检查点与旧证据、未终止任何外部 GPU 进程。

| 工作线 | 分支 | HEAD SHA | 关键产物 |
|---|---|---|---|
| A 归一化审计 | `experiment/r8-motion-normalization` | **`22d0a95`** | `report/R8_PROPRIO_NORMALIZATION_AUDIT.md`，16 测试，证据 517 KB |
| B VLA 42D 集成 | `feature/r8-vla-42d-smoke` | **`8d3eccf`** | `report/R8_VLA_42D_INTEGRATION.md`，12 测试 + GPU Smoke，证据 82 KB |
| C 独立评测准备 | `research/r8-independent-evaluation` | **`4f28417`** | `report/R8_INDEPENDENT_EVAL_PREP.md` + `protocol/r8_eval_protocol.json`，6 测试 |
| 基线（审计） | `audit/r8-proprio-scale-preflight` | `02acabc`（未改动） | ChatGPT 的 `r8_scale_audit.py` 原样运行 |

---

## 0. 七个问题的直接回答

**1. 当前 R8 归一化是否真的损失速度/加速度信息？损失多少？** — **VERIFIED 是，且频率很高。**
V/Acc 与位置共用 `pmin/pmax`：训练集 **30.7%/31.6%** 的运动条目被钉在 `-1`（开发 29.8/30.7%、
留出 31.8/32.8%）；**5 个维度**（`left_arm_2, left_gripper, right_arm_1, right_arm_2, right_gripper`）
**全部负向运动塌缩为同一个值**；**零运动映射到非零向量**
（`[0.445, -1, -1, 0.975, 0.217, 0.341, -1, 0, -1, -1, 0.964, 0.093, 0.451, -1]`）；
**16.3%** 的有运动条目与"零运动"不可区分；夹爪维度 96–98% 被裁剪。

**2. 新的有符号归一化在 DEV 上更好吗？** — **VERIFIED 是（幅度适中）。**
DEV L1：旧方案 0.001166 → 去掉裁剪 0.001123 → **有符号 q99 0.001113（−4.5%）** → q99.9 0.001137；
留出集（仅回顾性）0.001230 / 0.001190 / **0.001179** / 0.001192；修正量 R² 0.484 → **0.561**。
按预注册规则**只依据 DEV 选定** `signed_motion` + q99，并已写入冻结元数据。

**3. OpenVLA 的 42 维输入是否真正集成完成？** — **功能链路 VERIFIED，因果解码 OPEN。**
GPU 实测：真实 7B 加载（`llm_dim=4096`）、42 维投影层（16,957,440 参数）、前向 loss 15.63、
投影层梯度 2.01/20.86、LoRA 梯度 852 张量、**三个槽位分别影响 logits（0.3125/0.3438/0.3438）**、
扰动改变 9/14 token、解码到物理命令并应用夹爪策略、无 NaN、无未来泄漏。
**未通过**：KV-cache 与全序列 argmax 仅 8/14 一致（§3）。

**4. GPU Smoke 是否实际运行？** — **VERIFIED 是**，共 3 次受控尝试：
第 1 次因我的 harness 缺陷（生成时未 `eval()`，梯度检查点强制 `use_cache=False`）断言失败；
第 2 次因相位顺序导致 CUDA OOM（显存被三个外部进程占 23.7 GiB）；
第 3 次 **14/15 通过**，峰值显存 **21.88 GB**、wall **47.6 s**、期间外部进程未被干扰。
失败两次的原始日志均保留。

**5. 新的独立 Push-T 测试轨迹能否取得？** — **本机 VERIFIED 不能**：
100 集 Push-T 全部被 `split80.json` 使用（内容哈希清单显示 100/100 重叠、0 重复）；
`RoboDojo/data/demo`（20 集）是**另一个任务**（"Stack the three bowls"）；
`openvla_offline_bundle` 的 108 个 `push_T_ep*.hdf5` 每个仅 9,184 字节却是 (600,480,640,3) 声明 →
**占位测试夹具**；其余两个任务目录是不同任务。
获取需授权：A 官方 128 集（增量 2–3 GB）、B 用现有 expert 脚本新采 20–30 集、C 跨任务。
工具与冻结协议已就绪（`tools/r8_eval_manifest.py`，6/6 测试；协议哈希 `69ae5083…`）。

**6. 是否存在阻断小规模 GPU 训练的缺陷？** — **有一个：KV-cache 等价性（OPEN）。**
归一化缺陷已通过采用 `signed_motion` 冻结契约解决（不再阻断）；
测试集复用**不阻断训练**，但阻断任何"最终泛化"声明。

**7. 下一轮可立即推进的独立任务？** 见 §4 的依赖图，其中可并行的三项是：
(a) KV 判定（一次约 1 分钟 GPU + 模型加载）；(b) 独立集获取的可行性核对（只读/授权）；
(c) 图像特征 vs proprio 的受控对照设计（CPU 可先写工具与门槛）。

---

## 1. 工作线 A（P0）要点

* 复算并确认审计脚本的独立实现与生产契约逐元素一致（脚本内断言未触发）；
* 逐维/分臂/分集合的裁剪与塌缩统计（training/dev/holdout 全部一致）；
* 四种归一化方案的 DEV 对照 + 留出集回顾（不参与选择）；
* 冻结契约：`signed_motion`、q99、位置槽位不变、`v_scale`/`a_scale` 来自 TRAIN；
* 16 项回归测试（含"旧方案 5 维塌缩 / 31% 裁剪 / 零运动非零"的真实数据锁定与审计的 `[0,1]` 反例）；
* 判定：**必须在 7B 训练前切换**，但代理任务实测代价仅 4.5%，**不构成项目级阻断**。

## 2. 工作线 B（P1）要点

* 适配层（`r8_vla_adapter.py`）：42 维特征、每环境两帧历史、目标/命令、夹爪 DEV 策略、
  关节界裁剪、元数据 fail-closed；
* 独立 Smoke harness（`diagnostics/r8_smoke.py`，**不改冻结的 R6 脚本**），
  含 `--plan` / `--check-contract`（CPU 10/10）/ `--kv-diagnose`；
* 冻结契约生成器（`tools/make_frozen_contract.py`）把归一化 + R 目标分位写进元数据；
* GPU 实测 14/15（见 §0.4）；
* KV-cache 等价性 OPEN，诊断工具已实现但未取得判定。

## 3. 唯一未通过项：KV-cache 与全序列不一致

| 维度 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KV | 132 | 102 | 177 | 176 | 108 | 130 | 0 | 0 | 128 | 128 | 127 | 114 | 127 | 128 |
| 全序列 | 132 | 102 | 121 | 243 | 135 | 210 | 0 | 0 | 128 | 128 | 126 | 114 | 139 | 128 |

* 前缀步（只依赖 prompt）完全一致；不一致从**第一次使用缓存**开始 → 指向 bf16
  增量-vs-批量矩阵乘的数值敏感，但**未证明**；
* 28 维前身（R6）在同一机器上曾 14/14 通过，因此不是 42 维接口必然的结果；
* 判定工具 `--kv-diagnose` 已实现（逐步记录两路 256-bin logits 与 top1−top2 间隔），
  首次运行因 harness 缺陷中断（已修复、已编译验证），**未再占用共享 GPU**；
* 结论：**在判定与修复前，不得把 KV 解码视为已验证的部署路径，也不得据此启动 7B 训练**。

---

## 4. 依赖关系与下一步

```
[A 归一化]  已冻结 signed_motion(q99) ──┐
                                        ├──> [B 42D 集成] 功能链路 VERIFIED
[B KV 判定]  OPEN ──────────────────────┘        │
                                                 ▼
                                     [7B 小规模训练] 需：KV 通过 + 研究负责人授权预算
[C 独立评测集] BLOCKED(需授权采集/下载) ──> 最终泛化声明
```

可立即执行（无需新授权）：
1. `--kv-diagnose` 一次运行 → 判定"数值敏感 vs 结构性"；若数值，改为容差判定并 fp32 复测；
2. 用 A 的冻结契约重跑一次"归一化 × 种子"稳定性检查（CPU）；
3. 写"图像特征 vs proprio"的受控对照设计与门槛（CPU，不训练）。

需要授权：新 Push-T 轨迹获取（A/B 方案）、7B 小规模训练预算。

---

## 5. 交付清单

| 类别 | 内容 |
|---|---|
| 分支/SHA | A `22d0a95`、B `8d3eccf`、C `4f28417`（均在 `02acabc` 之上；main 与审计分支未动） |
| 新增代码 | A: `diagnostics/r8_norm_diagnostics.py`、`code/.../r8_normalization.py`、`training/r8_norm_ablation.py`、`tests/test_r8_normalization.py`；B: `code/.../r8_vla_adapter.py`、`diagnostics/r8_smoke.py`、`tools/make_frozen_contract.py`、`tests/test_r8_vla_adapter.py`；C: `tools/r8_eval_manifest.py`、`tests/test_r8_eval_manifest.py`、`protocol/r8_eval_protocol.json` |
| 测试 | A 16/16、B 12/12 CPU + 10/10 契约、C 6/6（合计 34 项 CPU 测试 + GPU Smoke 14/15） |
| 证据 | `evidence/r8_motion_normalization_20261020/`、`evidence/r8_vla_42d_smoke_20261020/`、`evidence/r8_independent_eval_prep_20261020/` |
| 失败记录 | A：`FAILURES.md`（5 条）；B：三次 Smoke 尝试的原始日志（eval-mode 断言、CUDA OOM、KV 不一致）；C：无失败（工具首次即通过） |
| 未解决 | KV-cache 等价性；独立评测集缺失；`r8_eval_protocol` 的清单未冻结（无数据） |

## 6. 对"是否值得把三帧历史 + 速度残差引入 OpenVLA"的判断

* **三帧历史（Acc）**：在匹配的 CPU 消融中带来 −29%（对比两帧）与 −33.9%（对比恒速），
  20/20 回合，三种子稳定 → **值得引入**（VERIFIED）。
* **速度残差目标**：L1 仅比直接 Δ 好 3.4%（R² 0.311 vs 0.301）→ **值得引入但不关键**，
  其主要价值是"零修正"回退语义与更可解释的监督（VERIFIED，幅度小）。
* **归一化**：必须先换成有符号运动缩放（VERIFIED 的缺陷修复，代价几乎为零）。
* **剩余不确定性**：KV 解码等价（技术门禁）、独立测试集（结论强度）、
  以及"R 目标在 7B 上是否仍有 3.4% 的收益"（HYPOTHESIS，需训练验证）。
