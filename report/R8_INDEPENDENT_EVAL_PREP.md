# R8_INDEPENDENT_EVAL_PREP

工作线 C（P1）：为 R8 之后的最终评测准备**独立、未被使用过**的 Push-T 轨迹。
本线**只做清点、方案与工具**，不启动任何数据采集、不修改正式数据集、不看任何模型结果。

| 项目 | 值 |
|---|---|
| 分支 | `research/r8-independent-evaluation`（基于审计 SHA `02acabc`） |
| 主数据 | `/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data`（100 集，32,304 帧） |
| 现有分割 | `split80.json`：80 训练 + 20 留出，**覆盖全部 100 集** |
| 工具 | `tools/r8_eval_manifest.py`（build / verify / disjoint / freeze），**6/6 测试通过** |
| 协议草案 | `protocol/r8_eval_protocol.json`（已冻结规则文本，待新数据到位后冻结清单） |

---

## 0. 摘要（中文）

1. **VERIFIED：本机不存在任何"未用过"的真实 Push-T 轨迹。**
   主数据集 100 集的内容哈希清单显示 **100 个唯一回合、32,304 帧**，
   与 `split80.json` 的并集**完全重叠（100/100）**；
   其他候选都不是可用的 Push-T 数据（详见 §1）。
2. **VERIFIED：旧的 20 个留出回合不能再当作最终测试集。** R2–R8 已在其上反复比较、
   调参与选方案；它只能作为"回归/探索"集。任何"最终泛化"声明必须等待新数据。
3. **IMPLEMENTED + TESTED：冻结工具链已就绪。** 清单工具能
   （a）对每集计算内容哈希（state+action+instruction+length），并给出数据集级 SHA；
   （b）复算校验（篡改单个数值即可被发现）；
   （c）检测重复内容；
   （d）检测与已用回合的重叠；（e）把"清单 + 协议 + git SHA + UTC 时间"绑定冻结。
4. **方案（需研究负责人授权）：** 三条获取路径与成本见 §3；建议 **A（官方 128 集）** 优先，
   B（按现有 expert 脚本新采 20–30 集）作为兜底。
5. **BLOCKED（仅此一项）：** 在拿到新数据并冻结清单之前，**不能宣称最终泛化能力**；
   这不影响 A/B 两条线的代码与 CPU 验证继续推进。

---

## 1. 数据清点（真实运行结果）

### 1.1 主数据集（Push-T，本机）

```
episodes=100 unique=100 frames=32304 dataset_sha256=6757696397863af8…
```

* `verify`：**ok**（清单与磁盘逐集一致，无缺失、无内容变化）；
* `disjoint`（对照 `split80.json`）：**n_overlap_with_used = 100，n_duplicate_content = 0**
  ⇒ 全部 100 集都已被 R2–R8 使用，**没有任何未用回合**；
* 覆盖统计（供新数据对标）：长度 156–773 帧（均值 323）、每回合 `mean|Δ|` 0.0046–0.0105
  （均值 0.00697）、每回合"明显运动帧"占比均值 0.383、25 Hz 恒定。

### 1.2 其他候选（全部排除）

| 候选 | 规模 | 判定 | 依据 |
|---|---|---|---|
| `RoboDojo/data/demo/arx_x5/data` | 20 集 | **不是 Push-T** | instruction 全部为 "Stack the three bowls together." |
| `openvla_offline_bundle/stage63_d0/sim_data/push_T` | 108 个 `push_T_ep*.hdf5` | **占位/测试夹具** | 每个文件仅 **9,184 字节**，却声明 `vision/cam_*/colors` 为 (600,480,640,3) uint8（约 552 MB/相机）⇒ 稀疏未写入；108 个文件合计 1.3 MB |
| `stack_bowls` / `sweep_blocks` | 各 100 集 | **不同任务** | instruction 与任务语义不同，只能做跨任务测试，不能替代 Push-T 最终集 |
| 其他主机（历史记录） | — | **未授权** | 需下载 + 授权；本轮未执行任何下载 |

**结论（VERIFIED）**：本机没有可用的独立 Push-T 轨迹；必须获取或新采集。

---

## 2. 新评测集的冻结协议（已写入 `protocol/r8_eval_protocol.json`）

### 2.1 纳入 / 排除

* 纳入：instruction 含 "Push the T-shaped block"、25 Hz、≥100 帧；
* 排除：与 `split80.json` 的 100 集有任何重叠；R2–R8 任何阶段（含探索）使用过的回合；
  2026-10-20 之前录制的回合。

### 2.2 划分与覆盖目标

* 回合级互斥：**5 个 dev + ≥15 个 final**；
* 覆盖必须至少跨越现有数据范围：长度 156–773 帧、`mean|Δ|` 0.0046–0.0105、
  明显运动帧占比 ≥0.38；物体初始位姿只在图像中可见，因此用**图像内容哈希多样性 +
  初始末端位姿散布**两项来验收（工具已支持内容哈希）。

### 2.3 指标与门槛（与 R8 现有口径一致，避免"换指标改结论"）

主要：物理 L1、修正量 R²、`|Δ|≥1e-3` 符号一致率、胜过恒速的回合数（目标 ≥90%）。
次要：Δ 余弦、分臂/逐维 L1、运动分层 L1、幅度比。
门槛：g1 L1 < 恒速；g2 R² ≥ 0.30；g3 符号 ≥ 0.75；g4 ≥90% 回合；
g5 手臂维度无系统性退化（夹爪沿用 DEV 策略）；g6 种子间差异 ≤10%。

### 2.4 防"看结果再改"机制

1. 先提交清单 + 协议（本文件与 `protocol/`）；
2. 用 `freeze` 把「清单 SHA + 协议 + git SHA + UTC 时间」写成一个文件并**先提交**；
3. 评测前再跑一次 `verify`；
4. 每个模型版本只评测一次；frozen 文件之后任何改动都会使评测失效；
5. 报告必须给出全部种子、逐回合表、frozen 哈希、被评测代码的 git SHA。

---

## 3. 获取方案（需授权，本轮未执行）

| 方案 | 内容 | 成本/依赖 | 风险 |
|---|---|---|---|
| **A（建议优先）** | 取官方完整 push_T（约 128 集）中未使用部分（≥20 集） | 约 2–3 GB 增量、需要下载授权；须核对与本机 100 集内容哈希无重叠 | 若官方集与本机 100 集重叠，可用部分可能 <20 集 |
| B | 用现有 expert 脚本在仿真中新采 20–30 集，随机化物体初始位姿 | GPU/仿真时间 + 采集授权；25 Hz、每集 20–30 s | 采集分布若与训练集过于相似，则成为"简单复制"，需要用 §2.2 的覆盖检查拦截 |
| C | 用其他任务（如 stack_bowls）做跨任务泛化 | 已有数据，无采集成本 | **不能**替代 Push-T 最终集，只能作为附加证据 |

**建议**：先执行 A 的可行性核对（只读、需要授权访问）；若 A 的可新增回合 <15，则并行 B。

---

## 4. 工具与用法

```bash
# 1) 为新数据目录建清单
python tools/r8_eval_manifest.py build --data-dir <new_pushT_dir> \
    --task-filter "Push the T-shaped block" --out manifest_new.json
# 2) 校验（复算哈希）
python tools/r8_eval_manifest.py verify --manifest manifest_new.json
# 3) 与已用回合的重叠检查（必须 ok=true）
python tools/r8_eval_manifest.py disjoint --manifest manifest_new.json \
    --split /home/dgut/pusht_runtime/split80.json
# 4) 冻结（在任何模型看到数据之前）
python tools/r8_eval_manifest.py freeze --manifest manifest_new.json \
    --protocol protocol/r8_eval_protocol.json --git-sha $(git rev-parse HEAD) \
    --out frozen_eval_new.json
```

测试（6/6）：清单确定性、单值篡改可检出、重复内容可检出、`verify` 报错、
与已用回合重叠可检出、`freeze` 正确绑定清单/协议/SHA。

---

## 5. 结论与状态

| 项 | 状态 |
|---|---|
| 本机是否存在未使用的真实 Push-T 轨迹 | **VERIFIED 不存在**（100/100 已被使用） |
| 旧 20 留出是否还能当最终测试集 | **VERIFIED 不能**（已被反复使用） |
| 冻结工具链 | **IMPLEMENTED + TESTED（6/6）** |
| 冻结协议草案 | **IMPLEMENTED**（`protocol/r8_eval_protocol.json`，待新数据后冻结清单） |
| 获取新数据 | **BLOCKED（需研究负责人授权）** |
| 本轮是否启动采集/下载 | **否**（未执行，遵守授权边界） |

**对最终结论的影响**：在拿到独立集之前，R8 的 0.001186 vs 0.001795（−33.9%）只能表述为
"在已被反复使用的旧留出集上的探索性离线证据"，不能表述为"已验证的泛化能力"。
