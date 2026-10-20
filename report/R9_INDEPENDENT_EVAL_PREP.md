# R9 Gate D — 新独立评测集准备（工具/协议就绪；数据获取 BLOCKED 待授权）

**工作包** D ｜ **分支** `research/r9-new-test-readiness` ｜ **基线** `review/r8-three-lane-contract-audit@5242237`
**结论**：工具与协议 **VERIFIED_CPU**；**新数据获取 = BLOCKED_DATA_ACQUISITION_AUTHORIZATION**（未下载、未采集、未在 FINAL 上看任何模型表现）。

---

## 1. 本机数据源清点（`evidence/r9_eval_prep_20261020/r9_data_sources.json`，只读）

| 来源 | 目录 | 结论 | 回合 | 单文件中位 | 指令 |
|---|---|---|---|---|---|
| 现有 push_T | `data/data/hdf5/push_T/arx_x5/data` | **USABLE_PUSH_T** | 100 | 74.6 MB | Push the T-shaped block … |
| RoboDojo demo | `RoboDojo/data/demo/arx_x5/data` | **DIFFERENT_TASK** | 20 | 68.1 MB | Stack the three bowls |
| offline bundle stage63 | `openvla_offline_bundle/stage63_d0/sim_data/push_T` | **PLACEHOLDER_FIXTURES** | 108 | **9,184 B** | 无 |
| stack_bowls | `data/data/hdf5/stack_bowls/arx_x5/data` | **DIFFERENT_TASK** | 100 | 116.9 MB | Stack the three bowls |
| sweep_blocks | `data/data/hdf5/sweep_blocks/arx_x5/data` | **DIFFERENT_TASK** | 100 | 153.9 MB | sweep the blocks |

* 现有 100 集总计 **7.3 GB**，**100 个内容哈希互不相同**，数据集哈希
  `6757696397863af8058b296a80f9c50ccfc1ed824c43e8d6f6f122cbd610040a`，合共 **32,304 帧**。
* 与 `split80.json` 做 disjoint：**overlap = 100 / 100**，重复内容 0 → **本地已无未使用的 push_T 回合**。
* bundle 的 108 个文件声明 (600,480,640,3)，实际每个 **9 KB** → 测试夹具，**不能**当评测数据。

## 2. 采集/下载预算（研究负责人决策用真实数字）

实测 **77.8 MB / 回合**（7.3 GB ÷ 100）：

| 路线 | 回合 | 预估体积 | 需要授权 | 已执行 |
|---|---|---|---|---|
| 官方增量（与旧集不重叠的 push_T） | 20 / 128 | **1.56 GB / 9.96 GB** | 数据下载 | 否 |
| 专家脚本新采（同任务/同机器人/3 相机/25 Hz） | 20 / 30 | **1.56 GB / 2.33 GB** | 仿真采集 | 否 |
| 跨任务替代（stack_bowls / sweep_blocks / demo） | 20 | 1.56 GB | **不可作为 push_T 最终集** | 否 |

> 跨任务数据只能做迁移/负对照，**不能**冒充同任务最终测试。

## 3. 清单工具链（继承 R8-C，哈希已固定）

`tools/r8_eval_manifest.py`（SHA256 `fccb4a57…`）提供 `build / verify / disjoint / freeze`；
内容哈希 = state ‖ action ‖ instruction ‖ 帧数。R9 证据：

| 产物 | 结果 |
|---|---|
| `manifest_existing100.json` | 100 集、100 唯一、32,304 帧、dataset hash `6757696…` |
| `verify.json` | ok=true，无缺失/无内容变化 |
| `disjoint.json` | ok=**false**，overlap=100 → 正是 BLOCKED 的证据 |

冻结闸门 **g1–g6** 已写入协议；`freeze` 会写入 `frozen_utc / git_sha / protocol / manifest_sha256`，
规则明确要求"必须在任何模型评测之前提交"。

## 4. 协议 `protocol/r9_eval_protocol.json`（r9.1，SHA256 `e5836191…`）

* 继承 R8 协议（`48ad7326…`，按 SHA 绑定，不可漂移）：指标、基线、门槛不变；
* 新增：内容级去重（文件名 + 内容哈希双查）、占位文件下限 100 KB、跨任务拒绝、
  DEV(5)/FINAL(≥15) 回合级互斥、FINAL 在冻结前不得被任何模型看见；
* 明确禁止：无授权下载/采集、冻结前评测、把旧 20 或旧 100 的任何帧当 FINAL、跨任务冒充、覆盖旧 HDF5/证据。

## 5. 覆盖目标（从现有集归纳，供新集对照）

| 维度 | 现有集 |
|---|---|
| 回合长度 | 156–773 帧（均值 323） |
| 每回合 mean\|Δ\| | 0.00462–0.01049（均值 0.00697） |
| 运动帧占比 | 均值 0.383 |
| 频率 | 25 Hz |

新集必须在这些维度上不"更容易"（否则等于降低难度），报告时逐项对照。

## 6. 授权后的一次性流程（已脚本化，本轮不执行）

```
g1 build    --data-dir <新目录> --task-filter push
g2 verify   必须 ok（内容哈希稳定）
g3 disjoint 必须 ok（与旧 100 无名称/内容重叠、内部无重复）
g4 size     回合数 >= 15
g5 freeze   写入 frozen_manifest.json（含 protocol sha + git sha）
g6 commit   在冻结提交之后、且仅在之后，才允许在 FINAL 上评测一次
```

## 7. 回归（13/13 + 继承 6/6）

`tests/test_r9_eval_prep.py`：协议 g1–g6 完整、协议按 SHA 绑定 R8、授权未执行、
FINAL≥15、禁止偷看、内容哈希确定性、**重复内容被检出**、重复导致 disjoint 失败、
verify 往返、旧 100 全部在用、disjoint=100、来源判定（USABLE/DIFFERENT/PLACEHOLDER）、预算为正且均未执行。
`tests/test_r8_eval_manifest.py`（继承）6/6。
另做负对照：把同一集在两个文件名下 symlink 成 4 个文件 → `n_unique_content=2 / n_duplicate_content=2`，disjoint 判 FAIL。

## 8. 边界

1. 本轮**没有**下载、没有采集、没有在 FINAL 上评测任何模型。
2. 现有 100 集全部进入 R2–R9，**不能**再被称作全新最终测试；旧 20 holdout 同理。
3. 只有研究负责人批准数据获取并通过 g1–g6，才可能产生新独立测试结论。
4. 即使新集就绪，结论仍是**离线一步预测**证据，不等于 RoboDojo 闭环成功率。
