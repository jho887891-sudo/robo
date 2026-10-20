# R10.2 B — FINAL 冻结路径的 CLI 级交叉审计（VERIFIED_CPU_TESTED）

**状态**：`VERIFIED_CPU_TESTED`（**不是** "新 FINAL 已就绪"：本地没有真实新采集数据）
**分支** `r10.2/final-freeze-audit` ｜ **基线** PR#14 `429e28a`

---

## 1. 回归（真实运行，不引用 R10 的 10/10）

| 套件 | 结果 |
|---|---|
| `tests/test_r10_freeze_execution_audit.py`（PR#14 新增） | **11/11 OK** |
| `tests/test_r10_freeze_guard.py` | **10/10** |
| `tests/test_r9_final_disjoint.py`（PR#11） | **9/9 OK** |
| `tests/test_r9_kv_same_prefix.py`（PR#12） | **12/12** |

PR#14 新增的 `freeze_checked()` 确实把闸门写进了代码：要求 guard 全 PASS、`small_set_override is False`、
`expect_total == 25`、**完整 40 位 Git SHA**、协议 hash 必须等于冻结值 `e5836191…`、输出路径不得预先存在，
并把 `r10_split_assignment`（5/20 名单）与 `r10_provenance`（protocol/used-manifest/guard-checks 三重 hash）
一起封进被 SHA256 的 manifest。

## 2. CLI 级审计（`tools/r10_2_b_freeze_cli_audit.py`，21/21 PASS）

每个用例都通过**真实命令行**调用 `tools/r10_freeze_guard.py --freeze-out ... --git-sha <40hex>`，
并检查**退出码**与**目标冻结文件是否存在**（负例必须 rc≠0 且文件不存在 / 已存在文件不被改写）。

### 正例（真实写盘）

| 检查 | 结果 |
|---|---|
| 冻结文件真实写出 | PASS，`frozen_ok.json` 15,889 B |
| 封存的回合数 | 25 |
| DEV / FINAL 名单 | **5 / 20**，互斥 |
| 每条记录同时携带 `content_sha256` 与 `state_action_sha256` | PASS |
| 封存 seed（`r10-dev-v1`）、协议 hash、旧 100 清单 hash、完整 git sha | PASS |
| `manifest_sha256` | 记录 |

### 负例（15 个，全部 rc=9 且未落盘）

| 用例 | rc | 冻结文件 |
|---|---|---|
| 旧回合改名 | 9 | 不存在 |
| 同 state/action 只改 instruction | 9 | 不存在 |
| 9 KB 占位文件 | 9 | 不存在 |
| 缺相机 | 9 | 不存在 |
| RGB 通道异常（2-D 数组） | 9 | 不存在 |
| 30 Hz | 9 | 不存在 |
| 帧数 <100 | 9 | 不存在 |
| 错误任务指令 | 9 | 不存在 |
| 回合数 24（≠25） | 9 | 不存在 |
| 候选内重复 | 9 | 不存在 |
| 协议漂移（改一个指标） | 9 | 不存在 |
| test-only override 试图冻结 | 9 | 不存在 |
| 冻结路径已存在（预置文件） | 9 | **原文件 hash 未变** |
| 候选目录为空（无法构建 manifest） | 9 | 不存在 |
| 截断损坏的 HDF5 | 9 | 不存在 |

## 3. 边界

* 所有 HDF5 均为**合成数据**，因此本轮结论只能是 `VERIFIED_CPU_TESTED`；真实 25 回合尚未采集（见 `report/R10_2_COLLECTOR.md`）。
* 本轮**未评测** FINAL，也未用 FINAL 调参。
* 证据：`evidence/r10_2/b_freeze_cli_audit.json|log`、四个回归日志、`tools/r10_2_b_freeze_cli_audit.py`。
