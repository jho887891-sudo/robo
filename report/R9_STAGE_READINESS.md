# R9 阶段就绪总览（一页地图）

**Mission** `RD-R9-GATE-20261020-01` ｜ **执行** DeepSeek ｜ **审计** ChatGPT ｜ **审批** 研究负责人
**基线** `review/r8-three-lane-contract-audit@5242237`（本次未改动） ｜ **main** `329614f`（未改动）
**机器可读清单** `mission/R9_MISSION_MANIFEST.json`

---

## 1. 四条工作线与 Gate 状态

| 线 | 分支 | HEAD | Gate | 标签 | 真实运行？ |
|---|---|---|---|---|---|
| A 冻结契约 | `experiment/r9-freeze-64train` | `f7b8b39` | **PASS** | VERIFIED_CPU_HDF5 | 是（真实 100 集 HDF5） |
| B 训练入口 | `feature/r9-vla-trainer-contract` | `385ff4d` | **PASS_CPU** | VERIFIED_CPU + MOCK_FORWARD + PENDING_GPU | 是（CPU，零优化步，未载 7B） |
| C KV 诊断 | `diagnostics/r9-kv-same-prefix` | `02735bc` | **C-CPU PASS / C-GPU BLOCKED** | VERIFIED_CPU + BLOCKED_WAITING_FOR_AUTHORIZATION | CPU 是；GPU 否 |
| D 独立评测 | `research/r9-new-test-readiness` | `f8e43b0` | **PREP PASS / DATA BLOCKED** | VERIFIED_CPU + BLOCKED_DATA_ACQUISITION_AUTHORIZATION | 是（只读清点/哈希） |

测试合计：A 15/15（含 7 项负对照）＋B 16/16＋C 10/10＋D 13/13（另继承 R8-C 6/6）；
流程性检查：A 独立复算 9/9、CPU 契约 10/10；B trace 11/11；C 自检 PASS（8/8 同前缀 + 2 项对照）。
**每一线都保留了失败迭代日志**（`evidence/*/FAILURES.md`），验收标准未放宽。

## 2. 这一轮真正解决的问题

1. **审计 F1 已修复并可复算**：目标 q01/q99 只来自 **64 TRAIN**（19,127 行），16 个 DEV 已排除；
   与旧 80 集拟合相比单维最大相对变化 **29.3%**，说明这不是名义修正。旧 80 集文件在新路径 **fail-closed 拒绝**（退出码 1），原文件 SHA 未变。
2. **42D/[S,V,Acc] + R 目标第一次穿过"近似真实"的训练链路**：真实 entrypoint 模块被导入、真实 prompt/label（15 个监督位）、
   真实 HDF5→42D（与手工重算偏差 0.0）、R 恒等、冻结 256-bin、解码回物理命令（范围内误差 1.59e-4 ≤ 一个 bin）、夹爪 6/13 恒速策略。
3. **KV 同前缀协议被验证并钉死**：相同前缀 8/8 一致（噪声底 8.0e-8）；位置偏移与缓存注入两种破坏都被检出；
   生产探针静态审计通过、旧 bug 形态（`[[W0 + tk]]`）不存在；失配时退出码 6，绝不假 PASS。
4. **独立评测只差授权**：本地 100 集 100/100 已用尽、0 未使用；跨任务数据被明确判为不可替代；采集预算按实测 **77.8 MB/回合** 给出（20/128 集 = 1.56/9.96 GB）。

## 3. 尚未解决（明确 BLOCKED，不是"通过"）

| 项目 | 状态 | 解锁条件 |
|---|---|---|
| 真实 7B 上 8/14 KV 失配的**原因** | BLOCKED | 一次有界 GPU 诊断（≤5 分钟含加载、≥24 GB 空闲），需 `R9_KV_GPU_AUTHORIZED=1` |
| 真实 7B 的 42D 前向/反向/loss | PENDING_GPU | 同上窗口 + 常量断言（见下） |
| 新独立 FINAL 测试集 | BLOCKED | 研究负责人批准下载或采集，然后 g1–g6 冻结 |
| 任何 7B 优化步 / 正式训练 | NOT AUTHORIZED | 需 A/B/C-GPU 收口 + 预算批准 |

**新发现的开放风险（HYPOTHESIS）**：环境里 `prismatic` 常量按 `sys.argv` 子串判定平台，
因此解析成 LIBERO 的 `ACTION_DIM=7 / NUM_ACTIONS_CHUNK=8 / PROPRIO_DIM=8`，而数据是 14 维；
旧入口 `evaluate()` 里的 `for k in range(ACTION_DIM)` 在该环境下只解码 7 个 token。
新 42D 入口不引用该常量（固定 14），但**任何 GPU 训练/评测前必须显式断言**，否则历史指标不可比。

## 4. 方法学修正（供审计核对）

审计给的 KV 判据 `margin ≤ 2·max|Δlogit|` 对任何真实 argmax 翻转**必然成立**，无法区分数值与结构原因。
R9-C 改为**噪声底分类**（≤0.05 → 数值上可能；>0.05 → 结构性可疑），阈值显式声明、仍**不**下根因结论。
该修正写在 `report/R9_KV_DIAGNOSIS.md` §4 与 `evidence/r9_kv_cpu_20261020/FAILURES.md` 第 3 行。

## 5. 依赖图

```
[Gate A 64TRAIN 契约 PASS] ──┐
[Gate B 入口 trace PASS] ────┼──> [需授权：一次 GPU KV 诊断] ──> [KV 判定] ──┐
[Gate C-CPU 协议 PASS] ──────┘                                            ├─> [7B 小规模训练申请]
[Gate D 工具/协议 PASS] ──> [需授权：数据获取] ──> [g1..g6 冻结] ─────────┘
```

## 6. 研究负责人现在可以做的三个决定

1. 批准 **一次** GPU KV 诊断（上限：≤5 分钟含权重加载、≥24 GB 空闲、不影响其他进程）→ 决定 KV 是数值还是结构问题。
2. 批准**数据获取路线**（官方增量 20/128 集 = 1.56/9.96 GB，或新采 20–30 集 = 1.56/2.33 GB）→ 决定最终测试集来源。
3. 是否在本轮之后提交**小规模 7B 训练申请**（需先完成 1，并接受"仅离线一步预测证据、无闭环成功率"的边界）。

## 7. 本轮明确**不**声称

闭环成功率、任何 7B 训练或泛化结论、KV 失配根因、旧 20/旧 100 是全新最终测试、
42D 入口已跑过训练步。
