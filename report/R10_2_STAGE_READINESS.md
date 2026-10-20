# R10.2 阶段就绪总览（四线并行执行结果）

**Mission** `RD-R10.2-MAXSAFE-20261021-01` ｜ Spec `ac2d1ee` ｜ 执行 DeepSeek ｜ 审计 ChatGPT
**分支**（各自独立，均已推送）：

| 线 | 分支 @ HEAD | 状态 | 关键证据 |
|---|---|---|---|
| A 资源释放 | `r10.2/resource-release@a3396eb` | **EXECUTED** | 23.300 → **44.639 GiB** 空闲；两个本项目 PID 优雅退出；共享 kit 未动 |
| B FINAL 冻结 | `r10.2/final-freeze-audit@8d86a25` | **VERIFIED_CPU_TESTED** | PR#14 11/11、guard 10/10、PR#11 9/9、PR#12 12/12、CLI 审计 **21/21** |
| C 唯一 KV GPU | `r10.2/kv-diagnosis@3e1d56b` | **VERIFIED_GPU** | **1 次**、退出码 0、**98 s**、**14/14 一致**、logits sidecar `534b026c…`、C5 复算 11/11 |
| D 前瞻研发 | `r10.2/42d-deploy-12task@a51ef15` | **CPU 通过 / 采集 BLOCKED** | R42 **22/22**、scope+control+offline **20/20**、`TASK_SCOPE_V1` 12 任务、采集 `BLOCKED_COLLECTOR` |

---

## 1. 逐 Gate

| Gate | 状态 | 说明 |
|---|---|---|
| A OWNER_PROVED / EVIDENCE_SAFE / NO_OTHER_CONSUMERS | **PASS** | 进程链、命令行指纹、仓库/端口归属；旧评测自 10-18 无产出且 2.18 GB 日志含 **4,696,992** 个 AssertionError；仅一条 CLOSE-WAIT 旧连接 |
| A 优雅结束 | **PASS** | client `SIGINT` → 37 s 退出；server `SIGINT` 未退 → `SIGTERM` 立即退出；**无 SIGKILL / pkill / docker stop** |
| B 冻结闸门 | **PASS（合成数据）** | 正例真实写盘 15,889 B，封存 25=**5+20** 名单与 content/state-action 双哈希、seed、协议 hash、旧清单 hash、完整 git SHA；15 个负例全部 rc=9 且不落盘 |
| C DIAG_EXECUTED | **PASS** | 启动前空闲 44.639 GiB ≥ 24；`timeout 300s`，实测 98 s；无 optimizer |
| C KV_EQUIVALENT | **PASS** | 同前缀 **14/14**，中位 margin 1.5625 ≫ 中位 max\|Δlogit\| 0.234375 |
| C ROOT_CAUSE_RESOLVED | **HYPOTHESIS_SUPPORTED** | R8 的 8/14 无法复算（当时无逐步 logits）；本次在同协议下 14/14，支持"旧差异来自比较协议"的假设 |
| D 42D 部署适配 | **PASS（CPU）** | 相机顺序/多 env 历史隔离/t0-t1/42D 手算/label mask/解码恒等/夹爪回退/速率限制/元数据 fail-closed |
| D 12 任务 TaskSpec | **PASS（静态）** | 12/12 在本机 inventory；只有 3 个任务有本地数据，其余保持 UNKNOWN |
| D 控制时序与离线指标 | **PASS（CPU）** | latency = **NOT_MEASURED**（无真实 7B 运行） |
| D 新采 25 回合 | **BLOCKED_COLLECTOR** | 无合规采集入口；`k1_*` 经只读审计判为不适用（未运行） |
| 训练 | **未授权 / 未执行** | `optimizer_steps = 0` |

## 2. 本轮真正拿到的新事实

1. **GPU 窗口是可以合法拿回来的**：旧评测早已停止产出（2 天零结果）却仍在空转报错；按授权优雅结束后空闲 23.3 → 44.6 GiB。
2. **KV 同前缀在真实 7B 上是等价的（14/14）**，且这是**可复算**的证据：每步 256-bin logits 已归档，CPU 独立重算逐项一致。
3. **bf16 数值差是真实存在的**（中位 0.234 logit），但远小于 margin，不是翻转来源；0.05 阈值未用于判因果。
4. **冻结工具真的会拒绝**：15 个负例（含改名复用、同轨迹换指令、协议漂移、test-only override、已存在文件）全部 fail-closed。
5. **CPU 回归抓到了真实缺陷**：R42 适配器原先把历史裁剪放在特征计算之前，导致加速度恒为 0；已修复并加断言锁住。

## 3. 仍然阻断 / 需要决定

| 项 | 状态 | 解锁条件 |
|---|---|---|
| 25 回合新数据 + 5/20 真实冻结 | `BLOCKED_COLLECTOR` | 指定**具体**且已审计的采集程序（或授权改造 `k1_*`）+ 独立 GPU 窗口 |
| R13 零优化步 7B Smoke（≤20 min） | 未申请 | 需新的 GPU 预算批准 |
| 真实 7B 训练 | 未授权 | 独立预算 + 技术 Gate（KV 已通过不等于可训练） |
| 11 个其他任务数据 / 12 任务训练 | 未授权 | 需新的采集与训练授权 |

## 4. 边界（不声称）

* 没有闭环成功率、没有 7B 训练、没有 FINAL 评测、没有真实新数据、没有 latency 数字。
* `09bd5f5e…`（上游 RoboDojo 参考）在本仓库不可解析，标注 UNVERIFIED。