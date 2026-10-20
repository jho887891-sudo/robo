# R10.1｜资源解锁：已获批的有条件优雅终止与后续 Gate

> **授权来源**：研究负责人在 2026-10-20 对“在确认评测属于本项目、结果已完整归档且无人依赖后，优雅结束旧 Push-T 长跑评测及其 :6020 策略服务，以释放 GPU；不包括其他人的进程或训练”的明确答复：“可以”。
> **状态**：`STOP_PROJECT_EVAL_CONDITIONALLY_AUTHORIZED_NOT_EXECUTED`。
> **本文件是指令和授权边界，不代表停止动作/GPU KV/数据采集已经发生。**
> **源仓库**：`jho887891-sudo/robo`；执行基线 `r10/cpu-gates@4af5d603c6410ec951ac8b1b138312bbdebb9c24`；审计补丁 Draft PR #14 (`review/r10-freeze-gate-resource-audit-20261020`)；`main@329614f54d3090b9b6ab120855ba50fbc93bdd1e` 保持只读。

## 1. 审批矩阵（不要扩大解释）

| 操作 | 状态 | 范围 |
|---|---|---|
| 只读核对旧 push_T 评测/PID/日志/结果与依赖 | AUTHORIZED | 可立即并行 CPU 实施 |
| 归档旧评测日志、run_id、分数/录像/进度、原始文件 SHA | AUTHORIZED | 只读原件，写新存档，不覆盖 |
| 优雅结束旧项目评测客户端和 :6020 策略服务 | **CONDITIONALLY_AUTHORIZED** | **必须先**确权、确认结果完整持久化且无其他消费者依赖；仅停止已核实的本项目旧任务进程 |
| 停止其他用户、共享 Isaac Sim 核心、整个 `rd_setup` 容器或未知 PID | **NOT_AUTHORIZED** | 不允许 |
| `kill -9`、`pkill`、`docker stop`、清理其他容器、任意 GPU 抢占 | **NOT_AUTHORIZED** | 不允许 |
| 一次 KV 诊断 | **ALREADY_AUTHORIZED** | R10 原额度 1 次，300 秒含加载，启动前空闲 ≥24 GiB，不训练；R10 记录 GPU 尝试次数 0 |
| 25 回合全新 Push-T 数据采集（5 DEV/20 FINAL） | **ALREADY_AUTHORIZED IN PRINCIPLE** | 必须有确证的安全采集器与独立资源窗口；此次“可以”**不构成**未知 `k1_*` 脚本的执行许可 |
| 7B optimizer step、正式训练、重复 GPU、超出 25 回合、合并 main、正式部署 | NOT_AUTHORIZED | 一律保留独立申请 |

## 2. DeepSeek 直接执行：并行 CPU，不等待小步骤回报

### A 线｜评测进程确权与证据保全（与 B/C 的只读工作并行）

1. 独立工作树及全新 `evidence/r10_1_<UTC>/` 目录；记录完整 Git SHA、用户、主机、`date -u`、环境。**重新核对进程，而非相信 2026-10-20 15:31:49 UTC 的旧快照。**
2. 只读采集：`nvidia-smi`、`ps -fp 3451345,3451404,3469786,3111650`、`pstree -aps`、`docker ps`、`docker top rd_setup`、`ss -ltnp`（可用时）、`/proc/<pid>/cmdline`、用户 UID、cgroup/container、服务 6020 对应消费者及评测客户端启动方式；保存原始输出与退出码。**旧 PID 若已变更，不按号码停止新的无关程序。**
3. 按 `run_id` 收集旧 10 路评测的已完成/进行中状态、分数文件、完整度、关键过程量、录像、最后输出时间及退出/恢复能力。留存清单/文件大小/sha256/路径与可重复校验命令。若尚有合法运行中的评测、结果不完整又无法安全恢复、归属不明或存在其他消费者：不停止，标 `BLOCKED_ACTIVE_EVAL` / `BLOCKED_OWNER` / `BLOCKED_EVIDENCE`，CPU 工作继续。
4. 三项确证全部成立（**本项目自有旧评测、已完成或安全停止点且关键证据已完整持久化、无其他依赖使用者**）时，才开始优雅结束。先查找现有 client 正常关闭/停止接口；优先使用该接口停止对应评测任务，等待其确认写盘，再通过其官方停止接口结束只服务本旧评测的 :6020 策略进程。若没有正常关闭接口，可对**再一次精确确认归属和命令行后的本项目 PID**使用定向 SIGINT、必要时 SIGTERM，等候退出并保留日志。**禁止 SIGKILL、按名称范围性 pkill、停止容器、清理系统任务、影响 Isaac Sim 共享基础进程。** 如进程未正常退出或存在不确定性，停止进一步信号并报告 BLOCKED。
5. 停止后重新读检：`nvidia-smi` 空闲（GiB，2^30）、非本项目 PID 仍在且未受干扰、老评测结果完整且 SHA 一致，保存两端快照与资源释放前后表。

### B 线｜冻结审计补丁远程 CPU 回归

1. 从 PR #14 固定审计 SHA 在独立 worktree 取代码；运行新增 `tests/test_r10_freeze_execution_audit.py`、原 `tests/test_r10_freeze_guard.py` 和 PR #11/#12 回归；报告实际 PASS/FAIL 与原始日志。不把先前 R10 的 10/10 当作本补丁测试。
2. 验证正式冻结只有真实 25 集、5 NEW_DEV / 20 NEW_FINAL、固定 r9.1 协议 SHA、全 40 位 Git SHA，且实际冻结 JSON 中保存两份名单和其 hash；test-only override 不允许冻结；任何失败不得返回 0 / 覆盖旧产物。
3. 如代码有最小可复现缺陷，独立分支补测试后做最小修复，不修改原始 R9/R10 实验证据。失败 2 轮仍未解决，只阻断 DATA-FREEZE，不影响安全的资源检查或已授权 KV。

### C 线｜只读审计未知采集器

查找远程的 `k1_final_take_action.py` 与 `k1_ori2_full_push.py` 实际文件路径/SHA/入口/导入链；只读确认其用途是否为 expert 新示范采集器，验证其对 `rd_setup`、三相机、25Hz、14D、HDF5 schema、随机种子、初始位姿变化、动作语义、保存路径隔离与文件覆盖保护。**不得运行、改写或把旧数据转存为新数据。** 形成“可用/必须最小改造/不适用”的证据表，交研究负责人决定是否授权该具体脚本；若远程现有官方 expert 采集接口已明确且在 R10 既有授权范围内，也先留出固定 SHA 的安全证据，不做未经确证的执行。

## 3. GPU KV 诊断：在资源合格时自动推进既有一次授权

A 线保全结束并确认项目专属资源可释放，或资源自行充足时：

- 重新测空闲显存 **≥24 GiB**，同时确认其他 GPU 用户没有新负载抢占；CPU P0 / 旧 64 TRAIN frozen contract hash / KV `r10_kv_logits.npz` sidecar 代码的回归仍 PASS。
- 如 `ACTION_DIM=7` 风险会使本 GPU 探针错误地以 7D 解码动作，则必须在真实入口增加 14D 预检或 fail-closed。新探针应断言 14 个步骤及窗口 `31744..31999`，不允许因错常量执行无效 GPU。
- 使用原 R10 已批准的**唯一一次**诊断，约束 `timeout 300s` 包括权重加载；外层 shell `set -o pipefail` 并即时捕获原始 `PIPESTATUS[0]`，保存原始 stdout/stderr、命令、启动/退出时间、GPU 快照、模型/代码/契约 SHA、14×256 KV/full logits、哈希、cache/prefix 时序和逐 token margin。退出非零、超时或 OOM 一律留证且**不自动重试**。
- `DIAG_EXECUTED`、`KV_EQUIVALENT`、`ROOT_CAUSE_RESOLVED` 三个 Gate 单独判断：14/14 一致可说明这次同前缀等价；若失配，原因保持 `OPEN_UNRESOLVED`，不能拿启发式 0.05 判 bf16 根因。**KV 诊断不解锁 7B 训练。**
- 如果安全归档后仍不满足 24 GiB，记录 `BLOCKED_RESOURCE`，不降低阈值或终止其他进程。

## 4. 新采线路与权限边界

- 25 集的新采任务**仍有授权**，但当前没有已确定安全的采集程序；本任务先完成 C 线只读审计。此次批准旧评测停机**不意味着**授权执行未验证的 `k1_*`。
- 只有已审核且在 R10 原授权范围内的**确定采集入口**与独立 GPU/容器窗口都具备时，才能继续 25 集采集；不属于该范围或改动运行机制的未知脚本，提交代码 SHA 与风险申请具体授权。
- 数据与 KV GPU 窗口串行；25 新回合中 5 NEW_DEV、20 NEW_FINAL，禁止旧 100 改名、旧轨迹重放、不同任务填充；任何正式冻结只能通过最新审计通过的真实 `r10_freeze_guard`，冻结后仍**不进行 FINAL 评测**。

## 5. DoD / 失败收口 / GitHub 与 ROUTE

- 独立分支含 `report/R10_1_RESOURCE_RELEASE.md`（归属与评测完成证据、停止动作前置判断、对每个具体 PID 的精确动作与结果、GPU 前后快照）、`report/R10_1_CAPTURE_ENTRY_AUDIT.md`、`report/R10_1_KV_RESULT.md`（如真正运行）、`mission/R10_1_MANIFEST.json`（每线 PASS/BLOCKED 与原始复算路径）； `evidence/r10_1_<UTC>/FAILURES.md` 不覆盖旧证据。
- `ROUTE.md` 只追加**已发生**的事项，包括本次授权与后来实际停止/测量/诊断/采集结果；不把授权写成执行成功。
- 受资源/归属/日志验证阻断时只停相关子线；CPU 归档与修复不等待独占 GPU。不得重复请求普通小步骤确认；涉及非本项目进程、未知采集器执行、训练、额外 GPU 与额外采集时必须另行批准。

**当前结论**：仅 `CONDITIONALLY_AUTHORIZED`，尚无远程停机/释放显存/7B KV/25 集成功证据。
