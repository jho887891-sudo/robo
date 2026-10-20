# R10 独立交叉审计与资源恢复执行边界

**审计对象**：`r10/cpu-gates@4af5d603c6410ec951ac8b1b138312bbdebb9c24`。  
**状态**：R10 原始 CPU 结果有 GitHub 原始归档支持；本分支冻结修复为 `IMPLEMENTED / REMOTE_TEST_PENDING`，未运行远程真实 HDF5、GPU 或采集程序。  
**保护**：`main@329614f`、R9 四分支、PR #11/#12、R10 原始证据与任何运行中 PID 均不修改。

## 一、独立核对

- R10 执行树与固定 HEAD 存在；`mission/R10_MISSION_MANIFEST.json`、`evidence/r10/`、回归日志、失败归档存在。
- 已存 GPU 资源快照时间 `2026-10-20T15:31:49Z`：total **47.402 GiB**、free **23.300 GiB**、旧项目服务与 Isaac Sim 进程占用，少于要求 **24.000 GiB**；GPU-KV = **BLOCKED_RESOURCE，0 次尝试**。
- 新采 25 集在报表中为 `BLOCKED_ENV`，无采集器可执行 SHA、无新 HDF5 或 FINAL manifest；不可改报 PASS。
- 探针 CPU sidecar 回归有证据，但 R8 真实 7B 历史 KV 失配 8/14，**根因仍 OPEN**。

## 二、审计发现及最小修复（未改旧证据）

### P0-FRZ-1：冻结命令可能在未冻结时返回 0 / test override

原 `tools/r10_freeze_guard.py` 的测试入口可通过 CLI `--allow-small-set-for-tests` 改变阈值；甚至在提供 `--freeze-out` 但缺失 `--git-sha` 时，若检查项通过会打印拒绝冻结，却仍以 0 返回。

**独立分支修复**：正式冻结须恰好 **25 个候选**、`5 DEV`、`20 FINAL`、完整 40 位 Git SHA；禁止 test override 联合正式 freeze；传入的输出已存在则 fail-closed，不覆盖证据。任意拒绝须返回非零。

### P0-FRZ-2：冻结产物不包含 DEV/FINAL 名单

原程序通过 `M.freeze(res['manifest'], ...)` 只封装全集内容哈希，虽然预先计算了 `assign_splits`，实际输出没有包含哪 5 集是 DEV、哪 20 集是 FINAL。

**独立分支修复**：`freeze_checked()` 把 `r10_split_assignment` （dev、final、seed）、`protocol_sha256`、`used_manifest_sha256`、`guard_result_checks_sha256` 写入要哈希封存的 manifest，且强制使用原定 r9.1 协议 SHA `e5836191ed4c90d2482211ec5e7266eb58444ad742f59cb66eab84972b889f24`。

### P1-RUN-1：GPU shell 日志管道退出码可能失真

R10 报告给出的
```bash
timeout 300s python diagnostics/r8_smoke.py ... 2>&1 | tee run.log
```
若未启用 `pipefail`，shell 通常把最后一个 `tee` 的成功退出码当作整条管道成功；即使探针超时、OOM 或返回 6，也可能误报 0。

**执行修正**：参见下面的 `R10.1` 指令，使用 `set -o pipefail` 并独立保存 `PIPESTATUS[0]` 和 sidecar 校验，不以 shell 成功独立判断诊断结果。

### P1-EVAL-1：共享进程归属未取得停止授权

评测与策略服务的 PID/命令名在快照中可见，但记录未证明当下仍占用、是否属于当前用户所有、是否有未归档的 10 路正式评测结果。**显存差 0.7 GiB 不能成为未经确认终止共享进程的理由。**

## 三、新增回归与待验

`tests/test_r10_freeze_execution_audit.py`：合成的真正 `freeze_checked` 文件写入测试，至少包括 DEV/FINAL 5/20 存入哈希 manifest、坏 Gate 拒绝、短 Git SHA 拒绝、旧目标覆盖拒绝、删减回合拒绝、DEV/FINAL 重叠拒绝、测试 override 拒绝、未完整 Gate 拒绝、协议 drift 拒绝和 CLI 假冻结负例。

**实测声明边界**：这些测试已写入 GitHub，**本对话环境无法访问远程 HDF5/Python runtime，尚未声称远程 PASS**。DeepSeek 必须在独立 worktree 真实重跑，附原始 stdout/stderr 与 SHA；不能用 R10 原本的 `10/10` 替代本审计测试的结果。

## 四、资源解锁执行规则（尚未批准停机）

### 第一阶段：只读归档/评测状态核查（可以立即做）

1. 针对 PID `3451345/3451404/3469786` 与 `3111650`，只读记录 `ps -fp`、`pstree -aps`、`nvidia-smi`、`docker ps/top`、容器运行命令、端口 6020 的监听进程、启动用户与父子进程、run_id、运行中回合进度和最近输出时间。
2. 对 10 路评测，判断**仍有产出/已完成/卡死**，查结果是否持久化（任务分数、过程量、视频、日志），写证据清单和哈希；不能假设「运行 2 天就一定应该停」。
3. 如果未确认这些进程是本项目可停止的评测与服务器，则保持原状并报告 `BLOCKED_OWNER`；禁止执行 `kill`、`docker stop` 或 `pkill`。如果归属确认但用户尚未授权停止，只提交优雅终止顺序/残余风险供批准。
4. 并行只读审计主机中存在的 `k1_final_take_action.py`、`k1_ori2_full_push.py`：固定真实文件 SHA，核对是否为 **expert 数据采集** 而不是旧闭环测试、是否写真实 HDF5、相机/时间轴/关节维度是否匹配，是否覆盖旧数据、是否需要 Isaac Sim/容器，是否存在仅用于旧任务的随机种子/路径。**不得运行未知脚本。**

### 第二阶段：得到明确的进程停止许可/独占窗口后

- 仅在用户确认为本项目自有、评测证据已持久化并同意结束后，优先使用官方/已有脚本的 **优雅退出机制**；依赖关系里通常先结束 eval client，再结束 :6020 策略服务；`3111650` 或无法确权的共享程序绝不停止。不要直接 `kill -9`、`docker stop`。
- 重新测量真实空闲显存 `>=24 GiB` 和负载；R10 CPU 修复回归及 frozen q01/q99 hash 检查通过后，**只执行一次**已授权 GPU KV 诊断（包含权重加载、≤300 秒），任何启动前失败不消耗 GPU 次数，启动后不自动重试。
- shell 执行必须保留管道原始退出码，例如：
```bash
set -o pipefail
set +e
timeout 300s python diagnostics/r8_smoke.py --kv-diagnose \
  --norm-contract evidence/r9_gate_20261020/frozen_contract_64train.json \
  --min-free-gb 24 --out "$NEW_OUT" 2>&1 | tee "$NEW_OUT/run.log"
kv_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$kv_status" > "$NEW_OUT/probe_exit_code.txt"
```
- 必须验证 JSON/NPZ 真实存在、SHA、14×256 列、prefix/mask/cache 长度、第一处分叉，且诊断执行完成与 KV 14/14 功能 Gate 分开。
- KV 工作和 Isaac Sim 采集**串行**；R10 已授权的 25 集任务可在确定安全的采集程序与独立窗口满足条件后执行。采集后先验 `r10_freeze_guard` 的真实 HDF5 Gate，再完成新 DEV/FINAL 5/20 封存；本轮不看 FINAL 模型表现。
- 任何 OpenVLA optimizer step、额外 GPU 会话、额外采集、全数据下载、合并 main、覆盖旧权重/证据、正式部署仍禁止。

## 五、交付与审核

- 独立分支 + SHA；原始命令与退出码；资源状态与 PID 所属证据；10 路评测结果归档情况；`k1_*` 代码审计结论；独立冻结回归实际结果；真实 GPU KV JSON/NPZ（如果条件满足）；25 集采集与冻结内容哈希（如果条件满足）。
- 将已经发生的 R10 结果作为补充附在 `ROUTE.md` 后面；**不重写原始失败报告、也不把未运行写成 PASS**。
- 当前决定建议：`STOP_SHARED_PROCESS=NOT_AUTHORIZED`、`UNKNOWN_K1_EXECUTION=NOT_AUTHORIZED`、`TRAINING=NOT_AUTHORIZED`。待研究负责人批准安全停机后才解除第一个标记。
