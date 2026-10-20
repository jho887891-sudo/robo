# R10.2 A — 旧 Push-T 评测与专属 :6020 服务的安全结束（已真实执行）

**状态**：`OWNER_PROVED` + `EVIDENCE_SAFE` + `NO_OTHER_CONSUMERS` 三项 PASS → **已优雅结束**
**分支** `r10.2/resource-release` ｜ **基线** `r10/cpu-gates@4af5d60`
**GPU 空闲**：23.300 GiB → **44.639 GiB**（GiB = 2^30）

---

## 1. 重新采集（不依赖 2026-10-20T15:31:49Z 的旧快照）

| 项 | 值 |
|---|---|
| UTC | 2026-10-20T16:40:01Z 起（`evidence/r10_2_20261020T164001Z/identity.log`） |
| 主机 / 用户 | 见 `identity.log`（`whoami=dgut`，UID=1000） |
| Git | `/home/T7/ojh/robot_actual/submission` 只读核对 |
| GPU | A6000 47.402 GiB 总，空闲 23.300 GiB，占用 24.102 GiB |

## 2. 归属与证据（三项 Gate）

### OWNER_PROVED
进程链（`docker_top.log`、`proc_cmdlines.log`）：

```
3451298 dgut   bash -c echo *** | sudo -S docker exec -e ROBODOJO_PROGRESS_LABELS=t,target_t \
                rd_setup bash /workspace/RoboDojo/_run_client.sh push_T OpenVLA_Pure 6020 1
3451305 root   sudo -S docker exec ... rd_setup bash /workspace/RoboDojo/_run_client.sh push_T ...
3451345 1234   bash /workspace/RoboDojo/_run_client.sh push_T OpenVLA_Pure 6020 1
3451351 1234   bash scripts/robodojo.sh client --task push_T --policy-name OpenVLA_Pure --policy-port 6020
3451356 1234   bash /workspace/RoboDojo/scripts/eval_policy.sh --task_name push_T --policy_name OpenVLA_Pure
3451404 1234   /isaac-sim/kit/python/bin/python3 -u src/eval_client/main.py --task_name push_T --num_envs 10 ...
3469786 dgut   venv_oft/bin/python XPolicyLab/setup_policy_server.py --config_path .../OpenVLA_Pure/deploy.yml --overrides port=6020 task_name=push_T
3111650 1234   /isaac-sim/kit/kit ...（共享 Isaac Sim 基础进程，未触碰）
```

仓库、命令、`deploy.yml`、`:6020` 端口与 `task_name=push_T` 都属于本项目，`user=dgut` 为项目账号。

### EVIDENCE_SAFE
* 客户端 stdout `/tmp/run_dbg2.log` **2,179,097,387 B**，sha256 `5259701f…6be206c6`，内含 **4,696,992 个 `AssertionError`**，
  说明该评测自 10-18 起处于破坏性错误循环；已保留 head/tail 各 2 MB 与完整哈希（未复制 2.18 GB 进仓库）。
* 容器 `/tmp/eval_result/_progress` 最后写入 **2026-10-18 13:01**，此后无任何结果文件；`find -mmin -60` 为空。
* 已归档 7 个进度文件（合计 ~260 KB）并逐个 sha256（`archive_sha256.txt`，`sha256sum -c` 全部 OK）。
* 容器 `eval_result` 下没有逐回合产物，故不存在"正在写盘的半成品"。

### NO_OTHER_CONSUMERS
* `ss -ltnp`：6020 仅本项目服务监听；`ss -tanp` 只有一条 **CLOSE-WAIT** 的旧连接（客户端已死）。
* `ps -eo pid,cmdline | grep 6020`：只有本项目评测链。

## 3. 实际发送的信号（逐 PID，SIGINT → 必要时 SIGTERM；从未 SIGKILL）

| PID | 角色 | 动作 | 结果 |
|---|---|---|---|
| 3451404 | 旧评测客户端（10 env，已空转 2 天） | SIGINT（16:42:05Z） | **37 s 后自行退出**，`RESULT=EXITED_ON_SIGINT` |
| 3469786 | 专属 :6020 OpenVLA_Pure 策略服务 | SIGINT（16:42:59Z）未退 → SIGTERM（16:43:59Z） | **立即退出**，`RESULT=EXITED_ON_SIGTERM` |
| 3111650 | 共享 Isaac Sim kit | **未发送任何信号** | 全程存活（`gpu_after.txt`） |

停机脚本对每个 PID 先做 `cmdline` 指纹校验（PID 复用防护），指纹不符即拒绝；
脚本内**没有任何 SIGKILL / pkill / docker stop** 路径（见 `scripts/r10_2_a_stop.py`）。

## 4. 前后对照

| 项 | 停止前 | 停止后 |
|---|---|---|
| 空闲显存 | 23.300 GiB | **44.639 GiB** |
| 占用显存 | 24.102 GiB | 2.763 GiB（仅共享 kit 2,486 MiB） |
| :6020 监听 | 有 | 无 |
| 共享 kit 3111650 | 存活 | 存活（未干扰） |
| 归档证据 | — | `sha256sum -c` 全部 OK（停止后复核） |

停止后空闲 44.639 GiB ≥ 24 GiB，直接满足唯一一次 KV 诊断的准入门槛（见 `report/R10_2_KV.md`）。

## 5. 边界

* 只停止了两个**已确权、已归档、无依赖**的本项目进程；未触碰 `rd_setup` 容器本身、未 `docker stop`、
  未停止共享 Isaac Sim kit，未影响任何其他用户进程。
* 停止不等于评测"成功"：该旧评测自 10-18 起没有产生任何逐回合结果，归档的是**失败/空转证据**，不是分数。

## 6. 证据

`evidence/r10_2_20261020T164001Z/`：identity / git / nvidia_smi / gpu_free / docker_ps / docker_top / port6020 / listeners / proc_cmdlines、
`archive/`（7 个进度文件 + sha256）、`logs/client_stdout_identity.txt`（2.18 GB 的哈希与大小）、
`stop_client.log`、`stop_server.log`、`scripts/r10_2_a_stop.py`、`scripts/r10_2_a_snapshot.sh`。
