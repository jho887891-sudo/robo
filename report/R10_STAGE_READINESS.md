# R10 阶段就绪总览（授权已记录 · 执行受资源阻断）

**Mission** `RD-R10-AUTH-20261020-01` ｜ **执行** DeepSeek ｜ **审计** ChatGPT ｜ **审批** 研究负责人
**分支** `r10/cpu-gates`（基线 `026d48e`，即 PR #12 审计补丁）｜ **main** `329614f` 只读、未改动
**授权内容**：1 次有界 7B KV GPU 诊断（≤5 分钟，≥24 GiB 空闲）+ 新采 25 个 Push-T 回合（5 DEV / 20 FINAL）

---

## 1. 逐项状态（授权 ≠ 执行 ≠ PASS）

| Gate | 内容 | 状态 | 证据 |
|---|---|---|---|
| P0.1 | 复验 PR #11 内容去重工具（旧 100 清单 + 负例） | **PASS_CPU** | `evidence/r10/test_r9_final_disjoint.log`（9/9）、guard 负例 4 项 |
| P0.2 | 复验 PR #12 退出语义（7 未授权 / 8 授权未执行） | **PASS_CPU** | `wrapper_unauthorized.log`=7、`wrapper_authorized_no_run.log`=8、回归 12/12 |
| P0.3 | 新增 `r10_freeze_guard`（真执行 g1–g6，不靠文档声明） | **PASS_CPU** | `evidence/r10/test_r10_freeze_guard.log`（10/10 + 7 负例） |
| P0.4 | 每步 256-bin logits 落 `.npz` sidecar + sha256 | **PASS_CPU** | `evidence/r10/probe_patch.json`、`test_r10_kv_probe.log`（16/16） |
| P0.5 | 缓存 preflight（关 GC、强制 use_cache、拒绝空 cache） | **PASS_CPU** | 同上（preflight 3 项 + 静态 4 项） |
| P0.6 | A 契约复验 / B 42D trace 复验 / 14D 与 token 窗口 | **PASS_CPU** | `a_contract_recheck.log`（9/9）、`b_trace_recheck.log`（11/11）、回归 15/15 + 16/16 |
| **GPU-KV** | 一次真实 7B KV/full 诊断 | **BLOCKED_RESOURCE** | `evidence/r10/gpu_resource_probe.json`：空闲 **23.300 GiB** < 24 GiB |
| **NEW-DATA** | 新采 25 回合 + 5/20 冻结 | **BLOCKED_ENV** | 无采集入口；`isaacsim` 宿主不可导入；容器正跑评测 |
| TRAIN | 任何 7B 优化步 | **NOT_AUTHORIZED** | 未初始化 optimizer，未进入训练入口的 train 分支 |

## 2. 为什么 GPU 判 BLOCKED_RESOURCE（不是失败，也不是没做）

| 项 | 实测 |
|---|---|
| 显存 | 总 47.402 GiB，空闲 **23.300 GiB**，被外部进程占用 **24.102 GiB** |
| 阈值 | R10 要求 ≥ **24 GiB**（2^30），缺口 **0.700 GiB** |
| 占用者 | `3111650` isaac-sim kit 2486 MiB；`3451404` isaac-sim eval client 5604 MiB；`3469786` venv python 16200 MiB（本项目 6020 端口策略服务，起于 10-18） |
| 本轮动作 | **未启动**诊断、**未**终止/抢占任何进程、未做第二次尝试 |
| 解锁条件 | 空闲 ≥24 GiB（例如研究负责人决定结束 10-18 起的长跑评测），即按 `timeout 300s` 一次执行 |

## 3. 为什么新采 25 回合判 BLOCKED_ENV

1. `scripts/robodojo.sh` 只有 `eval / server / client` 三个子命令 —— **仓库内没有数据采集入口**；
2. `isaacsim` 在宿主 `python3` 和 `venv_oft` 中均不可导入，模拟只存在于容器 `rd_setup`（`nvcr.io/nvidia/isaac-sim:5.1.0`）；
3. 该容器当前正在执行 `_run_client.sh push_T OpenVLA_Pure 6020 1`（10 路环境评测），并对应 16.2 GiB 的策略服务；
   再叠加采集会同时在 GPU 上争用，违反"不与 KV 诊断/共享进程冲突"的授权条件；
4. 授权明确禁止：下载官方增量、扩大采集量、改名复用旧数据、跨任务充数 —— 因此**不能用替代方案包装成完成**。

**已就绪的部分**：预算（25 × 77.8 MB ≈ **1.94 GB**）、schema 校验、内容/state-action 双哈希去重、
种子化 5/20 划分、freeze 前置闸门与证据格式，全部可复算（`r10_freeze_guard`）。资源一到位即可执行，不需要重新设计。

## 4. 本轮真正新增的可执行能力

* **不再"文档说通过"**：`tools/r10_freeze_guard.py` 从真实 HDF5 重算 7 类检查，任一失败退出码 9 且拒绝 freeze；
* **GPU 诊断产物可独立复算**：每步 256-bin logits 进 `.npz`，JSON 记录 sha256/shape/dtype/列定义；
* **授权不等于执行**：wrapper 有授权但未真实跑 GPU 时返回 **8**，不会伪装成功；
* **缓存/检查点冲突被显式处理**：kv_diagnose 关掉 gradient checkpointing 并强制 `use_cache=True`，cache 为 None 即拒绝。

## 5. 需要研究负责人决定的三件事

1. **GPU 窗口**：是否结束 10-18 起的长跑评测（`3451345/3451404` 与其策略服务 `3469786`）以腾出 ≥24 GiB；
   我不会自行终止他人进程。
2. **采集程序**：仓库缺少采集入口，需指定一个（或授权使用既有 `k1_*` 脚本）并给出独立 GPU 窗口；
   若继续不可得，请确认把"25 回合"降级为 `BLOCKED_ENV` 交付，而不是改用下载或跨任务数据。
3. **训练**：仍不授权；即使 GPU KV 诊断跑通，功能 Gate C 若失配仍为 OPEN，不解锁训练。

## 6. 明确不声称

KV 根因已定位、25 回合已采集、FINAL 可评测、7B 训练或闭环成功率。
