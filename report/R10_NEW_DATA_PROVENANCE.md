# R10 新数据溯源 — 采集入口缺失 + GPU 占用，NEW-DATA BLOCKED_ENV

**授权**：新采 **25** 个 Push-T 回合（5 `NEW_DEV` / 20 `NEW_FINAL`），来自新仿真执行/新随机初始状态
**状态**：`BLOCKED_ENV`（未采集、未下载、未复用旧数据）｜ **分支** `r10/cpu-gates`

---

## 1. CPU 零采集 dry-run（R10 §4.1 要求）结果

| 探测项 | 结果 |
|---|---|
| 采集/记录入口 | `scripts/robodojo.sh` 子命令仅 `eval / server / client`；未发现 `collect/record` 入口 |
| 模拟器可用性 | `isaacsim` 在宿主 `python3` 与 `venv_oft` 中**均不可导入**；无 `isaac-sim` 可执行文件 |
| 容器 | `rd_setup`（`nvcr.io/nvidia/isaac-sim:5.1.0`，Up 2 days）内运行 `_run_client.sh push_T OpenVLA_Pure 6020 1`，10 路环境评测 |
| HDF5 writer | 仓库中只有评测/转换类脚本（`utils/process_data.py` 为格式转换、IsaacLab 工具为 mp4↔hdf5），无 push_T 采集写入器 |
| 已知可写 HDF5 的项目脚本 | `k1_final_take_action.py`、`k1_ori2_full_push.py`（历史实验脚本，未文档化为采集器；需模拟器与 GPU 窗口） |
| 现有数据 | 旧 100 集全部已用于 R2–R9（内容哈希 100/100 唯一，dataset `6757696…`），**0 未使用** |

**结论**：仓库内没有可直接执行的 Push-T 采集入口，且唯一能跑模拟的容器正被一个长跑评测占用。
按授权"数据采集可使用现有已获准的 RoboDojo expert/仿真程序"——目前不存在这样的程序入口，
因此判 `BLOCKED_ENV`，而不是改写任务或换路线。

## 2. 采集预算（授权用量上限已固定）

| 项 | 数值 |
|---|---|
| 每回合体积（旧 100 实测均值） | **77,793,273 B ≈ 77.8 MB** |
| 25 回合预估 | **≈ 1.94 GB**（仅本地只读目录，不入 Git） |
| 频率 / 维度 / 相机 | 25 Hz、14D state+action、`cam_head`+`cam_left_wrist`+`cam_right_wrist` |
| 每回合最少帧数 | ≥ 100（协议 g4） |

## 3. 冻结链路已实现并通过回归（等数据即可执行）

`tools/r10_freeze_guard.py`（新增）在 freeze **之前**从真实 HDF5 重算 7 类检查：

| 检查 | 失败后果 |
|---|---|
| 真实 schema：state/action 四组关节 14D、三路 RGB 帧数组与帧数对齐、≥100 帧、有限值、文件 ≥100 KB | 拒绝 |
| 任务：instruction 必须含 "Push the T-shaped block" | 拒绝 |
| 频率：25 Hz | 拒绝 |
| 回合数：恰好 25（默认 `--expect-total 25`） | 拒绝 |
| 内部去重：name / content_sha256 / state_action_sha256 | 拒绝 |
| 跨旧 100：三类哈希与 `manifest_existing100.json`（绑定 dataset `6757696…`）比对 | 拒绝 |
| verify：从磁盘重建清单得到同一 dataset hash | 拒绝 |

回归 `tests/test_r10_freeze_guard.py` **10/10**，含 7 个负例：
改名复用旧回合（content overlap=1）、同一轨迹换 instruction（state_action duplicate=1）、9 KB 占位文件、
30 Hz、缺相机、错回合数、guard 未通过时拒绝 freeze。

**划分**：`assign_splits` 按 `sha256(seed + content_sha256)` 排序取前 5 为 `NEW_DEV`，其余 20 为 `NEW_FINAL`，
种子固定 `r10-dev-v1`，输出两份名单哈希，DEV/FINAL 互斥且可复算。

## 4. 资源占用实测（阻断原因）

```
free 23.300 GiB / 47.402 GiB；外部进程占 24.102 GiB
/isaac-sim/kit/kit 2486 MiB | isaac-sim eval client 5604 MiB | venv python 16200 MiB
```

采集必然占用 GPU（渲染 + 模拟步进），在另一评测正在跑、且空闲不足 24 GiB 时执行会直接违反
"不影响他人任务 / 与 KV 诊断串行"的授权条件 → `BLOCKED_ENV`。

## 5. 一旦资源与入口就绪的执行顺序（不改任务）

1. 指定并核对采集程序（或其 SHA），确认保存路径只指向新的空目录；
2. 在**独立窗口**跑恰好 25 回合（失败回合保留日志，不补数、不改名复用）；
3. `r10_freeze_guard --candidate-dir <新目录> --expect-total 25 --expect-dev 5`，
   全 PASS 才 `--freeze-out ... --git-sha <SHA>`；
4. 提交两份 manifest + 名单哈希 + sidecar（**不提交 HDF5**）；
5. 本轮仍**不评测** FINAL。

## 6. 明确禁止（本轮已遵守）

改名复用旧回合、复制旧 state/action 改 instruction、用 stack_bowls/sweep_blocks 充数、
未经再次批准改用官方下载、冻结前偷看 FINAL、把失败回合用旧数据补齐。
