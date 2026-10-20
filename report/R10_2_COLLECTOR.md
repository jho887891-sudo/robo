# R10.2 D1/D2 — 采集入口审计：无合规采集器，`BLOCKED_COLLECTOR`

**状态**：`BLOCKED_COLLECTOR`（**未运行**任何 k1_* 脚本、未采集、未下载）

---

## 1. 只读审计结果

| 候选 | 路径 / sha256 | 结论 |
|---|---|---|
| `k1_final_take_action.py` | `/home/T7/ojh/robot_actual/RoboDojo/k1_final_take_action.py`，`98ddc82b…3e52`，462 行 | **不适用** |
| `k1_ori2_full_push.py` | `/home/T7/ojh/robot_actual/RoboDojo/k1_ori2_full_push.py`，`562f0205…3f5e`，550 行 | **不适用** |

两者都是 **Isaac Lab 单环境(curobo 规划器)专家实验脚本**，不是 `expert demonstration` 采集器，理由逐条：

1. **写出的"数据"不是回合数据**：只在成功时写一个 HDF5，字段是 `success`、`action`(1×14 **全零**)、`qpos_right`、`ee_pos`、
   `T_pos`、`target_pos` —— **没有 state 时序、没有 action 时序、没有相机、没有 instruction、没有 frequency**；
2. **保存路径固定且会覆盖**：`save_dir = "/data/workspace/openvla_offline_bundle/expert_k1"`，
   文件名固定 `expert_success_000.hdf5/json`，**没有 run_id、没有唯一目录、多次运行互相覆盖**；
3. **没有相机采集代码**（grep `cam_*`/`colors` 无命中），无法满足三路 RGB 要求；
4. 依赖 `isaaclab.app.AppLauncher` 与容器内资产，宿主 python 无法导入；运行需要 Isaac Sim GPU 窗口。

仓库内也没有其它 push_T 采集入口：`scripts/robodojo.sh` 只有 `eval / server / client`。

## 2. 因此

* 不因文件名相似而执行；不把旧轨迹改名/重放；不用跨任务数据充数；不擅自改用官方下载；
* 25 回合采集保持 `BLOCKED_COLLECTOR`，等研究负责人指定**具体**采集程序与其 SHA（或明确授权改造其中一个 k1 脚本），
  并给出与 KV/其他用户不冲突的独立 GPU 窗口。

## 3. 已经准备好的一切（拿到数据即可执行）

* `r10_freeze_guard`（PR#14 版）真实执行 g1–g6：schema（三相机、25 Hz、14D、≥100 帧、有限值、≥100 KB）、
  任务指令、回合数 25、内部去重、跨旧 100 三重哈希去重、verify；任一失败退出码 9 且拒绝冻结；
* 冻结时把 **5 DEV / 20 FINAL 名单 + content/state_action hash + seed + 协议 hash + 旧清单 hash + 完整 git SHA** 封进被 SHA256 的 manifest；
* CLI 级 21/21 验证（含 15 个负例，见 `report/R10_2_FREEZE.md`）。