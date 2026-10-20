# R10.2 D3 — OpenVLA_R42 部署适配与 checkpoint 元数据（仅 CPU，22/22 PASS）

**状态**：`IMPLEMENTED + VERIFIED_CPU`（未跑真实 7B 推理，未训练）
**新模块**：`code/xpolicylab/policy/OpenVLA_R42/r42_adapter.py`（不改 OpenVLA_Pure 的 14/28D 历史路径）

---

## 1. 接口能力

| 能力 | 实现要点 |
|---|---|
| 图像入口 | `validate_images()` 强制 **dict + 严格相机顺序** `cam_head, cam_left_wrist, cam_right_wrist`、`uint8`、`HxWx3/4`；顺序/dtype 不符直接报错，不做隐式 resize |
| episode 历史 | `EpisodeHistory`：**每个 env_idx 独立**、只存**已观测**状态、`reset` 立即清空、时间戳倒退即报错；t0 → V=Acc=0，t1 → Acc=0 |
| 42D 特征 | `concat(S14, V14, Acc14)`，V=S_t−S_(t−1)，Acc=S_t−2S_(t−1)+S_(t−2)，用**冻结契约**归一化（S 位置界、V/Acc signed q99，零速度映 0） |
| 标签 | `build_labels()`：prompt 全部 IGNORE，**14 个动作 token + STOP(2)** 共 15 个监督位，token 必须落在 **[31744, 31999]** |
| 解码 | `decode_command()`：`A_hat = S + V + R_hat`；夹爪 6/13 用 `S+V` 恒速回退；关节界裁剪 + **每周期速率限制**（`max_rate × dt`），并回报裁剪/限速计数 |
| 元数据 | `build/validate/save/load_metadata()`：19 个必填字段（task_id、三个模型 hash、proprio_dim=42、布局、t0/t1 规则、相机顺序、prompt 来源、契约/split hash、codec、token 窗口、目标表示、步数/seed、夹爪策略、安全界、对齐版本） |
| fail-closed | `proprio_dim` 为 7/8/14/28、窗口被改、缺字段、task 不符、对齐版本不符、hash 非 64 位 → 全部拒绝 |

## 2. 回归（`tests/test_r10_2_r42.py`，22/22 PASS）

覆盖：t0/t1 边界、**t2 加速度与手算一致**（发现并修复了真实缺陷：历史裁剪早于特征计算导致 Acc 恒为 0）、42D 有限性、
multi-env 隔离与 reset、时间戳倒退拒绝、相机顺序/dtype 拒绝、token 窗口、label mask（15 监督位）、
`A_hat=S+V+R̂` 恒等、夹爪恒速回退、速率限制确实生效、元数据往返与 5 类错误元数据拒绝。

> 发现的缺陷（已修）：`observe()` 原先在计算特征**之前**裁剪历史，第三个观测起 `prev2` 丢失 → Acc 永远为 0。
> 这正是把 CPU 回归放在真实训练之前的价值；修复后有专门断言锁住。

## 3. 未做 / 未声称

* **没有**真实 7B 前向；`MOCK/IMPLEMENTED` 之外一律不写 PASS。
* 未启动任何 optimizer；未来训练需要独立预算凭证（`R9_TRAINING_AUTHORIZED=1` **不是** R14 的训练批准）。
* R13「零优化步 7B Smoke（≤20 min）」的申请材料已列为待批，本轮**未**跑额外 GPU。