# R8_VLA_42D_INTEGRATION

工作线 B（P1）：OpenVLA-7B 的 42 维历史 proprio + 速度残差目标的集成与 GPU Smoke。
未启动任何训练（无优化器步、无 3000/6000 步），未修改 R5–R7 代码与旧检查点，
未干扰其他 GPU 进程（三个外部进程 2.43 / 5.47 / 15.82 GiB 全程未被触碰）。

| 项目 | 值 |
|---|---|
| 分支 | `feature/r8-vla-42d-smoke`（基于审计 SHA `02acabc`） |
| 归一化契约 | 来自工作线 A（`experiment/r8-motion-normalization@22d0a95`）：`signed_motion`、q99、TRAIN-only |
| 冻结元数据 | `evidence/r8_vla_42d_smoke_20261020/frozen_r8_contract.json`（proprio 42 / velocity_residual / q01,q99） |
| 模型 | OpenVLA-7B pristine（`b5431a07…`，运行时校验）+ LoRA r32（110.8M）+ **42 维投影层（16,957,440 参数）** |
| CPU 测试 | `tests/test_r8_vla_adapter.py` **12/12**；`r8_smoke --check-contract` **10/10** |
| GPU Smoke | **已真实运行**：14/15 通过，1 项失败（KV-cache 等价），峰值显存 **21.88 GB**，wall 47.6 s |

---

## 0. 摘要（中文，带状态标签）

| 项 | 状态 |
|---|---|
| 42 维输入真正进入 OpenVLA（形状、投影层、前向） | **VERIFIED**（GPU：`projector_params=16,957,440`，`proprio_dim=42`，forward loss 有限） |
| 梯度流经新增投影层 | **VERIFIED**（fc1/fc2 梯度范数 2.01 / 20.86，fc1.bias 0.79，fc2.bias 2.54） |
| LoRA 梯度正常 | **VERIFIED**（852 个张量，最大范数 30.0） |
| **三个槽位（S / V / Acc）都真实影响输出** | **VERIFIED**（logit 最大变化 0.3125 / 0.3438 / 0.3438，阈值 0.05） |
| 输入扰动改变生成 token | **VERIFIED**（14 维中 9 维变化） |
| 速度残差目标 + 物理还原 + 夹爪 DEV 策略 | **VERIFIED**（CPU 10/10 契约检查 + GPU 解码路径；排除维度恰为 6、13） |
| 无未来帧泄漏 / 无 NaN·Inf | **VERIFIED**（破坏 S_{t+1} 后特征与目标逐位不变） |
| **KV-cache 与全序列 argmax 等价** | **FAIL / OPEN**：一次受控运行中 **8/14** 一致；数值敏感性 vs 结构性缺陷**尚未定论**（见 §3） |
| 旧 14/28 维模型兼容 | **VERIFIED**（元数据 fail-closed；宽度不符在加载前拒绝） |

**结论**：42 维集成的**功能链路已验证**（输入→投影→前向→梯度→槽位敏感性→解码→物理命令），
但**因果解码等价性未通过**，因此**还不能**宣布"R8 已具备 GPU 训练条件"；
这一项是进入 7B 训练前必须补齐的最后一个技术门禁。

---

## 1. 实现内容

| 文件 | 作用 |
|---|---|
| `code/xpolicylab/policy/OpenVLA_Pure/r8_vla_adapter.py` | 42 维特征构造（三帧 + 边界规则）、`ProprioHistory42`（每环境仅缓存最近两个**已观测**状态）、`build_projector`、`load_projector_state`（宽度校验）、`training_target`/`command_from_residual`、`decode_tokens_to_command`（含 DEV 夹爪策略）、`clip_command`、元数据 `metadata`/`resolve_metadata`（fail-closed） |
| `code/xpolicylab/policy/OpenVLA_Pure/r8_normalization.py` | 由工作线 A 提供并**原样引用**（`signed_motion` q99，零运动→0，符号保持） |
| `diagnostics/r8_smoke.py` | 独立 GPU Smoke harness（**不改**冻结的 R6 smoke）：`--plan` / `--check-contract`（CPU）/ `--kv-diagnose` / 默认 GPU 运行 |
| `tools/make_frozen_contract.py` | 把 A 的归一化方案 + R 目标的 train-only q01/q99 合并成**检查点元数据**并冻结 |
| `tests/test_r8_vla_adapter.py` | 12 项 CPU 测试（布局、边界帧、缓冲一致性、目标/命令、夹爪策略、编码界、元数据、真实投影层形状与参数量、真实 HDF5 无泄漏） |

**特征与目标（与工作线 A 冻结一致）**：

```
P_t = [S_t, V_t, Acc_t]  (42)      V_t = S_t - S_(t-1)      Acc_t = V_t - V_(t-1)
R_t = (A_t - S_t) - V_t            A_hat = S_t + V_t + R_hat
边界：t=0 → V=0,Acc=0 ；t=1 → Acc=0（训练、验证、部署同一实现）
夹爪：dim 6、13 使用 S+V（DEV 选定），其余 12 维使用模型
```

---

## 2. GPU Smoke 实测（真实运行，不是推断）

前置条件：冻结契约文件存在；显存空闲 **25.02 GB**（三个外部进程 2.43 / 5.47 / 15.82 GiB 在跑，
**未被终止**）；梯度检查点按空闲显存自动开启（数学等价）。

| # | 检查 | 结果 |
|---|---|---|
| 1 | 真实 7B 权重加载（pristine 动态模块） | PASS（`llm_dim=4096`） |
| 2 | 42 维投影层前向有限 | PASS（`proprio_dim=42`，参数 16,957,440） |
| 3 | 梯度到达投影层 + LoRA | PASS（见 §0 数值） |
| 4 | 三个槽位分别影响动作 logits | PASS（0.3125 / 0.3438 / 0.3438） |
| 5 | 输入扰动改变 token | PASS（9/14 维变化） |
| 6 | **KV-cache == 全序列 argmax** | **FAIL（8/14）** |
| 7 | token → 残差 → 物理命令 + 夹爪策略 | PASS（命令有限，排除维度 [6,13]） |
| 8 | 无 NaN/Inf、无未来泄漏 | PASS |
| 9 | 资源记录 | PASS（峰值 21.88 GB，wall 47.6 s） |

GPU 侧解码检查中 `residual_within_step=false` 属**预期**：该模型未经训练、生成 token 与真值残差
相差很远，超出"一个编码步"是正常的；编码界本身由 CPU 测试用真值 token 单独验证并通过。

---

## 3. 唯一失败项：KV-cache 与全序列不一致（OPEN）

| 维度 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KV | 132 | 102 | **177** | **176** | **108** | **130** | 0 | 0 | 128 | 128 | **127** | 114 | **127** | 128 |
| 全序列 | 132 | 102 | **121** | **243** | **135** | **210** | 0 | 0 | 128 | 128 | **126** | 114 | **139** | 128 |

**已知事实**：
* 前两步完全一致（132、102）——这两步之前只用到 prompt 前缀，两条路径计算完全相同；
* 不一致从**第一次使用缓存的自回归步**开始；
* 模型处于 `eval()`，同一 proprio、同一图像、同一 prompt；bf16 autocast；
* 同一 harness 的 28 维前身（R6）在同一台机器上曾以 14/14 通过，因此**不能**把结果归因为
  "42 维接口本身有害"。

**尚未完成的诊断**：`--kv-diagnose` 模式已写好（逐步记录两条路径的 256-bin logits、
top1−top2 间隔，判断"间隔 < logit 差 ⇒ bf16 数值敏感"还是"结构性缺陷"），
但其**第一次运行因我的 harness 缺陷中断**（`past_key_values` 未在循环中正确传递，
已修复并有编译验证），**在共享 GPU 上没有继续占用更多轮次**。
因此该模式状态为 **IMPLEMENTED，未取得判定**；KV 等价性为 **OPEN**。

**修复路线（下一步，需要一次 GPU 时段）**：
1. 运行 `--kv-diagnose`（约 1 分钟 + 模型加载）：看首次不一致步的"间隔 vs logit 差"；
2. 若为数值敏感：把该检查改为**容差判定**（logits 最大差 < 阈值，或 top-1 概率质量差），
   并在 fp32 下复测一次；
3. 若为结构性：检查 `enable_pure_autoreg` 的动作掩码与 KV 拼接的交互（R3 修复点），
   并补一个最小复现（单层/短序列）。

在 2 或 3 完成前，**不得**把 KV 解码当作已验证的部署路径。

---

## 4. 集成检查清单（对照任务书 B1/B2）

| 项 | 状态 | 证据 |
|---|---|---|
| 42 维输入真正进入 OpenVLA | VERIFIED | GPU 检查 1/2/4 |
| ProprioProjector 可加载与训练 | VERIFIED（前向+梯度）；训练本身未授权 | GPU 检查 2/3 |
| LoRA 参数与投影层梯度正常 | VERIFIED | GPU 检查 3 |
| 旧 14/28 维模型兼容 | VERIFIED | `resolve_metadata` fail-closed 测试 |
| 首帧/第二帧历史处理 | VERIFIED | CPU 边界测试 |
| 多环境历史缓存隔离 | VERIFIED | `reset`/env_idx 测试 |
| 训练与推理特征布局一致 | VERIFIED | 在线缓冲 = 离线向量（逐位） |
| 无未来帧泄漏 | VERIFIED | CPU + GPU |
| TRAIN-only q01/q99（R 表示） | VERIFIED | 冻结契约里记录来源与数值 |
| 256-bin 编码 / 自回归生成 / 解码还原 | VERIFIED（除 KV 等价） | GPU 检查 7 + CPU 编码界 |
| 关节上下界裁剪 | VERIFIED | `clip_command` 测试 |
| 夹爪 DEV 固定策略 | VERIFIED | 排除维度 [6,13]，三种子一致 |
| 检查点保存/加载与元数据 | VERIFIED（元数据层）；实际 7B 检查点未训练 | `metadata`/`resolve_metadata` |
| **KV-cache 等价** | **OPEN** | §3 |

---

## 5. 交付与复现

```bash
export PUSHT_HDF5=/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data
export PUSHT_SPLIT=/home/dgut/pusht_runtime/split80.json
export PUSHT_OFT=/home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_OFT/openvla_oft
export PUSHT_RUNTIME=/home/dgut/pusht_runtime
export PUSHT_MODEL=/home/dgut/pusht_runtime/model_cache/openvla-7b-pristine
export PUSHT_BOUNDS=/home/dgut/pusht_runtime/push_T_bounds.npz

python tests/test_r8_vla_adapter.py                       # 12/12
python diagnostics/r8_smoke.py --plan                     # 前置条件清单
python diagnostics/r8_smoke.py --norm-contract evidence/r8_vla_42d_smoke_20261020/frozen_r8_contract.json \
    --check-contract                                      # CPU 10/10
python diagnostics/r8_smoke.py --norm-contract <frozen.json> \
    --min-free-gb 20 --out <dir>                          # GPU smoke（需空闲显存）
python diagnostics/r8_smoke.py --norm-contract <frozen.json> --kv-diagnose --out <dir>   # KV 判定（待执行）
```

证据：`evidence/r8_vla_42d_smoke_20261020/`（冻结契约、GPU/CPU smoke JSON、三次尝试的日志、源文件 SHA）。
