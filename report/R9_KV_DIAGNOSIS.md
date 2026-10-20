# R9 Gate C — KV-cache 同前缀诊断（CPU 合约通过；GPU 判定 BLOCKED 待授权）

**工作包** C ｜ **分支** `diagnostics/r9-kv-same-prefix` ｜ **基线** `review/r8-three-lane-contract-audit@5242237`
**结论**：Gate C-**CPU** **PASS**；Gate C-**GPU** = **BLOCKED_WAITING_FOR_AUTHORIZATION**（未运行，不猜原因）。

---

## 1. 为什么必须重做（审计 F3 已修）

R8 的 GPU smoke 用"缓存自回归解码"对比"全序列重算"，两条路径一旦选了不同 token，后续输入历史就不同，
之后的 logit 距离/argmax margin 都不可比。审计基线已改为**每一步把 full 路径的 greedy token teacher-force 进缓存分支**，
两边始终看到同一历史。本线用一个**静态审计**把这个修复钉死，并补上缺失的 CPU 语义验证。

## 2. 归档证据的可用性（如实说明）

`evidence/r8_vla_42d_smoke_20261020/gpu_smoke/attempt3_partial_14of15.json` 只保存了 14 个 token 索引：

| 维度 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KV | 132 | 102 | 177 | 176 | 108 | 130 | 0 | 0 | 128 | 128 | 127 | 114 | 127 | 128 |
| full | 132 | 102 | 121 | 243 | 135 | 210 | 0 | 0 | 128 | 128 | 126 | 114 | 139 | 128 |

**没有逐步 256-bin logits、没有 margin、没有 dtype/首次分叉记录**，因此事后无法做数值-vs-结构的判别；
必须用审计后的探针在新契约上重跑一次。该失败记录完整保留，未被覆盖。

## 3. CPU 合约结果（`evidence/r9_kv_cpu_20261020/`）

环境：CPU、torch 2.4.1+cu121、transformers 4.40.2；微型 Llama（2 层，hidden 64，随机初始化，fp32），
window 取 8 个 token 以适配小词表；探针本身与生产探针共用同一套语义（同一前缀 teacher-force、逐步记录 Δlogit/margin）。

| 检查 | 结果 |
|---|---|
| 相同前缀（fp32） | **8/8 token 一致**，logit 噪声底 **8.0e-08** → 缓存传递/位置/掩码语义正确 |
| 对照 1：position_ids 偏移 +3 | **被检出**：中位 max|Δlogit| **1.75e-03**（≈噪声底 2.2e4 倍），argmax 未翻转 → 探针对结构性破坏敏感 |
| 对照 2：向缓存注入 1 行伪造 KV | **被检出**：**7/8** token 一致，**首次分叉 step 1**，Δlogit 最高 **0.0281** |
| bf16 vs fp32（同前缀、单次前向） | max|Δlogit| 2.16e-03，top1−top2 margin 0.133 → 在该微型模型上 bf16 不足以翻转 argmax（**不可外推到 7B**） |
| 静态审计生产探针（`diagnostics/r8_smoke.py`） | 5 项必需模式全部命中；旧 bug 模式 `[[W0 + tk]]` **不存在** |

## 4. 方法学修正：`margin ≤ 2·max|Δlogit|` 不能判别根因

审计给出的判据（margin ≤ 2Δ 则"与数值差兼容"）对**任何真实的 argmax 翻转必然成立**
（若要从 top1 翻到 top2，|Δ| 至少是 margin 的一半），因此它无法区分"数值敏感"与"结构性错误"。

本线改用**噪声底分类**，并把它写成显式阈值而不是结论：

* `mismatch_class = none` → `SAME_PREFIX_EQUIVALENT`
* 失配且 max|Δlogit| ≤ **0.05**（文档化的 bf16 logit 噪声量级）→ `OPEN_NUMERIC_PLAUSIBLE`（bf16 **可能**，不证明）
* 失配且 max|Δlogit| > 0.05 → `OPEN_STRUCTURAL_SUSPECT`（先查 cache/position/mask/token）

阈值 0.05 是**判据声明**，不是从 7B 实测得出的常数；GPU 重跑后应据实记录真实噪声底再更新。

## 5. 仍未解决的部分（明确 BLOCKED）

* 真实 7B 的 8/14 失配**原因未定**：既不能说"bf16 数值误差"，也不能说"缓存/mask 错"。
* 唯一能给出判定的动作：**一次**有界 GPU 诊断
  `python diagnostics/r8_smoke.py --kv-diagnose --norm-contract <64-TRAIN 冻结契约> --min-free-gb 24 --out <dir>`
  （≤5 分钟含权重加载，空闲显存 ≥24 GB，不影响其他进程）。
* 该动作**尚未获批**：`diagnostics/r9_kv_same_prefix.py --gpu-diagnose` 在缺少
  `R9_KV_GPU_AUTHORIZED=1` 时直接拒绝（退出码 7，证据 `gpu_guard.log`）。
* 重跑必须留下：每步 256-bin 全 logits、max/mean |Δ|、top1−top2 margin、argmax、首次分叉、dtype/device/模型 SHA。

## 6. 回归（10/10 PASS，`tests/test_r9_kv_same_prefix.py`）

三次确定性对照（clean / 小 Δ 翻转 / 大 Δ 翻转）分别得到 `SAME_PREFIX_EQUIVALENT` /
`OPEN_NUMERIC_PLAUSIBLE` / `OPEN_STRUCTURAL_SUSPECT`；失配时退出码 **6**（绝不假 PASS）；
teacher-force 序列等于 full 路径 greedy 序列；前缀长度逐步 +1；window 切片正确；
生产探针静态审计 PASS；**合成的旧 bug 探针被检出**。

## 7. 边界

1. 微型模型结论只证明**协议正确性**（缓存/位置/掩码/窗口/退出语义），不证明 7B 行为。
2. 未触碰 GPU、未启动训练、未修改审计的 `r8_smoke.py` 与其原始证据。
3. 在 GPU 判定完成前，**不得**把 KV 解码视为已验证的部署路径，也不得据此启动 7B 训练。
