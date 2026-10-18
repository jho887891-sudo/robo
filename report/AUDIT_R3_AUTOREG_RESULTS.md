# 第三轮根因审查与验证结果（自主执行）

分支 `audit/r3-autoreg-training-fix`（起点 `audit/r2-action-codec-fix@d63ad8a`）。
本轮**未重训 6000 步、未启动 12 任务仿真、未改主分支、未改旧检查点与数据集**。

时间线（UTC）：A 阶段 17:35 · B 阶段 17:39–17:50 · C 阶段 17:50–18:00 · D 阶段 18:00–（见各节实测时间戳）

---

## 1. 已确认的源码问题（前三轮累计，含本轮复验）

| # | 问题 | 源码位置 | 证据 |
|---|---|---|---|
| 1 | 训练标签中 12.23% 落在 PAD（token 32000），且整体偏移 1 个 token | `train_pushT_pure2.py`(旧) `tok()`；现 `action_codec.py` | 第二轮：94514 个标签里 11559 个为 PAD；第三轮复验通过 |
| 2 | 训练时 14 个动作输入位置被**置零** ⇒ 无法利用前序动作标记 | `modeling_prismatic.py` `forward` 的 `input_embeddings = input_embeddings * ~all_actions_mask` | 掩码 14×True；进入 LM 前 `max|emb| = 0.000000` |
| 3 | 部署侧缓存生成喂**真实**动作标记 ⇒ 训练/部署机制不一致 | `modeling_prismatic.py` `if input_ids.shape[1] == 1` 分支（`labels=None` 直通） | 部署 tokens 与“keep 机制逐 token 生成”逐 token 一致 |
| 4 | 训练脚本的完整序列离线评测同样置零 | `train_pushT_pure2.py` `evaluate()`；`diagnostics/eval_pure_openvla.py` | 两处都传 `labels=inp.clone()` |
| 5 | runtime 动态模块的**权威副本在基础模型目录**，HF 缓存是派生产物 | `model_cache/openvla-7b/modeling_prismatic.py` → `~/.cache/huggingface/modules/.../openvla-7b/` | 第三轮打补丁时实测：只改缓存会被覆盖（`AttributeError`） |

### 1.1 关键实测（本轮复验，原样输出）

    [1] all_actions_mask at the 14 action positions:
        [True, True, True, True, True, True, True, True, True, True, True, True, True, True]   (n_true=14)
        prompt positions masked=0, STOP position masked=False
    [2] masked max|emb| at 14 action positions = 0.000000   norms[0:4] = [0.0, 0.0, 0.0, 0.0]
    [2] kept   max|emb| at 14 action positions = 0.096680   norms[0:4] = [1.4651, 1.2338, 1.0597, 1.4613]
        参照：视觉 patch 行范数 ~107–213

结论（**已证实**）：训练时 14 个动作 token 的输入嵌入确实被清零；由于注意力是因果的、
且这些位置恒为 0，位置 k 的预测只依赖“前缀 + 位置下标”，纯 OpenVLA 因此被压平成**并行解码器**。

---

## 2. 修复（最小、可开关、默认不改变 OFT）

| 文件 | 改动 |
|---|---|
| `code/openvla_dynamic_modules/modeling_prismatic.py` | 新增 `set_action_input_masking(mode)` / `_resolve_action_input_masking(flag)`；`forward` 增加 `mask_action_inputs` 关键字；置零那一行改由开关控制（**默认 zero**） |
| `training/train_pushT_pure2.py` | `set_action_input_masking("keep")` + 训练/评测前向 `mask_action_inputs=False` + `run_meta` 记录 |
| `code/xpolicylab/policy/OpenVLA_Pure/model.py` | 前缀与回退前向显式 `mask_action_inputs=False`；`last_decode` 记录机制 |
| 编码器 | **未改**，仍是第二轮 `action_codec.py`（`MODE_FIXED` 用于训练；旧检查点解码仍走同一 `LEVELS` 栅格） |
| 注意力 | **未改**，仍严格因果 |

diff 见 `patches/r3_action_input_masking.patch`（131 行）。补丁后 4 份 `modeling_prismatic.py`
字节一致：`8d89d5f09d483651dee57ba2d672c2a901396e9a7ea5bb3e2e66e2d743eeefff`（50,580 B）。

---

## 3. A 阶段：纯 CPU 单元测试（通过）

    $ python -m unittest tests.test_action_codec tests.test_deploy_contract tests.test_action_masking_contract
    Ran 35 tests in 0.288s
    OK

    $ PUSHT_HDF5=... python -m unittest tests.test_label_dataset
    Ran 3 tests ... OK
    [fixed]  labels=94514  token range=[31744,31999]  PAD=0
    [legacy] PAD labels=11559/94514 = 12.23%   frames with >=1 PAD = 6596/6751 = 97.70%

    契约测试在修复前代码上：FAILED (failures=6) → 修复后：7/7 OK

**状态：已证实（全部通过）**


## 4. B 阶段：训练 / 离线评测 / 部署 机制一致性（GPU，全程实测）

脚本 `diagnostics/r3_train_infer_consistency.py`（真机加载 6000 步检查点，单帧真实样本）。

    [B1] train() vs eval() logits maxdiff = 0.000e+00  (769 vision tokens)
    [B2a] full-seq vs KV teacher-forced: logit maxdiff=1.250e-01 prob maxdiff=1.24e-02 tokens equal=True
          full=[212, 187, 146, 44, 81, 143, 1, 106, 1, 1, 255, 185, 189, 255]
          step=[212, 187, 146, 44, 81, 143, 1, 106, 1, 1, 255, 185, 189, 255]
          actions identical = True ; |logit| max ~ 26.2（该量级下 bf16 ulp = 0.125）
    [B2b] full-seq vs no-cache step-by-step（同一内核）：logit maxdiff = 0.000e+00
    [B3] cached greedy  = [212, 187, 144, 44, 81, 143, 1, 106, 1, 1, 255, 185, 189, 255]
         full recompute = [212, 187, 144, 44, 81, 143, 1, 106, 1, 1, 255, 185, 189, 255]  equal=True
    [B4] eager multimodal attention: shape [32, 820, 820], strict upper-triangle max = 0.000e+00
    [B5] train() 模式下篡改未来动作#11 -> 14 个位置的 maxdiff = [0.0 ×11, 0.25, 0.125, 0.125]

    === B RESULT ===
      B1_train_eval_identical            PASS
      B2a_tokens_identical_kv            PASS
      B2a_actions_identical_kv           PASS
      B2b_nocache_bitwise_identical      PASS
      B3_cached_equals_full              PASS
      B4_attn_strictly_causal            PASS
      B5_train_mode_no_future_leak       PASS
    R3_B_STAGE: ALL_PASS

判读（**已证实**）：

- 训练口径的“一次整段前向”与部署口径的“逐 token 前馈 + KV 缓存”**输出完全一致**
  （14/14 标记相同、解码动作逐位相同）；两者 logits 相差 0.125，正好等于该量级下的 1 个 bf16 ulp，
  而同一内核重算整段的对照（B2b）**逐位相同 0.0** —— 因此差异纯属精度累加顺序，不是机制差异。
- 训练与推理都是严格因果：真实多模态路径（820 token）在 eager 下注意力严格上三角 **max = 0.000e+00**；
  train() 模式下篡改“未来”动作标记，过去 11 个预测位置 logits 变化**恰好为 0**（只有位置 12/13 变，
  因为它们本就依赖第 11 个标记）。
- 顺便记录一个陷阱：本机 `attn_implementation=sdpa` 时若打开 `output_attentions=True`，
  HF 会退回 eager 且不建因果掩码，返回的注意力矩阵会显示“非因果”（上三角 max=1.0）。
  任何用打印注意力矩阵判断因果性的做法在本环境都会得出错误结论（报告第 5 节有专节说明）。

---

## 5. C 阶段：梯度检查（GPU，全新 base+LoRA+proprio，不加载旧适配器）

脚本 `diagnostics/r3_gradient_check.py`。为了不 OOM（GPU 上另有他人进程占用约 24 GB），
本脚本**不再加载部署用的 merged 模型**，只构建一份 trainable 结构。

    [keep] loss=18.3680 finite=True | LoRA grads: tensors=852 norm=4.589e+01 nonzero_frac=0.495 | proprio norm=2.000e+01
    [keep] d(loss)/d(action token embedding row) per dim:
           [5.81, 4.94, 7.87, 4.25, 3.91, 13.87, 7.69, 3.68, 4.23, 4.85, 7.76, 2.39, 2.05, ~0]
    [zero] loss=20.9951 finite=True | LoRA grads: tensors=852 norm=9.783e+01 nonzero_frac=0.495 | proprio norm=2.512e+01
    [zero] d(loss)/d(action token embedding row) per dim: [0.0 ×14]

    === C RESULT ===
      C1_loss_finite_both                PASS
      C1_lora_receives_grad              PASS
      C1_proprio_receives_grad           PASS
      C1_no_nonfinite_grad               PASS
      C2_keep_action_emb_grad_nonzero    PASS
      C3_zero_action_emb_grad_is_zero    PASS
    R3_C_STAGE: ALL_PASS

判读（**已证实**）：

- **keep 机制**下，14 个动作 token 位置（置零之前）的嵌入梯度 13/14 非零（1.9–13.9），
  说明后续位置确实把梯度回传到前序动作标记 —— 自条件通路在梯度层面存在；
- **zero 机制**下同样位置梯度**严格为 0.0** —— 这从梯度上证明了“置零切断了自条件通路”，
  与第 1 节的嵌入为零、第二轮的掩码实验互相印证；
- LoRA（852 个张量，49.5% 元素非零）与 ProprioProjector 都拿到有效梯度，损失有限、无 NaN/Inf；
- 第 14 维（right_gripper）在本次初始化/样本下梯度约为 0（该维在该帧的动作基本不变），
  不影响结论（其余 13 维足以证明通路存在，且逐维数据已如实记录）。

---

## 6. 100 个 push_T 专家回合的动作统计（可选工作，纯 CPU，已完成）

脚本 `diagnostics/action_delta_analysis.py`，100 个回合 / **32,204** 个转移（`/tmp/action_delta.json`）：

| 参照量 | BOUNDS 归一化 | 物理单位 |
|---|---|---|
| 恒等（预测 S[t]）与专家 A[t] | **0.017568** | 0.013733 |
| 重复上一步（预测 A[t-1]）与 A[t] | **0.008835** | 0.006903 |

    absolute action : mean|A| = 0.3827 ; mean|A - S| = 0.0137 ; corr(A, S) = 0.9983 ; R²(A|S) = 0.9966
    residual  Δ=A-S : mean|Δ| = 0.0137 ; std = 0.0361
    binning         : 绝对动作 bin 宽均值 0.00535（典型一步 = 2.67 bin）
                      残差       bin 宽均值 0.00105（典型一步 = 14.27 bin）

逐维（节选，全部 14 维在 JSON 里）：

| dim | mean|Δ| | 绝对 bin 宽 | 一步 = 多少 bin（绝对） | 残差 bin 宽 | 一步 = 多少 bin（残差） |
|---|---|---|---|---|---|
| left_arm_j0 | 1.45e-02 | 6.27e-03 | 2.32 | 7.90e-04 | 18.39 |
| left_arm_j3 | 1.50e-02 | 4.60e-03 | 3.27 | 1.10e-03 | 13.65 |
| left_gripper | 1.00e-02 | 3.92e-03 | 2.55 | 1.57e-03 | 6.38 |
| right_arm_j1 | 2.24e-02 | 8.86e-03 | 2.53 | 1.61e-03 | 13.94 |
| right_gripper | 8.61e-03 | 3.92e-03 | 2.20 | 1.57e-03 | 5.49 |

三条结论（**已证实**）：

1. **绝对动作几乎由当前状态决定**：`R²(A|S) = 0.9966`、`corr = 0.9983`。这意味着“绝对动作”这个
   目标本身把 99.7% 的方差交给了一个与动作无关的量，策略很容易退化成“复制状态”——这正是
   第一/二轮实测到的退化（模型误差 = 恒等策略的 4.19 倍）。
2. **专家每步只走 2.67 个 bin**（绝对表示），量化误差本身就占了单步位移的 1/3 量级；
   换成残差表示后同样一步占 **14.27 个 bin**，量化分辨率相对提升约 **5.3 倍**。
3. **“重复上一步动作”已经比“复制造状态”好一倍**（0.008835 vs 0.017568）——任何新训练都必须
   同时超越这两个平凡基线，否则不能说明学到了控制。


## 7. D 阶段：32 样本受限过拟合实验（四.D）

脚本 `training/overfit_r3.py`。硬约束全部写在代码里并由脚本自身校验：

| 约束 | 实现 |
|---|---|
| 32 个**固定**样本 | `pick_samples()`：val 回合按名排序、帧按固定步长（每 40 帧），落盘 `_snapshot/config.json` |
| 先保存配置与代码 | `snapshot()` 在训练前拷贝 `modeling_prismatic.py / action_codec.py / train_pushT_pure2.py / overfit_r3.py` 并记录 SHA-256 |
| 独立输出目录 | `/home/dgut/pusht_runtime/r3_overfit/{keep_run1,zero_run1}` |
| ≤300 次更新 | `MAX_STEPS=300`，`--steps > 300` 直接拒绝启动 |
| 只用 LoRA+proprio | OpenVLA-7B 主干冻结，trainable 127.7M |
| 修复后的合法编码器 | `action_codec.MODE_FIXED`（token ∈ 31744..31999） |
| 逐个生成评测 | `gen_actions()`：KV 缓存、喂自身预测，**不用教师强制** |
| 异常即停 | 非有限损失 → `FAILED_NAN_LOSS`；CUDA/OOM → `FAILED_RUNTIME`，均写 `status.json` + 日志 |

补充说明：本机 GPU 被其它进程占用约 24 GB，脚本自动开启**梯度检查点**（只影响显存与速度，
不改变计算数学）；同一脚本在 300 步内没有出现 NaN/OOM 之外的异常路径。

（训练与评测的实际数字见下一节的原始日志摘录与 `metrics.jsonl`。）

---


### 7.1 两组实验的配置（完全一致，只差动作输入掩码）

    keep 组：`set_action_input_masking("keep")`  + 前向 `mask_action_inputs=False`
    zero 组：`set_action_input_masking("zero")`  + 前向 `mask_action_inputs=True`（= 修复前的旧机制）
    共同：32 固定样本 / 300 次更新 / LoRA r=32 all-linear + ProprioProjector（127.7M 可训练）
          lr 5e-4 AdamW(wd=0) clip 1.0 / 编码器 MODE_FIXED / 梯度检查点（显存保护）
    快照：`_snapshot/config.json` 记录了样本列表与 4 个源码文件的 SHA-256
          （modeling_prismatic.py = 8d89d5f0…、action_codec.py = d7c81a37…）

### 7.2 训练过程（每 3 条日志抽样；完整曲线上传在 metrics.jsonl）

| step | keep loss | keep tok_acc | zero loss | zero tok_acc |
|---|---|---|---|---|
| 1 | 16.221 | 0.000 | — | — |
| 25 | 3.436 | 0.214 | 4.288 | 0.214 |
| 50 | 3.142 | 0.286 | — | — |
| 75 | 0.521 | 0.857 | — | — |
| 100 | 0.228 | **0.929** | 6.215 | **0.071** |
| 150 | 0.361 | 0.857 | 1.262 | 0.643 |
| 200 | 0.086 | 0.929 | — | — |
| 250 | **0.0001** | **1.000** | — | — |
| 275 | 0.005 | **1.000** | 0.562 | 0.857 |
| 300 | 0.501 | 0.929 | 0.297 | 0.929 |

（tok_acc 是“同一条样本的教师强制 14 位置标记准确率”，单样本更新，故有抖动；
keep 组在 step 250 达到过 **14/14 全对**，zero 组整段没达到过。）

### 7.3 真正的逐个生成评测（每次 eval 都重跑 32 个样本；不用教师强制）

| step | keep 生成 L1(物理) | keep tf_acc | zero 生成 L1(物理) | zero tf_acc | 恒等基线(物理) |
|---|---|---|---|---|---|
| 0 | 0.44788 | 0.022 | 0.45378 | 0.025 | 0.00834 |
| 100 | **0.08943** | 0.908 | 0.30704 | 0.388 | 0.00834 |
| 200 | **0.02488** | 0.980 | 0.19580 | 0.667 | 0.00834 |
| 300 | **0.02328** | 0.984 | 0.07464 | 0.815 | 0.00834 |

BOUNDS 归一化口径（step 300）：keep **0.0301** vs 恒等 0.0111；zero 0.1024 vs 0.0111。

动作变化量（step 300）：keep `mean|A_hat − S| = 0.02777` vs 专家 `mean|A − S| = 0.00834`；
zero 为 0.07910 —— 两组都**过冲**（预测位移大于专家位移），keep 过冲 3.3x，zero 过冲 9.5x。

### 7.4 结论

- **能学会（已证实）**：在 `keep` 机制下，32 个固定样本在 300 次更新内把教师强制的动作标记
  准确率从 2.2% 提到 **98.4%**（step 250 曾达 100%），说明修复后的因果自回归通路、
  编码器与梯度都正常工作，模型**具备拟合能力**，不存在“学不动”的结构性障碍。
- **修复带来的收益（已证实）**：同样 300 步、同样 32 样本，`keep` 的自由运行误差
  **0.02328** 对比 `zero` 的 **0.07464**，**快 3.2 倍**；在 step 100 时差距更大
  （0.0894 vs 0.3070，3.4 倍）。也就是说“保留前序动作标记”不仅更正确，在样本效率上也明显更好。
- **仍不足以超越恒等策略（已证实）**：两组在 300 步内都没能低于恒等基线 0.00834
  （keep 2.79x、zero 8.95x）。原因有两个，都是定量可见的：
  ① 这个子集每步位移只有 ~1.5 个绝对 bin（`mean|Δ| = 0.00834` vs bin 宽 0.00535），
     一步错 1 个 bin 就吃掉全部余量；
  ② 自由运行存在自回归误差累积：keep 的教师强制准确率 98.4%（14 位置全部正确的概率 ≈ 80%），
     剩下 20% 的样本一旦某个标记出错就整体偏离 —— 所以“教师强制很高、自由运行仍差”。
- **不能做的推广**：32 样本 × 300 步只回答“机制是否可学”，**不构成任务能力证据**；
  也不代表 300 步之后的趋势（keep 在 200→300 步仍在缓慢下降）。

---

## 8. 与恒等策略的定量对比（汇总表）

| 对象 | 口径 | 模型 L1 | 恒等策略 L1 | 比值 |
|---|---|---|---|---|
| 6000 步旧检查点（部署路径，第二轮） | BOUNDS，20 留出样本 | **0.05062** | **0.01207** | **4.19x 更差** |
| 6000 步旧检查点（部署路径，第二轮） | 物理单位 | 0.03493 | 0.00994 | 3.51x 更差 |
| 训练日志里的历史数字 | q01/q99 归一化（并行机制） | 0.0586 | — | 口径不同，不可直接比较 |
| 32 样本 × 300 步，**keep**（修复后机制） | 物理，自由运行 | **0.02328** | 0.00834 | 2.79x 更差 |
| 32 样本 × 300 步，**zero**（修复前机制） | 物理，自由运行 | 0.07464 | 0.00834 | 8.95x 更差 |
| 32 样本 × 300 步，**keep** | BOUNDS，自由运行 | 0.0301 | 0.0111 | 2.71x 更差 |
| 32 样本 × 300 步，**zero** | BOUNDS，自由运行 | 0.1024 | 0.0111 | 9.23x 更差 |
| push_T 专家数据（100 回合 32204 转移） | BOUNDS | —— | 恒等 0.017568 / 重复上一步 0.008835 | 两个平凡基线 |

读法：6000 步旧检查点（并行机制、完整数据集）是 4.19x；本轮 32 样本 300 步的 keep 已经把
“比恒等差”的距离缩到 2.79x，且是同预算下 zero 的 1/3.2 —— 但**仍未低于恒等**，说明
小样本过拟合不能替代正式训练，下一步必须按第 9 节做残差目标 + 完整数据。

---

## 9. 下一轮正式训练建议（不改代码，仅设计）

依据本轮数据（`R²(A|S) = 0.9966`、残差表示把单步位移从 2.67 bin 提到 14.27 bin、
“重复上一步”基线已优于“恒等”一倍）：

1. **首选：残差目标** `Δ = A[t] − S[t]`
   - 归一化与分箱都用 Δ 的逐维 1%/99% 分位（`action_codec.normalize/encode_indices` 直接复用）；
   - 动作解码后再加回 `S[t]` 得到绝对关节目标，控制接口不变（仍是 `set_joint_position_target`）；
   - 保留第三轮的 `keep` 自回归机制与第二轮的合法编码器（31744..31999，0 个 PAD）；
   - 训练时同时记录两个量化指标：`mean|Δ_pred|` 与 `mean|Δ_gt|`。
2. **必须超越的两个平凡基线**（否则不能声称学到控制）：
   - 恒等：BOUNDS 0.017568；重复上一步：BOUNDS 0.008835。
3. **离线评测改用自由运行**（`gen_actions` 口径），禁止再用教师强制准确率代表部署表现。
4. **规模**：先 3000 步（与例程一致），每 500 步做一次自由运行评估 + 保存；确认指标单调下降后再续训。
5. **可选的数据侧改进**：按阶段（接近/下压/对齐）分层采样，避免大部分帧都在“几乎不动”的状态。
6. **不做**：12 任务大规模仿真；在残差方案验证前不动 OFT 既有流程。

---

## 10. 尚未解决的问题（如实标注）

| 项 | 状态 |
|---|---|
| 残差方案的实际效果 | **尚未验证**（本轮按约束未改动作定义；仅给出统计依据与设计） |
| 6000 步正式训练 | **未执行**（禁止） |
| 闭环成功率 / 12 任务仿真 | **未执行**（禁止） |
| D 阶段能否“明显拟合”32 个固定样本 | 见第 7 节结论（受 300 步上限与 24 GB 显存限制） |
| 旧检查点在 keep 机制下重训后的表现 | 未执行 |
| `sdpa + output_attentions=True` 的 HF 假象 | 已记录为陷阱；不影响本轮结论（因果性只用 logits 层判定） |
| GPU 显存 | 他人进程常驻约 24 GB，本轮全部训练/验证都启用梯度检查点；若后续要跑 6000 步需先释放 |

---

## 11. 复现命令

```bash
cd /home/T7/ojh/robot_actual/submission
PY=/home/dgut/pusht_runtime/venv_oft/bin/python

# A) 纯 CPU 单元测试（38 项）
$PY -m unittest tests.test_action_codec tests.test_deploy_contract tests.test_action_masking_contract
PUSHT_HDF5=/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data $PY -m unittest tests.test_label_dataset

# B) 训练/推理一致性（GPU，需 ~4 分钟加载）
$PY diagnostics/r3_train_infer_consistency.py --out /tmp/r3_B.json

# C) 梯度检查（GPU）
$PY diagnostics/r3_gradient_check.py --out /tmp/r3_C.json

# D) 受限过拟合（GPU，各 300 步）
$PY training/overfit_r3.py --masking keep --steps 300 --out <dir_keep>
$PY training/overfit_r3.py --masking zero --steps 300 --out <dir_zero>

# 可选：100 回合动作统计（纯 CPU）
$PY diagnostics/action_delta_analysis.py --n 100 --out /tmp/action_delta.json

# 第三轮取证（掩码/嵌入/部署对照）
$PY diagnostics/r3_mask_probe.py --sample 0 --out /tmp/r3_probe.json
$PY diagnostics/r3_causality_scale.py --n 20 --free-n 5 --out /tmp/r3_causal.json
```
