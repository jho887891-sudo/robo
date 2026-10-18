# 第三轮审查报告：纯 OpenVLA 自回归训练的动作输入掩码

分支 `audit/r3-autoregressive-mask`（起点 `audit/r2-action-codec-fix@d63ad8a`）。
本轮**没有重训 6000 步、没有改旧检查点、没有改主分支、没有动第二轮结果**；
只做了取证、最小修复与验证脚本。

## 0. 结论摘要

1. **缺陷确认**：训练时传入 `labels` 后，14 个动作 token 位置的输入嵌入**全部被置零**
   （`input_embeddings = input_embeddings * ~all_actions_mask`），实测 14 个布尔值全为 True、
   进入语言模型前的动作位置嵌入 `max|emb| = 0.000000`。
2. **因此纯 OpenVLA 从来没有学过条件于前序动作标记**：训练阶段等价于 OFT 的**并行解码**
   （每个动作位置只看前缀 + 位置信息），而不是自回归。
3. **部署侧却是真自回归**：KV 缓存逐 token 生成走的是 `input_ids.shape[1] == 1` 分支，
   该分支 `assert labels is None` 并把真实 token 直接送进语言模型 —— **动作位置拿到真实嵌入**。
   实测：部署输出 == 独立实现的逐 token 生成（keep 机制）逐 token 一致。
4. **训练脚本自己的完整序列离线评测同样置零**（`labels=inp.clone()`），所以历史离线指标
   （`autoreg_L1 = 0.0779 / 0.0586` 等）测的是**并行机制**，不是部署机制。
   两个机制的差异已定量：20 个留出样本上 14 个动作位置平均 **12.30/14** 个标记相同、
   动作 L1 差 **0.00392**（自由运行时 13.00/14、0.00356）。
5. **注意力是严格因果的**（logits 层面验证：篡改第 11 个动作标记，前 10 个预测位置 logits 完全相同）。
   但要注意：本机 `attn_implementation=sdpa` 时把 `output_attentions=True` 打开会让 HF 退回 eager
   且不建因果掩码，**返回的注意力矩阵上三角非零**（`max=1.0`）——这是 HF 的陷阱，不是模型缺陷，
   用“打印注意力矩阵”判因果会得出错误结论（本文把这个陷阱单独记录，见第 5 节）。
6. **最小修复已实施并验证**：动作输入掩码改为可配置（默认仍是 zero，OFT 行为不变），
   自回归训练/评测/部署显式声明 `keep`。契约测试 7/7 通过（修复前同一测试 6 项失败），
   修复后 GPU 验证 T1–T7 见第 7 节。
7. **对原 6000 步训练有效性的影响**：该检查点是用“并行机制”训练的，它的**权重不会因为本次修复
   而变好**；部署行为在本轮修复前后完全一致（部署本来就喂真实标记）。
   修复的意义是：今后的自回归训练不再丢掉对前序动作的条件依赖，且三处机制从此一致。
   另外实测表明，这个检查点对前序动作标记的依赖本身就很弱（第 7 节 T7），
   所以“掩码缺陷”更像是**能力缺失**（学不到自条件），而不是把已有能力弄坏的“注入错误”。

---

## 1. 缺陷一：14 个动作输入位置全部被置零（问题 1）

### 1.1 源码位置

| 位置 | 内容 |
|---|---|
| `code/openvla_dynamic_modules/modeling_prismatic.py:578` | `all_actions_mask = self._process_action_masks(labels)` |
| `modeling_prismatic.py:495-505` | `_process_action_masks` = `get_current_action_mask \| get_next_actions_mask` |
| `modeling_prismatic.py:617-621`（修复前） | `else: all_actions_mask = all_actions_mask.unsqueeze(-1); input_embeddings = input_embeddings * ~all_actions_mask` |
| `prismatic/training/train_utils.py:8-32` | 两个掩码函数的实现（按 `labels` 的 `cumsum` 取前 `ACTION_DIM` 个非 `IGNORE_INDEX` 位置，且要求 `token > 31743`） |

### 1.2 实测：14 个布尔值

    样本 episode_0000066.hdf5 t=221，prompt_len=36
    labels（旧编码 token）= [31962, 31929, 31890, 31788, 31836, 31884, 31745, 31850, 31745, 31745, 32000, 31929, 31933, 32000]
    all_actions_mask at the 14 action positions:
      [True, True, True, True, True, True, True, True, True, True, True, True, True, True]   (n_true=14)
    prompt positions masked = 0 ; STOP position masked = False

注意：两个目标里含 `32000 = PAD`（第二轮已定位的编码缺陷），它们**仍然被算作动作位置**
（掩码条件是 `token > 31743`），所以 PAD 也被置零 —— 这一点与第二轮的结论互相印证。

### 1.3 实测：进入语言模型之前的嵌入

用 `register_forward_hook(with_kwargs=True)` 钩住 `language_model`，取 `inputs_embeds`：

    修复前（训练机制，动作位置被置零）: max|emb| at 14 action positions = 0.000000, norms = [0.0, 0.0, 0.0, 0.0, ...]
    对照（把掩码强制成全 False）      : max|emb| at 14 action positions = 0.096680, norms[0:4] = [1.4651, 1.2338, 1.0597, 1.4613]
    参照尺度：视觉 patch 行范数 ~107–213（同一个多模态序列里）

结论：**被置零这件事是事实，不是推测**；而且动作 token 的嵌入本身很小（~1.4），
相比视觉前缀（~10^2）低两个数量级 —— 这解释了下一节的“影响为何有限”。

---

## 2. 缺陷二：自回归条件依赖丢失（问题 2）

修复前，模型在动作位置 k 的输入是“零向量 + 位置编码”，注意力又是因果的，
所以位置 k 的预测 **只依赖前缀（图像 + 指令 + proprio）与位置下标**，与前序动作取值无关。
这与 OFT 的并行解码设计意图一致（OFT 用 `noisy_actions` + `_replace_input_embeddings`
把零替换成扩散去噪特征），但纯 OpenVLA 用的是自回归标记 -> 该机制把自回归“压平”成了并行。

**实测证据（同一批 20 个固定留出样本，GT 前缀 teacher forcing，只改掩码）**：

    mean tokens equal (masked vs kept) = 12.30 / 14
    mean |ΔA| L1                        = 0.00392
    （单样本 episode_0000066 t=221：14/14 相同、L1 = 0.00000）

自由运行（逐 token 生成）同样只改掩码：

    mean tokens equal = 13.00 / 14 ; mean |ΔA| L1 = 0.00356  （5 个样本）

也就是说：把前序动作标记“还回来”对输出影响很小 —— 因为**模型从来没学过用它**
（训练时它一直是 0），而且它的嵌入尺度只有前缀的 ~1/100。
这条对“缺陷有多严重”的判断很关键：**结构上确实丢了自条件能力，但在这个已经训练好的
检查点上，丢失的部分本来也没被学到。**

---

## 3. 缺陷三：部署侧喂真实动作标记 -> 机制不一致（问题 3）

### 3.1 源码位置

| 位置 | 内容 |
|---|---|
| `modeling_prismatic.py:533-556` | `if input_ids.shape[1] == 1:` 缓存生成分支，`assert labels is None`，然后 `language_model(input_ids=input_ids, ...)` —— **不做任何掩码替换** |
| `model.py:190-206`（部署，KV 缓存） | `self.vla(input_ids=nxt, past_key_values=past, labels=None, use_cache=True)` |
| `model.py:180-190`（部署，前缀） | `labels=prompt_ids.clone()` —— 提示段没有动作 token，掩码全 False，不置零 |

结论：**训练时动作位置是零向量，部署时是真实 token 嵌入** —— 条件机制不一致，问题成立。

### 3.2 实测：部署到底用的什么机制

    部署（model.Model.infer，KV 缓存，labels=None）tokens =
      [212, 187, 144, 44, 81, 143, 1, 106, 1, 1, 255, 185, 189, 255]
    独立实现的逐 token 生成（keep，不置零）tokens = 同上             -> 逐 token 完全一致
    独立实现的逐 token 生成（zero，置零）    tokens 与之相差 1/14 个位置

即：部署路径确实是“真实动作标记 + 因果注意力”的自回归（keep 机制），
与训练时的“全零 + 并行”机制不同。

---

## 4. 缺陷四：训练脚本的完整序列离线评测也置零（问题 4）

| 脚本 | 前向调用 | 是否置零 |
|---|---|---|
| `training/train_pushT_pure2.py` 训练循环 | `vla(input_ids=inp, labels=lab, ...)` | **是** |
| `training/train_pushT_pure2.py` `evaluate()`（自回归解码 + 完整序列） | `vla(..., labels=inp.clone())` | **是** |
| `diagnostics/eval_pure_openvla.py` `teacher_forced()/predict_autoreg()`（第一轮） | `vla(..., labels=inp.clone())` | **是** |
| `code/.../model.py` 部署（KV 缓存逐 token） | `labels=None` | **否** |

因此第一/第二轮报告的离线数字（`autoreg_L1 = 0.0779 → 0.0586`、`0.0126` 等）
都是“并行机制”的指标，**它们从来没有测过部署时的机制**。
两套机制的差距已在上文定量（12.30/14 标记、0.00392 L1 ），不足以解释闭环完全不动，
但足以说明“离线离线好、闭环不动”不能用同一套指标直接对比。

---

## 5. 注意力因果性与“未来动作信息泄漏”（问题 5）

### 5.1 结论：正常前向是严格因果的

用 **logits 层面**的扰动实验（两次调用都用 `output_attentions=False`，排除实现切换的影响）：

    确定性对照：同一输入跑两次，maxdiff = 0.000e+00
    篡改第 11 个动作标记：前 10 个预测位置的 logits 变化 = 全部 0.0
    正向对照（篡改第 4 个）：其后的预测位置 logits 变化 > 0

即：过去 → 未来有信息流，未来 → 过去没有。

### 5.2 陷阱记录：`output_attentions=True` 在 sdpa 下会给出“非因果”的假象

本机 `language_model = LlamaForCausalLM，`attn_implementation = sdpa`。当把
`output_attentions=True` 打开时（HF 会退回 eager 实现），

    layer0 注意力矩阵（32 heads, 50x50）：严格上三角 |attn| 最大 = 1.0000
    last layer：上三角最大 = 0.98828

并在同一调用里表现为“篡改未来 token 会改变过去位置的 logits”（maxdiff 0.1–0.7）。
**这不是模型缺陷，而是 HF 的行为**：sdpa 路径依赖 `is_causal=True`，在没有显式 4D 掩码时
退回 eager 就不会再建因果掩码，于是 introspection 得到的注意力矩阵是非因果的。

> 结论：任何“用 `output_attentions=True` 打印注意力矩阵来判断因果性”的做法在本环境里都会
> 得出错误结论。本文的因果性结论只基于 logits 扰动（`output_attentions=False`），
> 并已在修复后的验证里重复（第 7 节 T6）。

---

## 6. 最小修改方案（已实施，未重训）

### 6.1 设计原则

1. **默认行为不变**：`zero`（置零）仍是默认 —— OFT 的并行/扩散解码完全不受影响；
2. **显式声明**：纯 OpenVLA 自回归路径（训练、训练内评测、部署）显式声明 `keep`；
3. **因果性不变**：只改“动作位置喂什么”，不改注意力掩码，仍是严格因果（第 5 节 + T6）；
4. **复用第二轮编码器**：动作离散化仍走 `action_codec.py`（训练用 `MODE_FIXED`，旧检查点解码仍用
   `LEVELS[idx]` 栅格），本轮没有再碰编码数学；
5. 三处机制从此一致：训练 = 完整序列评测 = 部署（都是真实前序动作标记 + 因果注意力）。

### 6.2 修改点（4 个文件，见 `patches/r3_action_input_masking.patch`）

| 文件 | 修改 | 行/函数 |
|---|---|---|
| `code/openvla_dynamic_modules/modeling_prismatic.py` | 新增 `set_action_input_masking()` / `_resolve_action_input_masking()`；`forward` 增加 `mask_action_inputs` 关键字；置零那行改为受开关控制（默认 zero） | `_process_action_masks` 之后新增；`forward` 签名末；`forward` 内 `else:` 分支 |
| `training/train_pushT_pure2.py` | `vla.set_action_input_masking("keep")` + 训练/评测两处前向传 `mask_action_inputs=False` + `run_meta` 记录 | `main()` 训练循环前；`evaluate()` 前向；训练前向；`json.dump(...)` |
| `code/xpolicylab/policy/OpenVLA_Pure/model.py` | 前缀与回退两条前向显式传 `mask_action_inputs=False`；`last_decode` 记录 `action_input_masking="keep"` | `infer()` |
| `diagnostics/*`（新增，见第 10 节） | 探针、验证、取证脚本 | — |

补丁前后核心差异（modeling）：

```diff
             else:
-                # Replace the embeddings of the action tokens with zeros
-                # (Later on, the positional embeddings will be added to them)
-                all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
-                input_embeddings = input_embeddings * ~all_actions_mask
+                # [R3] 由 mask_action_inputs / self.action_input_masking 决定：
+                #   zero（默认，OFT 并行/扩散解码）：置零 => 各动作位置互不可见
+                #   keep（纯 OpenVLA 自回归）：保留前序真实动作标记嵌入
+                if self._resolve_action_input_masking(mask_action_inputs):
+                    all_actions_mask = all_actions_mask.unsqueeze(-1)
+                    input_embeddings = input_embeddings * ~all_actions_mask
```

### 6.3 一个必须记录的部署事实（修正第二轮的表述）

第三轮打补丁时发现：**运行时真正被 import 的动态模块，是 transformers 从基础模型目录里的
`.py` 复制到 HF 模块缓存之后的那份**。也就是说

    /home/dgut/pusht_runtime/model_cache/openvla-7b/modeling_prismatic.py   ← 权威来源
        ↓（transformers 每次加载时复制）
    /home/dgut/.cache/huggingface/modules/transformers_modules/openvla-7b/modeling_prismatic.py  ← 实际 import

只改缓存副本，下次加载会被模型目录里的原件覆盖（本轮第一次尝试就是这样失败的，实测报
`AttributeError: ... has no attribute '_resolve_action_input_masking'`）。
第二轮只记录了缓存路径、没记录这条复制链，属于记录不完整，本轮修正。

补丁后 4 份 `modeling_prismatic.py` 完全一致：
`8d89d5f09d483651dee57ba2d672c2a901396e9a7ea5bb3e2e66e2d743eeefff`（50,580 B），
分别在仓库副本 / 基础模型目录 / HF 缓存 / OFT 源码树。

---

## 7. 测试与真实结果

### 7.1 契约测试（纯 CPU，无需 GPU/数据集）

新增 `tests/test_action_masking_contract.py`（7 项，AST 检查而非文本搜索）。
同一个测试文件分别跑在**修复前（git HEAD）**与**修复后**的代码上：

    修复前：Ran 7 tests ... FAILED (failures=6)      # 只有 C7 通过（训练脚本没自己重写置零）
    修复后：Ran 7 tests ... OK                       # 7/7 通过

另外补了 `tests/__init__.py`：此前 `python -m unittest tests.xxx` 会被 site-packages 里同名
`tests` 包遮蔽而报 `ModuleNotFoundError`（第一/二轮 README 里的这条命令其实跑不起来，
应改用 `python tests/xxx.py` 或加了这个 `__init__.py` 之后再用 `-m`）。

### 7.2 GPU 修复验证（`diagnostics/r3_fix_verify.py`，真实运行 2026-10-18 17:32 UTC）

    [T1] default masking zero = True ; set_action_input_masking ok = True
    [T2] mask=True  == pre-fix masked? True   [212, 187, 146, 44, 81, 143, 1, 106, 1, 1, 255, 185, 189, 255]
    [T3] mask=False == pre-fix kept?   True   [212, 187, 146, 44, 81, 143, 1, 106, 1, 1, 255, 185, 189, 255]
    [T4] mask=True  max|emb| at 14 action positions = 0.000000
    [T4] mask=False max|emb| at 14 action positions = 0.096680
    [T5] free-run(keep) = [212, 187, 144, 44, 81, 143, 1, 106, 1, 1, 255, 185, 189, 255]
         deployment(KV) = [212, 187, 144, 44, 81, 143, 1, 106, 1, 1, 255, 185, 189, 255]  equal=True
    [T6] determinism maxdiff = 0.000e+00
         perturb future action#11 -> maxdiff = [0.0 x11, 0.309, 0.131, 0.094]
         perturb past   action#4  -> maxdiff = [0.0, 0.0, 0.0, 0.0, 0.453, 1.445, 0.625, 0.375, 1.203, 0.734, 0.906, 0.938, 0.688, 0.938]
         no-future-leak=True ; past-affects-future=True
    [T7] prefix actions GT vs all-255 (keep mode): tokens equal 11/14, |dA| L1 = 0.01094 (max 0.07858)

    === RESULT ===
      T1_default_zero                PASS
      T2_oft_regression              PASS
      T3_fix_effective               PASS
      T4_zero_when_true              PASS
      T4_nonzero_when_false          PASS
      T5_deploy_matches_free_run     PASS
      T6_deterministic               PASS
      T6_no_future_leak              PASS
      T6_past_does_matter            PASS
    R3_FIX_VERIFY: ALL_PASS

逐条含义：
- **T2**：修复后 `mask_action_inputs=True` 的预测与**修复前代码**逐 token 相同 —— OFT 行为没被改动；
- **T3**：`mask_action_inputs=False` 的预测与修复前“手动关掉掩码”的结果相同 —— 开关确实生效；
- **T5**：独立实现的逐 token 生成与部署 `Model.infer`（KV 缓存）逐 token 一致 —— 部署实现无偏差；
- **T6**：未来动作标记对过去位置的影响**严格为 0**，过去动作对后续位置有影响 —— 因果性成立；
- **T7**：把 14 个前序动作标记全部替换成极值后，仍只有 3/14 个标记改变、动作 L1 变化 0.01094
  —— 旧检查点对前序动作的依赖**很弱**（它训练时这些位置一直是 0）。

### 7.3 部署回归：修复不改变现有检查点的部署行为

用第二轮的统一离线评测（`eval_unified.py`，走部署同款加载路径）在**修复后**的代码上重跑 20 个样本：

    修复前（第二轮，15:19 UTC）: model BOUNDS L1 = 0.0506 / 物理 0.03493 ; identity 0.0121 / 0.00994 ; 4.19x
    修复后（本轮，17:28 UTC）  : model BOUNDS L1 = 0.0506 / 物理 0.03493 ; identity 0.0121 / 0.00994 ; 4.19x
    delta: mean|A_hat - S| = 0.03424 vs mean|A - S| = 0.00994 (3.446x)   —— 与第二轮逐位一致

原因很直接：部署本来就走 `labels=None` 的缓存分支（不置零），本轮只是把它显式写清楚。

---

## 8. 对原来 6000 步训练有效性的影响

| 问题 | 回答 |
|---|---|
| 旧检查点的权重被改动了吗？ | **没有**。SHA-256 仍为 `0d7b95cc…` / `7c5a24b2…`（第二轮核对过，本轮未动） |
| 旧检查点的部署行为变了吗？ | **没有**。修复前后部署路径逐位一致（7.3） |
| 旧训练“无效”吗？ | **不是无效，而是丢了一个能力**：它在动作位置永远看到 0，因此从来没学过“条件于前序动作标记”。它的预测实际是 `f(前缀图像+指令+proprio, 位置下标)`，即**并行解码器**。 |
| 这能解释闭环完全不动吗？ | **不能单独解释**。第二轮已经定量的主因是“A ≈ S 复制当前状态”（模型误差是恒等策略的 4.19 倍）；本轮又量化出掩码只影响 12.30/14 个标记、0.00392 L1，量级不足以造成“零位移”。 |
| 修复会让旧检查点变好吗？ | **不会**。`keep` 让训练机制与部署一致，但旧权重是在 `zero` 机制下学的；要受益必须重训（本轮禁止）。 |
| 历史离线指标还要不要看？ | 要看，但必须标注口径：它们测的是**并行机制**，与部署（自回归 + 真实前序标记）不是同一机制。 |

一句话总结：**这是一处真实存在的训练/部署机制不一致（且训练侧因此失去了自条件能力），
已修复并验证；但它不是闭环不动的直接原因，也不会让现有检查点变好。**

---

## 9. 下一步：小样本过拟合实验方案（**尚未执行，需要批准**）

目的：验证“`keep` 机制下自回归条件是否可学、能否复现训练集动作”，
**不是**验证任务能力，也不产生任何成功率结论。

| 项 | 设计 |
|---|---|
| 数据 | 2 个 episode（`episode_0000000/0000001`），每 5 帧取 1 帧 ≈ 70 个训练样本 |
| 模型 | OpenVLA-7B 冻结 + LoRA r=32 all-linear（与 6000 步同配置） |
| 编码 | 第二轮的 `MODE_FIXED` 合法编码器（31744..31999，0 个 PAD） |
| 机制 | `keep`（本轮修复后） + 因果注意力 |
| 超参 | AdamW lr 5e-4, wd 0, clip 1.0, 400 步, batch 1 |
| 对照组 | 完全相同的配置，仅 `mask_action_inputs=True`（旧并行机制） |
| 指标 | ① 训练帧 14 位置 token 准确率；② 自由运行（逐 token）动作 L1 vs GT；③ 前序动作扰动敏感度（T7 同款） |
| 成本 | 每组约 30–50 分钟 GPU；共 2 组 |
| 保护 | 输出到新目录 `/home/dgut/pusht_runtime/r3_overfit/{keep,zero}`，不覆盖任何现有检查点 |

判据（事先写死，避免事后解释）：若 `keep` 组的自由运行 L1 明显低于 `zero` 组、
且训练帧 token 准确率 > 90%，则说明“自条件机制是可学的、修复方向正确”。

**本轮到此为止，不启动该实验。**

---

## 10. 交付清单与复现命令

### 10.1 新增文件（本分支）

| 文件 | 用途 |
|---|---|
| `diagnostics/r3_mask_probe.py` | 取证探针：掩码布尔值 / 嵌入是否为零 / masked-vs-kept / 部署对照 |
| `diagnostics/r3_causality_scale.py` | 因果性取证（含 HF sdpa 陷阱）+ 20 样本机制差异定量 |
| `diagnostics/r3_patch_modeling.py` | 最小修复的幂等补丁脚本（4 份 modeling + 训练脚本 + 部署脚本） |
| `diagnostics/r3_fix_verify.py` | 修复后验证 T1–T7（含部署一致性与因果性） |
| `tests/test_action_masking_contract.py` | 7 项契约测试（纯 CPU；修复前 6 项失败、修复后 7/7 通过） |
| `tests/__init__.py` | 让 `python -m unittest tests.xxx` 可用（避免被 site-packages 的 `tests` 遮蔽） |
| `patches/r3_action_input_masking.patch` | 本轮改动的完整 diff |

### 10.2 复现命令

```bash
cd /home/T7/ojh/robot_actual/submission
PY=/home/dgut/pusht_runtime/venv_oft/bin/python

# 1) 契约测试（纯 CPU，秒级；修复前应失败 6 项，修复后 7/7 通过）
$PY -m unittest tests.test_action_masking_contract -v

# 2) 取证（需 GPU，约 4 分钟加载 + 1 分钟推理）
$PY diagnostics/r3_mask_probe.py --sample 0 --out /tmp/r3_probe.json
$PY diagnostics/r3_causality_scale.py --n 20 --free-n 5 --out /tmp/r3_causal.json

# 3) 修复（幂等；会同时更新仓库副本 / 基础模型目录 / HF 缓存 / OFT 源码树）
$PY diagnostics/r3_patch_modeling.py

# 4) 修复后验证（需 GPU）
$PY diagnostics/r3_fix_verify.py --out /tmp/r3_fix.json
```

### 10.3 仍未验证（不得当作通过）

| 项 | 状态 |
|---|---|
| `keep` 机制下的**训练效果** | **未执行**（需重训，本轮禁止；方案见第 9 节） |
| 修复后的闭环成功率 | **未执行**（未启动仿真） |
| 旧检查点在 `keep` 机制下重训后的表现 | **未执行** |
| 小样本过拟合实验 | **未执行，等待批准** |
| 其它任务的跨任务能力 | 不成立（检查点只在 push_T 上训练过） |
