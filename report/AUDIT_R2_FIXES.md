# 第二轮审查修复报告（分支 audit/r2-action-codec-fix）

本文件回答第一轮代码审查提出的 7 个问题。每个问题给出：**结论 → 源码位置 → 修改前后差异 →
执行过的测试与真实输出 → 仍需 GPU/仿真验证的部分**。

## 0. 本轮硬约束（全部遵守，可逐条核对）

| 约束 | 本轮状态 | 核对方式 |
|---|---|---|
| 不重新训练模型 | **未训练**（无任何训练进程、无新权重产物） | 新增文件里没有权重；`training/run_meta_*.json` 仍是原始两份 |
| 不修改主分支 | **未改** `main` = `329614f5` | `git ls-remote origin` |
| 不修改旧检查点 | **未改**；部署副本 sha256 仍与 `CHECKPOINT_IDENTITY.md` 一致 | `diagnostics/check_checkpoint_identity.py` |
| 不把动作改成残差 | **未做**；`A[t]` 仍是绝对关节目标 | `training/train_pushT_pure2.py` 中目标构造未变 |
| 不宣称恢复任务执行能力 | **没有**；本轮没有任何成功率结论 | 全文无 success rate 声明 |
| 只做诊断代码 + 最小修复 | 是 | 见第 8 节文件清单 |

分支起点：`audit/pure-openvla-collapse@3c2b84b` → 新分支 `audit/r2-action-codec-fix`。

---

## 1. 问题 1：动作离散化边界（PAD 作标签 + 1 个 token 偏移）

### 1.1 结论

修复前存在**两个成对缺陷**，同一份“编码”逻辑在训练/部署/诊断里各写一遍：

- **D1（整体偏移 1 个 token）**：编码写的是 `idx = np.digitize(v, LEVELS) ∈ 1..256`，
  取 token 时又写了 `+ 1`，于是训练标签落在 **31745..32000**；而部署端解码窗口是
  `logits[31744 : 31744+256]`，即 **31744..31999**。训练目标与解码窗口整体错开 1 个 token，
  窗口首位 `31744` 永远不可能是训练标签。
- **D2（边界溢出成 PAD）**：`v ≥ 1`（物理动作 ≥ q99，被 clip 到 +1.0）时 `digitize(+1.0) = 256`，
  于是 `token = 31743 + 256 + 1 = 32000`，正好等于基础模型的 `pad_token_id`（32000）。
  **PAD 被当成动作目标**，且它在解码窗口之外，部署时永远选不出来。

### 1.2 源码位置

| 位置 | 修复前 | 修复后 |
|---|---|---|
| `training/train_pushT_pure2.py:50` `tok()` | `np.digitize(...)` → 1..256，再 `+1` | `action_codec.encode_indices(mode="fixed")` → 0..255 |
| `training/train_pushT_pure2.py:47` `detok()` | 手写 `BIN_CENTERS[tokens]` | `action_codec.denormalize(decode_indices(...))` |
| `code/xpolicylab/policy/OpenVLA_Pure/model.py:44-45` | `BINS = 256; BIN_CENTERS = np.linspace(-1,1,256)` | 由 `action_codec` 提供常量与函数 |
| `model.py:226-227`（解码） | `centers = BIN_CENTERS[clip(tokens,0,255)]` | `decode_indices(tokens)` + `validate_token_ids` |
| 新增 | — | `code/xpolicylab/policy/OpenVLA_Pure/action_codec.py`（唯一实现） |

### 1.3 修复后的统一约定（`action_codec.py`）

```
ACTION_TOKEN_BEGIN_IDX = 31743
合法动作 token          = BEGIN+1 .. BEGIN+256 = 31744 .. 31999
PAD_TOKEN_ID = 32000 (text_config.pad_token_id)     STOP_INDEX = 2

encode: idx   = clip(digitize(clip(v,-1,1), LEVELS) - 1, 0, 255)      # 0..255 满射
        token = ACTION_TOKEN_BEGIN_IDX + idx + 1                      # 31744..31999
decode: v_hat = LEVELS[idx]        LEVELS = linspace(-1, 1, 256)
```

不变量（单元测试逐条断言）：任意 `v ∈ [-1,1]` 都落到合法动作 token；永不产生 PAD/STOP/越界；
`idx=0 ↔ v=-1` 与 `idx=255 ↔ v=+1` 精确；256 个 idx 全部可达；量化误差 ≤ `2/255 = 0.00784`。
保留 `linspace(-1,1,256)` 作为重建格点，是为了让修复只是“去掉多余偏移”，不改变取值栅格。

`MODE_LEGACY` 保留**仅用于复刻旧检查点**（它的标签是用旧编码烧进去的），并在测试里
把两个缺陷断言成“回归证据”。

### 1.4 实测：训练标签里到底有多少 PAD

20 个留出回合、94,514 个标签（`tests/test_label_dataset.py`，真实数据 + 训练落盘的 q01/q99）：

    [fixed]  labels=94514  token range=[31744,31999]  PAD=0
    [legacy] PAD labels=11559/94514 = 12.23%   frames with >=1 PAD = 6596/6751 = 97.70%

即：**修复前 12.23% 的动作目标是 PAD 标记，97.70% 的样本至少含 1 个 PAD**；
修复后 **0 个 PAD，全部落在 31744..31999**。

另外实测 `digitize` 的取值范围确实是 `min=1, max=256`（不是 0..255），D1/D2 由此坐实。

### 1.5 单元测试与真实输出

```bash
$ python tests/test_action_codec.py
...
test_boundary_table ... ok          test_minus_one ... ok
test_plus_one ... ok                test_zero ... ok
test_legacy_plus_one_yields_pad ... ok
test_legacy_minus_one_never_hits_window_start ... ok
test_legacy_range_exceeds_window ... ok
test_all_256_indices_used ... ok    test_all_values_map_to_legal_tokens ... ok
test_no_pad_even_after_clipping_extremes ... ok
test_random_values_map_to_legal_tokens ... ok
test_validate_rejects_illegal ... ok
test_gripper_all_values_legal ... ok   test_gripper_zero_one ... ok
test_raw_above_q99 ... ok              test_raw_below_q01 ... ok
test_bad_mode_rejected ... ok          test_encode_decode_idempotent ... ok
test_monotonic ... ok                  test_roundtrip_error_bound ... ok
----------------------------------------------------------------------
Ran 20 tests in 0.202s
OK
```

覆盖点：`-1 / 0 / +1` 三端点（含期望 idx 与 token 的边界表）、分位数裁剪（raw 超过 q99 / 低于 q01）、
夹爪 0 / 1（0→31744，1→31999，且不等于 PAD）、20 万个随机值满射性与合法性、
量化误差上界、编解码幂等、旧编码的 PAD 回归。

### 1.6 对已发布检查点的影响（重要，且未被“修好”）

- 旧检查点的权重里学的是**旧编码**（含 12.23% PAD 目标）。本轮**没有重训、没有改权重**。
- 部署端解码数学**未变**：`v_hat = LEVELS[idx]` 依旧是旧编码的逆映射（对 `idx ≤ 255` 一致），
  因此修复后的 `model.py` 在旧检查点上应给出与之前逐位相同的动作（decode 路径等价）。
- 也就是说：**动作离散化边界已被修正，但这一修正不会让旧检查点变好**——它只是让“以后的训练”
  不再把 PAD 当目标、不再整体偏移 1 个 token。任何“已恢复执行能力”的说法都是无依据的。

---

## 2. 问题 2：统一离线评测（现有 6000 步检查点 vs 恒等策略）

### 2.1 脚本

`diagnostics/eval_unified.py`（+ 共用的 `diagnostics/pure_bootstrap.py`）。要点：

- 直接实例化**部署同一个类** `XPolicyLab.policy.OpenVLA_Pure.model.Model`，因此
  adapter / `ProprioProjector` / `q01,q99` / BOUNDS 统计参数与线上服务走同一条加载路径；
- 输入：三路相机（`cam_head`/`cam_left_wrist`/`cam_right_wrist`，480x640 原始帧）+ 14 维本体状态；
- 样本：`fixed_samples.json` 的前 20 个，全部来自 20 个留出回合（与第一轮同一批）；
- 归一化：统一 **BOUNDS**（`push_T_bounds.npz` 的 amin/amax）；
- 同时报告物理单位误差、逐维误差、预测动作变化量 `mean|A_hat - S|` 与真实变化量 `mean|A - S|`、
  以及 token 合法性统计。

```bash
python diagnostics/eval_unified.py --n 20 --out /tmp/r2_eval.json
```

### 2.2 实测结果（GPU 真实运行，2026-10-18 15:19 UTC）

    模型   : BOUNDS L1 = 0.0506   物理 L1 = 0.03493
    恒等   : BOUNDS L1 = 0.0121   物理 L1 = 0.00994
    模型/恒等 = 4.19x（>1 表示比“照抄当前状态”更差）
    变化量 : mean|A_hat - S| = 0.03424  vs  mean|A - S| = 0.00994  (比值 3.446)

逐维（14 维全列，`model` = 检查点；`ident` = 恒等策略；`pred_dlt/gt_dlt` 为动作变化量）：

| dim | model_bnd | ident_bnd | model_phys | ident_phys | pred_dlt | gt_dlt |
|---|---|---|---|---|---|---|
| left_arm_j0 | 0.0454 | 0.0168 | 0.03911 | 0.01448 | 0.03632 | 0.01448 |
| left_arm_j1 | 0.0446 | 0.0170 | 0.05656 | 0.02160 | 0.06215 | 0.02160 |
| left_arm_j2 | 0.0339 | 0.0152 | 0.03806 | 0.01705 | 0.03390 | 0.01705 |
| left_arm_j3 | 0.0877 | 0.0205 | 0.05661 | 0.01326 | 0.04779 | 0.01326 |
| left_arm_j4 | 0.0942 | 0.0174 | 0.04807 | 0.00887 | 0.04168 | 0.00887 |
| left_arm_j5 | 0.0468 | 0.0151 | 0.03407 | 0.01098 | 0.03368 | 0.01098 |
| left_gripper | 0.1028 | 0.0040 | 0.05138 | 0.00198 | 0.05336 | 0.00198 |
| right_arm_j0 | 0.0172 | 0.0088 | 0.01863 | 0.00951 | 0.01749 | 0.00951 |
| right_arm_j1 | 0.0297 | 0.0103 | 0.03381 | 0.01178 | 0.03284 | 0.01178 |
| right_arm_j2 | 0.0118 | 0.0125 | 0.01106 | 0.01165 | 0.01801 | 0.01165 |
| right_arm_j3 | 0.0685 | 0.0121 | 0.04438 | 0.00783 | 0.04647 | 0.00783 |
| right_arm_j4 | 0.0814 | 0.0057 | 0.03184 | 0.00222 | 0.03331 | 0.00222 |
| right_arm_j5 | 0.0300 | 0.0098 | 0.01818 | 0.00597 | 0.01318 | 0.00597 |
| right_gripper | 0.0147 | 0.0039 | 0.00734 | 0.00193 | 0.00927 | 0.00193 |

token 统计（280 个位置 = 20 样本 x 14 步）：

    min_idx = 1   max_idx = 255   pad_or_illegal = 0
    frac_at_idx0 = 0.0            frac_at_idx255 = 0.1179
    mean_top1_prob = 0.5184       min_top1_prob = 0.0199

读法：
- 14 维里 **12 维比恒等策略更差**（只有 `right_arm_j2` 略好：0.0118 < 0.0125）；
- 末位 bin（idx 255）占了 **11.79%** 的输出，而 idx 0 一次都没用——这与 1.x 的结论一致：
  旧编码下最高一格只能靠 PAD(32000) 表示，模型永远学不到，只能在 255 处饱和；
- §mean|A_hat - S|§ 是 §mean|A - S|§ 的 **3.45 倍**：模型不是“不动”，而是在预测一个**幅度偏大**
  的绝对目标；闭环里它被 §set_joint_position_target§ 应用后因绝对值≈当前状态而被控制器“吃掉”
  （第一轮结论），两步现象并不矛盾。

“是否在复制当前状态”的定量检验（同一批 20 样本、14 维展平后）：

    corr(pred, state) = 0.99188      corr(gt, state) = 0.99939      corr(pred, gt) = 0.99254
    mean|pred - state| = 0.03424     mean|gt - state| = 0.00994

即预测与“当前状态”的相关性（0.992）几乎等于真值与“当前状态”的相关性（0.999），
而预测与真值的残差比“直接复制状态”大 3.4 倍 —— 与第 4 节 proprio 消融的结论一致：
**模型学到的是 A ≈ S（复制本体状态），只是复制得不够准**。

### 2.3 与第一轮 `autoreg_L1=0.0586` 的关系（不可直接比较）

训练日志里的 0.0586 用的是 **q01/q99 归一化**（训练脚本 `evaluate()` 的口径），本节 0.0506
用的是 **BOUNDS 归一化**（OFT 口径，也是恒等基线 0.0121 的口径）。两者尺度不同，**不能相减比较**；
可比的是“同口径下的模型 vs 恒等”：BOUNDS 口径下 **4.19x 更差**，物理单位下 **3.5x 更差**。

### 2.4 可复现性

JSON 里保留了 20 个样本的原始 §state / pred / gt / identity§（14 维数组）与帧号，
审查者可直接重算任意统计量，无需重跑 GPU：§/tmp/r2_eval.json§ 的 §raw§ 字段。
本次两次独立运行（15:19 与 15:28 UTC）逐位一致（§model 0.05062 / identity 0.01207§），
说明推理链路是确定性的。

---

## 3. 问题 3：部署侧任务指令处理

### 3.1 修复前的真实行为（代码证据）

`model.py` 旧版里：

- `_encode_obs_pure()` 把观测指令放进 `observation["instruction"]`（用 `extract_prompt`）；
- 但 `infer()` 只调用 `self._prompt_ids()`，而 `_prompt_ids()` 用的是 `self.instr`（来自 `deploy.yml`
  的 `prompt` 字段）——**observation["instruction"] 从未被读取**，也没有任何日志体现这一点。

再加上部署配置固定：`deploy.yml: prompt: "Push the T-shaped block ..."`，且
`start_ps_pure.sh` 只覆盖 `task_name/port/...`，不改 `prompt` ——
**三个任务（push_T / stack_bowls / hang_mugs）跑的都是同一条 push_T 指令**。

### 3.2 修复（最小、行为等价 + 显式化）

新增 `code/xpolicylab/policy/OpenVLA_Pure/deploy_contract.py`：

- `TRAINED_TASKS = ("push_T",)`、`TRAINED_DATASET`、`TRAINED_INSTRUCTION`（与训练脚本 `INSTR` 一字不差）；
- `resolve_instruction(config_prompt, task_name, obs_instruction, instruction_source="config")`
  返回实际指令 + 是否忽略了观测指令 + 是否属于训练域 + 告警列表；
- `instruction_source="observation"` 是**显式 opt-in**，且会附带“未经验证、不可作为泛化证据”的告警。

`model.py` 改动：
- 构造时通过 `resolve_instruction` 固定指令，并打印
  `[OpenVLA_Pure] instruction pinned from config (task=...): '...'`；
- 每帧调用 `_note_instruction()`：观测指令与固定指令不一致时**打印一次**警告；动作日志新增
  `instruction_used / instruction_source / obs_instruction / obs_instruction_ignored` 四个字段；
- `task_name` 不在 `TRAINED_TASKS` 时打印跨任务声明。

### 3.3 单元测试（`tests/test_deploy_contract.py`，8 项全通过）

    test_config_prompt_wins ... ok
    test_observation_instruction_is_ignored ... ok
    test_missing_config_prompt_falls_back_to_trained ... ok
    test_observation_source_is_opt_in_and_warned ... ok
    test_observation_source_without_instruction ... ok
    test_other_tasks_get_cross_task_notice ... ok
    test_trained_task_list ... ok
    test_prompt_text_matches_training ... ok
    Ran 8 tests in 0.001s  OK

### 3.4 GPU 侧实测（模型真实加载时的行为）

`eval_unified.py` 运行日志里可见修复后的策略服务打印：

    [OpenVLA_Pure] LoRA merged (cameras=3)
    [OpenVLA_Pure] instruction pinned from config (task=push_T): 'Push the T-shaped block to align it precisely with the gray T-shaped pad.'

### 3.5 明确声明

**该检查点只在 push_T 数据上训练**（80 个训练回合，6000 步）。所以：

- `stack_bowls` / `hang_mugs` 的失败**不是跨任务泛化结论**——它既不能说明模型“泛化失败”，
  也不能说明它“在本任务一定能做”，因为指令、任务语义、物体都没在训练里出现过；
- 反过来说，第一轮的“闭环不动”结论只在 **push_T + 固定指令** 这个条件下成立，本轮不扩大其适用范围。

---

## 4. 问题 4：本体状态（proprio）消融

### 4.1 脚本改进（`diagnostics/ablate_proprio.py`，旧版 `ablate.py` 保留不删）

| 旧版问题 | 新版做法 |
|---|---|
| 同时更换图像与状态（两个变量一起动，无法归因） | **固定同一帧的三路图像与同一条指令**，只改状态 |
| 只打印前 8 维动作 | **14 维全量打印**（动作、动作-状态、token、概率） |
| 没有任何 token 层信息 | 打印 14 步的 argmax idx、token id、top1 概率与 top3 候选 |
| 没有投影层信息 | 直接调用 `ProprioProjector`，打印 4096 维特征的范数/统计量/与真实状态的 L2、cos |
| 只有“零状态”一个对照 | 增加**归一化中点状态**（`state=(pmin+pmax)/2` → proprio_norm = 0）作为 OOD 对照 |

### 4.2 实测（GPU，固定图像 `episode_0000066.hdf5 t=221`，固定指令）

本体输入（14 维，归一化后）：

     dim                  real zero(raw0)        mid
     left_arm_j0        0.7368     0.4446     0.0000
     left_arm_j1        0.2993    -0.9999     0.0000
     left_arm_j2       -0.0615    -1.0000     0.0000
     left_arm_j3       -0.5366     0.9751     0.0000
     left_arm_j4       -0.0219     0.2171     0.0000
     left_arm_j5        0.1061     0.3409     0.0000
     left_gripper      -1.0000    -1.0000     0.0000
     right_arm_j0       0.0003     0.0003     0.0000
     right_arm_j1      -1.0000    -1.0000     0.0000
     right_arm_j2      -1.0000    -1.0000     0.0000
     right_arm_j3       0.9639     0.9639     0.0000
     right_arm_j4       0.0930     0.0930     0.0000
     right_arm_j5       0.4512     0.4512     0.0000
     right_gripper      1.0000    -1.0000     0.0000

动作输出（14 维全部）：

     dim                   real        zero         mid |real-zero|          gt
     left_arm_j0        0.22826    -0.00237    -0.40902     0.23063     0.26223
     left_arm_j1        1.63265     0.01746     1.31834     1.61519     1.61117
     left_arm_j2        1.04791     0.01367     1.23727     1.03424     1.05522
     left_arm_j3       -0.96804     0.00008    -0.86565     0.96812    -0.96976
     left_arm_j4       -0.12675     0.00125    -0.13463     0.12799    -0.10559
     left_arm_j5       -0.14719     0.00278    -0.28779     0.14997    -0.16242
     left_gripper       0.00392     1.00000     0.00392     0.99608     0.00000
     right_arm_j0       0.00063     0.00063     0.00063     0.00000     0.00000
     right_arm_j1       0.00896     0.00896     1.71161     0.00000    -0.00000
     right_arm_j2       0.00737     0.00737     1.33338     0.00000     0.00000
     right_arm_j3       0.00000     0.00000    -1.13486     0.00000     0.00000
     right_arm_j4       0.00046     0.00046    -0.27411     0.00000    -0.00000
     right_arm_j5       0.00167     0.00167     0.00167     0.00000     0.00000
     right_gripper      1.00000     0.00392     0.50196     0.99608     1.00000

动作 token 与概率（14 步）：

     step              idx  token   top1_p   zero_p top3(idx:prob)
     left_arm_j0       212  31956   0.1406   0.7479 212:0.141 216:0.071 213:0.071
     left_arm_j1       187  31931   0.0889   0.7453 187:0.089 191:0.078 190:0.074
     left_arm_j2       144  31888   0.1544   0.7362 146:0.154 144:0.154 145:0.120
     left_arm_j3        44  31788   0.0669   0.6142 44:0.067 46:0.052 45:0.052
     left_arm_j4        81  31825   0.0858   0.5814 81:0.086 87:0.076 79:0.059
     left_arm_j5       143  31887   0.1425   0.7668 143:0.143 148:0.086 138:0.086
     left_gripper        1  31745   0.9949   0.9942 1:0.995 3:0.002 5:0.000
     right_arm_j0      106  31850   0.9996   0.1660 106:1.000 107:0.000 105:0.000
     right_arm_j1        1  31745   0.9999   0.7702 1:1.000 2:0.000 3:0.000
     right_arm_j2        1  31745   1.0000   0.8567 1:1.000 2:0.000 3:0.000
     right_arm_j3      255  31999   0.8419   0.3101 255:0.842 254:0.146 252:0.003
     right_arm_j4      185  31929   0.9979   0.6243 185:0.998 184:0.002 182:0.000
     right_arm_j5      189  31933   0.9991   0.8035 189:0.999 188:0.001 190:0.000
     right_gripper     255  31999   0.9040   0.9998 255:0.904 254:0.084 245:0.004
     identical token positions: 6/14 ; max |action(real)-action(zero)| = 1.615187

ProprioProjector 输出特征（4096 维）：

     real_state  |f|=213.03  mean=-0.105  std=3.327  min=-11.80  max=11.31
     zero_state  |f|=355.81  mean=-0.083  std=5.559  min=-18.76  max=21.38
     mid_state   |f|= 63.96  mean= 0.012  std=0.999  min= -3.28  max= 3.47
     L2(real,zero)=280.13   L2(real,mid)=196.16   cos(real,zero)=0.617

### 4.3 结论（**这条更正了第一轮的一个表述**）

第一轮 README 写的是“proprio 消融：真实状态与全零状态输出**完全相同**（proprio 未起作用）”。
受控消融**复现不出**这个结论，实测是相反的：

- 把本体状态置零后，动作**塌缩到 ≈ 0**（0.0000–0.0175），14 步里只有 **6 个 token 与真实状态相同**，
  动作最大差 **1.615 rad**；投影特征 L2 差 **280.1**（cos 0.617）。
- 真实状态下的输出**紧跟当前关节角**（例：`left_arm_j1` 输出 1.6327、当前状态 1.6466、GT 1.6112）。
- 归一化中点状态（训练分布之外）下输出落到 **±1 饱和区**（1.32–1.71），说明输入一旦偏离训练分布，
  模型就退化为极端值。

因此更准确的表述是：**该检查点学到的是“把本体状态复制成动作目标”（A ≈ S），而不是“忽略本体状态”**。
这与第一轮的核心闭环现象完全一致（A ≈ S ⇒ 目标关节角 ≈ 当前关节角 ⇒ 机械臂几乎不动），
但**归因要从“proprio 通道失效”改成“绝对目标被学成了当前状态的复制”**。

第一轮的旧脚本 `diagnostics/ablate.py` 保留未删（可复核），但**其结论以本次受控实验为准**：
旧脚本同时更换图像与状态、只打印前 8 维、且用 `mean|action - state|` 这类对“复制状态”不敏感的指标，
容易把三种条件的差异读成“相同”。

> 这一条对后续方案选择有直接影响：如果再训练，光“修好感官通道”没用，
> 需要让目标变成**相对量**（残差 Δ）或引入能让“不动”付出代价的监督信号。
> 本轮**按约束没有做这个改动**。

---

## 5. 问题 5：部署实际加载的权重是否与 CHECKPOINT_IDENTITY.md 的 SHA-256 一致

新增 `diagnostics/check_checkpoint_identity.py`：从 `deploy.yml` 读出**线上真正加载的三个路径**，
从 `CHECKPOINT_IDENTITY.md` 解析期望的 SHA-256 与字节数，现场重算并逐项比对；
读不到文件时输出 `UNVERIFIED`（绝不写成“已验证”）。

```bash
python diagnostics/check_checkpoint_identity.py
```

实测输出（2026-10-18，本机可读权重）：

    file                                     size sha256(actual)                                                       status
    adapter_model.safetensors           484458600 0d7b95ccef30b8e79980511689980c62657e2c26c65c4b46ba20c62f47048803     MATCH
    adapter_config.json                       970 b20d0792d6f5c7ff261821a5f92ad8bfcd9a07c96430181648b3286c0e5653dc     MATCH
    proprio_projector--pure.pt           67373328 7c5a24b2aff65b3b0ef581e1459f740c7d69d25e517c7280c48c89118143572b     MATCH

    VERDICT: ALL_MATCH

结论：**部署加载的适配器 / 适配器配置 / proprio 投影层，与审查包记录的归档副本字节一致（3/3 MATCH）**。
结果已写进 `CHECKPOINT_IDENTITY.md` 的“部署侧实际加载核对”一节。
（审查者若在没有权重的机器上运行该脚本，会得到 `UNVERIFIED`——这是预期行为。）

---

## 6. 问题 6：实际运行的动态加载模块

### 6.1 机制与真实路径

基础模型 `config.json` 的 `auto_map` 决定一切（`trust_remote_code=True`）：

    "AutoConfig":             "configuration_prismatic.OpenVLAConfig"
    "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction"

所以被 import 的是 **HF 动态模块缓存里的一份副本**，不是仓库同名文件：

    /home/dgut/.cache/huggingface/modules/transformers_modules/openvla-7b/

策略服务由 `start_ps_pure.sh` 在**宿主机**（非容器）用 `venv_oft` 启动，因此这条路径就是线上代码。

### 6.2 真实 SHA-256 与差异

| 文件 | 字节 | SHA-256 | 与 `openvla_oft_src/prismatic/extern/hf/` 同名文件 |
|---|---|---|---|
| modeling_prismatic.py | 48,906 | `b5431a074c0025a12e46dc954a5e18d1d73477babb5ae42e3a12ab4b907f33a6` | `diff -q` 相同 |
| configuration_prismatic.py | 5,865 | `68cc5ae34f1b46af3168d8d479cb81bb776965653453fd904aa8eefb6c8f9f68` | `diff -q` 相同 |
| processing_prismatic.py | 12,713 | `2474a5c3fdd4ff15234924636dc0ead6419c89f49699193be47dbb87510a5425` | —（未在上游源码树中，单独收录） |

**差异为空**：实际运行的动态模块与 OFT 源码树中的同名文件**字节完全相同**（sha256 亦一致）。
第一轮“仓库没收录、无法核对”的问题，本轮以收录原样文件解决：`code/openvla_dynamic_modules/`
（含 `README.md` 与三条复现命令）。

---

## 7. 问题 7：第二阶段学习率调度（README 与日志不一致）

### 7.1 事实

用 `diagnostics/lr_schedule_forensics.py` 逐行解析两份原始日志（日志每 200 步记一次 lr）：

| 阶段 | lr 观测 | 可证明的下降步数区间 | 与 README 旧表述 milestone=1000 |
|---|---|---|---|
| 阶段一 | 步 800: `5.00e-04` → 步 1000: `5.00e-05` | `[801, 1000]` | **相容**（粒度不足，无法唯一定位） |
| 阶段二 | 步 1800: `1.00e-04` → 步 2000: `1.00e-05` | `[1801, 2000]` | **矛盾**（1000 不在区间内） |

原始输出：

    == stage 1 : pure3c_train.log ==
    points=16 steps=1..3000
    lr values seen: 5.00e-05, 5.00e-04
    lr drop: 5.00e-04 @step 800  ->  5.00e-05 @step 1000   (true drop step in [801, 1000])
    VERDICT: COMPATIBLE with claimed milestone=1000 (allowed interval [801,1000]; NOT uniquely determined)

    == stage 2 : pure3c_cont.log ==
    points=16 steps=1..3000
    lr values seen: 1.00e-05, 1.00e-04
    lr drop: 1.00e-04 @step 1800  ->  1.00e-05 @step 2000   (true drop step in [1801, 2000])
    VERDICT: CONTRADICTION - claimed milestone=1000 outside the logged interval [1801,2000]

### 7.2 处理

- **不猜**：没有 `--lr_milestone` 的落盘记录（`run_meta*.json` 不含该字段，launcher 未留存），
  阶段二的真实取值只能界定在 `1801..2000`，例如 1900 与 2000 会产生完全相同的日志。
- `training/README.md` 已更正：删除“阶段二在步 1000 降到 1e-5”的错误表述，改为上表 + 复核命令，
  并注明阶段二的重建命令缺少 `--lr_milestone`。
- 该缺失项已写入 `MISSING.md`（“阶段二的实际 --lr_milestone：缺失且与 README 旧表述矛盾”）。

---

## 8. 交付清单

### 8.1 新增文件（本分支）

| 文件 | 对应问题 | 说明 |
|---|---|---|
| `code/xpolicylab/policy/OpenVLA_Pure/action_codec.py` | 1 | 统一编解码（唯一实现，纯 numpy，含 `validate_token_ids`） |
| `code/xpolicylab/policy/OpenVLA_Pure/deploy_contract.py` | 3 | 指令固定 / 训练域声明的显式契约 |
| `tests/test_action_codec.py` | 1 | 20 项边界单元测试 |
| `tests/test_deploy_contract.py` | 3 | 8 项指令/训练域单元测试 |
| `tests/test_label_dataset.py` | 1 | 3 项真实标签合法性集成测试（缺数据自动 skip） |
| `diagnostics/pure_bootstrap.py` | 2,4 | 公共引导层（部署同款加载路径） |
| `diagnostics/eval_unified.py` | 2 | 统一离线评测（三路相机 + 14 维状态 + BOUNDS + 物理单位 + 逐维 + 变化量） |
| `diagnostics/ablate_proprio.py` | 4 | 本体消融（固定图像/指令；14 维全量 + token 概率 + 投影特征） |
| `diagnostics/lr_schedule_forensics.py` | 7 | 学习率调度取证 |
| `diagnostics/check_checkpoint_identity.py` | 5 | 部署权重 SHA-256 核对 |
| `code/openvla_dynamic_modules/{modeling,configuration,processing}_prismatic.py` + `README.md` | 6 | 实际运行的动态模块原样收录 |
| `patches/r2_codec_instruction.patch` | 1,3 | 改动前后完整 diff（261 行） |
| `.gitignore` | — | 排除 `__pycache__` |

### 8.2 修改文件（最小修复）

    code/xpolicylab/policy/OpenVLA_Pure/model.py | 109 ++++++++++++++++++++++---   (修改前 254 行 → 修改后 322 行)
    training/train_pushT_pure2.py                |  31 ++++++++--                   (211 → 226 行)
    2 files changed, 123 insertions(+), 17 deletions(-)

要点：
- `model.py`：(a) 解码走 `action_codec` 并加 `validate_token_ids` 断言；(b) 指令固定 + 告警 + 动作日志字段；
  (c) 新增 `last_decode`（token / token id / 每步 top3 概率）供诊断使用。**未改任何推理数学**。
- `train_pushT_pure2.py`：(a) `tok()/detok()` 改为 `action_codec`（`MODE_FIXED`）；(b) 训练循环内
  `assert_legal_token_ids`；(c) `run_meta` 增记 `codec_mode`。**未改网络、损失、优化器、数据管道**。

同步落地的部署副本（字节一致，sha256 已核）：`RoboDojo/XPolicyLab/policy/OpenVLA_Pure/{model.py,action_codec.py,deploy_contract.py}`
与 `/home/dgut/pusht_runtime/{action_codec.py,train_pushT_pure2.py}`。

### 8.3 全部执行过的测试与真实结果

| 命令 | 结果 |
|---|---|
| `python tests/test_action_codec.py` | **Ran 20 tests … OK** |
| `python tests/test_deploy_contract.py` | **Ran 8 tests … OK** |
| `PUSHT_HDF5=… python tests/test_label_dataset.py` | **Ran 3 tests … OK**；fixed PAD=0；legacy PAD=11559 (12.23%) |
| `python diagnostics/lr_schedule_forensics.py` x2 | 阶段一 COMPATIBLE、阶段二 **CONTRADICTION** |
| `python diagnostics/check_checkpoint_identity.py` | **VERDICT: ALL_MATCH**（3/3） |
| `python diagnostics/eval_unified.py --n 20`（GPU） | 模型 BOUNDS L1 **0.0506** / 物理 **0.03493**；恒等 **0.0121** / **0.00994**；比值 **4.19x** |
| `python diagnostics/ablate_proprio.py`（GPU） | 见第 4 节 |

合计 **31 项单元/集成测试全部通过**（真实输出见各节）。

---

## 9. 已核实 vs 仍待 GPU / 仿真验证

### 9.1 已核实（本轮有真实输出）

1. 动作离散化的两个缺陷（D1 偏移 1 token、D2 边界溢出成 PAD），以及修复后标签 100% 合法（`PAD=0`）；
2. 旧编码在 20 个留出回合上产生 12.23% PAD 标签、97.70% 的样本含 PAD；
3. 现有检查点在同一批 20 个固定留出样本上**比恒等策略差 4.19x（BOUNDS）/ 3.5x（物理）**；
4. 部署端解码输出的 280 个 token 全部合法（`pad_or_illegal=0`），且 11.79% 落在末位 bin；
5. 观测指令被忽略（代码 + 契约单元测试）；三任务共用同一条 push_T 指令（`deploy.yml` + `start_ps_pure.sh`）；
6. 部署加载的 adapter / config / proprio 投影层与归档副本 3/3 SHA-256 一致；
7. 实际运行的动态模块路径、SHA-256，且与 OFT 源码树字节一致（差异为空）；
8. 阶段二学习率下降发生在步 1801..2000，与“milestone=1000”矛盾，且无法唯一恢复。

### 9.2 仍待验证（**未执行**，不得视为通过）

| 项 | 为什么没做 | 需要什么 |
|---|---|---|
| 修复后编码器的训练效果 | 本轮禁止重训 | 用 `MODE_FIXED` 重新训练一个检查点（同样的 6000 步），再跑 `eval_unified.py` |
| 修复后策略的闭环成功率 | 本轮不接仿真 | RoboDojo 一个完整回合（push_T），命令见 `deploy/` 与第一轮 README |
| 旧检查点在修复后解码路径下与旧行为逐位一致 | 需要闭环或逐步日志对照 | 一次 `PURE_ACTION_LOG` 闭环回合，与新代码输出 diff |
| `stack_bowls` / `hang_mugs` 的跨任务能力 | 检查点未在这些任务上训练，做也不构成泛化结论 | 需要相应任务的训练数据与新检查点 |
| 指令对输出的实际影响（消融） | 观测指令被忽略是**代码级**结论；未单独跑“换指令”的 GPU 对照 | 可用 `instruction_source="observation"` 显式开启后对照（未验证路径） |

---

## 10. 复现命令汇总（含前置条件）

```bash
# 前置：PY=/home/dgut/pusht_runtime/venv_oft/bin/python  （CPU 测试只需 numpy/h5py）
cd /home/T7/ojh/robot_actual/submission

# 1) 纯 CPU，无需数据集（秒级）
$PY tests/test_action_codec.py
$PY tests/test_deploy_contract.py
$PY diagnostics/lr_schedule_forensics.py --log logs/pure3c_cont.log --stage 2 --claimed-milestone 1000

# 2) 需要数据集与本地权重
PUSHT_HDF5=<push_T hdf5 目录> $PY tests/test_label_dataset.py
$PY diagnostics/check_checkpoint_identity.py

# 3) 需要 GPU（加载 OpenVLA-7B + 6000 步适配器，约 15 GB 显存）
$PY diagnostics/eval_unified.py --n 20 --out /tmp/r2_eval.json
$PY diagnostics/ablate_proprio.py --out /tmp/r2_ablate.json
```

若在别的机器上跑 GPU 部分，需要先设置：
`PUSHT_RUNTIME`（run_meta/bounds/split/fixed_samples 所在目录）、`PUSHT_ROBODOJO`（RoboDojo 根）、
`PUSHT_HDF5`（数据集）、可选 `PUSHT_DEPLOY_YML`、`PUSHT_BOUNDS`、`PUSHT_SAMPLES`、`PUSHT_SPLIT`。
GPU 部分默认走 `PUSHT_SAMPLES/fixed_samples.json` 的前 20 个样本，**不做随机采样**，保证可复现。
