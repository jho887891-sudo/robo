# 训练（产生 6000 步检查点的真实脚本）

## 文件
- `train_pushT_pure2.py` — 实际训练脚本（原始版本 11,790 B，`sha256 4c8fa3c0…`；本轮修改后的文件见 `../patches/r2_codec_instruction.patch`）
- `run_meta_3000.json` / `run_meta_6000.json` — 两个阶段落盘的配置与指标
- `../logs/pure3c_train.log`、`../logs/pure3c_cont.log` — 两个阶段的原始训练日志

## 启动命令（**重建**，非原样留存）

运行时未保留 shell history（`~/.bash_history` 无相关记录），launcher 脚本也未落盘。
以下命令由三处硬证据重建：argparse 默认值、`run_meta*.json` 的 `lr/steps`、日志行
`loaded LoRA from ... (879 tensors)`。

```bash
# 阶段一（3000 步，lr 5e-4 为 argparse 默认值；run_meta_3000.json 记录 lr=0.0005）
python train_pushT_pure2.py --cams 3 --proprio 1 --steps 3000 --lr 5e-4

# 阶段二（续训 3000 步；run_meta_6000.json 记录 lr=0.0001）
python train_pushT_pure2.py --cams 3 --proprio 1 --steps 3000 --lr 1e-4 \
  --init_ckpt /home/dgut/pusht_runtime/run_pure3c/checkpoints/final
```

### 学习率调度：README 旧表述已更正（第二轮审查问题 7）

旧版本这里写着“MultiStepLR 在步 1000 处 x0.1”，并据此重建了阶段二命令。用日志逐行核对后，
这个说法**只对阶段一成立**，对阶段二是错的：

| 阶段 | 日志中 lr 首降的观测位置 | 真实下降步数可证明区间 | 与 milestone=1000 是否相容 |
|---|---|---|---|
| 阶段一 | 步 800 还是 5.00e-04，步 1000 已是 5.00e-05 | [801, 1000] | **相容**（日志粒度 200 步，无法唯一定位） |
| 阶段二 | 步 1800 还是 1.00e-04，步 2000 已是 1.00e-05 | [1801, 2000] | **矛盾**（1000 不在区间内） |

复核命令（纯 CPU，几秒）：

```bash
python diagnostics/lr_schedule_forensics.py --log logs/pure3c_train.log --stage 1 --claimed-milestone 1000
python diagnostics/lr_schedule_forensics.py --log logs/pure3c_cont.log  --stage 2 --claimed-milestone 1000
```

实测输出（原文）：

    == stage 1 : pure3c_train.log ==
    lr values seen: 5.00e-05, 5.00e-04
    lr drop: 5.00e-04 @step 800  ->  5.00e-05 @step 1000   (true drop step in [801, 1000])
    VERDICT: COMPATIBLE with claimed milestone=1000 (allowed interval [801,1000]; NOT uniquely determined)

    == stage 2 : pure3c_cont.log ==
    lr values seen: 1.00e-05, 1.00e-04
    lr drop: 1.00e-04 @step 1800  ->  1.00e-05 @step 2000   (true drop step in [1801, 2000])
    VERDICT: CONTRADICTION - claimed milestone=1000 outside the logged interval [1801,2000]

因此：**阶段二的实际 `--lr_milestone` 无法从现存材料唯一恢复**（只能确定落在 1801..2000 之间；
例如 2000 与 1900 会产出同样的日志）。本轮不做猜测；该值已列入 `../MISSING.md`，上面的阶段二命令
只能视为“milestone 未知”的示意。

## 训练配置（脚本内实参）

| 项 | 值 | 来源 |
|---|---|---|
| 样本 | train 80 / val 20（日志：`train=80 val=20`） | 日志 |
| chunk | 1（`NUM_ACTIONS_CHUNK=1`） | 日志 |
| 动作维度 | 14 | 日志 |
| 相机数 / proprio | 3 / 1 | 日志 |
| LoRA | r=32, alpha=16, dropout=0.0, all-linear | `build()` |
| 优化器 | `AdamW(lr=args.lr, weight_decay=0.0)` | 代码 |
| 调度 | `MultiStepLR(milestones=[lr_milestone], gamma=0.1)`，argparse 默认 milestone=1000 | 代码 |
| 梯度裁剪 | `clip_grad_norm_(ps, 1.0)` | 代码 |
| 保存 | 每 `--save_every`（默认 1000）步：`vla.save_pretrained(<dir>/lora_adapter)` + `torch.save(pp.state_dict())` | 代码 |
| 离线评估 | 每次保存后 `evaluate(...)`：自回归解码 14 个 token → 反归一化 → 与 GT 比 L1，并报 pred_std/gt_std/uniq | 代码 |
| 随机种子 | 0 | argparse 默认 |

## 实际参与更新的参数
- **仅** LoRA 适配器（`target_modules="all-linear"`，含视觉塔，约 1.1 亿参数）与 `ProprioProjector`（14→llm_dim）。
- OpenVLA-7B 主干（含视觉塔原始权重）**冻结**，代码中未解冻任何主干参数。

## 损失
```python
inp = cat([prompt_ids, action_ids(BEGIN+t+1), [STOP]])
lab = cat([IGNORE_INDEX * len(prompt_ids), action_ids, [STOP]])
out = vla(input_ids=inp, labels=lab, pixel_values=pv, proprio=pr, proprio_projector=pp)
loss = out.loss   # 只在动作 token + STOP 上计算交叉熵
```
即：**只对 14 个动作 token + 1 个 STOP 计损失**，提示词与图像前缀全部 `IGNORE_INDEX`。

## 动作目标构造（关键）

- 目标 = `A[t]`（HDF5 `action/*`），已用数据验证 `A[t] == S[t+1]`（绝对关节目标，见 `../data/README.md`）；
- 归一化：`norm = clip(2*(A - q01)/(q99 - q01) - 1, -1, 1)`，q01/q99 为训练集动作的 1%/99% 分位；
- 分箱与 token id：**第二轮已改为统一实现** `../code/xpolicylab/policy/OpenVLA_Pure/action_codec.py`
  （`MODE_FIXED`）：`idx = clip(digitize(norm) - 1, 0, 255)`，`token = 31743 + idx + 1 ∈ 31744..31999`；
- 解码：`v = LEVELS[idx]`，`LEVELS = linspace(-1, 1, 256)`，再反归一化回物理单位。

### 已发布的 6000 步检查点用的是**修复前**的编码

旧写法是 `idx = digitize(norm) ∈ 1..256`，再 `+1` 取 token，导致：

1. 标签整体落在 **31745..32000**，比部署端解码窗口（31744..31999）多偏移 1 个 token；
2. `norm ≥ 1`（物理动作 ≥ q99，被 clip 到 +1.0）时输出 **token 32000 = pad_token_id**。
   实测 20 个留出回合：**11,559/94,514 = 12.23% 的标签是 PAD**，97.70% 的样本至少含 1 个 PAD。

因此本轮修复**只对今后的训练生效**：旧检查点未重训、未改动，权重里烧的是旧编码，部署端
只能继续按同一栅格解码（数学上等价，见 `../report/AUDIT_R2_FIXES.md` 问题 1 的说明）。
