# 训练（产生 6000 步检查点的真实脚本）

## 文件
- `train_pushT_pure2.py` — 实际训练脚本（11,790 B，`sha256 4c8fa3c098b91b1f9117df4d…`，与运行目录中的副本一致）
- `run_meta_3000.json` / `run_meta_6000.json` — 两个阶段落盘的配置与指标
- `../logs/pure3c_train.log`、`../logs/pure3c_cont.log` — 两个阶段的原始训练日志

## 启动命令（**重建**，非原样留存）
运行时未保留 shell history（`~/.bash_history` 无相关记录），launcher 脚本也未落盘。
以下命令由三处硬证据重建：argparse 默认值、`run_meta*.json` 的 `lr/steps`、日志行
`loaded LoRA from ... (879 tensors)` 与 `lr=1.00e-05`（MultiStepLR 在步 1000 处 ×0.1）。

```bash
# 阶段一（3000 步，lr 5e-4 为 argparse 默认值；run_meta_3000.json 记录 lr=0.0005）
python train_pushT_pure2.py --cams 3 --proprio 1 --steps 3000 --lr 5e-4

# 阶段二（续训 3000 步；run_meta_6000.json 记录 lr=0.0001，日志显示步 1000 后降到 1e-5）
python train_pushT_pure2.py --cams 3 --proprio 1 --steps 3000 --lr 1e-4 \
  --init_ckpt /home/dgut/pusht_runtime/run_pure3c/checkpoints/final
```

## 训练配置（脚本内实参）

| 项 | 值 | 来源 |
|---|---|---|
| 样本 | train 80 / val 20（日志：`train=80 val=20`） | 日志 |
| chunk | 1（`NUM_ACTIONS_CHUNK=1`） | 日志 |
| 动作维度 | 14 | 日志 |
| 相机数 / proprio | 3 / 1 | 日志 |
| LoRA | r=32, alpha=16, dropout=0.0, all-linear | `build()` |
| 优化器 | `AdamW(lr=args.lr, weight_decay=0.0)` | 代码 169 行附近 |
| 调度 | `MultiStepLR(milestones=[lr_milestone], gamma=0.1)`，默认 milestone=1000 | 代码 |
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
loss = out.loss   # 仅在动作 token + STOP 上计算交叉熵
```
即：**只对 14 个动作 token + 1 个 STOP 计损失**，提示词与图像前缀全部 `IGNORE_INDEX`。

## 动作目标构造（关键）
- 目标 = `A[t]`（HDF5 `action/*`），已用数据验证 `A[t] == S[t+1]`（绝对关节目标，见 `../data/README.md`）；
- 归一化：`norm = clip(2*(A - q01)/(q99 - q01) - 1, -1, 1)`，q01/q99 为训练集动作的 1%/99% 分位；
- 分箱：`np.digitize(norm, linspace(-1,1,256))` → token id `= ACTION_TOKEN_BEGIN_IDX + t + 1`；
- 解码：`center = linspace(-1,1,256)[token]`，`A = (center+1)/2*(q99-q01) + q01`。
