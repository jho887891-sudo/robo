# 检查点身份（不上传权重）

| 项 | 值 |
|---|---|
| 本地路径 | `/home/T7/ojh/robot_actual/pushT_pure3c_6000/checkpoints/final` |
| LoRA adapter | `lora_adapter/adapter_model.safetensors` — **484,458,600 B** — `sha256 0d7b95ccef30b8e79980511689980c62657e2c26c65c4b46ba20c62f47048803` |
| proprio 投影 | `proprio_projector--pure.pt` — **67,373,328 B** — `sha256 7c5a24b2aff65b3b0ef581e1459f740c7d69d25e517c7280c48c89118143572b` |
| adapter 配置 | `adapter_config.json` — 970 B — `sha256 b20d0792d6f5c7ff261821a5f92ad8bfcd9a07c96430181648b3286c0e5653dc` |
| 基础模型 | OpenVLA-7B（`openvla_offline_bundle/models/openvla-7b`，3 个 safetensors 分片，约 15 GB） |
| 训练步数 | 阶段一 3000 + 续训 3000 = **6000** |
| LoRA 超参 | r=32, alpha=16, dropout=0.0, `target_modules=all-linear`（含视觉塔） |
| 动作表示 | 256 个离散 bin；q01/q99 见 `training/run_meta_6000.json` |
| 离线指标 | `autoreg_L1 = 0.05863387882709503`（20 留出样本，首步，BOUNDS 归一化） |
| final 与 step3000 | LoRA 与 proprio 的 sha256 **完全相同** → `final` 即续训第 3000 步产物 |

复算命令：
```bash
sha256sum pushT_pure3c_6000/checkpoints/final/lora_adapter/adapter_model.safetensors \
          pushT_pure3c_6000/checkpoints/final/proprio_projector--pure.pt
```
