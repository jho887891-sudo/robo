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
| 动作编码约定 | **旧编码（legacy）**：`token = 31743 + digitize(v) + 1 ∈ 31745..32000`，其中 `digitize(v)=256` 落到 PAD(32000)。第二轮已定位并修复，但**本检查点未重训**，所以部署端解码仍必须用同一栅格（见 `report/AUDIT_R2_FIXES.md` 问题 1） |

复算命令：
```bash
sha256sum pushT_pure3c_6000/checkpoints/final/lora_adapter/adapter_model.safetensors \
          pushT_pure3c_6000/checkpoints/final/proprio_projector--pure.pt
```

## 部署侧实际加载核对（第二轮，2026-10-18 现场复算）

审查者关心的不是“归档副本”，而是**部署时真正读进去的那几个文件**。线上路径来自
`XPolicyLab/policy/OpenVLA_Pure/deploy.yml`（策略服务由 `start_ps_pure.sh` 在宿主机启动）：

| deploy.yml 字段 | 实际路径 | 字节 | SHA-256 | 与上表比对 |
|---|---|---|---|---|
| `adapter_dir` | `/home/dgut/pusht_runtime/model_cache/pure3c/lora_adapter/adapter_model.safetensors` | 484,458,600 | `0d7b95ccef30b8e79980511689980c62657e2c26c65c4b46ba20c62f47048803` | **MATCH** |
| `adapter_dir` | `/home/dgut/pusht_runtime/model_cache/pure3c/lora_adapter/adapter_config.json` | 970 | `b20d0792d6f5c7ff261821a5f92ad8bfcd9a07c96430181648b3286c0e5653dc` | **MATCH** |
| `proprio_projector_path` | `/home/dgut/pusht_runtime/model_cache/pure3c/proprio_projector--pure.pt` | 67,373,328 | `7c5a24b2aff65b3b0ef581e1459f740c7d69d25e517c7280c48c89118143572b` | **MATCH** |

结论：**部署加载的适配器与 proprio 投影层，与本文件表格记录的归档副本字节一致（3/3 MATCH）**。
复算方式（可复现，需能读取本地权重；读不到时会输出 UNVERIFIED 而不是“已验证”）：

```bash
python diagnostics/check_checkpoint_identity.py            # 默认读 deploy.yml + 本文件
python diagnostics/check_checkpoint_identity.py --out /tmp/identity.json
```
