# 仍未纳入 / 缺失清单（如实标注，无替代品）

更新：2026-10-18（第二轮审查）。已关闭项标注 **已关闭**，仍然缺失的项保持如实标注。

| 项 | 状态 | 说明 |
|---|---|---|
| 训练启动命令的原样留存 | **缺失** | 无 shell history、无 launcher 脚本；`training/README.md` 中的命令为依据 argparse 默认值 + run_meta + 日志行重建，并已标明 |
| 阶段二（续训）的实际 `--lr_milestone` | **缺失（且与 README 旧表述矛盾）** | 日志只能证明学习率首降发生在步 ∈ [1801, 2000]；与“milestone=1000”矛盾，且无法唯一恢复。见 `diagnostics/lr_schedule_forensics.py` 与 `training/README.md` 的更正 |
| 阶段一（3000 步）适配器权重 | 未上传（按安全要求） | 只上传哈希与配置；需要时可按 `CHECKPOINT_IDENTITY.md` 的哈希核对本地文件 |
| RoboDojo 演示数据集（100 episode） | 未上传（按安全要求） | 见 `data/README.md` 的官方获取方式与结构说明；标签合法性集成测试在缺数据时自动 skip（skip 不计为通过） |
| 仿真资产（Assets 目录） | 未上传 | 体积大且为第三方资产 |
| stack_bowls / hang_mugs 的逐步动作链日志 | **缺失** | `PURE_ACTION_LOG` 只在 push_T 诊断回合开启（`evidence/push_T/action_chain.jsonl`）；其余任务仅有过程量 `progress_*.jsonl` |
| 上游依赖源码 | 未复制 | `XPolicyLab/utils/process_data.py`、`prismatic/*`（openvla_oft 内的 OpenVLA 实现）、`scripts/robodojo.sh` 均属上游仓库，通过 `patches/` 与 `code/` 说明其调用关系；不重复搬运 |
| 实际运行的 trust_remote_code 动态模块 | **已关闭** | 已原样收录到 `code/openvla_dynamic_modules/`（modeling / configuration / processing 三份，附 SHA-256）；实测与 `openvla_oft_src/prismatic/extern/hf/` 同名文件字节一致 |
| 修复后的动作编码器的训练验证 | **未执行（本轮禁止重训）** | `action_codec.MODE_FIXED` 已通过 20 项单元测试与真实标签合法性测试，但**没有任何模型是用它训练出来的**；它是否能改善闭环执行，本轮无法回答 |
| keep 机制下的训练效果/小样本过拟合实验 | **未执行（等待批准）** | 第三轮只做机制修复与验证；重训方案见 `report/AUDIT_R3_AUTOREGRESSIVE_MASK.md` 第 9 节 |
| 残差动作目标的实现 | **未实现（本轮按约束未改动作定义）** | 仅有统计依据与下一轮设计（见 report/AUDIT_R3_AUTOREG_RESULTS.md 第 6/9 节）；已证实 R2(A|S)=0.9966、残差表示分辨率 x5.3 |
| 32 样本过拟合实验的长程趋势 | **未探索** | 只跑满 300 次更新上限；keep 组在 200-300 步仍在下降 |
| 修复后策略的闭环/仿真验证 | **未执行** | 本轮只改诊断代码与最小的部署指令/编解码逻辑，未启动 RoboDojo 仿真，也没有任何成功率数据 |
| 部署侧重跑（新代码下的 1 个回合） | **未执行** | 旧检查点在修复后的解码路径下行为应与之前一致（解码数学未变），但这一点未经闭环实测确认 |
