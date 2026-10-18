# 仍未纳入 / 缺失清单（如实标注，无替代品）

| 项 | 状态 | 说明 |
|---|---|---|
| 训练启动命令的原样留存 | **缺失** | 无 shell history、无 launcher 脚本；`training/README.md` 中的命令为依据 argparse 默认值 + run_meta + 日志行重建，并已标明 |
| 阶段一（3000 步）适配器权重 | 未上传（按安全要求） | 只上传哈希与配置；需要时可按 `CHECKPOINT_IDENTITY.md` 的哈希核对本地文件 |
| RoboDojo 演示数据集（100 episode） | 未上传（按安全要求） | 见 `data/README.md` 的官方获取方式与结构说明 |
| 仿真资产（Assets 目录） | 未上传 | 体积大且为第三方资产 |
| stack_bowls / hang_mugs 的逐步动作链日志 | **缺失** | `PURE_ACTION_LOG` 只在 push_T 诊断回合开启（`evidence/push_T/action_chain.jsonl`）；其余任务仅有过程量 `progress_*.jsonl` |
| 上游依赖源码 | 未复制 | `XPolicyLab/utils/process_data.py`、`prismatic/*`（openvla_oft 内的 OpenVLA 实现）、`scripts/robodojo.sh` 均属上游仓库，通过 `patches/` 与 `code/` 说明其调用关系；不重复搬运 |
| 模型代码的 trust_remote_code 缓存版本 | 未复制 | 实际加载的是 `~/.cache/huggingface/modules/transformers_modules/openvla-7b/modeling_prismatic.py`（HF 动态模块缓存）；本轮已观测到它与仓库内同名文件不是同一份，需审查者注意 |
