# R25-M2 D0：双任务部署前置契约设计（2026-10-22）

## 目标与边界
从固定的 `research/r24-training-convergence-audit@a3db2a64a3d60664f51439461d0bfc197d41508d` 追溯 M2 step2000 的训练身份、2-task normalizer、真实训练提示词、任务 ID、动作 codec、三相机42D输入和安全限制。避免把 R15 单任务 bridge 的失败「修成」静默错误。

本设计是 **D0-A：可验证的 M2 部署输入/输出契约与 fail-closed 预检查**，而不是宣布官方仿真已可运行。正式 7B 加载、Isaac Sim、RoboDojo 控制端到端 smoke 是后续 D0-B，缺少远端模型文件时保持 BLOCKED。

### 源码确认的断点
- `r42_metadata_bridge.build_stage_metadata` 重建的是 R15 64/16 TRAIN/DEV + 单task hash，不能装载 M2 80/20 + per-task payload。
- `R42DeployServer.__init__` 对整个 `metadata.normalizer` 调用 `NormalizationContract.from_metadata`，不支持多任务外层容器。
- `R42DeployServer.build_prompt` 使用 Push-T 固定 `INSTRUCTION`。M2 训练通过 `m2_train_shared._make_prompt` 调用 `PurePromptBuilder("openvla")`，需要每任务原始文本和 include_task_token。
- `R42DeployServer.act` 调用 `decode_with_metadata` 不传 `task`，M2 应 fail-closed。
- `model.py` 的观察指令不匹配时仅 WARN+IGNORE，不允许在多任务部署中继续沿用此行为。
- R24离线夹取的 `pmin/pmax` 是训练数据范围，不是可证明的 ARX-X5 物理关节安全范围。禁止当作权威机器人硬限。

## D0-A 实现接口
新模块 `code/xpolicylab/policy/OpenVLA_R42/m2_deploy_contract.py`，只读 CPU：
```
build_session_metadata(checkpoint_dir, task,
                       normalizer_path, splits_path,
                       stage_evidence_path, safety_contract_path)
    -> (metadata: dict, attestation: dict)

M2SessionContract(metadata, task).observe(state, timestamp?, env_idx?)
M2SessionContract(metadata, task).decode(bins14, state14, velocity14,
                                         prev_cmd14?, dt?)
M2SessionContract(metadata, task).expected_prompt
```
- checkpoint **必须**有 `MANIFEST.json`, `trainer_state.json`, `sampler.json`, `lora_adapter/adapter_model.safetensors`, `proprio_projector_42.pt` 等全状态材料，逐项校验 SHA，不依赖 PyTorch/CUDA。
- trainer checkpoint schema: `r15-full-state-v1`，且 `global_step=2000` 对真实M2；函数不硬编码 step 数但要求 checkpoint 与 stage evidence 一致。
- trainer `data_contract` 中 task set/ID、`normalizer_sha256`、每任务block哈希、`split_sha256`、`camera_order`、`n42`、`base_dim`、`gripper_dims`、`action_tokens=15`、`target_representation=velocity_residual`、`alignment_version=r42-v2` 都与训练器/源码/外部冻结文件一致。
- stage evidence `schema=m2-train-evidence-v1`，`prompts`、`prompt_sha256`、`include_task_token` 不能缺失；这些字段要与 `training/m2_train_shared.prompts_for` 重新生成并交叉校验，**训练实际 tokenizer token 序列必须在 GPU smoke 再比**，CPU不冒称完成。
- 正式安全合同必须独立提供且有可溯源依据：`schema=r25-m2-robot-safety-v1`、`source_kind=robot_joint_hard_limits`、`verified_by`，每任务 `amin/amax/max_rate` 全部为14维有限数、严格递增。校验真实性还需物理设备/仿真配置独立复核；仅JSON声称 verified 不构成真实验证。
- 输出一个**固定任务**的 session metadata，包含多任务原normalizer payload哈希及明确 `task_id=selected_task`、`serving_task`、`trained_prompt`、`include_task_token`、`safety_contract_sha256`；safety_limits采用所选任务已提交的外部合同，不从normalizer的 pmin/pmax猜测。
- `M2SessionContract` 使用 `R.normalizer_for_task(payload, task)` / `r8_normalization.NormalizationContract.from_metadata`，通过 `R.EpisodeHistory` 输出真实 `[S14,V14,Acc14]` 归一化后的42D（零速度t0，零加速度t1）；`decode` 显式传 `task=task` 并应用 `R.decode_command` 的夹爪+限幅和可选 rate rule。
- 对未知任务、错误任务ID、外部安全合同不存在、哈希不一致、TRAIN/DEV交叉、错误camera维度、prompt不一致、缺失checkpoint字段、normalizer被改变均拒绝。
- 不直接调用原 `R42DeployServer`/ `model.py` 进行实际推理，不在这次改动中偷偷兼容 M2。D0-B 必须另行完成 per-task real-backend serving adapter 才能报完整闭环 READY。

## 测试和门禁
先写失败测试：真实仓库 `data/m2_task_normalizer_2task.json` 验证；临时密封checkpoint/manifest/stage和单独测试安全合同；至少包含正例、任务污染、normalizer篡改、metadata错误、缺失证书、source_kind非法、任务指令错、42D/R42解码路径和rate校验。测试使用 numpy+pytest，不能依赖 CUDA 或 HDF5；GitHub Actions先观察失败（RED），完成实现后 GREEN。

持续保护：历史R15部署元数据不可改；main、checkpoint、HDF5、WAL、旧FAIL不能修改。不调用 `optimizer.step`，GPU 0训练步。

## 验收状态
- **D0-A PASS**：GitHub CI真实测试通过，合法meta可由模拟密封checkpoint重建，非法meta确实被拒绝，真实2-task normalizer原样读取，R15回归没有变化。
- **D0-B BLOCKED**：真实M2密封checkpoint在原服务器的复核、生产 safety contract 签定、训练token parity、7B GPU smoke和官方闭环尚未跑。所有部署成功率均不可声明。
