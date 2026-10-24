# R26 独立安全复核结案报告（2026-10-24）

**原基线**：`fix/r25-m2-deployment-contract@274b851818b76a3f875b0a1a401a4bdb66358948`，上游 R24 原始证据 `a3db2a64a3d60664f51439461d0bfc197d41508d`。

**审计分支**：`review/r26-m2-safety-entrypoint-20261022`。**范围**：纯CPU模型入口与审计边界；0新优化步；未使用服务器checkpoint/GPU/IsaacSim，未修改main、历史数据、R15启动器、旧FAIL。

## 一、复核发现及实码依据

1. **默认RoboDojo启动器从未指向M2**。`code/xpolicylab/policy/OpenVLA_R42/r42_serve.py`直接导入 `OpenVLA_R42.model.Model`（旧R15），`deploy.yml`仍写 `task_name: push_T`和R15单任务checkpoint。因此不能把新增 `model_m2.py` 的CPU测试通过描述为「M2已经可以从正式启动命令运行」。为避免误放真实控制权限，原启动器保持不动，闭环状态仍是BLOCKED。
2. **测试桩动作可能被误当真实机器人输出**。先前仅在 `device.startswith("cuda")` 时拒绝被注入的假后端。CPU伪后端通过合法 `Model.get_action` 输出14D结构。已改为必须显式`allow_stub_action_for_contract_test is True`且`device=="cpu"`，同时在动作出口重复检查；`allow_real_7b_smoke`不能代替CPU测试标志。
3. **环境变量隐式切换CV**。`R42_POLICY=cv`原先足以让M2在调用14-token解码时返回恒速token，却没有外部明确授权或政策标识。已要求额外`allow_cv_diagnostic is True`，否则拒绝，并将 `policy_mode=vla/cv_diagnostic`与`serving_task`透传至RoboDojo模型输出。CV继续按被选任务的q01/q99编解码。**原R15环境变量逻辑未改**。
4. **假Replay缺失三相机像素**。`M2TaskDeployServer.replay`过去对含processor的真实模型使用`pixel_values=None`，并可能误称回放复现。现对含processor的模型一律拒绝该CPU-only回放；只有无真实processor的测试桩能回放，并标明`replay_scope=cpu_stub_only`。
5. **训练与部署token可能不一致**。M2训练器 `training/m2_train_shared.py`记录 `prompt_token_ids_sha256`，旧D0-A只校验英文指令文本和其SHA。现强制从 stage evidence 读取两任务token哈希，选择会话时绑定所选任务hash；`server.build_prompt`在真实prefill前重建token id列表并计算 `MT.sha256_json(ids)`，不匹配直接异常，**不执行VLA前向**。
6. **注入metadata可伪造任务描述或缺失来源**。先前session只核查提示词非空、元数据基本SHA长度。现对齐冻结 `MT.prompts_for([task], include_task_token)` 的实际文本，并要求`m2_checkpoint_manifest_sha256/m2_training_evidence_sha256/m2_data_contract_sha256/m2_safety_contract_sha256`字段完整且格式合法。**格式完整仍不等于身份真实性**，认证需要独立读取远端文件重算、绑定checkpoint和stage证据。

## 二、TDD原始证据

| 实验 | Run | 验证内容 | 结果 |
|---|---:|---|---|
| 旧接口RED | 39790287109 | 未授权CPU桩、隐式CV、Replay等六项 | 6 FAILED / 26 PASSED |
| 权限与CV修复GREEN | 39790857187 | 更新老CV测试为显式opt-in | 32 PASSED + 32历史PASSED / 2 SKIP |
| tokenizer RED | 39791421913 | 缺少token hash、hash不绑定、错误token放行 | 3 FAILED / 33 PASSED |
| tokenizer GREEN | 39791679105 | stage token hash及prefill前核查 | SUCCESS |
| 模型外层policy_mode RED | 39791950344 | model envelope缺少policy_mode, CV开关未透传 | 2 FAILED / 36 PASSED |
| 模型外层GREEN | 39792255950 | 每次Model.get_action可见policy_mode/serving_task | 38 PASSED + 32历史PASSED / 2 SKIP |
| 注入metadata RED | 39792653974 | 伪训练提示词、缺来源哈希被放行 | 2 FAILED / 38 PASSED |
| **注入metadata GREEN（本轮固定验收）** | **39793227231** | 全部安全回归 | **40 PASSED + 32历史PASSED / 2 SKIP** |

CI链接规范：`https://github.com/jho887891-sudo/robo/actions/runs/<run-id>`。以上为真实GitHub Actions运行结果，且RED均已留证，不重写历史日志。

## 三、当前结论

- **VERIFIED（CPU）**：未经明示授权的CPU桩被拒；CV必须双重opt-in且输出政策身份；含processor的伪Replay被拒；对真正训练token序列的hash校验会拒绝错误token；错误任务提示词与缺失元数据来源字段会拒绝。冻结R15原文件未改。所有上述结论的证据仅限GitHub Actions的CPU测试及测试桩和真实GitHub M2 2-task normalizer。
- **IMPLEMENTED**：M2真实训练prompt的token hash验证、任务限定stub服务、新元数据完整性字段及格式校验；**尚未在原训练服务器真实checkpoint上复核**。
- **BLOCKED**：真实M2完整checkpoint哈希比对、真实7B GPU Smoke、Isaac仿真状态/接触轨迹、经独立验证的joint hard limits和prev_cmd/dt速率安全；M2真实Action输出仍被设计性禁用；R15标准启动器并未配置为M2。
- **HYPOTHESIS**：训练不足、LR、泛化、残差表征的因果归因仍未完成，不允许用此次CPU测试声称已解决Push-T闭环。

## 四、给Codex/DeepSeek的低复杂度后续执行

只有本报告以及完整脚本准备完毕后，远程执行者可做固定命令运行、产出JSON和hash，不决定架构或放开控制。
- 在具备checkpoint的服务器**只读**重算 `sampler.json` 曝光帧清单；精准输出TRAIN-SEEN/UNSEEN。
- 只读检查M2 checkpoint/stage evidence存在性、manifest内各文件SHA、prompt_token_ids_sha256、数据split；未知或不匹配一律 BLOCKED，不重训。
- 只进行GPU只推理 smoke（需单独预算/既有授权和真实安全条件），不调用`optimizer.step`，不执行600步任务。
- 不允许擅自改`model_m2.get_action`禁令、旧R15入口、checkpoint/HDF5/WAL、main。

**此Review PR只要求合入R25的安全修复，不要求合并main，不代表M2已具备正式闭环能力。**
