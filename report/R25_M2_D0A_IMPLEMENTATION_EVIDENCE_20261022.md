# R25 M2 D0-A：ChatGPT 直接实施与审计报告

日期：2026-10-22。基线：`research/r24-training-convergence-audit@a3db2a64a3d60664f51439461d0bfc197d41508d`。独立任务 Issue #32。所有代码只在 `fix/r25-m2-deployment-contract`，**不改 main，不覆盖原M2模型/数据，不运行优化步骤，不运行Isaac Sim闭环**。

## 1. 设计判断（基于实码的边界）

旧入口 `OpenVLA_R42/model.py`、`r42_deploy_server.py` 服务 R15 单任务：R42DeployServer 初始化时直接以整个 metadata.normalizer 构建一个 NormalizationContract，提示词也固定为 Push-T，decode 没有 task 参数。M2 数据真实存在 per-task normalizer，不能用允许错误 hash 的方式启动旧入口。

R24把TRAIN `pmin/pmax` 当成离线安全截断，而它们并非已证明的机器人关节硬限。M2要部署时必须有**独立安全合同**，不能从训练数据推导机器人的物理安全边界。因此D0-A采用显式外部合同及 CPU-only fail-closed 预检，未取得真实物理证明不开放实机动作。

独立模块：
1. `m2_deploy_contract.py`：读取真实2-task normalizer、严格校验由M2训练器保存的 full-state checkpoint/Manifest/data_contract、split/stage evidence/prompts/mix/任务ID及各项SHA；单任务 session 输出完整多任务normalizer、所选任务外部安全界限、来源hash。42D历史与14D codec解码复用已存在R42权威实现。**没有独立安全合同直接报错**。
2. `m2_deploy_server.py`：M2专用单任务实例，复用R42的14-token输出、图像与KV接口，但任务绑定显式、normalizer显式、推理语句从M2训练 `_make_prompt` 构造、`decode` 显式 `task=...`；错指令报错，不忽略。
3. `model_m2.py`：M2专用ModelTemplate与RoboDojo相同四键动作输出和四相位延迟，拒绝缺少任务ID、错误观测指令、旧R15 metadata静默fallback；真实 GPU backend 必须显式 `allow_real_7b_smoke=True` 且检查底座config/LoRA/projector SHA。**即使GPU加载完成，XPolicyLab ModelTemplate 的真实get_action仍拒绝机器人动作**，因为该接口尚未取得真实 `prev_cmd/dt` 的安全速率限幅合同。CPU注入stub可用于控制契约测试，不构成部署。
4. R15原 `model.py`、`r42_deploy_server.py`、`r42_metadata_bridge.py` 与冻结checkpoint **没有修改**。

## 2. 测试先行、RED -> GREEN 可复核

| Gate | GitHub Actions Run | 原始结果 |
|---|---|---|
| TDD-1 RED: M2 metadata module 尚无 | 39772943673 | 预期 `ModuleNotFoundError: m2_deploy_contract` |
| TDD-1 GREEN | 39773357022 | 10项新增PASS + 32项旧PASS、2 SKIP |
| TDD-2 RED: M2 per-task server 尚无 | 39773944163 | 预期 `ModuleNotFoundError: m2_deploy_server` |
| TDD-2 GREEN | 39774159578 | CI SUCCESS，任务分离和桩后端解码验证 |
| TDD-3 RED: M2 ModelTemplate 尚无 | 39774477294 | 预期 `ModuleNotFoundError: OpenVLA_R42.model_m2` |
| TDD-3 GREEN | 39774706656 | 23项新增PASS + 32项旧PASS、2 SKIP |
| TDD-4 初版测试夹具错误 | 39775014113 | 初版没有关闭假RGB分辨率校验；修测试本身，无生产改动 |
| TDD-4 RED: 真实机器人动作缺少安全禁止 | 39775110795 | 预期拒绝原因 `real M2 backend requires base_model_path`，并未触发拟验证的rate安全门 |
| TDD-4 GREEN | 39775228828 | CI SUCCESS，真实控制动作被独立 fail-closed Gate 拦截 |

已在GitHub Actions Ubuntu/Python3.10运行上述测试，使用真实 `data/m2_task_normalizer_2task.json` 和 `data/m2_splits.json`，配备**仿真密封的checkpoint及显式标注为 synthetic 的安全合同**。它验证纯CPU接口，不证明真实7B checkpoint或物理限位已匹配。

## 3. 状态定义

### VERIFIED（GitHub Actions）：
- 真实2-task normalizer被读入并用于CPU测试；
- 模拟密封checkpoint的Manifest和原始文件哈希不一致会 fail-closed；
- 两任务有不同的 prompt / q01/q99 / 42D历史，14D输出基于R42真实codec；
- 旧R15相关元数据回归无退化；
- 对错误任务、错误指令、非法安全边界、缺失硬限合同、normalizer/相机顺序不一致会拒绝；
- stub-only RoboDojo ModelTemplate入口可以返回指定14D动作格式；
- 真实后端的机器人动作 API 在安全速率/部署Gate未满足时 fail-closed。

### IMPLEMENTED（尚未在远程硬件验证）：
- `build_session_metadata` 支持对真实远端封存M2 checkpoint做静态身份复核；
- `M2TaskDeployServer` 复用真实R42 14-token/KV backend接口，生成M2任务特定prompt；
- `model_m2.Model` 可执行真实7B **显式授权加载 smoke**，但真实GPU加载及token parity尚未执行。

### BLOCKED：
1. 原机器 `runs/m2_two_2000_20261021T123314Z/checkpoints/step2000` 原始全部文件不在本ChatGPT运行环境，不能实际对真实full-state做证明。
2. 安全合同中 `source_kind=robot_joint_hard_limits` 与 `verified_by` 只是格式和可溯源声明，真正设备或Isaac Sim的物理limit仍需独立对照，不能当作完成硬件认证。
3. 尚未实现/验证真实控制器 `prev_cmd`、`dt` 和max_rate的端到端安全限制，因此 **真实 ModelTemplate.get_action 的动作输出被主动封锁**。
4. 真实7B加载、训练token序列校验、KV smoke、官方Push-T PRIVILEGED EE/物体位姿/contact闭环均未运行，也不可宣称成功率。
5. 训练步数为0，H1/H2/H3/H4的因果归因结论不因本工程Gate改变。

## 4. D0-B 最小可判定验收（申请，不擅自执行）

- 原服务器只读调用 CLI 生成新目录下的 M2 meta（提供 sealed checkpoint、normalizer、split、stage evidence、经过独立设备/仿真核实的joint hard limits）。
- 任务 `push_T/stack_bowls` 分别复算 metadata、输入42D、14 token ids、task prompt wrapper/token sha；LoRA、projector、base config hash比对。
- 开展真实7B GPU只推理smoke：观察同checkpoint同输入的 action token、RAW残差以及SAFETY受限动作，不发送机器人，记录GitHub原始日志/hash/WAL。
- 对伺服端真实 `prev_cmd/dt` 与节拍、安全边界进行独立回归，确认限速生效后才考虑放开ModelTemplate真实动作发出，**还需研究负责人正式部署授权**。
- 后续官方DEV闭环保留特权目标/末端/接触轨迹及新鲜官方_result，不能把R15闭环作为M2成功率。

## 5. 研究决策
优先完成R25 A线真实TRAIN-SEEN采样核验和B线GT裁剪不可消除误差，并取得D0-B远程凭证。未有上述证据时不批准同配方+6000步或更换action chunk监督目标。

**自审范围：** 单独新增M2模块，风险点为真实7B prompt-token parity、RoboDojo远端registry加载入口、物理限幅来源和控制速率，它们都在无证据时 fail-closed/明确阻断；本报告不把“通过CPU接口”误称为“真7B部署成功”。
