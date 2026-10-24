# R26｜M2 推理入口可达性与失效关闭审计设计

**来源基线：** `fix/r25-m2-deployment-contract@274b851818b76a3f875b0a1a401a4bdb66358948`
**实施类型：** R25 D0-A 的边界修复与审计，非新的闭环部署架构。目标是令没有物理安全认证的路径**绝不被误认作已可部署**。
**证据标准：** GitHub独立审计分支、RED→GREEN 回归、原始Actions日志。没有GPU与原 checkpoint 的行为一律明确标注 BLOCKED。

## 问题与原因（源代码可独立检查）

1. **真实启动路径不可达 M2**：`code/xpolicylab/policy/OpenVLA_R42/r42_serve.py` 直接 `from OpenVLA_R42.model import Model`；`__init__.py` 的 `Model` 属性也转向 `model.py`。它们都不是 `model_m2.py`。原 `deploy.yml` 固定 R15 Push-T checkpoint，`setup_eval_policy_server.sh` 固定 `policy_name=OpenVLA_R42`。即使给启动脚本参数 `task_name=stack_bowls`，不能据此确认载入了M2。R15代码不改动；在新模型未完成实际安全认证前也不允许把默认Model切到M2。该条报告为 **BLOCKED_ON_ENTRYPOINT**。
2. **CPU桩被当作机器人动作的风险**：`model_m2.Model.get_action` 仅检查 `model_cfg.get("backend") is None`。任意提供 `backend=FakeBackend` 且 `device=cpu` 的外部配置，都可以在不设置“测试模式”的条件下，通过WebSocket ModelTemplate 得到格式合法的机器人关节目标。既然控制器可能只检查动作格式，测试桩与真正的 VLA 行为不能共用默认真实动作接口。
3. **诊断用CV环境变量可能静默覆盖模型**：`m2_deploy_server.M2TaskDeployServer._decode_tokens` 会因为 `R42_POLICY=cv` 直接返回CV token。它原本是研究基线；若在任意VLA评测环境继承这一变量，日志却被误用为“模型输出”，研究结论会失真。需要显式 **opt-in**，并在输出中报告政策模式。旧R15环境变量行为不变。
4. **replay缺少图像和真实7B上下文**：`M2TaskDeployServer.replay` 调用 `next_token_logits(prefix, None, feat)`。这可用于某些CPU桩，但真实视觉模型前向需要图像，不能声称这是可靠的训练帧回放。应对 processor 已加载的运行环境拒绝执行并记录原因，不得生成伪回放证据。
5. **不能误把元数据格式认证等同于物理认证**：`safety_contract.source_kind` 与 `verified_by` 为软件格式声明，真实性仍需交叉核对机器人/仿真控制器的物理关节限位与速率限制，及来自伺服的 `prev_cmd/dt`。这次不解除真实 `Model.get_action` 禁令。

## 限定实现（不改变用户已冻结的架构）

- **仅在新增的M2路径修改**：`model_m2.py` 和 `m2_deploy_server.py`。R15的 `model.py`、`r42_serve.py`、`deploy.yml`、`r42_metadata_bridge.py` 保持只读。
- M2 `Model` 的桩后端动作出口需要显式测试授权 `allow_stub_action_for_contract_test=True`，并要求 `device == "cpu"`；没有这个标志时在构造阶段拒绝注入的后端。不允许通过 `allow_real_7b_smoke=True` 间接开启测试桩。
- `M2TaskDeployServer` 新增 `allow_cv_diagnostic=False`；`R42_POLICY=cv` 且标志未提供时拒绝；只有两者都明确才执行CV，并输出 `policy_mode=cv_diagnostic`。默认正经推理输出 `policy_mode=vla`。
- `replay` 有真实图像processor时必须 `fail-closed`，不得用 `pixel_values=None` 假冒真实7B回放。CPU桩可继续回放，但日志明确说明 `replay_scope=cpu_stub_only`。
- 绝不修改旧测试的基本物理契约。只有显式 CPU stub 夹具需补 opt-in。用测试证明保护有效而非只检查配置字符串。
- 额外提交静态入口检查，对 `r42_serve.py` 的启动模型做精确依赖核对；如果后续修改启动器，测试必须明确失败，要求重新进行安全审核。没有GPU时**不能宣称部署可用**。

## 验收门槛

- Gate R26-0：测试先 RED，新负向测试必须实际抓出上面至少 3 类问题。
- Gate R26-1：新增修复 GREEN + 旧 R25 CPU 测试绿色；公开真实GitHub Actions job logs。
- Gate R26-2：独立审计报告按 `VERIFIED/IMPLEMENTED/BLOCKED/HYPOTHESIS` 列出剩余接口差距，Draft PR基于PR #35分支，不合并。
- Gate R26-3（未来）：取得原M2 checkpoint、可靠joint hard limits、real7B GPU smoke/tokenizer prompt parity以及privileged state官方DEV闭环，并把真任务结果与控制日志一一对应。未经研究负责人授权不执行正式部署或训练。

**保护范围：** main、HDF5、checkpoint、WAL、旧日志、旧FAIL、先前PR #31/#33/#34/#35 证据一律不可改；本轮新增optimizer steps为0。
