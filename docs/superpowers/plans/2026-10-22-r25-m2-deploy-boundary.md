# R25 D0-A：实施计划（2026-10-22）

基线：`research/r24-training-convergence-audit@a3db2a64a3d60664f51439461d0bfc197d41508d`
设计：`docs/superpowers/specs/2026-10-22-r25-m2-deploy-boundary-design.md`
分支：`fix/r25-m2-deployment-contract`

**Task 1 (RED):** 新建 `tests/test_r25_m2_deploy_contract.py` 和最小CI工作流 `.github/workflows/r25-m2-contract.yml`，所有逻辑按设计先写失败回归（模块缺失是预期红灯，需看到运行日志）。

**Task 2 (GREEN):** 最小实现 `m2_deploy_contract.py`：
- `build_session_metadata` 完整离线hash验证、任务/归一化/提示词签定及外部安全限幅；
- `M2SessionContract` 真正按照任务完成42D观察和14D动作残差解码；
- 不修改R15和checkpoint；
- 运行GitHub Actions看实际结果，不以语法通过替代回归。

**Task 3 (REVIEW):** CPU脚本与JSON实际数据快照核对、逆向错误注入，单独产出证明：数据/代码路径/哪些还未在远端验证，建立 Draft PR（不合并）。

**Task 4 (后续阻断D0-B):** 在具备远端checkpoint/HDF5/安全认证合同后，由执行器另行完成真实7B加载/token parity和控制接口 smoke，不属于本轮已授权资源。失败只阻断D0-B，不回滚 A/B CPU安全验证。

预算：本次 0 optimizer steps，不启动任何训练。
