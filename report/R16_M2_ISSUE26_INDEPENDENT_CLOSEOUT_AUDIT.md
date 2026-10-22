# R16 / M2 Issue #26 收官交叉审计（2026-10-22）

执行者冻结：feature/r15-true-continuation@9a973206a0fe71e44c65dfdbed34b925842a1234，前版基线b57d16a2579ee69f90703b49ca2b96955329dc8b。单独审计分支，本审计**没有访问远端GPU进程、没有重新运行7B、没有修改main、旧WAL、HDF5、checkpoints或FINAL**。只基于GitHub归档的原始JSON、源代码、commit与提交日志核对；服务器路径引用和真实checkpoint byte hash需由远端执行者复核。

## 一、可以核对到的新事实

- 执行者从b57d16a增加9个提交至9a9732。M2 2任务push_T+stack_bowls 2000步已在执行报告登记REACHED_TARGET/DONE，step2000来自runs/m2_two_2000_20261021T123314Z/checkpoints/step2000；GPU job 7742.6s、TWO_TASK_PILOT已记7952.0s（不同费用口径：wall和含更早run/bucket调整），修账过程包含旧+9937.7s冲正和+2058.7s限定重记，WAL真实性仍需服务器原始链和时间戳验证。
- M2旧action_window_probs漏769视觉前缀；新修复的evidence/dev_eval_v2_aligned.npz档案哈希17f8559b95cc411327915bb2c914a0c486dc88b3c3ef037a246b1843c888e0f1，且mission/m2_p6_two_corrected_report.json报告20/20 PASS/exit0。两任务各8 DEV帧、同normalizer、corrected256bins窗口 CE push_T 2.1214、stack_bowls 3.0963；跨task prompt的own win率1.0，wrong prompt penalty最低0.653 nats。以上是**已提交机器报告中的数值及源码逻辑**，而不是本审计在远端独立重算整个模型权重。
- R16完整prefix+KV增量解码：mission/r16_kv_equivalence.json 32个输入**全部**至少有动作bin分歧，k0一致、k1开始失配，logits差最大31.39，缓存per-call p50 1459.8ms vs完整前向5906.9ms (4.05x)，FAIL_FUNCTIONAL_PARITY，不得投入实际动作控制。tools/r16_three_way.py对8帧记录 reference=train inference==full deploy 8/8、reference==incremental cached 0/8，局部化问题在增量分支，但尚未找到可部署修复。
- R16只缓存视觉：mission/r16_vision_cache.json 报告32输入所有14动作bin一致，logit worst_abs=0；p50 full3058.873ms vs cached3058.237ms，收益~0.0208%，不提供可观加速。其计时范围与之前官方WS server end-to-end p50 7754ms并非严格相同，不能将3058ms写成已经达到25Hz。同步模拟能否合法等待模型和真正官方成功仍NOT_MEASURED。
- 任务数据：mission/r12_task_status.json 12任务真实TaskSpec已整理、3任务历史HDF5各100集、9任务无已验收数据，fold_clothes/pour_liquid_into_cup专家两项BLOCKED_DATA。没有证据表明12任务联合训练或官方FINAL成功已完成。

## 二、独立新发现：P6的物理 no-op L1 公式算错（已最小修复，未远端跑绿）

tools/m2_gate_check.py::baseline_bins(payload,task,state,action,t) 现有代码：
```python
out["l1_noop"] = float(np.abs((s[t] - s[t]) - a[t]).mean())
```
no-op定义为A_hat=S[t]，正确值必为mean(abs(S[t]-A[t]))；原码是mean(abs(A[t]))即**全零关节预测**，不是no-op/identity。故旧P6报告的push_T no-op L1=0.380568和stack_bowls=0.462869不是合法no-op基线，不能用于动作物理改善论证（这不影响bin NLL里r_noop=-V的构造，故不应凭此一项推翻所有P6窗口NLL）。本审计在独立分支只改此一行，加真实HDF5 fixture回归，源码待执行者远端运行红绿并用原冻结DEV重新计算真实no-op L1，旧报告原样保留、另写勘误。

## 三、P6=PASS的严格语义和需要补的研究Gate

- 此PASS是**经修复后的离散256bin概率分布**在每任务8帧上的机器协议PASS，不意味着关节物理L1超过恒速，也不意味着官方12任务成功。P6报告中没有模型解码的SAFE physical14D L1、动作夹爪/关节速度层误差、部署限幅介入率或episode级可信CI；CV物理L1分别仅0.000943663、0.002196996，已经很强。应明确将`P6_machine=PASS`与`P6_capability=NOT_VERIFIED`分开，**先补上述关节物理指标与DEV episode扩展再放行4任务新增正式GPU阶段**。
- P6 forget/no-catastrophic-forgetting检查直接读取stage_evidence.json中的旧`ce_curves`，它们来自修复前的错误视觉prefix窗口。虽同错误位置内部可做描述性趋势，不能用它们证明正确action-token意义上的`no catastrophic forgetting`。mission/m2_p6_erratum.md也承认旧11 save points绝对值不可与新final CE比较，且尚未重算。需将此Gate状态标`LEGACY_UNVERIFIED`，需要时有界重算step0/200/.../2000的正确DEV窗口曲线，不覆盖旧曲线。
- NLL采用修订后的eps=1e-3 one-hot平滑点基线，数学上是确定概率模型的合法log score，但可调eps会严重影响同CV的差值，不等于物理动作精度。应做eps敏感度分析和matched SAFE L1，而非把pass NLL推出“胜恒速机器人控制”。
- KV three-way样本的full路径112个14维输出中有42个bin为0/255（37.5%）；32输入full路径448个bin中154个为0/255（34.375%）。这是端点高占比的诊断信号，**不直接证明物理动作越界或数据污染**。需同true_bins/end-bin率比较，量化quantile clipping、动作SAFE介入率、gripper分量与任务差异，防止高边界预测掩盖物理不匹配。
- 不得把GitHub Issue#26的工程任务各项"已有PASS或负结果已记录"当成整个R15-R23项目完成；R16真实控制时延依旧远超40ms，官方PushT success未知、12任务官方success未知、9任务数据缺失。

## 四、下一执行顺序（原权限内自动化，不要求用户转述小步骤）

CPU并行：
1. 远端用原同frame/normalizer验收本分支no-op L1最小补丁RED→GREEN；生成old/new字段差异与源SHA。
2. 审查旧corrected P6归档真实hash及源模型step2000全状态Manifest/LoRA/投影/数据样本；修正DEV物理动作score+clip率、每任务 episode-macro对CV/no-op，增加DEV已有20回合新预冻结诊断 cohort，不使用FINAL。
3. 评估R16已失败的增量KV路径：保留safe full默认、隔离Full FF vs language_model.step的position/attn/LoRA/pure mask input，建立k1首分歧的逐层trace和证伪假设；两个不同假设试验还不成功就将该后端标BLOCKED_MECHANISM，不把4x速度当可用结果。
4. 12任务合法专家与HDF5布局元数据并行推进，布料液体无真实专家仍标BLOCKED_DATA。

GPU重型严格排队：先完成有界的M2 DEV物理评测和R16首分歧诊断，依Gate可用时执行官方同步DEV1集真实对象进度；先确认官方是否支持慢速推理墙钟与物理timestep分离，不能偷偷改评分器、无效长timeout、越权启动4/12任务GPU大训练。真实P6能力Gate未通过不能按机器P6 PASS直接扩容。

每条路线输出源SHA、真实原始GPU/session日志、stdout/exit、样本索引/episode split及hash、预测NPZ、费用WAL idempotent事件、恢复全状态校验、真正Draft PR和ROUTE append-only。严禁main和老证据改变，禁止用FINAL模型选择。安全失败只阻断依赖支线。

## 证据位置

- 执行者：mission/m2_p6_two_corrected_report.json；evidence/dev_eval_v2_aligned.npz；mission/m2_p6_erratum.md；mission/r16_kv_equivalence.json；mission/r16_three_way.json；mission/r16_vision_cache.json；mission/r12_task_status.json；report/R15_R23_STATUS.md。
- 源码：tools/m2_gate_check.py::baseline_bins(),::forgetting_check()；training/m2_train_shared.py::action_window_probs()；code/xpolicylab/policy/OpenVLA_R42/r42_deploy_server.py::_TorchBackend/step_cached。
- 证据分级：SOURCE VERIFIED + EXECUTOR RAW ARTIFACT PUBLISHED，对远端 checkpoint/WAL GPU完整独立复算属于PENDING REMOTE；不声称ChatGPT远程任务已完成。
