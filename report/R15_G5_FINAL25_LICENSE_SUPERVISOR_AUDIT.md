# R15 G5 FAIL / R16 / M2 / FINAL25 / LICENSE 独立交叉审计
日期：2026-10-21。执行基线 feature/r15-true-continuation@9305563a2cf947739ebcc8cf428a11e2542dec90。主分支 main@329614f54d3090b9b6ab120855ba50fbc93bdd1e 保护不变。本文件基于已提交源码与证据，无新 GPU、无新 FINAL，不能证明正在运行的 r4 任务已经完成。

## 已核实的 G5 FAIL 与因果边界

mission/r15_gate_G5_1000.json 真实记录：冻结DEV16集×2帧32个样本、同一帧列表sha 67b43204...；SAFE14D episode macro L1: CV=0.00204748078249，VLA step1000=0.00253845500629，相对CV劣化23.9794%；四个速度分层0/4达到预注册改善5%；G5.pass_rule false，verdict FAIL，exit 3。旧G5不能事后放宽，继续该路线+2000步无证据支持，应阻断。同时step1000 correction R²=-0.120786、CV R²=0.028358（SAFE口径），残差并未证明有效。

由于A[t]=S[t+1]且V=S[t]-S[t-1]，CV的逐坐标预测误差确等于残差R=(A-S)-V的幅值，这是代数恒等式。TRAIN-only 42D->R线性模型R²负，仅说明此线性历史特征预测器不足，不证明非线性历史信息无效，也不证明“只有RGB才包含信息”。若进行新的研究实验，应先冻结可区分视觉依赖/时序误差/量化稀疏/动态噪声的少量诊断假设；不从G5 FAIL推导视觉信息不可学习。

执行者报告1664累计实际optimizer updates、其中1000唯一父checkpoint lineage步骤、664废弃段/重复算步；需在远程运行 tools/r15_ledger_lineage.py 并对比每段原始WAL/real checkpoint MANIFEST/hash，不能仅文字宣称逐项已独立验收。废弃664步**仍是实耗的计算**，不能以“唯一lineage只有1000步”把它们自动从资源账本或授权的累计优化步口径中免除。区分 lineage steps 和 all executed steps，在进一步训练前由独立账本证明两者。

## 发现新的P0：Supervisor Gate可以被错误放行

mission/supervisor_queue_v5.json 有两个G5 job命令末尾追加了 shell 分号 echo记录exit，然后 exit 0，使得真实G5 subprocess rc=3时，该job对supervisor仍报告rc=0。tools/mission_supervisor.py 的 dep_state 只认依赖job的状态 DONE，没有读取/核验官方Gate JSON。结果是失败的Gate可能被误标DONE。

更严重：任务 m1_2000_seg3_to_3000 的 depends_on 仅是 loop_pushT_stage1，而不是经SHA锁定并真正PASS的G5结果；另一个 m1_2000_step1000_to_3000 的依赖又指向旧门控路径。虽然执行者报告现有state中+2000阶段是BLOCKED，但这取决于此前loop的失败而不是牢固的G5门控不变量：未来某次loop成功或队列迁移可能错误触发已禁止的额外优化步！

P0要求：先读取活体supervisor状态与GPU进程归属，不停止别人的进程，也不能双启动同一卡的新supervisor；保留旧队列/失败日志。只为项目自有未来stage3建立新v6队列与content-addressed G5_PASS attestation；runner验证已冻结G5 verdict、cohort SHA、预测SHA和父checkpoint SHA，明确将 FAIL/CANNOT_EVALUATE 传递为非零状态并阻断训练。诊断性闭环允许独立于G5 FAIL进行 DEV，但 DEV诊断成功不代表有权启动+2000正式训练。补RED/GREEN：真实FAIL JSON + rc3，shell包装后仍无法将train stage3置READY；loop成功不能绕过G5；队列重启无重复step。

## R16真实范围

evidence/r15_loop_attempt1..3是有价值的集成失败源：attempt3 stage JSON仍NO_RESULT，server backend ready只是7B加载成功，不是动作执行完整、延迟合格或Push-T成功。至执行sha9305563新增 set_num_images_in_input(3) 及CPU回归；执行者声称r4在服务器继续跑，**其结果不在所核对GitHub固定提交中**。继续独立进行有界DEV闭环/真实延迟、图像tensor与14动作/安全验证，禁止把0成功/成功率未测混淆。确认r4实际状态再决定最小修复，不擅停外部服务。

## M2范围

现有3任务push_T、stack_bowls、sweep_blocks各100 HDF5（本地数据审计报告）；2任务160 TRAIN/61171有效帧行及M2 task-conditioned trainer/P6 CPU门控已IMPLEMENTED，但尚无真实M2 GPU训练与官方成功率。其他9任务数据未知或没有合规专家，其中布料与液体两项BLOCKED_DATA。三任务共享14D、3RGB、25Hz及A[t]=S[t+1]语义有源码与本地审计证据。每任务单独normalizer是必要的；实际部署适配仍需补齐，不能以训练侧CPU测试替代真实推理。旧HDF5不记录layout/seed，因此TRAIN/官方FINAL场景独立性无法证实；报告应明确UNKNOWN，不可写严格新场景泛化。

## 决策1：正式采用官方25集，保留可追溯修订

对上游源 RoboDojo-Benchmark/RoboDojo@b8e5eed9fc11fc3da8e4b142fc60acf22036efa5 的只读核实：task/RoboDojo/config/_task.yml task-specific eval_nums=25；src/eval_client/main.py的EVAL_NUM只可以min(EVAL_NUM,eval_num)降低25而不是提高30。因此研究方**接受每任务25次官方版本的primary FINAL**，12任务最多300次真实仿真回合，既有30次仅作为旧计划，不修改官方任务计分器。要求正式第一回合之前冻结新版本协议/模型checkpoint+SHA/实际seed-layout/命令/官方success方法、是否被先前选模触碰；未曾用于选择的证据不足时不能额外宣称untouched。目录1为建议主要layout，0/2如果额外试验需要预注册为独立组，不能事后挑最好。评测实际有效分母由官方stable过滤口径决定，必须同时报告启动数与计入数。**官方SIM FINAL25 != 新专家HDF5的25回合DEV5/FINAL20**，后者目前仍未合规完成。

## 决策2：许可证精确披露而非擅自解释

同一固定上游代码commit b8e5eed...，LICENSE文件原文是MIT License（2025 Yue Chen版权），README尾部却宣称 RoboDojo Non-Commercial Research License / 非商业研究使用。上游最新main的LICENSE与README已经都标MIT。这个同commit冲突是真的；不可把旧README限制删去或武断说其法律效果。当前非商业内部研究可继续，论文/材料应并列注明固定SHA、LICENSE和README声明；发布前逐项检查Assets、HuggingFace数据、XPolicyLab、第三方模型权重、依赖许可和下载条款。商业应用或捆绑第三方资产公开分发前单独寻求明确授权意见。不能笼统说整套数据/资产/模型均MIT。

## 验收与交付

独立开发分支实现 v6 supervisor G5语义检查、训练step总账和红绿负例；不修改原v5状态或main；追加ROUTE和正式验收文档。独立CPU数据/12TaskSpec/专家脚本继续；重GPU与Isaac串行。不扩张失败的R15模型训练步数；有可信机制假设可另做预注册诊断，不能假装其已获新优化步许可。M2独立已授权训练仍需数据/部署契约Gate。真正闭环结果以官方日志/视频/成功数，不以runtime ready或离线L1代替。