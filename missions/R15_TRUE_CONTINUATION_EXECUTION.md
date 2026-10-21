# 给 DeepSeek：R15 连续训练修复与自动继续完整指令

本指令在研究负责人已有的一揽子 R11–R16 授权内执行，无需重新申请R15训练预算。执行者先看 report/R11_R15_INDEPENDENT_RESUME_LEDGER_AUDIT.md，不得绕过新发现的P0。

## 基线与状态

- 执行源 feature/r11-p0-closeout@69031eda6ee32439aa75d71b227a66a7dcb78aa2（原始 Draft PR#18）；当前审计分支是对它的独立新增。
- 总资源授权 plan/r11-r16-delegated-pusht-20261021@83b579415588e60572768ec0e11f5868520fa695，32GPU-hours总额，R15训练桶14h，R14已用100/100更新，R15正式0/3000，R13已用169.7s，R14已用857.8s，整体账本已用1027.5s；所有新任务计入剩余，不重置。main@329614f54d3090b9b6ab120855ba50fbc93bdd1e只读。
- R13 verified GPU0step和R14过拟合有原始日志，但未有DEV泛化。R11 collector BLOCKED_EXPERT；不要暂停与R15无依赖的CPU工作。
- 用户给出的R15 300-step命令现在**不可以原样直接执行**，需要先修真正累计训练和防覆盖预算漏洞，不然会得到3个互不相干模型与超过3000步。

## Lane T：先完成真续训和预算守卫（独立CPU开发）

1. 用纯CPU小Torch模型或结构等价fixture分别测试：原R15 fresh300与fresh1000不是相同优化路径；ledger被外部覆写时不能检测；默认save_every=0没有里程碑checkpoint；在正式300步前补RED证据。
2. 在新执行分支实现基于完整checkpoint的续训：保存/恢复7B pristine模型身份、随机新LoRA、42D projector、AdamW状态、LR scheduler、Python/NumPy/torch CPU/CUDA RNG、sampler状态、global_step、数据列表hash、normalizer/codec/camera/版本、optimizer格式版本。所有文件先临时写/确保可读/校验hash，再atomic rename到全新目录，避免覆盖旧ckpt。恢复前verify base sha、LoRA/proj、完整状态/数据合同完全一致；错误/缺失fail-closed。若原始训练的随机数在LoRA初始化后seed，改为模型初始化前seed并加回归，这改变新训练起始分布但历史R13/R14证据不改。
3. 实现不可覆盖追加式单写者 GPU/step事件WAL及atomically updated projection ledger，run_id、CPU/GPU start/end、每个optimizer.step已发生/commit及GPU秒数；跨worktree共享必须专用单账本路径，不能被Git工作树push/checkouts覆盖。重建只允许取原始不可变WAL+ckpt并写新修正声明，不能覆盖历史证据。重复run_id/step和账本倒退必须拒绝。崩溃或计费不确定则安全阻断。
4. 在开GPU前做CPU中断恢复一致性：连续训练N步 vs K步→保存恢复→N-K，检查权重/优化器moment/LR/sampler下一样本/RNG/global_step一致；ledger负预算、并发竞争、残缺 checkpoint、sha修改、跨任务resume、resume旧14D、resume前后修改q01/q99、旧20误入、损坏WAL、历史计数清零必须拒绝。
5. 正式训练的 --target-global-step 只可 300、1000、3000；从0→300是首次，300→1000只补700，1000→3000只补2000。不能用三个独立fresh 300/1000/3000；不得把R14过拟合100步算成R15训练，也不得从R14权重续训。step300/1000/3000均有完整模型+optimizer+scheduler+RNG+manifest/hash；GPU利用额外每≤100步崩溃恢复点，save_every=0必须fail。
6. 执行者可以自主选择标准PyTorch checkpoint实现细节，但 checkpoint必须包含真实同种子的所有参数/状态；不能仅存LoRA和projector再初始化AdamW而谎称等价。任一CPU Gate无法验证就BLOCKED_IMPLEMENTATION不启动R15。

## Lane E：DEV评测器与对照（与T CPU并行）

1. 冻结64TRAIN/16DEV来源和相同DEV cohort/frame_pool t≥2，保存清单、帧mask、joint14D动作语义、夹爪处理、预处理、q01/q99、训练源sha；严格 episode macro，不允许旧20或新20FINAL参与模型选择。
2. 实现 CPU基线 identity A=S、constant-velocity A=S+V、TRAIN-only affine和已有MLP（如果原weights可读）；对同一批DEV样本输出 physical joint14D macro L1、per-arm/gripper、R²、sign/magnitude ratio、clip、运动强度分层、动作不动率，配对per-episode，保留 prediction .npz和summary。无权重的基线标NOT_AVAILABLE，不捏造。
3. 在GPU跑真实VLA前先用synthetic和本地只读HDF5验证指标口径，所有模型都走相同mask；离线分数不等于PushT闭环成功率。
4. Gate阈值预注册；300步以后真实预测/模型sha必须进入同一评测脚本；1000步允许继续条件按原Spec：相对CV宏L1改善≥5%或预注册运动分层的配对可信改善，不能因训练loss下降就继续3000。L1与任务进度仍分开。

## Lane D：Push-T expert新采（与T/E CPU独立）

1. 只读熟悉现有IsaacSim仿真状态、t和target_t、合法控制/IK接口；研究可利用privileged state为专家控制，但**训练的VLA输入不许有特权状态**。
2. 新建独立的scripted geometric expert（push direction/contact waypoint/IK/curobo、碰撞/成功判断），先构建最小可执行的合格专家策略而非空随机/全零；静态审计成功场景、种子和质量规则，CPU stub负例。
3. 在原R11 collector4h GPU子预算和40尝试总额内串行验证真实成功行为，第一25合格episode写唯一HDF5、3RGB、25Hz、S/A14、>=100帧、seed/scene provenance、content/state-action hashes，严格新目录、大小上限，绝不覆盖已有文件。如果专家不能成功则BLOCKED_EXPERT，不利用GPU无目标碰运气。
4. 成功后调用PR14强制实际冻结5NEW_DEV/20NEW_FINAL，不在冻结前看模型结果，不用于新训练。数据分支的延迟不阻止T/E和合格的R15 TRAIN-only工作。

## Lane S：R15 GPU真实分段（仅在T+E CPU Gate全绿后）

- 所有真实GPU/Isaac采集作业串行，启动前核对GPU可用、共享进程、磁盘、所有SHA及WAL、资源余量；每job外层timeout不超过R15/总额度。一次失败记费，停止自动重试，保留原始stderr。
- 从pristine base + declared new LoRA seed运行一次0→300 updates；保存全状态step300与预测NPZ；CPU独立复算DEV vs CV/identity和训练零塌缩，对G4做预注册判定。
- 满足G4才从step300完整恢复，只补700达到step1000；相同DEV cohort做G5指标；若L1/运动Gate不满足，停止正式训练诊断而不是硬上3000。
- 满足G5才从step1000完整恢复，再补2000至step3000；按DEV（非FINAL）选最佳step checkpoint，固定checkpoint和证明来源，绝不看20新offline FINAL选模。任一阶段异常只暂停S，其余CPU继续。
- 如果时间或GPU排队让训练无法在当前窗口完成，必须用已授权的可持续作业调度与GPU timeouts、独立WAL在用户提供的执行环境里安全运行；不能在聊天中承诺会在关闭窗口后继续工作。若执行环境不支持持续任务，只交付已完成的checkpoint和BLOCKED_EXECUTION_WINDOW，不能重跑先前steps代替恢复。

## Lane R：R16仿真闭环准备（并行CPU）

- 只完成新42D checkpoint经deploy server/IsaacSim接口的签名、三路RGB图像顺序、per-env历史，不能偷偷将7D/libero、14/28D旧norm导入；
- 准备 p50/p95/p99真实控制时序打点和安全限速/接触进度，模拟器官方success/score不可改；
- R15合格后20累计DEV回合1→5→20，检查物体移动和不动策略；在合格DEV之后固定checkpoint+协议+30新seed且预算足才做一次独立SIM FINAL；不允许用20HDF5新FINAL充作30SIM FINAL。

## 自动继续和交付

总预算32GPU-hour、R15桶14h、≤3000formal optimizer steps、100overfit已用完、collector≤40attempt/4h、DEV≤20、SIM FINAL≤30；都必须按真账本累计，包括失败。任何未来训练调用不允许绕过正式授权书和参数检查。非GPU代码/回归平行完成；遇资源/数据/机制失败只停受影响的子线。

交付独立branch SHA、真实Draft PR、raw logs/checkpoint和WAL SHA、DEV预测/配对baseline、独立代码审计与缺陷回归、更新版mission/R11_R16_MISSION_MANIFEST.json、report/R15_STAGE_READINESS.md、ROUTE.md append-only。禁止main修改、历史覆盖、数据泄漏、发布/真机。项目负责人不用再审批这一已授权预算内的R15微步骤；超预算或变更大架构才必须请示。

**目标是能复算的一条正式模型从0到300/1000/3000的连续优化轨迹，不是绿色测试列表，不是单个32帧过拟合。**
