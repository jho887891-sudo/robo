# R15–R23 自主研发总任务 V2：12任务统一 OpenVLA/RoboDojo + 闭环验收

版本 FULL-SCOPE-2.0，编制日期 2026-10-21；本文件是交给 DeepSeek 的一次性大任务总合同，并非已运行或已得到结果。

**项目唯一仓库**：jho887891-sudo/robo；不混入 CAD、底盘或指纹项目。
**研究负责人的目标**：基于已执行的 OpenVLA-7B+3RGB+42D/14D，最终交付 ONE 共享多任务机器人 VLA，对 RoboDojo 已冻结12个 Generalization 任务完成真实的数据、训练、部署、官方闭环评测、泛化与交叉审计；Push-T 仅是第一关，绝不能项目到此为止。

**授权生效方式**：GitHub 中保存 Spec/Issue 是交付，不会启动远程 GPU。研究负责人把此任务书或对应 Issue 的明确授权指令转发给可执行远程 shell 的 DeepSeek，才是执行许可。现有不受限于聊天窗口的研发许可仅适用于项目自有现有服务器；禁止新采购云算力、额外收费、抢占其他进程。新增 11 任务 expert HDF5 采集和下文 M2 20k optimizer-step 总额属于**扩展授权项目**，只在研究负责人正式转发本完整授权包后才生效，不得凭过去R11旧预算自动推断。

## I. 最新已验证起点，不能假装已经完成

Protected main @329614f54d3090b9b6ab120855ba50fbc93bdd1e。
执行者R15 @ca62c490f3655a3aa1ce30a2ebde046ee93bb475；已有真实7B正式optimizer300步，checkpoint step50..300，32帧(16DEV×2)指标VLA physical14D macro L1=0.002134356982，恒速CV 0.002110443304，VLA差约1.13%；残差R²=-0.023977。R14小样本过拟合100步已完成，但不计入R15。12任务实际 TaskSpec 与数据以当地最新代码为准；旧20 Push-T多次用于模型选择，不能冒充未触碰FINAL。新25专家HDF5没有收集，R16闭环还没有做。
独立审计 review/r15-300-audit-20261021@8adefe1d695217f1eec79339d433b362909dd180，记录四个阻断点：resume日志调用 sampler.next()消耗一步；757.7s实际R15 wall而WAL仅记576.967s步耗时，及额外DEV7B推理未入账；方向/幅度统计用绝对关节位置而非A-S；DEV原始解码有约6.03%关节坐标超界，G4 safety PASS存在口径矛盾。审计Draft PR#21 https://github.com/jho887891-sudo/robo/pull/21，执行者PR#20 https://github.com/jho887891-sudo/robo/pull/20。
不得覆盖旧checkpoint/原始WAL/日志。优先CPU红灯→最小修复→绿色回归→确定下一训练许可。不要求重跑前300步。

## II. 项目级终点：四个层级，不能偷换

M1 单任务真实能力：R15 42D 真实 checkpoint、R16 policy server 与 RoboDojo Push-T闭环。一次冻结新30 seed官方成功率，保存失败录像/位姿进度/真实推理延迟及基线。同样本离线joint L1≠闭环成功。
M2 12任务接口完成：12/12实际任务、camera/language/robot embodiment/action/control/task rule经本机源码审计；ONE可加载checkpoint+task-conditioned shared policy支持每任务输入与动作输出，不能仅以静态TaskSpec假称算法能力。
M3 12任务能力评测：冻结 12×30=360 个从未用于训练/选模的官方仿真seed（若官方评测协议不同则事前锁定等效契约）；12/12输出真实success/failure、对象进度、录像、CI、错误分析。拟内部目标：每任务≥3/30成功且宏成功率≥25%，这是建议研究目标不是官方基准要求；正式首次FINAL前预注册。若未达标明确FAILED_CAPABILITY，并保留可复现模型。
M4 科研升级（后置并行的CPU调研）：证据支持才启用多视角几何/短期记忆/Planner/active perception/semantic world state；同数据同预算匹配消融才论证收益；这些不是12任务能力的替代品。
R23再完成repo可复现模型+冻结协议+数据与模型哈希+checkpoint+官方任务成功率+延迟+资源消耗+科研负结果+独立审核，不未经授权合main或进行真机部署。

## III. 官方任务清单（本机inventory与版本必须核实）

stack_bowls；push_T；pack_objects_into_box；fold_clothes；hang_mugs；sweep_blocks；pour_liquid_into_cup；make_toast；arrange_largest_number；sort_nesting_dolls_by_size；store_laptop_and_headphones；stack_blocks。

逐项真实核对task ID、官方规则、robot model/joint order/关节类型/相机数及RGB颜色、任务指令、语言分辨、控制频率/超时/关节范围、动作/夹爪语义、expert获得方式、资产许可、原始数据和版本 hash。已有本地数据只有 push_T、stack_bowls、sweep_blocks 的历史报告曾显示各100回合，其余不能凭文件名假装存在。遇到非arx_x5/非14D任务，建立显式 embodiment adapters、任务特定归一化或设BLOCKED_INTERFACE，不允许强行把不同机器人压成14D。12任务范围一经freeze，不以简单任务替代高难任务；_random variants单独作为OOD评测而不混入主12。

## IV. 自动化总体原则：一个程序驱动完整项目

实现 persistent mission supervisor（可用systemd user service、tmux+受控watchdog或已有合法任务调度器，选择实际环境可用机制），核心特性：
- 无状态聊天窗口退出不影响**已经由服务器可靠启动的**依赖任务；DeepSeek不可声称自己聊天窗口关闭后仍会思考、监控或干预，真正自动化来自已部署可验证的任务程序。
- 工程DAG：每任务 ID/输入commit SHA/依赖/状态/运行命令/起止UTC/原始exit/负例/输出hash/下一个合法Gate；多次运行幂等且不覆盖；断点恢复不重做已完成steps。
- CPU并行上限4条轻作业；同一张A6000的重型训练、IsaacSim collect、IsaacSim rollout串行独占；先测其他进程归属，绝不停止他人任务/共享IsaacSim kit。
- 只在**真实依赖**时等待；数据专家若阻断，只停依赖它的任务训练，模型/部署/基线/TaskSpec及现有合法数据试点继续。
- 每个重GPU job有可验证的外层timeout、OOM/卡住watchdog、显存和磁盘水位、全状态checkpoint、不可覆盖事件WAL，失败秒数与步数照算；单次job最长6h（正常拆段）并定时存档，但不设置原先错误的32小时项目总硬停止。启动/耗时/耗资必须有真实账本。超过阶段优化step/episode上限必停，而不是无限试错。
- 故障自修上限：同一根因≤2次不同假设驱动的最小修复；失败后出FAILURE/BLOCKED分支、运行独立CPU线，不能无界同参重试。关键数据泄漏/动作越界/评测作弊/非授权费用立即停止涉及风险的分支。
- GitHub分支/真实Draft PR/完整SHA/原始日志只是可审计证据；GitHub Issue和报告不等于服务器已执行，不能称计划DONE。

## V. 一次性扩展权限，取消无必要的GPU总墙钟硬封顶但仍有实验预算

### 正式续训 R15 与Push-T R16
之前获准R15 formal≤3000步（当前300已用，接下来只补700到1000，若G5 PASS再补2000到3000）仍作为**M1第一条既定对照路线**，不得自行增加它的更新步。R14 100步已用尽。完成M1 PushT闭环的已有DEV≤20回合，单次新SIM FINAL≤30 seed规则继续保留。对于原32h/各桶本次使用“项目现有服务器内部执行”而非跨聊天绝对停止限制：下线保护是单任务6h、step和episode限定+watchdog，而不是忘记GPU预算；超过现有权限的实验类别在本授权包生效后按此更新。

### M2新多任务正式训练（新增完整项目权限，仅转发后生效）
ONE 共享OpenVLA-7B LoRA+task-conditioned input/proj/explicit embodiment adapter（必要时可先训练CPU toy模型），**多任务正式optimizer-step总额≤20,000（不含既有M1 ≤3000，上述额度彼此独立）**。按任务规模有界阶段：2任务≤2,000累计步；4任务累计≤6,000步；12任务累计≤18,000步；留出不超过2,000步供**一条**严格配对、有机制假设的消融/修复后训练，同样算进20k而不能绕过总额。阶段必须有旧任务遗忘对照/独立每任务DEV/同样本和同推理路径基线；任一阶段没学习信号则停该分支诊断，而非消耗完整额度。分离的3B/student/全参数7B/VLA骨干大改仍需研究负责人决策，不在自动权限中。

### M2新数据与仿真专家（新增完整项目权限，仅转发后生效）
- 可审计已授权现有服务器的官方或依法获得的各任务sim APIs、资产与许可；允许只为上述12任务开发受限脚本化/IK/CuRobo/官方合法expert生成器。不能修改成功判定，不得将 privileged sim state喂进待学习VLA；expert可以使用 privileged state但必须在HDF5记录来源。
- 数据准备目标：**每任务最多收集100条真实、不同episode且质量合格的专家轨迹**，包括PushT可在现有合法集上增补；每任务尝试≤250次，**跨任务最多3000 attempts**，初步按 TRAIN80/DEV20按episode/scene隔离；已有合法旧数据先审计后决定能否进入TRAIN/DEV（旧PushT20历史泄漏不进FINAL）。如某任务难以获取专家，记录 BLOCKED_DATA，继续该任务的控制/asset/source审计，不用随机/no-op充成专家。
- 数据没有公式可用的任务不能假生成expert。每任务真实三相机或TaskSpec真实可用的视觉流、时序、robot proprio/actions、英文指令、任务身份、25Hz仅在该任务真实仿真合同为25Hz时要求，time index对齐，存储hash。数据总新写入≤250GiB，超限停止数据写入/先压缩与抽样，不许删旧文件。已采数据只能进入全新目录、临时fsync→rename与双内容hash；不可重用新FINAL作训练。
- 官方SIM FINAL最多12任务×30新seed各一次，且只在模型+协议预注册冻结后执行；DEV rollout每任务最多20次、阶段1→5→20累计，按真实sim运行时间外层timeout。FINAL不可用来调模型；如果部分任务无法执行，必须给出不可实现证据和缺失样本数，不捏造360个success/failure。
- 若服务器剩余磁盘/显存/任务许可证不足或需要新外部收费，停受影响支线而非申请自动购买。

## VI. 并行任务矩阵：不可只做容易的5件事

工作线 A（从现在）：R15真实续训P0源码问题/账本漏记与DEV安全口径：恢复后日志不能消费sampler样本；整job与推理都记GPU耗时且旧账本immutable纠偏；absolute joint位置sign/ratio改为真正ΔA-S、残差R运动指标；RAW OOB与SAFE动作分别统计，采样率/physical clip预注册。CPU红灯→补丁绿灯→重复resume等价→step300父checkpoint源/hash→无断点跳步→只补700，G5通过才补2000。R15 checkpoint不是M2模型，不能占用多任务20k预算。
工作线 B（立刻CPU并行）：R16真实 policy server+真实3RGB/42D history、14个自回归动作token+EOS/codec、按checkpoint metadata failclosed、时间延迟 p50/p95/p99、观测/动作生效时钟、gripper/rate/joint安全；同seed DEV1→5→20成功或无进度回放；只有锁定模型/官方sim版本/seeds才执行PushT单次30 FINAL。
工作线 C（立刻CPU并行）：12 TaskSpec清单、Robot/scene/version/source/license/action/camera/control原代码核验，现有3任务 HDF5静态/时序/关节接口审计；建立多任务 dataloader / sampler / TaskToken+真实语言Prompt / 任务独立统计与codec / explicit action embodiment；CPU synthetic/real negative tests及train-deploy tensor parity。
工作线 D（立刻CPU并行）：各任务数据来源/专家引擎调研与最小功能实现；优先PushT 25 new及可用第二任务，按已冻结成功器做实际接触+回撤，而不只是到目标；通过小规模成功 smoke 才执行正式采集；可并行编写其他任务脚本化 experts；高难fold/pour/pack等不能冒充已有通用专家，按机械和模拟物理特性选真实可行动作规划或标blocker。
工作线 E（数据真实合格后）：ONE OpenVLA-7B M2训练底座+2→4→12任务渐进，独立 TaskSpec 及adapter，训练/DEV/FINAL隔离，明确处理task imbalance、small-data overfit、long-tail与少动作现象；每任务记录success proxy且与闭环严格分开。按DEV gate自动前进，不达到2任务有用表现时先查联合语言路由、动作空间、数据/归一化，不硬上12任务20k步。序列化GPU，不和R15、IsaacSim并发。
工作线 F（从现在CPU准备，模型准备后实测）：12任务DEV sim loop、统一official-eval wrapper、自动统计官方成功率、per-task和macro、Wilson95% CI、object target progress、failures/videohash、no-op/CV/legacy/单任务/共享VLA等配对任务/种子；真实control latency与action safety。所有12任务进入官方唯一新FINAL之前，封存checkpoint/seed list/version/codec/prompt并生成GitHub commit。
工作线 G（可并行研究不大量GPU）：多视角几何对齐/历史记忆/Planner/active perception/短期world model候选机制与原始失败分类，先做源码可行性和不耗GPU的合约/对照设计；只有M2能力基线且失败模式支持的一个小方向才在20k剩余≤2k更新内做一项严格matched消融。未经过负责人新架构批准不重写7B骨干/换3B大训。
工作线 H（独立持续交叉审计）：所有PR/source SHA、raw GPU/PID/WAL、真实HDF5内容hash、Train/DEV/new FINAL隔离、模型checkpoint real bytes、预测NPZ独立复算、同口径confidence bounds、静态/负例测试；发现问题独立审计分支最小补丁，保留失败证据，不允许对历史证明进行覆盖。

## VII. Gate图与自动继续条件

P0_SOURCE: official installed source, tasks, shared GPU, original R15 checkpoint+manifest hashes/WAL & TRAIN/DEV policy; failure only blocks dependent line。
P1_M1_RESUME: sampler301 same as uninterrupted; CUDA checkpoint equivalence preflight if budget; FULL 300 parent sha; GPU计费真实load+save+inference；DEV physical metrics/RAW vs SAFE；P0 red/green -> 自动补R15 +700。
P2_M1_G5: original frozen 16DEV×2frames primary metric, CV 0.0021104433, target L1 ≤0.002005 (或原预注册 motion-quartile 3/4 strata gate) + safety + no collapse；若不通过 STOP R15正式额外2000，自动CPU诊断并保留checkpoint，不等同整个项目终止。
P3_M1_ROLLOUT: real 14token deploy/backpressure+latency+object progress+safe reset; DEV1→5→20；无进度/unsafe时stop sim支线诊断；具备真实非零进展且checkpoint/官方规则/seeds冻结，才触发一次M1 SIM_FINAL30。
P4_TASKSPEC_12: 12/12具体interface/官方rules/robot embodiment/provenance，UNKNOWN不可偷偷推断；TaskSpec先可以归档已验证字段。
P5_EXPERT_VALID: 每任务真实行动控制和成功器对齐，先≥3合法独立成功轨迹并质量审查，才放大采集；某任务失败不阻塞其它任务。评测集不接触训练。
P6_M2_TWO: 两任务真实数据与共同/显式adapter契约，训练≤2000，验证任务路由不是固定PushT prompt；每任务相对no-op/CV有可信信号，兼顾遗忘，成功才4任务。
P7_M2_FOUR: 4任务累计≤6000，防灾难遗忘/归一化泄漏，4任务DEV改善或明确缺口，成功才推广12任务。
P8_M2_TWELVE: 12任务数据与接口满足即可ONE shared model累计≤18000，评测不落后简单基线且资源可控；如个别任务BLOCKED，仍保留共享checkpoint和工作中的子集、不得称12能力PASS。
P9_OFFICIAL_FINAL: 新SIM seeds/模型/时间/协议预冻结，12任务每任务至多30次唯一FINAL，后验不回调参；官方per-task success与Wilson CI、macro报告、失败视频、时延、安全事件；旧20样本/场景不能冒充完全新FINAL。
P10_RESEARCH_ABLATION: 仅有机制假设与matched baseline（同sample/sim seeds/steps）才启用可选≤2000 M2步，记录 null result。
P11_DELIVERY: 12-task统一strategy源/模型/metadata/加载与仿真命令、数据和预处理版本SHA、原始日志+NPZ/HDF5/hash、真实DraftPR、独立核对、ROUTE append、未达成任务逐项BLOCKED/FAILED/VERIFIED。

## VIII. 项目负责人绝不承担微步骤转述

执行者要在远程服务器安装并实际验证 one-shot supervisor 及安全队列，确认已运行任务不会因会话断开而停止；所有无需人决策的正常开发、低风险小修、CPU回归、已授权数据收集、已授权阶段GPU与仿真继续；只有代码/授权范围/资金/新训练step总额/换模型骨干/真机/改main/替换官方评测规则/违反安全和隐私的风险才请求研究负责人。

对于无法持久运行的基础设施，真实报告BLOCKED_PERSISTENCE并交付脚本与日志。不能承诺AI助手在聊天结束后自己持续操控服务器。

每阶段自动固定：run_id+actual full SHA、origin parent checkpointsha、可执行命令+exit+time、原始GPU与sim日志、全部实际数据hash+模型权重hash、ledger WAL、TRAIN/DEV/FINAL provenance、DEV NPZ、官方progress/success和录像hash、控制时延、单位/帧率；GitHub分支与真正Draft PR，ROUTE.md仅追加真实进展。不上传大 HDF5/模型权重或秘密到GitHub，使用可复算存储清单与外部保存路径。

最后向负责人只汇总关键研究决策：M1 PushT能力真实值；M2 12/12工程是否可运行以及每任务成功率；M3各任务数据瓶颈和失败机制；M4严格消融是否有效；预算/权益/安全边界；未完成可继续事项和依赖。不以“窗口时间不足”代替可持久化任务调度。

**现在安排A/B/C/D/H立即并行；E/F/G遵循真实依赖自动启动。目标是交付ONE可用的12-task VLA和官方仿真结果，不是完成一份大计划或停在300步。**
