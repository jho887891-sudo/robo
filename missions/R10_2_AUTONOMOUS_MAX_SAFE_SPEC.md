# R10.2 最大安全自主执行 Spec（可直接交给 DeepSeek）

任务 ID：RD-R10.2-MAXSAFE-20261021-01
状态：AUTHORIZED_SCOPE_DEFINED / EXECUTION_NOT_STARTED
适用仓库：jho887891-sudo/robo。项目只涉及 RoboDojo/OpenVLA 具身智能研究。
目标：研究负责人休息期间，DeepSeek 在现有授权内，无须小步请示，自主并行推进**全部无依赖的 CPU 研发/测试/文档/环境审计**；在安全资源 Gate 成立后自动用掉**已经获批的一次**真实 KV GPU 诊断；审查并准备已批准的 25 个新 Push-T 回合采集。任何无法获得权限或资源的行动，不伪造成功，只停受影响的分支并留证。

## 0. 不可修改的基线与许可边界

固定 SHA（必须在真正运行前重新 git fetch 验证；有漂移先报告，不自动 rebase main）：
- main 329614f54d3090b9b6ab120855ba50fbc93bdd1e（只读）。
- R8 audit 52422372e5eacdcd0ee841ef2b91bdc5a71066c7。
- R9 A frozen f7b8b39e2394772ae6b2d8d929c7a0ba0fb10ed9；B trainer 1cd50f1ae447a73289fd3a2dfd24eb14e973681d；C KV 02735bc2f16d60275ece01526f68860feafd5054；D eval f8e43b0ca6ccaa8a5711e338ba12e0f3be69328f。
- 独立 PR #11 content hash 93faece89a101b4fc31071781d6297eab121783a；PR #12 KV wrapper 026d48e0e5efbdee58eb31185a8bb135b0ffba09。
- R10 CPU evidence r10/cpu-gates@4af5d603c6410ec951ac8b1b138312bbdebb9c24，Gate证据提交 e67a3c72fabae3acc89a525a547f3df1b3a6afa9。
- PR #14 冻结补丁和有条件安全停机任务 review/r10-freeze-gate-resource-audit-20261020@429e28affd67f9643a122bdb5bc0f958f9befa43。
- 总项目计划 PR #15 plan/robodojo-project-completion-r11-r21-20261021@8a95b7cf6fe878a257fe03fa7b7d11950b0ec915。
- 上游 RoboDojo 任务库存参考 @09bd5f5e762b949f5858ece31347ce62057b3a5e；不能无审计升级本地仿真环境。

**本次可执行：** 不超过一次真正的 7B KV/full GPU 同前缀诊断（开始时空闲显存≥24 GiB，包含权重加载的总时长≤300秒，无 optimizer.step，不干扰他人）；新录制 25 个 Push-T expert 回合，5 DEV/20 FINAL（采集入口必须先有真实源码证明、属于此前获准的既有安全 expert/仿真程序且独立 GPU 窗口可用；未证实的 k1_* 脚本不得擅跑）；**已确认本项目所有、结果安全落盘、无人依赖**的旧 Push-T eval client 与专属 :6020 policy server 可优雅结束；所有无需 GPU 的代码审计、source-level patch、合成测试、旧 HDF5 只读验证、TaskSpec/12任务静态设计、文档与 Route 归档。

**本次不可执行：** 任意 7B 或其他模型 optimizer.step；新一次 7B Smoke 或 KV 第二次 GPU；11 个其他任务的数据下载/采集；改 main / merge PR；改写旧日志、旧 HDF5、权重；发布或真机部署；未经审计执行未知采集脚本、kill -9 / pkill / docker stop / 杀共享 Isaac Sim、抢 GPU。用户说“全部搞”指**现有授权内可推进的工作**，不等于取消资源和安全决策权。

预算：本次 CPU 作业以 8 小时墙钟作为上限，最多 4 个并行轻量工作树；长任务遇资源冲突须降低并发；不复制旧 7.3GB HDF5 到 GitHub；不下载外部新数据。KV GPU 只允许前述唯一 300 秒实例，启动后失败不得自动重跑。采集最多 25 次新回合尝试，不用旧回合补数；GPU/Isaac Sim 采集与 KV 排队串行。超预算、磁盘不足或无法确权时标 BLOCKED，其他 CPU 路线继续。

## 1. 并行依赖与自动继续

A｜资源：A0 原评测快照 → A1 项目归属/完整证据/无他人依赖三项 Gate → A2 条件达标才安全关闭专属旧任务 → A3 新显存快照 → C4 唯一获批 KV GPU（还依赖 C0–C3）。
B｜FINAL 安全：B0 PR #11/#12/#14真实 CPU 回归 → B1 修复冻结模式漏洞和名单真正封存 → B2 旧100只读内容 hash 验证与负例 → B3 新集冻结工具准备；新数据不存在时最多 VERIFIED_CPU，不声称 FINAL 已冻结。
C｜KV：C0 固定环境/model/contract SHAs → C1 64 TRAIN 归一化/14D/真实 same-prefix 前置 → C2 14×256 logits sidecar 与 cache preflight 实测 CPU → C3 只做 GPU 指令预检 → A3 PASS 后 C4 真实一次 → C5 CPU 独立重算 logits。
D｜前瞻研发：D1 只读 k1_* /expert采集入口审核 → D2 CPU 虚拟 HDF5 writer/采集器 schema → D3 新 42D 部署 adapter + checkpoint 元数据 CPU 回归 → D4 12 TaskSpec /数据缺口 → D5 控制频率测量/离线和官方闭环评测器 CPU → D6 文档、ROUTE 和 Gate tracker。互相无依赖部分全速并行。未知采集器仅审计，不据文件名执行。

自动继续规则：CPU Gate满足继续下一 CPU 阶段；资源与 C Gate满足可直接运行已批的一次 C4，不用再问；采集必须**已确证采集入口属于既有已获准合规程序**才进入25 attempts，否则 BLOCKED_COLLECTOR；缺新7B权限时 D3 CPU 实现/负例继续，不开GPU。只要一个分支失败，暂停其后续依赖，不停止其他线。

## 2. A：旧 10 路评测安全结束（条件许可已授）

1. 重新采集 UTC、hostname/whoami/UID、git、nvidia-smi、ps/pstree/cgroup、docker ps/top、ss 6020、客户端/服务端 stdout/stderr。旧快照 2026-10-20T15:31:49Z 的 free 23.300GiB 不是实时值；PID 3451345/3451404/3469786/3111650 必须防止 PID 重用，精确检查命令、owner、父进程和容器。
2. 按 run_id 汇总旧评测 10 个环境实际状态：已完成/仍有产出/停滞、last update、score、progress、视频、文件大小及 sha256、是否有未刷盘结果。只读保全、不要覆盖原文件。
3. 必须同时有 OWNER_PROVED + EVIDENCE_SAFE + NO_OTHER_CONSUMERS 三个明确 PASS 才能发送任何退出动作，否则 BLOCKED_OWNER/ACTIVE_EVAL/EVIDENCE，**不可停止进程**。
4. 优先 client 自带优雅结束接口，等待数据落盘；再次核验专属 :6020 无其他依赖后再关闭服务；只对具体确权的旧项目 PID 可定向 SIGINT，必要时 SIGTERM 并记录理由；严禁 SIGKILL、按名称全局终止、docker stop、停共享 Isaac Sim kit。
5. 停止后验证旧结果 hash 未变化、其他用户进程存活、GPU free GiB 前后。即使空闲仍不足24GiB，也不能降低 KV 阈值或继续抢占。
6. 产物：report/R10_2_RESOURCE_RELEASE.md、原始前后进程/评分清单、确权证据与退出码。没有关停则写 BLOCKED，而不是成功。

## 3. B：正式 FINAL 冻结路径 CPU 交叉审计

1. 在独立 review worktree 从 PR #14 SHA 起重新运行 tests/test_r10_freeze_execution_audit.py、tests/test_r10_freeze_guard.py、PR #11 的 tests/test_r9_final_disjoint.py 和 PR #12 退出码回归。原R10 10/10不代表PR14新冻结路径通过。
2. 合成与真实旧100只读混合负例，特别要实际以 CLI 运行并检查 EXIT_CODE 与**目标文件不存在**：旧回合改名、只换instruction的相同state/action、占位9KB、camera缺失、RGB异常、30Hz、<100帧、wrong task、n !=25、DEV/FINAL交叉、missing/full sha不对、protocol hash drift、test-only override、freeze path已存在、无法构建 manifest、候选内重复。所有失败必须fail-closed。
3. 正例真正调用 freeze 写文件，不只做静态 "refusing to freeze" 字符串测试；冻结文件哈希保护 5 NEW_DEV+20 NEW_FINAL 的实际**文件名、content_sha256、state_action_sha256、seed、协议hash、旧100清单hash、git full sha**，不得仅把候选全集 hash 封存。
4. 64 TRAIN /16 DEV/旧20 曾选型，R10 25新数据不在本地时仅标 VERIFED_CPU_TESTED，不能标 new FINAL ready；发现 bug 独立分支最小修复和负例，两轮失败只卡真实冻结分支。原失败日志不得覆盖。

## 4. C：唯一 7B KV 机制诊断的严格 Contract

C0–C3 必须 PASS 后可自动使用旧授权的一次 GPU：
- frozen 64 TRAIN（19,127行）目标 q01/q99，不能用混 DEV 的旧80、临时20分位数；签名、split sha、signed V/Acc q99 均固定；S=14D，V=14D，Acc=14D；检查 zero motion→0。
- 14 个自回归 action token、合法 256-bin IDs [31744,31999]；显式断言，不能受旧 prismatic LIBERO ACTION_DIM=7 污染。
- full-sequence 与 KV 每一步使用**同一个 teacher-forced prefix**；保存 KV/full logits 形状 [14,256]、float32、两路token、first mismatch、prefix_len/cache_len、mask/position信息、top2 margin、max/mean|Δlogit|、npz sha256、模型/代码/contract sha。旧“8/14”没有逐步 logits，不能冒充已经复算。
- CUDA KV 必须禁用 gradient checkpointing，use_cache=True；cache None 或 dtype/shape异常立即FAIL；预先通过CPU tiny模型的同前缀/位置偏移/伪cache负例与sidecar roundtrip测试。
- GPU 当时空闲 >=24 GiB、共享进程无冲突、唯一执行总≤300秒包含权重加载，不初始化optimizer，不重复。shell 用 set -o pipefail；保留 timeout 进程的真正退出码，不以 tee成功算PASS。
- 结果用三个互不替代状态：DIAG_EXECUTED、KV_EQUIVALENT、ROOT_CAUSE_RESOLVED；14/14一致才 KV_EQUIVALENT，本次发生8/14或其他失配仍 OPEN_UNRESOLVED，0.05数值阈值仅启发式而非bf16因果证明。
- 一旦真实启动消耗唯一额度；OOM/timeout/外部抢占/缺日志都留原始stderr并停止C4；CPU复算其已产生输出、其他线继续，不重试GPU。

执行示意：先确保脚本命令与固定 --help/contract 文件吻合，再在新输出目录调用 timeout 300s python diagnostics/r8_smoke.py --kv-diagnose --norm-contract <64TRAIN-frozen.json> --min-free-gb 24 --out <new-run-dir>，用 pipefail 和单独 probe_exit_code.txt 保存真实程序状态，之后独立检查JSON/NPZ/hash。此处不要求用户回来授权第二次。

## 5. D1/D2：找出真正可采集新数据的安全入口

1. 远程只读定位 k1_final_take_action.py、k1_ori2_full_push.py、existing RoboDojo expert/Isaac Sim接口，记录真实路径、文件sha256、依赖图、运行环境(容器/宿主)、是否属于 expert 生成而非 evaluator 或回放旧轨迹。
2. 每个候选审计：是否可创建独立新的随机初始条件/seed；真正 25Hz；三路 cam_head/left_wrist/right_wrist RGB；同一 T 时间轴下 state/action [T,14]（left_arm6+left_ee1+right_arm6+right_ee1）；A_t对应下一步绝对关节目标；instruction正确；>=100真实帧/回合；保存全新HDF5目录无覆盖；不存在旧100回放、不同任务数据冒用或 privileged scene state作为 VLA 输入。
3. 未审计/未获具体许可的 k1_* 不可运行。可在CPU实现 dry-run schema/writer stub 和生产入口的安全外包装，拒绝危险 path、空相机、错维度。真正 expert 程序合规且证据证明属于此前已获准使用的既有采集程序时，待KV GPU串行完成并获得独立 GPU 窗口，最多25次新采尝试；否则提交代码SHA与风险报告 BLOCKED_COLLECTOR，不因睡觉自行更换官方下载。
4. 如果成功采集，逐回合真实HDF5读检、hash与旧100及内部去重、确认不同初态/seed/场景覆盖；B线真实 freeze gate PASS 才写5 NEW_DEV /20 NEW_FINAL（FINAL模型结果本轮不看、不用于训练）。实际有效数不足25不能用旧数据补数，也不能超出25 attempts。

## 6. D3：新42D/R训练与部署接口仅做 CPU 研发

新增新策略 adapter（例如 code/xpolicylab/policy/OpenVLA_R42），不得隐式改旧 OpenVLA_Pure 14/28D历史checkpoint、既有train_pushT_pure2.py或旧部署配置。
- Input：三个原始RGB HWC uint8，每路[B,H,W,3]，相机顺序头/左腕/右腕；使用训练同一processor/resize/颜色顺序，不另做未授权256×256 resize；真实prompt必须被tokenizer读取，不可永远固定Push-T指令。
- Episode-local history：每个env_idx隔离S_t/S_(t−1)/S_(t−2)观测，reset即清空；绝不来自未来行动/下帧或另一个env；t0 V=Acc=0，t1 Acc=0。原始42D concat(S14,V14,Acc14)；V=S_t-S_(t−1)、Acc=S_t−2S_(t−1)+S_(t−2)。历史时间戳必须有序，同个episode。
- Frozen train-only normalization：S位置bound按64 TRAIN；V/Acc各自signed_motion q99（零速度映0），训练、CPU部署使用同文件与hash；量化范围内误差≤一个bin（约视分位数而定），越界剪切率单独统计。
- Label：R_t=(A_t−S_t)−V_t [B,14]，固定token IDs [31744,31999]，14 action token + stop/EOS；以现有tokenizer真实prompt为准计算causal mask/IGNORE与EOS，不可硬编码错误position；输出A_hat=S+V+R_decoded，夹爪6/13按DEV冻结fallback，关节物理bounds和每周期rate/速度安全界。
- Checkpoint meta强制包含 task_id、base/LoRA/projector完整hash、proprio_dim=42、输入layout与t0/t1规则、摄像头布局、prompt来源、frozen contract/split/codec、训练步数/seed、gripper/安全设定、对齐版本。legacy模型、缺meta、错误dims/hash、prismatic 7D默认全部fail-closed；隔离原始checkpoint。
- CPU测试覆盖 multi-env interleave/reset、错timestamp、t0/t1、42维数值手算、零motion、image order/dtype/resize、指令路由、14-bin窗口、decode恒速一致性、gripper/policy bounds、mock checkpoint存读、错误metadata和缺相机；所有假GPU模型行为必须标 MOCK/IMPLEMENTED，不得声称真实7B已跑。
- 加上未来真实训练“预算凭证”硬守卫与禁optimizer默认；R9_TRAINING_AUTHORIZED=1 不是未来R14训练批准。准备R13零优化步真7B Smoke ≤20min的申请、输入输出张量和DoD，但本次不跑额外GPU。

## 7. D4：12任务统一策略的 CPU 提前建设

官方标准Generalization候选12任务（必须本机版本核对再冻结TASK_SCOPE_V1）：
stack_bowls、push_T、pack_objects_into_box、fold_clothes、hang_mugs、sweep_blocks、pour_liquid_into_cup、make_toast、arrange_largest_number、sort_nesting_dolls_by_size、store_laptop_and_headphones、stack_blocks。
- 从本机任务配置/类/资产/robot/控制接口/相机/频率/指令/成功条件静态生成TaskSpec逐项表和machine-readable registry：task ID，instruction_source，robot embodiment，joint order/dim/units/action semantics，camera order/shape/fps，state history和normalize/codec contract，score/timeout，data provider, episode count/sha/LICENSE，风险与UNKNOWN缺口。
- 不能因为任务名一样就假定全部14D、同一个gripper/动作上下界/25Hz/三摄像头；任何不兼容任务必须标明需embodiment adapter，不能强制映射后称兼容。未知的expert数据数量/来源必须写UNKNOWN。
- CPU编写未来shared-LoRA 2→4→12 task-conditioned registry、language routing、per-task train-only normalizer/schema、sampler骨架及负例：错误task、fixed Push-T prompt被使用、未注册机器人、用DEV/FNAL统计时fail。不可采集其他11任务、不可训练。
- 区分12个训练任务和12个 *_random OOD布局，不把42个官方任务等同项目当前12任务。官方仓库可能修复过RGB顺序/观测帧，先比对本机版本，不擅自升级仿真容器。

## 8. D5：控制时间轴、评测协议与科研证据

- 旧部署0.93s/动作与原始demo 25Hz（40ms每帧）属于不同时间口径。只用CPU创建逐步timestamps、相机采集、policy server queue、14-token decode、真正sim step、action hold时长/动作延迟监测框架；p50/p95/p99和各env history更新约束。无真实7B运行则 latency metric=NOT_MEASURED；绝不能声称能实时25Hz。
- 旧100旧20只作为研究分析不叫全新FINAL，R10新5 DEV+20 FINAL是**离线动作数据**；正式闭环需要将来另冻结独立sim seed，不能混同。只写无需额外GPU的统计工具和synthetic regression。
- 离线指标 physical14D L1 帧均与回合宏均、恒等、S+V恒速、TRAIN-only affine、42D MLP proxy、correction R²、sign、cosine、运动强度/逐关节/夹爪误差、剪切率/动作幅度/不动率；只有同一批样本同一口径可比。
- 仿真报告器从原生JSON读取 task official success/score、object-to-goal位移、stall、动作安全、官方最大步数、超时、30个将来sim seeds的成功数和Wilson95%CI；允许用历史只读或synthetic fixtures调试，不运行新7B正式closed loop。
- 每个正式Gate指定模型SHA/seed/protocol/data/运行环境、日志/NPZ/视频hash，严禁FINAL看完再选模。后续R16 M1、R21 M2均需要新的仿真/GPU资源授权。

## 9. 独立验收、提交与停止

每条工作线独立git worktree/branch，先红灯测试再修复（同缺陷最多2轮），原始失败保留，不覆盖历史R2-R10 evidence。最多并行4条轻量CPU，重GPU/Isaac serial。
提交至少：
- report/R10_2_STAGE_READINESS.md：逐线Gate VERIFIED_CPU/IMPLEMENTED/VERIFIED_GPU/BLOCKED、原始复算路径、依赖和下一授权点；
- report/R10_2_RESOURCE.md、R10_2_FREEZE.md、R10_2_KV.md、R10_2_COLLECTOR.md、R10_2_42D_DEPLOY.md、R10_2_12TASK.md、R10_2_CONTROL.md；无数据时写BLOCKED原因和上游证据，不假PASS；
- mission/R10_2_MISSION_MANIFEST.json：base和head SHA、原始退出码/日志、真实资源/用户归属、安全结束动作、KV实际启动次数/是否sidecar、已采回合数/有效数/冻结5+20身份、optimizer_steps=0、无main改动、资源耗用、失败列表；
- evidence/r10_2_<UTC>/ 原始 stdout/stderr/commands/exits/sha、测试结果/失败轮次/机器环境/进程前后快照/内容哈希；
- ROUTE.md append-only 更新**已经真实完成**的成果与失败、commit链接；不得把 SPEC/授权写为执行PASS。
- 独立Draft PR，main、旧R9/R10分支/老HDF5/权重均原样保护。输出未来R13、R14、R15等新增GPU/训练预算一次性待批提案，不自动开始。

无须用户醒来逐步批复：现有授权内从一个PASS自动推进到下一个。确权失败、共享进程依赖、OOM/超300s、资源门槛不符、未知collector、数据污染/泄漏、未批准optimizer、超25attempts、需要第二次GPU/覆盖数据即停相应依赖分支；CPU其它线照常完成。所有自称 VERIFIED 的事项必须有真实固定SHA、原始日志、独立复算；实施代码但缺环境测试仅称 IMPLEMENTED。

## 10. 面向研究负责人的最终简报

完成后一次性汇报固定HEAD/PR及manifest和总结：①旧评测是否已安全归档、关闭了哪些确权PID及GPU free before/after；②唯一KV实际是否执行、14/14结果和证据；③新采集25有效与5/20冻结是否真实完成，不完整的为何BLOCKED；④42D部署与12-task CPU建设完成程度及测试；⑤模型optimizer_steps必须仍为0；⑥哪些新GPU/数据/训练/架构决定需要研究负责人进一步授权。不得要求负责人夜间人工转发小日志。

本Spec提交≠DeepSeek已运行；接收Spec的执行者应当立刻在其远程环境按依赖执行，不能把“计划已写”填成“研究已完成”。
