# RoboDojo / OpenVLA：从 R10.1 到完整项目验收（MASTER-1.0）

编制：2026-10-21。范围仅为 jho887891-sudo/robo 的 RoboDojo/OpenVLA 具身智能项目。本文件是未来阶段路线，不代表训练或评测已执行。起点：r10/cpu-gates@4af5d603c6410ec951ac8b1b138312bbdebb9c24、独立审计 @429e28affd67f9643a122bdb5bc0f958f9befa43。main@329614f54d3090b9b6ab120855ba50fbc93bdd1e 不修改。R10 原批准：一次 ≤300 秒真实 7B KV 诊断（含权重加载，≥24GiB 空闲）；新采 25 个 Push-T（5 DEV/20 FINAL）；只允许在确权/结果完整归档/无人依赖条件下优雅结束本项目旧评测与专属 :6020 服务。所有后续训练、追加 GPU、新任务数据采集、合并 main、真实硬件部署都需要研究负责人新的明确批准。

## 1. 什么才叫项目完成

**M1 单任务科研闭环**：新 OpenVLA-7B 42D + 速度修正量 R 真正走通训练、推理、checkpoint 恢复、RoboDojo Push-T 闭环；能产生任务成功，而不是仅有低离线关节 L1。未来在尚未使用的 30 个仿真评测 seed 上一次性验收，建议研究目标 ≥10/30 成功且不违反动作安全契约（这是拟议的内部目标，不是 RoboDojo 官方门槛，首次正式评测前锁定）。提供成功率 Wilson 95% CI、位姿进度、动作幅度、失败录像、同 seed 的 no-op/恒速/旧基线。

**M2 12 任务统一策略**：ONE shared VLA（LoRA / projector / 自回归头）支持已冻结的 12 个任务，真实任务语言与每任务输入/输出契约正确；接口支持 12/12，官方 sim 成功率评测覆盖 12/12。工程功能 PASS 不等于研究能力 PASS。建议能力达标线：12 个任务各自 ≥3/30 成功且任务宏平均 ≥25%，正式 FINAL 之前预注册；如果未达到，明确 FAILED_CAPABILITY，不包装成“完成全部任务”。

**M3 长期研究目标**：在 M2 验证后的统一策略上，检验语义场景/多视角几何、Planner、主动感知或短程世界模型能否提升任务表现。只有 matched-baseline、消融和独立评测支持的收益，才能称为研究贡献。可以得到可靠的负结果。

**正式交付**：SHA 可复算的 12-task checkpoint、元数据/数据哈希/推理和评测脚本、原始日志/录像/任务成功率、消融和独立审计、append-only ROUTE；未获授权不合并 main、发布权重、部署真机或提交官方 leaderboard。

## 2. 精确的 12 任务范围

官方 RoboDojo 有 42 个仿真任务。这里选其 Generalization 维度 12 个标准任务为默认候选：stack_bowls、push_T、pack_objects_into_box、fold_clothes、hang_mugs、sweep_blocks、pour_liquid_into_cup、make_toast、arrange_largest_number、sort_nesting_dolls_by_size、store_laptop_and_headphones、stack_blocks。

来源：RoboDojo-Benchmark/RoboDojo@09bd5f5e762b949f5858ece31347ce62057b3a5e 的 scripts/internal/task_inventory.py；12 个对应的 *_random 场景是单独的 OOD 测试，不偷算进 12 个训练任务。**PROJECT TASK_SCOPE_V1 尚未冻结**：先与服务器实际任务、robot embodiment、assets/camera、原先项目认定的“12 个任务”交叉核对；有差异让研究负责人一次性决定范围，不得为了易做暗换任务。

官方 RoboDojo 当前主要是 eval-only benchmark；不能假设 eval/robodojo.sh 自带专家数据采集与 HDF5 写入；旧 k1_* 脚本尚未完成源代码与安全审计。特别记录官方 2026-09 的 get_obs 帧偏移修复及 RGB 字节通道顺序修复，所有后续数据、训练、推理和 sim 必须记录各自实际版本和像素级合同，不得就地无审计升级运行容器。

## 3. 模型与数据设计，先锁住真实接口

M1 输入：三路 RGB（head/left_wrist/right_wrist，顺序及预处理固定）；真实观测的 14D 当前关节 S_t、同一环境/同一回合的 S_(t-1)/S_(t-2)；42D P=[S,V,Acc]，V=S_t-S_(t-1)，Acc=S_t-2S_(t-1)+S_(t-2)，初始边界按冻结规则置零且无未来泄漏；英文任务指令实际进入 VLA，不使用固定 Push-T prompt 伪装成多任务。

归一化：仅 64 TRAIN 拟合 S 位置边界、V 和 Acc signed-motion q99、R 目标 q01/q99；DEV 只允许选模型/夹爪策略；旧 20 已被 R2-R9 反复使用，不叫 FINAL。预测目标 R=(A-S)-V，经 TRAIN-only 256-bin 固定 codec 映射 31744..31999，监督 14 action token + STOP，decoder 还原 A_pred=S+V+R_pred；夹爪 6/13 用 DEV 锁定策略，物理关节范围和 rate clamp、安全边界同训练/部署。历史旧模型和新模型 metadata、projection 维度、时间轴、动作方向不能隐式兼容；发生不一致 fail-closed。真正训练时检查 label CE causal shift、之前动作 embedding 保留、LoRA 和 42D projector 梯度、冻结 normalizer/checkpoint source sha。

**现存部署 model.py 是旧 Push-T 专用（TRAINED_TASKS 只有 push_T，proprio_history 支持 14/28D），R9 的 42D 训练入口不是 42D 部署路径。** 下一阶段必须增加隔离的新部署适配器与 checkpoint 往返，不可只凭 CPU mock loss 训练。

**控制时钟硬 Gate**：历史纯 VLA 推理约 0.93s/动作，数据 25Hz（40ms/帧）。这些不代表同一时钟。实测 7B 单次 14token p50/p95/p99、服务 transport、policy command 生效到下一 observation 的物理步数、sim 是否停顿/保持旧动作。基于实测再选择稳定的 action hold/velocity safety 或另行授权 chunk/faster student。不得假装 0.93s 推理是 25Hz 闭环。

M2 每任务建立 TaskSpec：task_id、真实 prompt、robot/action order/units、三相机 schema、观测周期与 proprio history、TRAIN-only 状态/运动和残差 quantile、gripper、物理边界、数据来源与 hash、官方评测规则。如果某任务不是共同 14D action，则必须显式建立 embodiment adapter 并测试，不能胡乱强压到 14D。一个共享 LoRA + projector 的初始多任务设计，任务均衡采样、运动状态分层、episode/scene 分组防泄漏。先 2 →4 →12 个任务，每次防遗忘评测。

## 4. 阶段顺序和量化 Gate

| 阶段 | 要解决的问题和交付 | 准入/退出条件 | 资源授权 |
|---|---|---|---|
| R10.1 当前 | 核实并归档 10 路旧评测；安全优雅结束专属进程；PR#14 FINAL 冻结回归；只读 k1 源码审计；并行 12-task CPU 任务清单 | 结果 hash、进程归属与无人依赖、GPU 前后快照；freeze 真实回归；固定采集器候选 SHA | 安全停机已有条件授权；不能 stop 容器/其他人进程 |
| R11 | 唯一 7B 同前缀 KV/full 诊断，14×256 logits，正确 cache/prefix/position/mask；独立复算 | DIAG_EXECUTED 与 KV_EQUIVALENT 分开；不一致仍 OPEN，不能宣称 bf16 根因 | 仅使用已有一次 300s GPU 额度 |
| R12 | 新采 25 个 Push-T，5 NEW_DEV/20 NEW_FINAL，真实 HDF5/schema/三相机/25Hz/14D 验证；跨旧100内容+state/action哈希、随机初态、正式 freeze | 25 个真实回合、冻结 JSON 含 5/20 名单和协议/源文件 SHA；模型触碰 FINAL 前 GitHub commit | 25 个新采已批准；未知 k1 执行仍需确证安全/具体脚本批准 |
| R13 | 真实 7B 42D 训练/推理链路 0-step GPU smoke；新部署适配器、自动断言 14D（旧 prismatic 可能解析 7D）和 metadata/save-load | 真 7B forward/backward gradient、14action tokens、frozen codec、42D/14D 形状、相同前缀与部署图像转换、0 optimizer | 新增 GPU 授权（建议上限一次 20min） |
| R14 | 32 TRAIN 帧、覆盖多个训练回合的小样本 overfit，固定 optimizer 与数据；测试梯度和条件能力 | ≤100 次更新内 TRAIN token loss/acc 确有学习，生成物理动作不塌缩；无模型加载/保存不一致 | 单独申请 ≤100 steps / ≤45min |
| R15 | Push-T 真训练，64 TRAIN /16 DEV，300→1000→3000 step 按块继续；同帧无动作/恒速/MLP/14D 对照，必要的历史与目标消融；每块单独存档 | DEV 同口径离线 L1 比恒速好；R²/方向/动作幅度不虚高；负例、保存、梯度、取样统计 PASS；失败先查机制而不追加步数 | 300、1000、3000 各阶段单独预算；未批准 |
| R16 | Push-T 部署和时间控制；DEV 1→5→10 回合，锁定模型后正式 30 个新 sim seeds 一次性 FINAL；官方对象进度与成功判定 | M1 独立闭环成功数、30-seed CI、动作限幅/超时/无进度，证据录像、同 seeds no-op/CV/14D；未达标明确 FAIL | 评测与后续修复 GPU 单独批准 |
| R17 | 12-task installed version/TaskSpec/embodiment+数据/资产许可盘点（与 R10.1/R11 可先做 CPU） | 12/12 冻结任务清单和逐项 action/camera/instruction 合同，缺项明确 BLOCKED | CPU 可安全并行 |
| R18 | 逐任务专家数据准备：2个任务试点，再4、再12；每任务 TRAIN/DEV/从未接触 FINAL、hash + seed + scene 隔离；不得训练用 R12 20 FINAL | task-level 数据覆盖/质量可重算，成功轨迹与失败轨迹统计，按真实任务帧数估算预算 | Push-T 25 不等于 11 任务采集授权；新数据须批准 |
| R19 | 单模型多任务接口：TaskSpec 注册、prompt 路由、12个动作契约、每任务训练分位数、通用 42D projector/LoRA、episode 历史缓冲 | 2→4 任务样例 exact train/deploy codec，跨任务 prompt 真的变化，zero future leakage，不相容任务 fail-closed | 新重型 GPU smoke 需要授权 |
| R20 | 2→4→12 共享 VLA 训练，任务均衡、留意遗忘；共享模型 vs 独立/简单基线消融。7B 显存不够时提 3B/student 备选给负责人裁决 | 12任务 DEV 宏/逐项 metrics、同预算 seeds、遗忘矩阵、模型固定 SHA，不能因为 macro 好掩盖 zero-success | 重大 GPU 新授权；不默认换 3B |
| R21 | 官方 12 任务独立仿真评测；先 DEV，再锁定各任务 30 个从未用于模型选择的 sim seeds，合计拟议 360 rollouts；可另评 *_random OOD | 12/12 真实任务 success counts/CI/视频/失败类别，M2 工程 PASS 与能力 PASS 分开；版本/seed/code frozen | 大规模仿真新批准，不能边测 FINAL 边选模型 |
| R22 | 研究升级：场景几何/semantic world state、Planner、active perception、短程世界模型，只在失败原因有证据时一次引入一个组件，等数据/算力匹配消融 | 与 M2 baseline 同数据/seed/更新数，增益或无增益有独立证据，不能宣称未证实创新 | 新架构研究和额外 GPU/数据需批准 |
| R23 | 完整技术归档：冻结 base+LoRA+projector、TaskSpec、代码/数据/归一化/评测哈希、启动命令、12-task 统计和 CI、演示视频、性能安全、论文/报告、独立交叉审计 | 可复现、所有 FAIL/OPEN 原样公开给负责人、ROUTE 只追加，合并与部署需单独审批 | 无权默认修改 main/发布/真机部署 |

顺序依赖：R11（KV 真诊断）+ R12（冻结新集）+ R13（7B Smoke）→ R14 → R15 → R16(M1)。R17 可与 R10.1/R11 CPU 并行；R17+R18+R19 → R20 → R21(M2)；R22(M3) 可以在已有 M2 baseline 后按证据启动；最后 R23。重型 GPU 和 Isaac Sim 采集/评测排队独占；CPU 不受相应失败牵连。

## 5. 研究决策树：避免陷入无限训练

- R11 KV 14/14：缓存候选可进入 R13 实测；KV 仍失配：保留每步 full logits，CPU 查 prefix/cache/position/attention；真实 full recompute 可作为诊断替代路径，但必须核实推理时间/语义，在未获准第二次 GPU 前不偷偷重跑。根因若 OPEN，不能标 KV_EQUIVALENT。
- R12 collector 未找到：只读审计并做最小接口设计，收口 BLOCKED_COLLECTOR；不可把旧 100 改名、其他任务或 9KB 占位文件充作新 FINAL。
- R13 42D 前向、梯度、ckpt 载入、14D 动作任一失败：只修最小原因并加负例，不启动 optimizer；最多 2 轮修复失败则相应子线 BLOCKED。
- R14 32TRAIN overfit 不能学习：STOP 大训练，查 actions labels、mask、normalizer、LoRA 与梯度；不靠 3000/6000 步掩盖。
- R15 DEV 倒退或模型接近不动：STOP 追加训练，按同批帧比较恒等、恒速、控制周期和每关节动作方向；先诊断再申请算力。
- R16 出现动作控制超时：检查 25Hz 采样与 0.93s 推理之间差异，明确 hold / safety 等价后才继续。若 M1 达不到指标，留下可重复负结论，不把离线 L1 当成功率。
- R18 少某任务数据：只暂停该任务训练，继续其他 TaskSpec；但 M2 完成不可跳过未覆盖任务。
- R20 7B 48GB 不可训：冻结硬件/吞吐实测，再由负责人决定更小模型、蒸馏/低秩/流水策略；不能自动换架构。避免以大模型数据泄漏获取“泛化”。
- R21 FINAL 被查看做超参：此 FINAL 立刻降级为 DEV，需另建真正新 FINAL，不许重命名“独立”。
- R22 planner/主动感知若无 matched gain：报告无收益，保留共享 VLA 基线。

## 6. 成本、时间和停止预算（全部只是申请上限，不代表批准）

| 类别 | 计划单阶段硬上限（今后独立申请） | 说明 |
|---|---|---|
| R11 已批准 | 一次真实 7B GPU ≤300s 含 load，空闲 ≥24GiB | 0次已执行，不因本计划自动启动 |
| R13 7B 0步 smoke | 一次 ≤20min | 无 optimizer |
| R14 过拟合 | ≤100优化步且 ≤45min | 两个条件同时限制 |
| R15 分段 Push-T | 先 ≤300步/2h；若进展再 ≤1000累计步/5h、≤3000累计步/12h | 每块新增授权，非保证执行时长 |
| R16 DEV | 1/5/10 回合递增，按 1 回合实测确定总 GPU/sim 时间 | 若不动或无进度停止 |
| R18 新多任务数据 | 先 2任务可行性，再任务清单/bytes/episode 测量后预算 | 25集 Push-T 旧估约1.94GB，不可套到其他任务 |
| R20 多任务 | 2任务试点 ≤4 GPU小时；4任务 ≤8 GPU小时；12任务先根据吞吐新申请 | 不自动并行重型训练 |
| R21 正式多任务评测 | 拟议 12×30=360 sim rollouts | 以实际单回合时长测量、由负责人单独批准 |
| R22 研究创新 | matched ablation 另行立项、给定 seeds/步数/预算 | 不默认执行 |

GPU小时 = steps × 实测 seconds_per_step / 3600 + load/save/评测开销；评测时长 = rollouts × 实测秒/回合 / 3600。不要凭 48GB 显卡名推断剩余显存，也不能把旧推理 0.93s 简单乘以 25Hz 数据帧率得出真实闭环时间。时间安排为依赖型：M1 预计数周到数月，M2/数据/评测可能再需要数月；没有任务数据和共享 GPU 空闲证据之前不能承诺完成日期。

## 7. 统一实验规约

相同数据、相同抽帧、相同归一化、相同计算口径才对比。训练/DEV/最终冻结严格以 episode、content hash、state_action hash、环境 seed 和场景分离。R2–R9 的旧 20 留出回合已经被多次选型，不能说是 untouched FINAL。R12 5 DEV/20 FINAL 的数据是离线关节预测评测，不能等同 20 个仿真闭环测试回合。在线成功率与离线 joint L1 必须分别汇报。

离线：physical joint L1（帧平均与回合宏平均）、mean action magnitude vs expert、R²、sign/cosine、per-joint、运动强度分层；no-op、constant velocity、TRAIN-only affine、MLP 和同条件 VLA 14D baseline。训练 loss/token acc 是诊断指标，不能等同动作/任务成功。

闭环：原生 success count/total + Wilson 95% CI、官方 score、目标物体距目标位置/旋转误差随时间变化、关节安全约束、运动停滞、延迟 p50/p95/p99、回合长度、重启/失败日志；任务逐项和宏平均同时报告。FINAL 前固定 checkpoint/protocol/sim code/seed，FINAL 后不选新超参。

正式阶段至少交付 GitHub 固定分支和完整 commit SHA、命令、原始 stdout/stderr/退出码、环境和资源、freeze/dataset/模型 hashes、完整 logits/预测存档、失败日志、独立复算及 ROUTE append-only。每 Gate 用 IMPLEMENTED、VERIFIED_CPU、VERIFIED_GPU、CLOSED_LOOP_VERIFIED、BLOCKED_x、FAILED_x 之一，不以文字宣称成功。

## 8. 研究负责人与执行 AI 的决策边界

负责人只需在关键研究选择上介入：12个任务最终集合、每项新 GPU 与数据预算、适配其他 embodiment / 3B / 大架构、合并和正式部署。ChatGPT 做机制审计/验收、指定最小区分性实验；DeepSeek 并行执行无依赖 CPU 工作、GitHub 原始证据、失败保留与已获授权任务。当前只批准 R10.1 有条件安全停机、一次 R11 5分钟 GPU KV、R12 25集 Push-T 新采原则；未知采集脚本执行及后续训练都不包含在授权内。

## 9. 当前直接执行的工作包

1. R10.1 先只读确认 10路旧评测结果已持久化、进程属本项目且无人依赖，才可优雅停本项目旧评测客户端与专属 :6020 服务；不得停其他人的 Isaac Sim、共享容器、使用 kill -9。
2. 并行远程 CPU 复验 Draft PR#14 正式 freeze 写入和 DEV/FINAL 名单哈希；修复任何问题不重写历史 R10。
3. 并行只读审计 k1_final_take_action.py / k1_ori2_full_push.py 实际源码/入口/写 HDF5 逻辑，固定 SHA；未知脚本不得执行。
4. 并行 CPU 对照官方 TaskScope 12 标准任务与服务器安装版本，生成 12 个 TaskSpec 骨架及缺失信息清单，禁止新数据下载。
5. 空闲显存实测 ≥24GiB 且 CPU 前置条件通过，按既有授权仅一次 timeout 300s KV 诊断，保留每步14×256完整 logits/原始命令/PIPESTATUS，失败不重试。
6. R11/R12 实际结果经独立审计后，把 R13 ≤20min 0-update GPU Smoke 预算作为下一个研究负责人审批点，不提前开训。

参考：
- https://github.com/RoboDojo-Benchmark/RoboDojo/blob/09bd5f5e762b949f5858ece31347ce62057b3a5e/scripts/internal/task_inventory.py
- https://github.com/RoboDojo-Benchmark/RoboDojo/blob/09bd5f5e762b949f5858ece31347ce62057b3a5e/README.md
- https://github.com/jho887891-sudo/robo/blob/429e28affd67f9643a122bdb5bc0f958f9befa43/missions/R10_1_AUTHORIZED_RESOURCE_RELEASE.md
