# RD-R11-R16 | Push-T 一揽子授权与自动执行完整 Spec

版本：DELEGATION-1.0 / 2026-10-21
项目唯一范围：jho887891-sudo/robo 的 OpenVLA / RoboDojo 具身智能。
任务性质：DeepSeek 远程多阶段实现、合规数据采集、单卡有界 GPU 训练、模拟器闭环评测、独立证据归档；ChatGPT 负责交叉审计。
**本文件是可直接转交研究负责人授权 DeepSeek 使用的预算模板：收到研究负责人转发整份指令或明确采用此预算后才视作授权生效；GitHub 中单独出现本 Spec 不会启动实验，也不能冒充批准。**

## 0. 总目标与一次性交付目标

研究负责人已经要求：扩大有限权限、停止每个小步骤反复申请，让执行者按明确依赖从现有 R10.2 连续工作，直至 **真实 OpenVLA-7B 的 42D/R Push-T 闭环实验有可复算结果**，而非仅产生测试/PR 计数。

在授权包内自动完成：
A. 审计/修复真实 42D 训练与部署、生成新 Push-T expert HDF5 采集入口，获得最多 25 条合格新示范并冻结 5 DEV/20 FINAL；
B. GPU 0-update 7B 双路径 Smoke、端到端 checkpoint/deploy replay；
C. 已批准的受限训练机制过拟合、分段正式训练、同口径 DEV 诊断；
D. RoboDojo Push-T 受限 DEV 闭环，满足门槛后一次性独立闭环 FINAL；
E. 12 任务数据和接口盘点、已有数据的兼容 TaskSpec 和预算内可选两任务先导；ROUTE 与独立审计。
无依赖 CPU 线全程并行，所有实 GPU/IsaacSim/collector 作业在同一 A6000 队列串行，不能抢占其他进程。

最后必须交付 M1 的实际 VERIFIED / FAILED_CAPABILITY / BLOCKED 结论与真实证据，不能用“阶段开发完成”代替成功率。

## 1. 固定基线，保护，不假合并

主仓库 main@329614f54d3090b9b6ab120855ba50fbc93bdd1e 只读。
R8 审计 @52422372e5eacdcd0ee841ef2b91bdc5a71066c7。
R9 A TRAIN-only 冻结 @f7b8b39e2394772ae6b2d8d929c7a0ba0fb10ed9。
R9 B 42D 训练入口 @1cd50f1ae447a73289fd3a2dfd24eb14e973681d。
R9 KV @02735bc2f16d60275ece01526f68860feafd5054。
R9 FINAL 协议 @f8e43b0ca6ccaa8a5711e338ba12e0f3be69328f。
R10 CPU @4af5d603c6410ec951ac8b1b138312bbdebb9c24。
R10 独立冻结补丁（Draft PR#14）@429e28affd67f9643a122bdb5bc0f958f9befa43。
R10.2 A 停机 @a3396eb052c62a64ddc7813f86f6b44c55e1c9b2。
R10.2 B FINAL 合成冻结审计 @8d86a25993cc6bc7a1f7f44f6c14a477623e6c70。
R10.2 C 唯一真实 7B KV 同前缀 14/14 @3e1d56b83703988201d7886535242db4ed135ea4。
R10.2 D R42 CPU 模块与12任务范围 @e7cd9bb3cd8c7ec727c6ddd863b77a24c2fb7b30。
R10.2 汇总在 D 分支；Full MASTERPLAN @8a95b7cf6fe878a257fe03fa7b7d11950b0ec915。
以上均是独立历史；R10.2 KV 的旧额度**已使用 1 次**，不能以本任务名义重复。

必须建立独立 worktree/分支（feature/r11-collector、feature/r11-r42-integration、experiment/r12-7b-smoke、experiment/r13-train-pilot、experiment/r14-pusht-dev、research/r15-multitask-readiness、review/r16-independent-audit 等按真实依赖组织）。不要合并旧 PR 历史或覆盖 evidence 来省事，可在新的集成分支挑选固定原 commit 补丁并记录 cherry-pick 来源。保存每轮 Git SHA、原始失败、测试结果、权重数据 hash。不得改变 main、旧 HDF5、旧 checkpoints、原始实验证据。任何新目录在运行前检查不是符号链接指向保护数据；不得上传大 HDF5、模型权重、敏感路径凭据到 GitHub。

## 2. 一次性可委托资源总额（研究负责人转发本授权时生效）

### GPU 总额度

仅使用项目现有授权服务器中可用的 **单张 RTX A6000 48GB**。所有真正占用 GPU 的新任务**总计 ≤32 GPU-hours**，是所有加载、运行、训练、GPU模拟器/采集及验证的累计墙钟，不能将不同 worktree 重叠并行以绕过额度；以每作业启动/结束UTC与 GPU PID/监测取证。

分桶硬上限（更早达到总上限或具体上限则停止）：
- 新 expert/Isaac Sim 采集与调试：**4.0 GPU-hours**，仅 25 合格回合、最多 40 次尝试，磁盘新写入≤15 GiB。
- R13 真实7B零步前向/反向、KV验证、checkpoint/deploy Smoke：**0.5 GPU-hours**（总≤30分钟）。
- R14 仅 TRAIN 小样本 overfit：**1.0 GPU-hour**，最多 100 optimizer steps。
- R15 真实 Push-T 训练：**14.0 GPU-hours**，**至多 3000 个正式 optimizer steps 累计**（不是每个检查点重新 3000）。
- Push-T 开发闭环仿真：**4.0 GPU-hours，最多20个DEV回合**，在 gate 1→5→最多20 内逐步解锁。
- Push-T 正式一次性 sim FINAL：**6.0 GPU-hours，最多30个全新冻结seed**，仅在模型和评测 protocol 已冻结时进入。
- 现有 stack_bowls/sweep_blocks 的 **可选双任务先导**：**2.0 GPU-hours**，若M1已形成真实正进展且旧本地数据经合法划分和契约审计；该先导优化步从上述 3000 个正式总步数中扣减，不能额外扩展，且不可提前消耗 M1 必需的 GPU 配额。
- 未分配故障/加载余量：**0.5 GPU-hour**，仅修复同一已批准任务的机械失败，**不是第2次KV探针或任意额外实验许可**。

累计硬上限：32h；每个分桶上限同时生效，剩余额度可保留但不得擅自跨桶转移。未达到某个 Gate 即不可使用其后续桶。显存按真实剩余测算并服从其他用户优先，不停止/抢占其他人GPU进程。每GPU作业设置外层 timeout 和 start/end UTC、退出码、nvidia-smi；超过预算立即停止受影响分支而不是偷偷延长。

### 数据与仿真

- 授权**新建并运行**一个经过静态审计与无GPU stub 回归的 Push-T expert 采集器/安全 wrapper，调用已获准的 RoboDojo/Isaac Sim expert API，不得篡改官方计分器、读取未来信息进入 VLA 策略或冒用旧100。原始 k1_* 仅可作为实现原理参考，**不能直接当采集器**，也不能复用它的1×14全零“action”文件。
- Push-T **最多40个新仿真 episode attempts**，目标得到**最先出现的25个合格、独立 episode**；5 NEW_DEV +20 UNTOUCHED_OFFLINE_FINAL，采集接受标准、seed序列和失败记账在任何 episode 启动前冻结。采集器在重复/失败/失步时可最小修复最多2轮，不改变数据准入阈值；可用数据不足25则 BLOCKED_DATA，本任务继续训练链路，但不得正式 offline FINAL。
- 上述25个episode只用于独立DEV/FINAL，不进入64TRAIN监督、目标量化分位数或模型梯度。旧100中R2-R9使用过20个“留出”回合不是新FINAL。旧64 TRAIN与16 DEV的划分、codec、q01/q99冻结不变。
- 允许至多20个 Push-T **DEV模拟器闭环回合**，在每次失败后留对象移动/路径/控制时钟日志。
- **拟议30个全新 Push-T模拟器 FINAL seed**，必须在查看其模型结果前预注册和锁定；一次性锁定一个模型 SHA/版本/调参策略/评测命令/时间预算，至多30回合，不逐次用FINAL反馈改模型。不同于25个HDF5的离线FINAL。内部科研目标可在此包预注册为至少10/30成功，官方成败字段是最终裁判；如模型未达 DEV 进步门槛，跳过FINAL保留它的独立性。
- 已有 stack_bowls / sweep_blocks 100个本地episode各自用于CPU来源审计；若进行两任务GPU先导，仅使用经过认证的已有数据且训练/DEV episode完整隔离；**不批准下载/新采其余11任务**、不批准12任务大训。
- 储存新 HDF5≤15 GiB，权重checkpoint存储预留由现有磁盘实测可用空间判定（建议至少30 GiB保留），磁盘/容量不足不能删除/覆盖旧成果。不得自动采购云服务或增加新收费资源。

### 训练/安全

允许经 Gate 之后真实 OpenVLA-7B LoRA/42D projector 的学习优化步，总正式优化步最多3000，另独立 ≤100TRAIN-only overfit；必须显式记录 steps/sec、peak GB、gradnorm、每次输出 SHA 与 early-stop 条件。不得擅自全参7B训练或替换为3B、改变目标定义/Codec、启动真机/外部网络部署。

允许在独立实验/审计分支做普通最小修复与测试、改专属的 RoboDojo wrapper/collector/adapter，**不允许改官方 success/reward、动作语义、测试集、旧原始数据或评测阈值**。任何需要更换仿真版本或改机器人资产都必须停下相应线报告。

## 3. 关键依赖与并行图

CPU A：真实 collector writer/static audit → 合成正负例 → 真实最多40 attempts → 25 新回合 HDF5 → B 线冻结。
CPU B：PR#14/#11真实 CLI 回归与旧100哈希对比 → 真实新5/20冻结。
CPU C：R42 42D/label/metadata 与7B training/deploy代码接口交叉审计 → GPU Smoke → 100步过拟合 → 300→1000→3000(累计)有界训练 → DEV评测 → 只有正式冻结后SIM FINAL。
CPU D：官方12任务/现有3任务数据/TaskSpec/语言路由/控制时钟/实验复现/报告自动化。CPU先行不等待GPU。可选2任务试点必须M1表现和剩余额度满足才运行；无数据的新9任务标数据缺口，不许挤兑主线。
GPU队列只允许一条重任务：collector与7B smoke/training/sim互斥。机器若正在运行别人作业，则排队或BLOCKED_RESOURCE；不能为了速度打断共享服务。最多4条CPU轻量工作树独立并行，GPU任务严禁与heavy IsaacSim采集抢占。

## 4. 阶段P0：训练前真正的源码级 P0 审计与回归

R10.2 已在一个 pristine 7B GPU样本上以同前缀比较KV/full **14/14**；这是有价值实测，但不是训练/部署闭环的认证。本次不重做该历史KV诊断。须用新的R42真实 checkpoint在R13真实forward路径单独验证训练/部署等价（计入R13的30分钟配额）。

必核的已知源代码风险，不允许凭22/22 CPU测试跳过：
1. R42 r42_adapter.py validate_images 允许 H×W×4 的 uint8，而本项目训练定义的是三路RGB H×W×3；除非实现逐源证明和像素一致的显式 RGBA→RGB，否则直接拒4通道并加负例。
2. EpisodeHistory.observe() 拒绝 timestamp < last，但 timestamp==last 也可能导致重复观测/零速度误差；除非sim协议明确允许等时，否则应拒绝非递增或有明确丢帧规则，多env、reset 和重启完整负例。
3. decode_command() 先joint clamp再按 prev_cmd 和dt rate limit；如果 prev_cmd 不合法，限速输出可能重新超出关节界。必须物理安全双重检查：入参 14D finite、rate>=0、dt>0、prev_cmd合法，输出最终同时满足 physical limits 与 rate rule，否则拒绝动作或触发定义的安全hold；测试反例。
4. R42 build_labels()手写 token/bin/STOP/causal mask 与 R9 r9_trainer_contract、旧 ActionCodec、HF shift 逐位对照。必须使用统一 codec源头或有对全256bins/边界/随机14D的逐位完全相等证明；STOP=2和PAD32000不能混用。模型 input_ids/labels长度、prompt末尾、14 action tokens、EOS和后续attention/position确定相同。
5. R9 train_pushT_svacc.py 的TP.build()/pixel_values/LoRA初始化、原生prismatic ACTION_DIM=7/NUM_ACTIONS_CHUNK=8/PROPRIO_DIM=8路径，禁止任何7D/8D默认值污染 14D动作编码/解码。R10.2 KV探针也记录了prismatic默认7D但探针不引用它；新训练/部署不能因此被视为安全。
6. 训练目标 R=(A-S)-V、当前S/过去S/Acc只有历史真实观测，frame>=2，A[t]为下一步绝对关节目标；S/V/Acc [B,42]、R [B,14]、argmax logits [B,14,256]，必须明确样本mask、时间轴、gripper dim6/13。禁止用未来state构建输入或把不同env的history串起来。
7. Frozen normalizer真实 64 TRAIN fit、16 DEV隔离、量化输出与tensor端同一 SHA，不可把旧80归一化或旧14D projector读错。随机初始化的新 LoRA不继承旧有 PAD bug权重。严格元数据19字段和base+LoRA+projector/hash+训练步数/seed+camera/instruction/encoder+normalized bounds；checkpoint roundtrip前后动作输出一致。
8. R10.2 D 是CPU适配器，不是完整7B部署server。必须新建实际模型加载、正确processor/preprocess、LoRA/42D projector/metadata、batch reset/env_idx、法定14 token自回归+可选择的正确KV缓存、动作物理安全、回放无更新，然后才能声称42D end-to-end。
9. 相机原始480×640 3 RGB order head/left/right，训练/部署不许偷偷256 resize/BGR转换；先像素级tensor对照，未匹配拒绝GPU Smoke。
10. 若新数据 collector 仅生成1×14零动作/位姿/成功标志，不含三路真实RGB/state/action时序，直接拒绝；不得修改旧K1脚本原件。

所有 P0 先CPU红灯负例→独立分支最小修复→绿色全部回归。同缺陷最多2轮修复尝试，仍失败时阻断依赖训练线，其他数据采集或TaskSpec照常推进。为P0补充静态/真实HDF5 regression，不能以mock forward认作真实7B通过。

## 5. R11：真正创建新 Push-T 专家采集器并执行25集

**已明确在本授权里允许编写和运行经过审计的新采集器，而不再因“旧 k1 不适用”永远等待具体脚本名。** 允许从已批准的Isaac Sim/RoboDojo expert规划接口或现有专家示范数据生成器建新 HDF5 writer；必须先只读定位实际专家策略，若不存在可执行可靠专家轨迹源，不准把任意随机/全零轨迹包装为 expert，标 BLOCKED_EXPERT 并继续其他独立分支。

采集器应以 NEW_ROOT + run_id + unique episode_id原子方式输出，不覆盖现有任何文件，保存真实：
- episode时序 S[T,14]、A[T,14]，左右臂6/夹爪1及机器人单位顺序固定；A[t]为下一步绝对关节目标，不是全零假标签；
- cam_head / cam_left_wrist / cam_right_wrist 三路原始RGB uint8，每帧同观察t的图像和真实state一致，记录帧/仿真时间戳，无缺流；频率25Hz；
- 真实英文instruction、task=push_T、robot=arx_x5、随机初态/seed/expert config版本hash、结果/时长/失败原因；每回合≥100有效帧；
- 按冻结protocol预注册质量/成功与拒绝规则，max40 attempts、first25 valid，由真实专家尝试产生，失败留原始原因而不是后验挑模型好看的样本。不得使用R12的FINAL做专家源回放/训练；
- 两层写入保护：新输出目录不得存在；每回合先.tmp并fsync/关闭验证后rename为唯一.hdf5；硬文件size/byte-limit；不会清理旧文件或“补写”已有episode。
单集CPU读检不依赖GPU，全体写毕复核时序、14D finite、25Hz、三真实相机、长度、instruction、state action physically plausible与A[t]=S[t+1]语义近似、独立seed/scene、近重复；与旧100 content_sha256和state_action_sha256、file name全拒重合，内部亦然。

PR#14 freeze guard 要在真实25集上验证，写入5 DEV/20 FINAL 文件名+hash、协议sha、used100哈希、full Git SHA；**冻结 commit 推送后才能让模型接触新FINAL结果**；新FINAL不作为训练/模型选择数据。数据真实未收满25时标BLOCKED_DATA，不虚构冻结。新5 DEV可用来复核模型泛化，20 FINAL只在锁定模型后按预注册协议做一次。

## 6. R13：真实7B 42D ZERO-UPDATE Smoke（≤30分钟GPU）

P0 CPU Gate与真实计算环境准备通过后在专属GPU队列自动执行，一次真实加载 pristine base 7B、正确 3RGB、LoRA随机新初始化与可训参数清单、42D projector；执行少量TRAIN-only batch的teacher-forced loss/grad、一个合法14token推理、KV/full同前缀、checkpoint meta保存/重新加载（如果时间/显存足够），全程 optimizer_steps=0。R10旧实测14/14仅作为probe基准，不取代这里。

严守张量：
- input S,Sp,Spp [B,14]→42D [B,42] (B>=1)，三图等价processor输入；
- Rgt [B,14]→ bin indices 0..255→14 token ids 31744..31999；
- input_ids [B,L+15] 及 attention_mask 同长，labels前L个 -100且对14 action+STOP=2监督；实际HF teacher forcing/casual loss shift依据真实源码确认并存每位置loss，不能凭数组长度猜；
- projector 42→llm hidden（VLA模型的真实llm_dim，不硬写）；LoRA参数有梯度，base冻，标签无PAD；无未来state泄漏；
- decode action [B,14]有界、夹爪策略/关节速率/瞬时安全，与CPU逐位或量化容差对齐；
- save/load后相同输入token logits在允许bf16偏差内可复核，meta/normalizer/action order一致。

smoke的目标是达到 VERIFIED_GPU_ZERO_UPDATE，不是宣称真正训练或闭环。若失败，优先2轮P0最小修复、重新分配本桶剩余时间；不得超过GPU总额/桶。若无法完成，BLOCKED_GPU，R14/R15停止，其余CPU或采集继续。

## 7. R14：受限训练机制验证（≤100 steps、≤1GPU小时）

R13 PASS后自动在64 TRAIN中的32个轨迹多样采样帧上启动<=100 updates，仅使用TRAIN，不用DEV或新FINAL。固定随机种子、学习率、LoRA/projector设置、模型源sha、batch采样/step、codec、gradnorm和期望物理动作，0/20/40/60/80/100存完整checkpoint meta/hash和固定TRAIN预测存档。目标至少验证：有非零动作条件梯度、TRAIN CE在有噪声情况下呈下降趋势、能过拟合少量标签、生成非全零且可解码动作；其具体阈值应从起始CE和学习曲线预注册，不根据旧FINAL调整。比较冻结视觉/图像打乱/状态打乱，判断模型是否只靠恒速不读视觉（CPU基线可并行）。

若无法过拟合100步，立即 STOP 正式训练，报告诊断（标签/shift/视觉输入/梯度/学习率/LoRA权重更新/正则/控制层）；不能盲目扩大训练步数。只允许本桶≤100步，不得复用旧有6000步PAD版本权重当作新正确实验。

## 8. R15：正式 Push-T 训练（累计≤3000 steps、≤14 GPU小时）

只有 R14 证明真正学习信号+R13 checkpoint/部署入口可加载才自动进入。使用原先64 TRAIN episodes +16 DEV，TRAIN-only frozen q01/q99，DEV每次保存预测，不看新20 FINAL；所有模型/量化/decoder/gripper/hist/camera约束固定。原则仅一条 primary 42D/R 训练主线，若参数很多则先固定简单配置，避免用额外GPGPU打没有因果判别的随机搜索。

单GPU续训顺序：step0→300→1000→最多3000，**累计计数**。每阶段独立 Gate、checkpoint SHA、完整预测NPZ、DEV episode macro L1、修正量R²、动作幅度比、非零率/方向/运动分层、恒等A=S与CV A=S+V、TRAIN-only affine/MLP同样本对比。

自动继续阈值（预先冻结且允许按统计不确定性用DEV作判定）：
- 300步后：损失数值有限、合法动作token占比=100%、生成动作物理合法、至少有明确信号超过零步/恒等回归，若输出静止且DEV worse than CV则暂停诊断而不是继续；
- 1000步后：DEV macro physical L1 至少比CV改善5%或已经在至少三条预注册运动分层中显示稳定可信改进，动作幅度比/方向与expert同阶，不能出现连续多个epoch退化、模型忽略图像/文本；
- 3000步：仅在1000步Gate通过且剩余GPU预算充足自动续训；若DEV不提升或坍缩可提前停，选择DEV最佳 checkpoint，但新FINAL保持未知。
如统计波动使门槛不确定，追加不耗GPU的配对置信区间/episode层诊断，再决定继续或BLOCKED，绝不把旧20当“真实FINAL”。保存每个阶段可以独立复算的预测记录和DEVscores。

**训练成功不等于M1成功**：真实闭环里必须保持图像/动作/控制周期正确，并最终达到官方成功率；离线L1无论多低都只是中间Gate。

## 9. R16：真实部署控制时序、仿真闭环与一次正式FINAL

新部署入口必须处理两个关键差异：HDF5帧率25Hz=40ms，但旧7B约0.93s/动作（只是历史，不是当前42D推理测量）；必须记录每次真实14-token预测p50/p95/p99、RPC/排队/视觉预处理时间、sim physics step、观测周期与动作hold持续时间。观察是否每次policy请求暂停物理步，不能声称能实时25Hz。所有回合用官方未修改score/success逻辑，有joint bounds/速度限幅、失效/超时安全停止。

DEV仿真配额：Gate 1个回合检查接口、读取图像/历史和14D安全 → Gate5个回合看是否实际对物体产生进度及无重置串线 → Gate20回合总DEV预算（含前面1和5，累计20，不是1+5+20）。每一步存动作/状态/三相机/物体目标进度和score以及错误录像。输入不能看特权真值，特权state仅用于诊断指标。用同seed no-op / 恒速基线若能在剩余配额内评测，必须明确消耗次数；严格记录 rollout 数数，不能借口基线单独无限跑。

若连续5个DEV回合完全无物体进度或动作接近无动作策略，停止消耗剩余sim预算，集中 CPU分析编码、时序、视觉、控制和 gripper，修复需在授权GPU/更新步额内；若动作超界/接口错误立刻停并留证。优先让机器人可重复有物体进度，再判断成功率。

允许独立 FINAL 的先决 Gate：
1. 锁定模型 checkpoint/代码/环境版本、摄像头/任务目标/safety、推理/控制协议，唯一fixed seed list和原生评测脚本；
2. 20个新的离线 HDF5 FINAL如采集成功，只做已冻结offline评测一次，不能作为闭环成功率替代；
3. 对闭环单独预注册30个全新仿真seed且与DEV/已用场景可证明区分，模型在DEV里有非零真实任务进度、无动作安全违规、控制时序可完成sim回合；若不满足就**保留FINAL未使用**并报告BLOCKED_M1_CAPABILITY；
4. 在6GPU小时和30次上限内一次运行正式sim FINAL，完整保存每个run原始score/success、progress/videos/动作日志；不得边看FINAL边调超参；如只跑了部分seed，报告部分有效数和失败原因，不补超配额回合。

M1内部目标：建议事先固定成功率至少10/30，且动作安全违规0；同时报告真正成功数、Wilson95%CI、official mean score、物体最终距离/角度/进度、动作幅度、latency p50/p95/p99及失败类别。达到则 M1_RESEARCH_TARGET_PASS，未达到则 FAILED_CAPABILITY（仍是有价值的项目结果）；不足数据/评测环境则 BLOCKED，而不是虚构通过。

## 10. 多任务支线：先静态12任务，条件允许做2任务CPU/小GPU

任务集合仍为官方12个generalization标准任务：stack_bowls、push_T、pack_objects_into_box、fold_clothes、hang_mugs、sweep_blocks、pour_liquid_into_cup、make_toast、arrange_largest_number、sort_nesting_dolls_by_size、store_laptop_and_headphones、stack_blocks。R10.2 TASK_SCOPE_V1确认12在本地inventory，但很多robot/prompt/camera/动作语义UNKNOWN。先逐个源码与数据实查，不猜，不以文件名推断14D。

现有本地数据只有PushT、stack_bowls、sweep_blocks各100回合：只读检查合法性、数据目录/hash/拆分/机器人动作dim/相机色彩/任务成功轨迹、语言来源，并设计每任务TRAIN统计单独 normalizer，不能把PushT q01/q99套到其他任务。实现静态TaskSpec registry/shared-policy接口、任务prompt route、多任务sampler、防泄漏和CPU负例，最多只用上述已有数据。

可选2任务GPU先导只有在M1有可用表现、剩余额度且至少一个新增任务具备完整可信的监督数据、相同机器人接口或显式embodiment adapter时启动，最多2GPU小时且所有优化步计入3000总步数。只试PushT+1个已有任务，输出同口径 per-task DEV 和忘却检测，不能把它写成12任务完成；若M1还没过门槛，则这2h保留给未来，不强制花完。

不许可其他9个没有本地数据任务采集，不许可12任务真正训练和闭环全面评测；为下一次一揽子授权准备来源、成本、存储、设备和GPU throughput预算，绝不反复索要普通技术决策。

## 11. 交叉审计、失败停止和科研交付 DoD

每阶段至少有独立branch HEAD full SHA，真实GPU模型/代码/环境hash，数据与冻结manifest sha，命令/真实程序exit/PID/UTC起止，GPU/CPU资源时间和峰值，原始stdout/stderr + FAILURES.md，不覆盖历史；token日志与预测NPZ（不是只写数字）、仿真score和视频/轨迹hash，评测脚本/配对复算命令，检查所有训练样本实际source而非仅文件名声明。出现7D污染、相机4通道错误、EOS错位、时间泄漏、旧final误用、无限step训练、假collector/数据重复、OOM或者即将超过GPU上限，必须STOP受影响线并保留证据。

专门做一次交叉审计：CPU审计分支比较R42新train↔deploy同一完整样本的图像张量、proprio42、14监督token label、投影输出、logits、解码物理14D。real GPU Smoke后再对相同checkpoint独立复核保存/恢复。独立审核的补丁单独PR，不覆盖DeepSeek原始失败。

总报告：
- report/R11_R16_MASTER_EXECUTION.md：每阶段 IMPLEMENTED/VERIFIED_CPU/VERIFIED_GPU/CLOSED_LOOP_VERIFIED/BLOCKED/FAILED，固定baseline/checkpoint SHA和真实原始日志链接、剩余GPU小时、采集尝试数与25有效数、独立冻结5/20、所有DEV/FNAL评分；
- mission/R11_R16_MISSION_MANIFEST.json：预算每桶、累计GPU时长、真实optimizer steps（R14和R15分开）、训练/DEV/FINAL完全隔离证明、每个分支source SHA和相互依赖、所有失败计数/停机原因；
- evidence/r11_r16_<UTC>/真实源日志/NPZ/评测manifest/sim视频hash/数据 hash与所有被拒负例；
- 新的checkpoint metadata、R42部署接口单元测试和GPU Smoke/恢复日志、task scope与采集器、数据采集/冻结证据、评测汇总计算器；
- ROUTE.md **只追加实际完成的进展**、负结果与机制解释；不把未来计划写成实测PASS。
- 每条实现/独立审计线 GitHub 分支和真实Draft PR链接（不是 /pull/new/ 创建页面）；不默认合并到 main。
- 最终写清未达到M1原因，以及下一次升级12任务的具体数据/GPU预算；不向负责人请求逐个步骤审批，只在**任何新增超本包的费用/额度、真实设备部署、main改动或引入额外任务数据**时停下请求新的研究决定。

## 12. 自动继续算法与预算守卫

执行逻辑（生产必须具备机读budget.json和总账，不靠聊天记录数步）：

- 每个GPU job启动时从唯一任务账本读取 remaining_total_gpu_seconds 和 per_bucket_remaining_seconds；记录job_id、基线SHA、GPU PID/完整cmd；先通过GPU_FREE/PROTOCOL/DEPENDENCY，才启动；退出后无论成功、OOM或超时都计入真实elapsed；预设外层 timeout=min(本桶剩余，总剩余)，不允许负账本。
- collector最多40 attempts，达到25有效/单桶4h/磁盘15GiB任何一项先到即停止；失败/重复不补超过40，也不先采30再挑较好的25。
- 0步7B Smoke通过才能R14；≤100step TRAIN overfit通过才能R15；R15 300→1000→3000按已冻结DEV Gate；无动作或超过资源随时停；R16只在明确成功/前置协议下进入新30sim FINAL。
- CPU任务无依赖持续推进、生成真实代码、tests与PR；当某任务BLOCKED，交付能够运行的最小版本并如实写阻断。不要让所有工作线卡在一个未知专家脚本上。
- 如真正需要更大模型预算、补新任务海量数据、二次GPU缓存探针、全参训练、实验合并main、删除旧资产或真机，必须请求研究负责人明确批准；其余授权范围内全部自动继续到完成或可验证的停止条件。

**立即开始工作时，先固定本 Spec SHA 和现有R10.2实际分支SHA、列出安全GPU任务队列、核查环境，然后一边开发collector，一边做R42真实训练/部署P0交叉审计，剩余CPU做任务数据盘点与文档。无须研究负责人再逐个批复小实验。**
