# DeepSeek 实施指令：R15 G5=FAIL 后自动推进 M1/M2，安全修复和官方FINAL25

研究负责人已委托继续整个 R15–R23 12任务项目。本指令不扩大未获批准的外部费用、真机或额外 M1 正式训练额度。固定执行基线 feature/r15-true-continuation@9305563a2cf947739ebcc8cf428a11e2542dec90；独立审计 report/R15_G5_FINAL25_LICENSE_SUPERVISOR_AUDIT.md。main 不动，历史 WAL/HDF5/checkpoints/FINAL 不覆盖。

## 一次性研究决定
1. 接受本地已固定 RoboDojo 版本官方 **每任务25回合的primary SIM FINAL**（12任务最多300，PushT M1正式25），但任何单次FINAL前须新建独立的V2冻结commit包含checkpoint SHA、真sim版本、任务、真实layout seed dir、计分规则、启动命令、已用DEV与FINAL隔离证明。改动原30计划只能append协议修订，不得重写原计划。不能把历史训练HDF5因缺layout ID宣称100%新场景独立。新专家HDF5原25条5DEV/20FINAL是**另外的离线评测目标**，尚未完成，不可混淆。
2. 固定RoboDojo上游b8e5eed的LICENSE是MIT而README同SHA写Non-Commercial Research；当前main两个文件MIT。内部非商业科研继续；任何公开分发需逐项核验下载Assets/HF数据/XPolicyLab/submodules/weights的独立许可，不将整个资产包简单冠名MIT。论文准确并列标注冲突。

## 紧急 P0：调度器不能将 FAIL 变 PASS

读 mission/supervisor_queue_v5.json、tools/mission_supervisor.py、mission/r15_gate_G5_1000.json。G5 machine verdict FAIL/exit3，step1000 SAFE DEV L1 0.002538455比CV 0.002047481更差，0/4 strata胜。部分旧gate命令的shell exit0会掩盖失败；stage3 job只依赖loop task而非不可篡改G5_PASS。先查远端真实supervisor PID/state/项目GPU归属（不停止别人的进程），保留v5证据，写v6队列与显式content-addressed Gate事件，G5 FAIL或CANNOT时任何stage3在调度路径都无法READY。添加至少6条CPU负例：exit3而shell0、旧json FAIL、hash错、G5子job DONE却无PASS、DEV loop成功绕过训练Gate、supervisor重启/追加job重跑。合格且无资源冲突后切换项目自身队列；不重训已经跑过的300/500/1000。禁额外+2000 R15 formal steps；若需恢复该同假设训练，必须先新机制证据和研究授权，不能用M2配额偷偷给M1续训。

## R16 闭环独立执行，不因为G5 FAIL停工

立即查询第4次loop的实际日志/state/官方result JSON/PID/GPU计费，不要声称“正在运行”或“已成功”而不看现状；若失败，根据真实前向日志做最小修复并跑CPU回归。接下来验证三路RGB tile/channel parity与vision_backbone set_num_images_in_input(3)、ARX 14D合同、实际真checkpoint+LoRA+42D projector、14token动作、物理bounds和gripper、安全hold、per-env t0/t1/t2 reset、action周期、serving RTT p50/p95/p99。先单回合DEV确认官方env.step/任务进度，记录Policy实际生效动作与奖励、录像哈希，随后有条件累计1→5→20DEV。M1 R15 VLA@300/@1000 和CV/no-op同seed比较，**即使VLA离线更差仍可做有界诊断闭环**；不得把DEV当FINAL。只有模型/规则/seed冻结与安全/非零进度 Gate达成，才触发一次官方 PushT SIM FINAL25。

## 残差为何难：做机制诊断而非盲训

A[t]=S[t+1]使恒速误差恰等于|R|。线性42D->R不能预测仅否定此线性基线。CPU计算TRAIN与DEV按动作/阶段/运动四分位的R直方、R占256codec bins、在量化后R=0概率、R受训练归一化裁剪比例、源相机/目标位置分层、时序采样误差。无新训练地做冻结checkpoint视觉全遮挡/随机打乱与本体历史打乱的匹配DEV预测，对比恒速及无视觉变化的R分布；需要GPU时同一有界诊断任务按真实时间计费，保留image relevance与动作进度证据。没有证据不要宣称只有RGB信息，也不要用失败1000步证明整个VLA不可行。

## M2 不必等待 PushT 完全成功

现有push_T+stack_bowls 160 TRAIN/61171可用帧、两任务 per-task 42D/14D +语言合同、CPU共享trainer/P6 gate已存在；先修训练和真实部署链的多任务 normalizer_payload/load_frozen_contract/metadata与整个checkpoint SHA对齐，用小CPU逐张图片/时序回放加负例验证复用与错误任务ID拒绝。HDF5没有layout/scene ID：可以做 episode-level DEV开发研究，但暂不能声称严格 scene-level held-out，不能随便使用官方FINAL作训练。验证M2独立预算/WAL/真实sample stream后，在原已经授权的 M2正式训练累计最多20k里有界启动**两任务共享7B阶段累计最多2000优化步**，按每任务DEV/遗忘/任务辨识/恒速No-op离线与DEV闭环的预注册 Gate；如果无增益应立刻机制诊断，不能越过两任务Gate去4或12任务。
并行推进第3任务sweep_blocks、其余9任务专家/TaskSpec/数据来源缺口，不能全零轨迹冒充示范。布料和液体BLOCKED_DATA独立保留；专家不可靠的线不启动海量sim，不阻断已有两任务可验证工作。

## 官方25 V2合同与许可证
根据源码对 task-specific eval_nums25 + eval_num=min(EVAL_NUM,config)的核实生成 V2 pre-final manifest，更新research目标为每任务>=3/25、macro>=25% (这是项目内部目标不是官方阈值)。报每任务 attempted episodes、valid denominator、success数、Wilson95CI，任务macro以及运行环境sha；如果第1个FINAL曾经跑过则不能事后换协议，必须标历史结果不可比。许可证报告列 pinned/LICENSE 和 pinned/README 原始逐项版权，不覆盖旧文件。上游仓库main MIT声明与旧README noncommercial冲突不能自行对外公布法律结论。

## 自动执行/边界/DoD
无需研究负责人每次提交700步/1个任务批准。坚持服务器持久监督任务运行、CPU独立作业并行、heavy GPU与IsaacSim串行、单job超时/失败保全、旧1664总执行steps和1000有效lineage分别计数，二者GPU均记费；未证明G5_PASS不得增2000同线步数。不得修改main、旧数据/权重/WAL、官方计分器、真机、私有资产发布；独立GitHub工作树和真实Draft PR。

交付：v6 supervisor真实FAIL不能启动stage3测试、官方25 versioned freeze、r4真实结果与R16 1episode status及latency、M1 matched DEV机制诊断、2-task多任务train/deploy exact parity/合格时真实GPU0-step+训练与逐任务DEV、12任务数据进度、许可逐项出处、原始日志/哈希/ROUTE只追加。做到真实任务结果或可验证BLOCKED，不以窗口结束作为服务器任务终点。