# R11–R14 交叉审计与 R15 真续训/预算 P0 Gate

审计日期 2026-10-21。独立审计基于 feature/r11-p0-closeout@69031eda6ee32439aa75d71b227a66a7dcb78aa2；原始证据 PR#18 https://github.com/jho887891-sudo/robo/pull/18。未在本审计会话运行任何新 GPU、优化步或仿真。

## 1. 证据核对

- P0 CPU：evidence/r11_p0/green_regression.log 记录 250 passed / 3 skipped；源代码/红灯证据在提交中，状态 VERIFIED_CPU_SOURCE。
- R13：evidence/r13_zero_update_smoke_20261020T194055Z/raw.log 和 smoke_evidence.json 显示实际 pristine 7B 加载、LoRA/projector 梯度、loss 14.4278、optimizer=0、14 legal tokens、checkpoint/reload 保持 token 一致。GPU 99.7 s + 首次崩溃 70 s。支持 VERIFIED_GPU_ZERO_UPDATE（仅一个 TRAIN 输入），不支持泛化声称。
- R14：evidence/r14_overfit_20261020T200030Z/raw.log + overfit_evidence.json，32 TRAIN episode 各取1固定帧、100 个optimizer steps，固定帧 CE 14.8141→0.125，支持 OVERFIT_SIGNAL_PASS，但无 DEV 预测/视觉消融/任务闭环。14 维 bins 中 9→13 种仅说明离散多样性，不代表动作物理幅度或任务成功。
- 预算：原始 R13 169.7s（含失败70s）加 R14 857.8s，总1027.5s。R14 raw.log 从19:46:12 ledger start 到20:00:30 ledger end约858秒，overfit_evidence.utc_start=19:47:22 在模型加载之后，所以不是同一个计时起点。当前 reconstruction events 把R14 start写为20:00:30却在其后列出19:48..20:00里程碑，时间不单调，应区分 reconstructed_timestamp 和 actual_log_timestamp，不能作为原始逐事件记录。原始账本曾被覆盖，这是已发生的预算管理事故。
- R11 采集：BLOCKED_EXPERT，没有新25回合；R15和R16尚未运行。现有PR#18才是实际Draft PR，不可用/pull/new/创建页面替代。

## 2. P0-RESUME-1：当前R15不是累计续训

training/r11_train_pushT_r42.py 的 run_train（约L426–550）每次调用 build_model()，从pristine base及新随机LoRA/projector初始化；每次创建新的 AdamW+MultiStepLR；明确拒绝 args.init_lora；for step in range(1,args.steps+1)从1开始；保存checkpoint只含LoRA、projector和meta，没有optimizer/scheduler/RNG/sampler状态。并且默认save_every=0。训练随机种子在build_model之后才设置。

因此三次命令 --steps 300、--steps 1000、--steps 3000 会产生3个独立从零训练的模型，总共4300 formal steps，既不是累计训练，又超过授权的3000步。R15_GATE = BLOCKED_IMPLEMENTATION，必须先修复并做CPU回归。不能以R14固定帧CE下降替代。

## 3. P0-LEDGER-2：预算账本目前非故障安全

training/r11_train_pushT_r42.py 的 ledger_book() 用 open(LEDGER, w) 直接覆盖整个JSON，没有不可变WAL事件、排他锁、原子rename或事务检查。用户已经报告该账本被mirror push重置为0，再根据日志重建。正式run_train每次optimizer.step之后未及时持久更新 formal_optimizer_steps_used；ledger_budget只按 GPU秒数，不核验累计正式步数 ≤3000；程序异常时main只记wall秒，不足以恢复实际中断前步数。

必须完成完整GPU秒数和步数双重守卫，发生覆盖/崩溃/重复记账均 fail-closed，而不是允许旧计数清零后继续。

## 4. P1-METRIC-3：缺真正的DEV Gate

R15续训的允许条件不仅是训练loss下降，还要在预先冻结的16 DEV回合相同帧集/时间mask上比较 physical14D macro L1、identity A=S、constant velocity A=S+V、TRAIN-only affine与MLP、correction R²、motion-strata、符号/幅度比、夹爪与动作不动率，并保存VLA预测NPZ、模型SHA、同口径配对结果。未实现这些指标不能声称G4/G5通过。

## 5. 交给DeepSeek的最小修复验收

A. 先无GPU构建真续训：新的 --resume-from 指向已冻结的本项目全状态checkpoint；包含base源哈希、LoRA、42D projector、AdamW state、scheduler、global_step、Python/NumPy/torchCPU/torchCUDA RNG、采样器状态、数据/归一化/codec/seed/训练参数SHA。checkpoint必须原子写入（临时文件→fsync→rename）、不可覆盖；加载时严格检查文件哈希及版本。冻结R14 100步模型不得作为R15正式起点。

B. 单一训练 run 在pristine起始，step0→300首次段；恢复step300父checkpoint只补700更新到累计1000；恢复step1000只补2000到累计3000。禁止300+1000+3000三次各自从零训练。要提供 CPU 小torch或等价测试：同种子同数据的N步连续训练与K步→保存→恢复→N-K步，权重/optimizer动量/LR/step/RNG/下一个样本一致。显式拒绝 --save_every 0、错误checkpoint、错误global_step、超授权步数。

C. 将GPU/正式optimizer账本设为单写者、原子/带锁、带唯一run_id与不可覆盖追加事件WAL；每次step之后可追溯step_committed、每个重启做ledger与真实ckpt一致性校验；总额度剩余+R15桶剩余+3000总步数三项启动前fail-closed；崩溃保留实际用秒、任何记账不确定则停止等待审计，不能默认重建清零。不得修改原始R11账本事件；另建带来源的更正/新账本。

D. CPU先实现DEV评测器+数据无泄漏回归，真实只读HDF5匹配基线。绝不能因R14过拟合良好就跳过DEV配对。

E. D的专家PushT geometric/IK expert设计与R15评测器并行CPU推进，先证明在新seed上真实产生成功动作，再在原collector 4h/40 attempts额度内尝试生成25集；不可用旧100改名或全零轨迹凑数。

F. 修复/验收后才可花原R15桶执行300步；检查G4后续1000/3000，失败留证、独立工作线继续。R16需要真实42D deploy server连接并测量policy latency p50/p95/p99，再DEV1→5→20后是否允许正式30新seed FINAL。所有在现有32h/3000formal步总授权内，无须研究负责人再次批复。

所有修改独立实现分支+Draft PR/失败原始log/绿灯test/固定SHA；主分支、旧数据、权重、历史证据保持不变。ROOT ROUTE.md只追加已证实的审计与真正执行结果。当前R15不是已准备好可安全启动的状态，不得用原命令直接跑GPU。
