# R16实测7B时延 / M2视觉前缀Gate紧急交叉审计与一体化自动继续
日期 2026-10-21。执行者固定源 feature/r15-true-continuation@b57d16a2579ee69f90703b49ca2b96955329dc8b。
此文件在独立审计分支，不证明远端GPU或M2当前job已结束；不得修改main、旧日志、checkpoint、源HDF5、旧DEV预测证据。

## 核实的机制 / 真实状态

1. **真实7B闭环链可执行，但速度不可用于25Hz实时控制**：源码 r42_deploy_server._decode_tokens() 逐次追加14个action token，每次调用TorchBackend.next_token_logits(prompt+ids, pixel_values, 42D)，后者对全prompt重新执行model，传入全部三路视觉tensor/42D，并调用CPU logits转换。执行者报告20动作真实测量p50=7754ms、p95=8509、p99=8552；14次forward占5998ms，中值194倍于40ms。早期24.6ms来自stub，不能冒充真7B。该数字主要见report/R15_R23_STATUS.md 与 mission/R15_R23_EXECUTOR_LOG.md；**evidence/r15_loop_attempt6的原始测量文件尚不在此次固定Git tree里**，需执行者推送原始日志/时间戳/同步方法/完整计数、固定SHA后再标VERIFIED_INDEPENDENT。当前是VERIFIED_SOURCE + EXECUTOR_REPORTED_LATENCY。
2. **多任务P6门控的动作概率提取确有实码错误**：training/m2_train_shared.py::token_ce_from_logits依据 `pad=logits_rows-label_rows` 正确偏移769个视觉前缀；但是 action_window_probs() 从 `n_prompt-1` 读取，未加 `pad`。evaluate_frame给ce_window、14D pred_bins、counterfactual task routing与P6 Gate的均是错误的视觉token位置，不是合法action token位置。原M2初始报数例如ce_window8.4266/9.8413不能据此声称模型能力或已达P6；ce_full是另一统计量，原则上正确但仍需对照实际forward检查。
3. **M2训练GPU账本再次漏记非step时长**：training/m2_train_shared.py::run写每个step_commit的gpu_seconds=time.time()-st，但job_end的gpu_seconds=0.0。每次存大checkpoint、DEV16帧×2任务×2 prompts、多路vision forward与进程加载都没有被算到M2 job GPU秒。与过去R15中已经修过的类似。请以remote raw start/end和WAL补不可变修正；在后续正式GPU job上固定book完整wall seconds+全job异常兜底，不可double-charge。
4. **P6对照定义不严谨**：tools/m2_gate_check.py将点预测命中计作0 nats，失配直接赋ln256，记为“CE_baseline”。这是人为定义的bin错误惩罚/分数，不是真实由归一化预测分布定义的概率负对数似然。若继续以cross-entropy名义作正式SOTA比较，需提前固定同样的256-bin概率平滑策略（p=(1-ε)·onehot+ε/256，ε预注册）或把它改标启发式hit-penalty score，**同时**要求匹配帧的物理14D关节 L1、逐任务DEV闭环/动作进度和任务路由反事实实验，不能凭旧CE对照一项宣称成功。旧P6数值与门槛不篡改；建立新版本Gate且旧样本上结果标探索性/协议更新。
5. **M2恢复日志旧风险**：training/m2_train_shared.py调用sampler.next()取得日志中的“next frame”，再对sampler.position减1。当前FrameSampler仅按position%len(perm)读取无额外shuffle，所以多数情况恢复同一索引，但不应以有副作用的next()读取日志。替换为纯只读peek，CPU验证连续vs保存恢复的下一条跨tasks帧完全一致，特别在perm边界/中断恢复及all random状态下。
6. G5_M1=FAIL 保持，不得借 M2正式额度绕开旧PushT extra2000；P6正式裁决必须由修正后的archive/Gate独立复算得出。场景layout溯源UNKNOWN，官方SIM FINAL25已同意，但不能拿DEV选择模型时触碰FINAL。

## 自动任务A（必须优先）：M2 CE窗口修复与负例

独立审计分支已最小修补：
- action_window_probs(logits,n_prompt,n_text_tokens=labels_length) 正确使用 pad=logits_length-n_text_tokens，取 pad+n_prompt-1 起的14个合法窗口。
- evaluate_frame对实际task与counterfactual不同prompt分别传入各自labels长度。
- 新版eval_archive写 action_window_alignment=vision-prefix-aware-v2；load_eval_archive拒绝旧历史错误P6 archive。历史证据保留，不覆盖。
- tests/test_m2_dev_ce.py增加视觉prefix无prefix同bins/概率测试、错误长度拒绝和旧P6 archive拒绝。**这是源码修补，不是已远端跑绿；需DeepSeek在新实现分支做CPU red/green和实际M2 real-forward对照、验证新输出与teacher-forced文本对齐的14个label逐位一致**。
- 现有m2_two_2000_r5如果已在GPU运行，不可因为聊天审计随意kill；由有远端权限的执行者查PID、checkpoint与进程所有权/是否正执行评测，遵守原资源边界。旧checkpoint、optimizer状态仍可保留，并在fix之后从同一有效模型重新生成DEV预测NPZ和corrected P6 Gate。旧archive既不删也不用于Gate PASS。

## 自动任务B：真实 7B 的 latency 分解与原语义缓存（GPU重活串行）

按“诊断→语义等价→加速→落仿真”的依赖执行，不等待研究负责人每个小实验回来审批：
1. 固定checkpoint=R15@300或@1000，固定三路同RGB、42D、prompt、action codec 14token；严格源SHA、模型SHA及随机性/视觉预处理完全相同。诊断GPU<=30min独立桶，真实20次热身后64次样本（按可承受范围），nv synchronize正确阶段。将preprocess (PIL/CPU-to-GPU)、DINO/SigLIP vision features、proprio projector、prefill 769 visual+text tokens、每次14 decode、logits GPU->CPU和websocket RTT分别计时；记录p50/p95/p99、峰值显存。原始报告的7554/7754对照只可与同GPU同sample同batch同token数比较。
2. 新增可选 `cached` 推理后端（不得改训练模型架构或权重），**在每个控制step内部**一次vision+42D+prompt prefill，保存attention/past_key_values，然后仅以已生成的新action token做增量decode直到满14个。不同帧/环境/任务prompt/相机状态、reset、prev cmd变化时严禁跨环境复用旧前缀；维护mask、RoPE position/cache index、legacy prompt trailing space, multimodal token embedding排列，明确训练路径是pure-autoreg且要保留动作条件，不得用OFT action mask覆盖之前action embedding。
3. **同一batch真实7B数值等价性Gate**：32个冻结TRAIN/DEV开发输入（不读新FINAL）和至少3条回放动态序列，Full-Recompute 与 Cache 每一k=0..13输出合法256 bin窗口的 logits 差异统计(max_abs/p99)、argmax的14/14 token完全一致、最终decoded SAFE joint14D一致、无额外STOP/PAD、图像/42D按帧变化后的cache invalidation、跨env/reset不泄漏；不同长度prompts、两任务prompt与相机tile/频道正负样本均测。不一致不切换正式server；如bf16误差致改变bin，留完整sidecar+首次分叉定位，不隐瞒。
4. 最小启动改动只能在独立R42 deploy新backend feature flag下；原full backend仍可切回。profile在同GPU条件下至少20个动作真实数据，p50/p95/p99、全推理/LLM增量/视觉塔/拷贝与内存，证明真实加速比。缓存只能减少重复视觉/prefix，**不保证<40ms**；没有实测不承诺两数量级。
5. 如缓存后仍大于40ms：a) 对于**同步步进**的离线RoboDojo仿真，先确认它是否只影响实际墙钟、不影响仿真时序和控制稳定性。若同步等待策略输出且官方协议允许，仍可做小规模有界DEV真实闭环（报告推理wall），不能错误宣称40ms设备实时；b) 若要真实25Hz，后续是动作chunk/并行非自回归头、蒸馏3B/student、两速率控制等重大架构选项，需要负责人认可架构/新预算，不能把原缓存优化默认为已达标。
6. 建议Gate性能阶梯（非重新声明官方要求）：prefill一次和每action14次增量准确→内部benchmark真实speedup≥3x阶段值得继续优化，能达到p95<200ms才适合进一步原动作控制研究；真实25Hz要求端到端p95≤40ms并满足安全及实际时钟，可根据控制策略另立经批准的时序协议。没有速度改善也可交付严谨profiling/根因/失败保全。

## 自动任务C：M2 ledger / supervisor /训练与评测同步修复

- 继续保存现有合法M2 r5检查点、不重新训练。先在CPU搭模拟恢复和坏archive/错误prefix/cross-task prompt/不完整WAL/延迟冲突测试。
- 对M2的每个job，计费真 GPU job wall（建模加载+所有steps+save+评测），用WAL时间与clock兜底，避免已有M1 8764s与M2独立20k额度混记。训练运行中只能由进程所有者有据干预，不能误杀别的进程；结束后利用原始日志追加不可覆盖调整。
- 真实训练达到step2000、确证parent full checkpoint与opt/sampler/RNG后，按新版CE对齐重新生成8 DEV frames/task的per-task窗口分布和反事实prompt预测、物理14D L1/R²/动作幅度/无动作率，并与同sample的no-op/CV以及task-specific历史模型对照。保留P6旧门槛、新metric erratum。8frames/task极小，必须标探索性，使用20 DEV episodes的预先锁定扩展cohort做置信度/不确定性分析，但不触碰新FINAL。
- 旧公式CE与平滑定义另做方法审计。如果真正模型仍不能超过简单基线，阻断M2 4任务扩容并自动做相同帧、同数据、配对诊断，不盲目上6k/20k步；同时推进12TaskSpec、9缺数据任务的专家设计与合规取证。

## 自动任务D：部署/交付

- M1 offline G5=FAIL、真实PushT闭环SUCCESS UNKNOWN，M2训练RUNNING仅以持久job/原WAL查活后才可认为RUNNING；r5从启动到当前执行结果不能从GitHub固定SHA断言完成。
- 不修改main、旧checkpoint/NPZ、旧FINAL，GitHub独立PR、SHA、负例、GPU profiling JSON、latency raw, token logits and cache validity, P6 corrected archive hash, WAL correction, 12-task真实工作线继续。
- 安全停止：invalid token/越界/任一prefix-cache不等价只停部署优化；M2旧gate错误只停P6晋级决策而非删训练；资源进程不明只停受影响重GPU；重大模型/预算决策交负责人；其余CPU独立继续。**不因聊天窗口终止远程已经持久部署的作业**。
