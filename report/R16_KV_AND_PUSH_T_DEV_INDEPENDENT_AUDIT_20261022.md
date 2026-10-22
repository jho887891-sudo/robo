# R16 KV 编码根因 & 官方 Push-T DEV 闭环独立交叉审计（2026-10-22）

固定执行源：`feature/r15-true-continuation@569677b5b6ec403458178c7036e2ef00e493052f`；保护 main@329614f54d3090b9b6ab120855ba50fbc93bdd1e。**本审计独立阅读已提交源码、原始JSON与历史，未远程重跑7B、未修改旧WAL/HDF5/checkpoint/FINAL、没有启动GPU。** 不能将GitHub原始JSON的存在冒充重新GPU独立计算。

## I. VERIFIED_SOURCE：真正的R16第一根因是诊断工具，不是 KV Cache 语义错误

旧 `tools/r16_kv_equivalence.py` FULL 对照loop把合法动作bin `0..255` 直接追加进下一次forward的 `input_ids`，而真实合法动作token是 `TOKEN_BEGIN_IDX+1+bin = 31744..31999`。cached loop却正确执行bin→token的映射，因此人为制造从k=1开始的巨幅错误、假32/32全部不同、假reference==full而cached不相同。旧 `tools/r16_three_way.py` 也在reference与full两条path中共同喂了bin索引，所谓“三方参考一致”属于共同错误，不是独立真值。必须保留旧错误数据并显示SUPERSEDED，绝不能继续引用旧31.39 logits/0-255饱和作为真实模型前向缓存故障。

最新源码 `tools/r16_kv_equivalence.py::decode_full/decode_cached` 已统一经合法ID映射，并有 `needs_legal_token_check` 拒绝误喂bin。tests/test_r16_kv_decode_tokens.py 有四项针对该错误的CPU stub回归（执行者报告通过，审计未在真实远程运行）。`evidence/r16_kv_multiprobe.json` 显示真KV长度805=36文本+769多模态、相同token下k1三种增量实现合法256bin logits最坏差0.03–0.125；旧例 t155 从此前约17.7缩到0.125。故最强当前事实为**旧报告的重大divergence属于非法参考输入；正确参考下保留BF16级数值差异**。原先PR #28的H1-H5属于已经被新证据缩小/否证的旧诊断假设，不应再重复执行旧诊断任务。

## II. 关键口径：STRICT_PARITY_FAIL ≠ NOISE_LEVEL_PARITY

- 原本要求缓存与FULL的**所有14token argmax完全相同**。这个硬Gate在修正后仍**没有通过**。
- `mission/r16_kv_equivalence_v3.json` 原始summary：32开发输入、`argmax_identical_all=false`、`n_diverged=32`、`verdict=DIVERGED`；这是 `tol=0.01` 的严格logits门槛，将大量仅微小数值差的输入也判为DIVERGED。不可将它重新描述为 EQUIVALENT。
- `mission/r16_kv_forced_parity.json` 的另一种**新口径**使用`tol=0.5`；fixed teacher inputs的worst abs logit=0.5，逐token teacher-forced argmax同样出现5次与FULL不同（k2=1,k3=1,k11=2,k12=1，共5/448）；自回归FREE RUN也恰5/448次bin变化，约1.12%，每次单个相邻codec bin。Verdict=`NOISE_LEVEL_PARITY`是**近似数值容差Gate**，不是 bitwise/strict action parity。这个新口径若作为今后 DEV 的软件试运行准入，需要另行声明适用范围，保留旧严格FAIL；不能偷偷撤销曾经的硬Gate。
- 同版本的`mission/r16_kv_equivalence_v3.json`内最坏FREE RUN logits差1.3125，原因可能包括动作token翻转带来的序列级联；它与teacher-forced worst0.5不是同一指标，不应表格混称“最大logit差0.5”。
- `worst_physical_residual_max_abs=0.000157434...` 是14维`R`的**原生数值单位差**。ARX双臂维度包括6关节+1夹爪、左右共14项；关节通常是角度量，夹爪又有自身量纲。不能全部换算成“0.16毫米”。除非通过运动学正解对TCP位置变化逐坐标实测并标单位，报告只能写“max |ΔR_dim|=1.57e-4 native units”。真实 SAFE 14D动作、关节限速、夹爪阈值、末端位移以及长期闭环分叉还要另做配对验证。
- 448个位置来自32个输入×14token，其中包含2个任务的交叉wrong-prompt输入，不是448个独立episode，也不是官方控制动作位姿安全大样本。不能以1.1%直接推算闭环成功率不受影响。

## III. R16真实加速：成立但仍非实时

- 同GPU批量64动作报告（`mission/r16_kv_equivalence_v3.json`）：FULL p50/p95/p99=5060.1/5162.4/5211.5ms；缓存1145.1/1175.1/1210.9ms，p50速度比4.42x。实验对照在一个脚本同种输入，证据支持为 `EXECUTOR_RECORDED_REAL_7B_SPEEDUP`。脚本顺序先测全量后测缓存，需确保GPU热状态/clock公平，未来paired/interleaved ABBA只为提升严谨性，不阻断已测重要进展。
- 官方推理慢速同步DEV缓存路径日志显示600控制步后 total p50/p95/p99=1582/2021/2595ms，p50约40倍于40ms真实25Hz周期；`R42_KV_CACHE=0`默认不变。缓存路径可以作为显式留痕的开发仿真（non-realtime exploratory sim），不能由4.42倍加速或单次仿真结果宣传真机实时。
- 真7B缓存数值近似，而非严格14/14一致；如果未来要正式替代FULL serving，需新的**部署安全准入Gate**：按R15 step1000及M2 step2000不同真实checkpoint分别做同输入same-seed成对SAFE14D/运动/夹爪/TCP物理影响和缓存失效实验，且设置明确回退到FULL。不能因为M2 step2000上的近似等价，就假设R15 step1000闭环检查点也逐token近似。保留完整Full为标准。

## IV. 官方 Push-T 两个0/1，不是0%能力结论，也不能由reparse当fresh

- `evidence/r16_loop_dev_episode1/_result.json`：FULL对应1官方DEV episode，layout0，success=false，官方统计0/1。报告这个episode为600个控制steps，p50≈5529ms，墙钟3842.9s。
- `evidence/r16_loop_dev_cached/_result.json`：缓存相关目录复制有同样格式结果，layout0，success=false，官方0/1；`live_latency_cached.txt`有600步p50=1582ms，显示缓存服务端真实输出，但**结果归属仍需原始启动日志/进程run_id/result path证明**。
- 特别是`evidence/r16_loop_dev_cached/push_T_stage1_cache_reparsed.json` 中明确 `reparse_only=true`，`exit=null`，文件说明在05:10:30Z**没有启动新client**；它取的是容器内已有 `2026-10-22_04-48-44/_result.json`。因此**这个reparse事件本身不能支持“新一轮缓存闭环driver rc=0”**。应提交与04:48官方结果对应的原client stdout/stderr、真实run_id/exit/开始结束UTC和PolicyServer开关日志，独立确定其在`R42_KV_CACHE=1`下生成并确实执行600步。若能锚定，0/1结果就是合法缓存DEV实验；若不能，就标`RESULT_PROVENANCE_UNVERIFIED`，不能偷用旧episode凑独立评测。
- `tools/r15_loop_dev.py::newest_result_in_container`目前按task路径与`find -L ... -mmin`找最近结果，而非按唯一run_id与运行开始/结束/配置匹配；这在多次同seed/layout/ckpt重测时**可能错误关联其他run**。应在独立修复分支对真实执行要求锚定新建result directory/在启动前记录目录清单并验证 mtime≥job开始、job结束，并以官方result唯一path/hash及stdout tally一致为PASS；reparse要绑定明确历史结果路径不能靠latest。保留历史证据，不覆盖旧。
- 两个0/1仅证明这两个**DEV运行记录**中的结果均失败。即使真正同样layout0 seed0，也不能由n=1推出缓存“判定不变”或模型成功率=0。Wilson95% CI [0,0.7935]显示不确定性很大。失败根因可能是模型动作能力不足、safe clipping/控制器、任务物体初始条件、仿真时序或末端归位等；须从官方对象状态/动作轨迹与同场景CV/No-op基线分别排除，不能用离线G5_FAIL直接宣称闭环失败因果已经被证明。

## V. M2阻断独立于R16成果

此前独立 PR #27发现tools/m2_gate_check.py的no-op物理L1使用`mean(abs((s[t]-s[t])-a[t]))`，等价全零关节预测而非保持当前关节S。其补丁在独立分支尚待执行者应用并真实回归；M2 P6=PASS是每任务8帧、概率NLL口径的机器结论，并未独立证明SAFE物理动作优于CV且原forgetting curve仍用了旧视觉prefix错位CE。因此不应仅凭P6=PASS自动升级多任务训练，应保持数据与双任务部署接口CPU工作并行，做真正能力诊断。

## VI. 优先下一步：保留成果，不盲训

1. **立即CPU安全工作**：保留撤回声明及旧报告；为所有动作 token ID/codec断言和官方result唯一run_id匹配写负例；纠正文档中的“0.16mm”和“36层”误写，真实多探针可见layer0..31共32项（不覆盖历史正文，可只追加勘误）；审计R15 step1000 cached对照的输入/decoder/normalizer/部署开关。
2. **重型GPU无新优化步任务**：同R15 checkpoint、同DEV帧/任务，比较缓存与FULL的RAW14D/SAFE14D、关节及夹爪动作差异、控制速率/clip率、通过FK的末端TCP单位米位移（若FK资产与坐标可靠），跨不同motion stratum的一步及连续rollout分叉，给适用`DEV_SYNC_ONLY` vs `DEPLOYABLE`独立Gate。
3. **闭环针对能力而不是速度**：确认官方result确与cached session绑定；提供600步物体位姿/目标距离/夹爪/动作执行/归位谓词逐时间戳的曲线，最低成本有判别力的同布局CV/no-op和训练模型配对试验，优先定位失败发生在抓取、推动、定位还是撤臂；别只看0/1终点，不能把DEV当FINAL。
4. **算法架构的科研决策**：纯7B逐动作自回归在1.58s/决策时距真实25Hz差40倍。如果目标是先证明官方12任务闭环，可继续合法同步仿真、任务数据与能力研究；如目标是机上真25Hz，必须独立评估单次VLA推理成本、动作chunk/高低频规划+控制、学生策略蒸馏及更小模型，重大模型训练/部署与预算由负责人决定。禁止承诺纯KV可达25Hz。
5. **项目管理**：独立工作线继续12任务合规专家和任务接口，绝不因PushT一次失败冻结所有CPU线；GitHub原始SHA、真日志、被否证结论、追加的勘误、真实job账本和独立Draft PR为交付。

## 分类结论

- VERIFIED_SOURCE：非法bin输入就是旧对照工具重大分叉的直接原因；源码修复和回归存在。旧归因已撤回。
- EXECUTOR_MEASURED：修复后real7B速度4.42x，强制同token worst0.5，free5/448 flips，缓存慢速DEV600步 p50=1582ms；原始GPU文件与result目录在GitHub已归档，本独立审计没有重新运行GPU。
- NOT_VERIFIED_STRICT_PARITY：新工具仍输出DIVERGED；应清楚保留严格Gate与新噪声级Gate区分，未经批准不能把噪声级变成“位级完全一致”。
- OFFICIAL_DEV_RECORDED：FULL n=1失败；缓存目录有n=1失败文件但实际result关联需要锁run_id，reparse不是一轮新episode。
- BLOCKED_REALTIME_AND_CAPABILITY：25Hz未达成；12任务成功率未测，PushT n=1失败与M2 P6概率成绩不能证明闭环能力。
