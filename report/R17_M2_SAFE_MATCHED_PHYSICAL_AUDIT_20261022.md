# R17 / M2 交叉审计：成功确认闭环负结果，发现物理基线未同口径（2026-10-22）

## 执行及证据基线

执行者 `feature/r15-true-continuation@f34cc44495d34a9d42f15d265947c654cf8248c6`。main保持`329614f54d3090b9b6ab120855ba50fbc93bdd1e`。本分支为独立CPU源码和归档审计，**无远程GPU/仿真权限，未重跑真实7B/IsaacSim**；不覆盖任何历史数据、账本、checkpoints、预测或FINAL。已在GitHub独立核查`mission/m2_physical_audit.json`, `mission/m2_physical_curve.json`, `evidence/r17_cv_baseline/{loop_record.json,motion_compare.json,_result.json}`与实际源码。

## I. 可接受的成果（VERIFIED_SOURCE / EXECUTOR_RECORDED）

1. Push-T R15 step1000 full官方DEV单回合完成600步且官方success=false。恒速CV在相同布局0/seed0的官方client下有完整600步、官方`_result.json`=0/1、`result_is_fresh=true`和stdout `Success nums: 0, Fail nums: 1`。这说明在**这一个布局的开发评测中**两个策略都没有完成任务，绝不能由n=1写总体成功率0%。注意CV驱动记录 `exit=141`（并非0），尽管完整官方结果和600步视频存在；要分别记录“完成官方回合且失败”与“driver进程返回141”，并查退出原因，不得声称CV客户端退出码0。
2. 每100步头相机帧灰度差 `0.106, 0.092, 0.071, 0.020, 0.031, 0.009`（VLA）和CV `0.098,0.112,0.096,0.048,0.053,0.032`，表明**图像变化减少**；结合视频目视T未推到目标、双臂回缩，为“无reach-and-push”提供证据。灰度差不能独立测量真实关节速度、物体位移或接触；应按任务真值状态新增object 3D pose/contact/success subpredicate/safe command trace验证“t240后完全静止”和真实因果机制。
3. 已经订正no-op为`mean(abs(S-A))`，保存 `l1_zero_position_LEGACY`。CPU预测归档来自同一个M2 step2000 checkpoint，按corrected视觉前缀评估、8DEV frames/task。P6概率PASS与物理闭环能力未知/失败可以并存。
4. CV不是严格数学上的零残差：真实 serving 用 `token_ids_for_residual(zeros)` 将0编码到最近的256-bin，再经`decode_command`。CV具有小量量化偏差；对照也必须实际走同一codec/limit/gripper path。

## II. 本轮新P0：SAFE物理比较未执行 matched treatment，直接影响2→4晋级Gate

在执行者 `tools/m2_physical_audit.py`（v1）：
```python
raw = S[t] + v + r_hat
safe = np.clip(raw, pmin, pmax)  # only MODEL bounded
noop = S[t]                      # unbounded
cv = S[t] + v                    # unbounded
```
实际运行的`r42_adapter.decode_command`除bound clipping之外，还把模型推出来的**gripper dims(6,13)**残差丢弃，改为`S+V`，但v1的 `safe` 未做fallback。且v1对CV没有应用同样的量化codec、joint SAFE裁剪。这不是在同一控制接口上的fair comparison，不能直接作为“两个任务SAFE physical model输给CV”的最终证据。

现有执行者原始数值：
| Task | MODEL raw(v1) | MODEL safe(v1) | CV raw(unclipped) | MODEL-safe minus CV-raw |
|---|---:|---:|---:|---:|
| Push-T | 0.00089834 | 0.00144959 | 0.00094366 | +0.00050593 |
| stack_bowls | 0.00231544 | 0.00700685 | 0.002196996 | +0.00480986 |

PushT在**RAW下反而微小优于未裁剪CV**，而SAFE-v1相对MODEL raw增加约0.000551；stack_bowls SAFE比RAW多约0.004691。因此当前“CV胜过MODEL SAFE”的物理差距主要可能来自不同的处理路径，不能据此断言训练一定毫无能力。这里也**不能反过来宣称MODEL胜过CV**：v2实际结果必须重算，且8DEV帧的paired-bootstrap CI原本就跨过0（模型-CV：push_T[-0.00012,0.00163]；stack_bowls[-0.00046,0.01212]）。

## III. 新增独立匹配评测实现（待远端执行）

已在独立分支创建：
- `tools/m2_physical_audit_v2.py`，默认输出**新文件**`mission/m2_physical_audit_v2.json`，绝不覆盖v1。模型、CV均使用`r42_adapter.decode_command()`，相同实际256-bin codec和同一任务`pmin/pmax`，自动应用gripper fallback，no-op也施加相同bounds；报告新老均值、每帧差、CV/model裁剪维数、真实DEV target/state超限次数、同帧paired CI。
- `tests/test_m2_physical_safe_parity_v2.py`四个负例：CV也要裁剪；模型的gripper残差不被实际命令使用；no-op也同bounds；不裁剪CV会虚假提高其相对真实可部署策略的评分。
- 此实现**仅匹配per-task TRAIN pmin/pmax的物理位置限幅**，没有prev_cmd/dt所以没有率限制（标志rate_limiter_applied=false），也未证明这些数据bounds与实际R15/M2 checkpoint `safety_limits`始终相同。严谨的`DEPLOY_PARITY`仍需从封存checkpoint真实metadata/伺服调用收集约束，对每个joint/夹爪和控制步做往返对照；CPU测的数学闭式指标不能直接叫做实际600步闭环命令。
- v2代码与测试在本独立分支仅 IMPLEMENTED，未远程执行CPU/7B。**在它的实际结果及现场metadata核验前，M2 Gate修订为REASSESSMENT_REQUIRED（之前FAIL保护性保持、不得贸然启动2→4扩训）**，不是直接将FAIL翻转为PASS。

## IV. 方法学过度归因需要收敛

- “covariate shift已经由CV 0/1证明”：实测CV单布局0/1与VLA0/1支持**从静止难以启动**的假设，但要证明模型真的遭遇离线→在线运动状态分布漂移，至少对比真实rollout初始 `V_t` 分布、TRAIN示范的相同片段 `V_t`、网络当时`R_hat`、对物体接触距离的变化。仅灰度帧差或0/1不足以因果断言。
- 平均per-frame的`resid_r2`（14维单帧R2再平均，特别gripper、接近常数分量）并不等价于**全体残差元素上的全局R²/方向相关性**，且small eight frames不可靠；不能把R²≈0直接翻译为方向随机。需要打乱的残差零预测、cosine方向相似度与运动分层比较。
- 对raw vs safe裁剪的差异做来源划分（真实物理joint limit、TRAIN统计pmin/pmax过紧、gripper fallback、rate clamp）。检查GT动作是否超出TRAIN bounds，因为越界示范会导致任何安全策略被迫偏离GT；但如果只是训练统计包络不等于机械臂物理硬限位，必须在权限内提出独立安全设计，不自动放宽真实设备安全边界。
- 12TaskSpec/CPU专家数据线仍继续，不因两个负回合或Gate重审停止。
- 25Hz方案属于另一个架构议题。报告称已交`mission/r17_realtime_architecture_analysis.md`，在固定HEAD f34cc44 GitHub树未找到这个文件，请执行者核对真实文件路径/commit，不能将未入GitHub树的报告标VERIFIED_DELIVERY。

## V. 下一阶段自动化DoD（不强制新训练）

1. CPU运行v2测试→同冻结DEV8帧每任务产生新v2报告，保存hash；独立复算每一列、所有gripper/arm分组及paired差/CI，原v1报告加勘误不覆盖。
2. 对照实际M2 checkpoint metadata里的`safety_limits`、pmin/pmax、速度限幅与gripper fallback。补CV/Noop/模型真实部署 decoder SAME-action regression，含故意SAFE越界负例。
3. Push-T已有600步视频与run_id/client stdout，新增只读或安全DEV特权状态object XYZ/orientation、目标距离、接触、关节state/动作时序与阶段成功谓词，不泄漏给VLA视觉策略训练。
4. 如果新模型SAFE/CV在每帧/每episode无可信优势，保持2→4 Gate=FAIL；若v2证明修复后的SAFE能力明显不同，则先冻结新评测protocol、扩展已授权DEV和有界单任务闭环，不直接追加无证据训练。
5. GitHub独立审计PR/失败日志/不可变WAL/hash/append-only ROUTE。远程执行者自主继续已授权CPU/有界sim诊断；重型GPU/IsaacSim资源保护照旧，无人为小时配额，不购买新GPU、不动main/旧证据、未经负责人授权不开始大规模新训练。

**结论等级**：官方full与CV的同布局失败结果 EXECUTOR_RECORDED、源码存在physical comparator不公平 VERIFIED_SOURCE、v2实现 IMPLEMENTED、模型真实有无可用物理能力 REASSESSMENT_REQUIRED、covariate-shift因果解释 HYPOTHESIS。
## VI. 不能跨checkpoint做闭环因果结论

- 当前具有完整官方600步闭环负结果的VLA策略是 **R15 step1000** checkpoint；M2双任务每任务8DEV帧物理审计和2000步过程曲线来自**另一套M2 step2000** checkpoint。两者模型参数、训练过程、任务条件可能不同。**不能用M2的R²和残差0.47比值解释R15 step1000当时为何冻结**，也不能拿R15的一次0/1宣称M2 step2000闭环能力为0。
- 下一轮应在同checkpoint上收集真实闭环+离线物理配对，不能跨权重拼接一个“已验证根因”。同一任务下真实T没有移动是当前录制回合的事实，但“模型缺乏任何reach能力”属于更广的研究问题，必须检查物体相对末端的3D距离曲线与对应权重。
