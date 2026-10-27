# RoboDojo / OpenVLA 项目状态（唯一日常交接入口）

> 状态快照：2026-10-27。此文档是**项目索引**，不是无需核验的最新运行结果。
> 仓库：[`jho887891-sudo/robo`](https://github.com/jho887891-sudo/robo)。
> 每次接手先读取本文件及根目录 `AGENTS.md`，随后通过 GitHub 获取分支最新完整 SHA；以实际代码、原始日志和可复算证据为准。
> 只讨论此仓库的 RoboDojo/OpenVLA 项目，其他机器人/CAD/羽毛球项目不混入。

## 30 秒状态摘要

**当前研发位置：R45-D CPU 交付已完成，R45-E 运行时缺陷修复待完成。GPU Smoke、真实 GPU 吞吐基准和正式 5 Epoch 训练均未获授权，未运行。**

| 事项 | 状态 | 固定证据 |
|---|---|---|
| R44 训练器与 `joint_delta_v1` 数据契约 | IMPLEMENTED / CPU VERIFIED | [R44 trainer @ 7598a744](https://github.com/jho887891-sudo/robo/commit/7598a7449652a5d8dc9b6cca883fac3af0d2d56c) |
| R45-D 独立分支交付 | VERIFIED | [R45-D @ 0b35b13](https://github.com/jho887891-sudo/robo/commit/0b35b13f94d71ca15b6704a3b6bd0cffb2ad4e57) |
| R45 数据读取、缓存与计时模块 | IMPLEMENTED，训练路径仍有运行时缺陷 | [r45_data.py](https://github.com/jho887891-sudo/robo/blob/0b35b13f94d71ca15b6704a3b6bd0cffb2ad4e57/training/r45_data.py) |
| R44/R45 CPU Gate | VERIFIED（报告口径） | [CPU 证据](https://github.com/jho887891-sudo/robo/tree/0b35b13f94d71ca15b6704a3b6bd0cffb2ad4e57/evidence/r45-cpu-gate-20261027) |
| R45-E 静态交叉审计发现 | VERIFIED CODE FINDINGS，尚未修复/复测 | 见下方“阻塞缺陷” |
| 实际 GPU 显存、samples/sec、提速倍数 | UNKNOWN / BLOCKED | **没有 GPU 实测** |
| OpenVLA-7B 完整 5 Epoch 训练 | BLOCKED | **未经授权，不得启动** |

### SHA 索引

- `main`：`329614f54d3090b9b6ab120855ba50fbc93bdd1e`。这是较早的纯 OpenVLA 闭环实验快照，**并不代表最新研发进度**。
- R44 代码：`experiment/r44-trainer-implementation-20261026@7598a7449652a5d8dc9b6cca883fac3af0d2d56c`。
- R45-D 代码与证据：`experiment/r45-integrated-throughput-20261027@0b35b13f94d71ca15b6704a3b6bd0cffb2ad4e57`。
- R45-D CPU 代码快照：`3582134541b826e0ee2bd2072ebeb6786a342432`。最终 HEAD `0b35b13` 比该 SHA 只增加了归档 CPU 证据的提交；不要把它描述为新一轮代码测试。
- R45-B/C 是历史交叉审计，结论已并入本文；详情保留在 R45 分支的 `docs/r45_b_audit.md`、`docs/r45_c_baseline_audit.md`。

## 固定研究与数据契约

- OpenVLA-7B；基座冻结，训练 LoRA 和 42D 状态投影器。
- LoRA 目标按 R44 真实训练路径固定为 `q_proj/v_proj`、`r=32`、`alpha=16`，不得从 R45 某个配置类中的可选 `all-linear` 默认值推断正式配置。
- 输入：3 路 RGB，顺序 `cam_head, cam_left_wrist, cam_right_wrist`，外加 42D 状态历史特征。历史构造、归一化以基线代码和冻结元数据为准。
- 输出：14 维关节动作目标的离散编码（14 个 action token + STOP）；R44 目标 `joint_delta_v1`，即 `Delta_t = A_t - S_t`。TRAIN-only 分位数归一化。
- 数据：3 任务，TRAIN **108,780** 样本，DEV **28,268** 样本。回合与采样严格隔离。
- 目标：**5 个完整 Epoch**，每轮每个 TRAIN 样本恰好访问一次；不得以抽样训练代替完整 Epoch。
- 旧 20 个 Push-T 留出回合已被多轮模型选择使用，不可当作新的最终独立测试；离线关节 L1 不等于闭环任务成功率。

## R45-D 的实测/实现边界

R45-D GitHub 分支的 CPU 证据：

- R44 契约：**15/15 PASS**。
- R45 契约：**4/4 PASS**。
- R45 流水线、CPU 假模型和检查点测试：**12/12 PASS**。
- 未授权 GPU 基准：**退出码 2**，无产物目录。
- 运行环境报告：Python 3.10.12；Torch 2.5.1+cu124；Transformers 4.46.3；PEFT 0.13.2；h5py 3.16.0。
- 原始测试输出与退出码：[`evidence/r45-cpu-gate-20261027/`](https://github.com/jho887891-sudo/robo/tree/0b35b13f94d71ca15b6704a3b6bd0cffb2ad4e57/evidence/r45-cpu-gate-20261027/)。

**不能宣称：**真实 7B 前反向训练已 PASS、24 GB 显存足够、训练吞吐已加速、完整 Epoch 训练成功，或 GPU benchmark 已完成。

## 未解决的高优先级运行时问题（R45-E）

以下为针对 **R45-D SHA 0b35b13** 的源码核查，尚无 RED→GREEN 修复证据：

1. [`training/r44_train.py`](https://github.com/jho887891-sudo/robo/blob/0b35b13f94d71ca15b6704a3b6bd0cffb2ad4e57/training/r44_train.py)：TRAIN 循环仍使用旧 `episode_images(...)`，但日志引用本作用域没有赋值的 `data_timing`；且 `sample_start` 在数据加载后才开始，不能算端到端耗时。这会阻断正式训练路径。
2. 同文件的 `evaluate_dev(...)` 内引用未传入、未在本作用域创建的 `r45_reader`，DEV 路径有确定性未绑定变量错误。
3. [`diagnostics/r45_gpu_bench.py`](https://github.com/jho887891-sudo/robo/blob/0b35b13f94d71ca15b6704a3b6bd0cffb2ad4e57/diagnostics/r45_gpu_bench.py)：用 `time.sleep(0.001)` 模拟真实训练步骤，却可能输出 `PASS` 和 `samples_per_s`。**不能使用该脚本的输出作为 GPU 提速证据**；还需校验实际独立硬超时和 30 分钟预算约束。
4. [`training/r45_train.py`](https://github.com/jho887891-sudo/robo/blob/0b35b13f94d71ca15b6704a3b6bd0cffb2ad4e57/training/r45_train.py)：`build_model("real")` 明确不可用，该文件目前是 CPU 实验入口；中途停止检查点把游标写成 `len(idxs)`，有漏样风险；同时应检查不完整累积批次的处理。
5. [`training/r45_contract.py`](https://github.com/jho887891-sudo/robo/blob/0b35b13f94d71ca15b6704a3b6bd0cffb2ad4e57/training/r45_contract.py)：默认 `all-linear` 与正式 R44 `q_proj/v_proj` 不一致。必须明确区分实验配置与已批准的正式训练配置。
6. R45 `RuntimeReader` 有缓存，但正式 R44 TRAIN 循环尚未真正使用其图像接口；不能推断已有多 Worker 预取加速。应检查文件句柄上限、释放与 CUDA 异步计时。

当前任务：**R45-E 最小修复 + 真实调用路径 CPU 回归**，先 RED 再 GREEN；不要只用源代码字符串存在性作为集成测试。

推荐新审计分支：`audit/r45-training-runtime-fixes-20261027`，基线严格采用 `0b35b13f94d71ca15b6704a3b6bd0cffb2ad4e57`，执行前核对其是否已经存在更晚审计提交。

### GPU Gate，尚未获授权

R45-E 完成并复核后，单独申请以下资源，**申请不等于授权**：

1. 一次性真实 OpenVLA-7B 三相机 + 42D 前向/反向/优化器 GPU Smoke，实测峰值显存。
2. 最多 30 分钟单卡、强制外部超时的稳定吞吐量基准，比较相同样本/相同指标下的基线与优化路径。
3. 仅在实测吞吐、训练器恢复、GPU 安全和预算全部通过后，再由研究负责人授权完整 5 Epoch 训练。

禁止自行改用模拟 `sleep`、CPU 测试、开发集指标或历史估算替代真实 GPU 验证。

## 证据入口与清理原则

**常读的只有本文；历史报告按需点链接。** 这是减少窗口交接成本的主要办法，避免新建 R46、R47 文档重新描述此前所有步骤。

| 项目 | 处理方式 | 原因 |
|---|---|---|
| `README.md` 历史闭环报告 | 保留，提供本文件入口 | 它记录的是较早闭环证据，不是最新状态 |
| `docs/r45_b_audit.md`、`docs/r45_c_baseline_audit.md` | 保留作为历史证据，普通接手不重复阅读 | SHA 冲突已解决 |
| `docs/superpowers/plans/`、`specs/` | 当历史设计资料，不作为唯一状态来源 | 减少重复计划 |
| `training/r45_train.py` CPU 假模型入口 | 待最小修复后评估改名/归档，**现在不删** | 存在真实测试依赖 |
| `diagnostics/r45_gpu_bench.py` | 原位修复/替换占位逻辑，禁止直接删历史证据 | 防止性能结果造假 |
| `evidence/`、旧失败原始日志、全部 HDF5/权重/检查点 | **绝对不做自动删除** | 研究可追溯性 |
| 大量旧分支 | 先单独盘点可达性和 SHA 依赖，列表审批后再归档/删除 | 避免丢失唯一证据 |

**未经明确列出路径并核验唯一来源前，不删除代码、旧分支或历史文件。** 可安全清理的临时缓存、空目录、已被跟踪且可重建的测试输出，必须限定在独立工作区内操作。

## 每次完成任务如何更新本文

更新该文档时：

1. 先读最新 GitHub 提交及原始证据，固定 SHA，区分 VERIFIED / IMPLEMENTED / BLOCKED / HYPOTHESIS。
2. 更新顶部状态、SHA 索引、当前缺陷和唯一下一步；已经解决的缺陷移到历史证据链接或删掉**状态摘要中的重复文字**，不要删原始失败证据。
3. 仅在独立分支提交文档与最小测试；禁止未经授权修改 `main`。
4. 不把新建提示词或聊天口头报告当作 GitHub 交付；必须记录固定 SHA、日志和失败原因。
5. GitHub 上任何文件都不是自动实时同步的聊天记忆。新会话应重新读取最新版本，并核验分支 HEAD。

## 新会话恢复指令

> 继续 `jho887891-sudo/robo` OpenVLA/RoboDojo 项目。请先通过 GitHub 读取根目录 `AGENTS.md` 和 `docs/PROJECT_STATUS.md`，核验其中 R44/R45 分支 HEAD 是否更新，按 VERIFIED / IMPLEMENTED / BLOCKED / HYPOTHESIS 接续唯一下一步。不要重做已留证通过的阶段；不改 main、不运行未经批准的 GPU/正式训练、不覆盖 HDF5/权重/检查点。只在研究关键决策时找我。
