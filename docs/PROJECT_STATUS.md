# RoboDojo / OpenVLA 项目状态（唯一日常交接入口）

> 状态快照：2026-10-28。此文档是**项目索引**，不是无需核验的最新运行结果。
> 仓库：[`jho887891-sudo/robo`](https://github.com/jho887891-sudo/robo)。
> 每次接手先读取本文件及根目录 `AGENTS.md`，随后通过 GitHub 获取分支最新完整 SHA；以实际代码、原始日志和可复算证据为准。
> 只讨论此仓库的 RoboDojo/OpenVLA 项目，其他机器人/CAD/羽毛球项目不混入。

## 30 秒状态摘要

**当前研发位置：R45-F 服务器真实 TRAIN 三相机 CPU I/O profile 已回传 PASS（退出码 0），但正式 R44 全套 CPU preflight 仍需核对模型、normalizer、manifest 和 OFT 路径；真实 7B GPU Smoke、性能基准和 5 Epoch 正式训练均未授权。PR #51 继续作为唯一待合并集成分支。**

| 事项 | 状态 | 固定证据 |
|---|---|---|
| R44 训练器与 `joint_delta_v1` 数据契约 | IMPLEMENTED / CPU VERIFIED | [R44 trainer @ 7598a744](https://github.com/jho887891-sudo/robo/commit/7598a7449652a5d8dc9b6cca883fac3af0d2d56c) |
| R45-D 独立分支交付 | VERIFIED | [R45-D @ 0b35b13](https://github.com/jho887891-sudo/robo/commit/0b35b13f94d71ca15b6704a3b6bd0cffb2ad4e57) |
| R45 数据读取、缓存与计时模块 | IMPLEMENTED / GitHub CPU CI VERIFIED；真实 CPU I/O proxy 已回传 PASS | [PR #51](https://github.com/jho887891-sudo/robo/pull/51) |
| R44/R45 CPU Gate | VERIFIED（报告口径） | [CPU 证据](https://github.com/jho887891-sudo/robo/tree/0b35b13f94d71ca15b6704a3b6bd0cffb2ad4e57/evidence/r45-cpu-gate-20261027) |
| R45-E 静态交叉审计发现 | 主要代码问题已在 PR #51 修复并通过 CPU 回归；原 R45-D 历史缺陷仍保留 | [PR #51](https://github.com/jho887891-sudo/robo/pull/51) |
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

## R45-E 直接修复交付（2026-10-27）

- **修复 PR（独立审计分支）**：[#50](https://github.com/jho887891-sudo/robo/pull/50)，分支 `audit/r45-direct-runtime-fixes-20261027`，代码/CPU CI 固定 SHA `a9685ae53a39d90589ca1a5d97bdb039e1ce4549`，以 R45-D `0b35b13` 为基线，**尚未合并到 R45 正式分支**。
- **VERIFIED CPU**：GitHub Actions [R45 Runtime CPU Gate run 40477763136](https://github.com/jho887891-sudo/robo/actions/runs/40477763136) 全部成功；真实运行的 CPU 测试：R44 delta 15/15、R45 contract 4/4、运行时回归 6/6、**合成 HDF5 + Torch2.5.1 CPU 假模型训练/中断恢复 3/3**；Python 编译检查通过。实验仅用 GitHub CPU runner，未启动用户训练服务器 GPU。
- **已修复**：正式 R44 TRAIN 的 Reader 和 `data_timing` 绑定、完整样本计时与 GPU 同步、DEV reader 注入；RuntimeReader 的句柄上限与缓存 LRU、相互独立的阶段计时；R45 假模型的已提交优化器步数游标、梯度累积尾批次、恢复契约；正式 LoRA `q_proj/v_proj`；模拟 GPU Benchmark 永久拒绝 `PASS`，已授权但尚未实现真实引擎时退出码 3。
- **尚不可声称**：CPU stub/假模型通过并不等于 OpenVLA-7B GPU forward/backward/显存/性能验证。新的 benchmark 刻意保持 `BLOCKED_NOT_IMPLEMENTED`，不会输出伪吞吐或初始化 CUDA。
- **下一个 Gate**：在真实服务器复核 PR #50 CPU 契约与环境、确认修复能安全进入正式 R44 后，以独立审批的 GPU Smoke 验证 7B 真实计算。未获研究负责人授权不启动任何 GPU 任务。不要重做已通过的 GitHub 6+15+4+3 CPU CI。
- **合并限制**：本 PR 以 R45 分支为目标，尚未合并；代码未经合并前，下节问题描述仍适用于冻结的 **R45-D SHA `0b35b13`**。

## R45-F 真实 TRAIN CPU I/O 现场复核报告（2026-10-28，最新）

- **执行端报告：PASS，退出码 0**。固定执行 SHA `5ddddc1b4b926cb9ee467c1fa7305aff935e2314`；服务器隔离 worktree `/tmp/robo-r45f-cpu-real-20261028` 干净，未修改仓库/数据、未接触 GPU。**以下性能数值由 DeepSeek 现场执行回传，我已核验 GitHub 工具实现，但尚未直接取得服务器原始 report 字节做独立复算。**
- **数据定位**：`/home/T7/ojh/robot_actual/data/data/hdf5`；执行端确认位于 `/dev/vdb2` 的 `fuseblk` 已挂载目录；TRAIN **240/240**、DEV **60/60** 文件存在且非空；真实 TRAIN 建索引 **108,780** 样本。
- **CPU profile 固定样本**：三任务各 40，共 120 个 TRAIN 样本，重复两次 **240 次**；旧逐次重开 HDF5 路径和 R45 Reader 的三相机**恒等 CPU 图像变换代理**张量相同。此检查**不包含生产 resize/normalize，也不等价于完整训练数据契约预检**。
- **读图时延（报告值）**：旧路径中位数 **44.040 ms/样本**，R45 Reader 中位数 **20.677 ms/样本**，中位比值约 **2.13×**（两条独立统计量的比值，不是严格配对提速）；HDF5 JPEG 字节读取均值 **1.259 ms**，JPEG 解码与代理变换 **13.303 ms**，CPU tensor 准备 **7.488 ms**。
- **缓存与 Worker（报告值）**：256 MiB 解码图像缓存命中 **0/240**，120 张三相机张量约 **331,776,000 B**，大于缓存上限 **268,435,456 B**；只读 1/2/4 worker 的解码代理吞吐 **53.6/123.5/157.2 样本/s**（含启动）。该 Worker 任务**没有接入正式训练器**，不宣称生产训练可以达到对应数值。
- **真实现场证据**：`/tmp/r45f-cpu-execution-20261028/`；`data-root-discovery-20261028T074515Z.json`、`profile-20261028T075000Z/report.json`、`samples.jsonl`、`selected_samples.json`、目录内与外层 `SHA256SUMS`，执行端报告校验全部通过；如需独立 VERIFY，需取得原始 JSON、SHA256 内容和执行日志。
- **由结果驱动的 CPU-only 代码变更**：正式 `training/r44_train.py` 中 `RuntimeReader(cache_bytes=0)`，保留 HDF5 句柄复用，因为完整 TRAIN 每样本一轮仅访问一次，不值得让大缓存留存无法复用的解码张量；`diagnostics/r44_gpu_smoke.py` 改为使用相同 Reader 以提高真实训练与 Smoke 输入路径一致性，修正直接运行的项目根导入路径；新增 CPU 回归测试。**这些代码变更尚无 GPU 运行结果、加速收益仍 UNKNOWN**。
- **GitHub CPU 验证**：[R45 Runtime CPU Gate run 40652969926](https://github.com/jho887891-sudo/robo/actions/runs/40652969926) **36/36 PASS**（发现工具 3、R44 15、R45 contract 4、运行时 9、合成 JPEG profile 2、Toy stop-resume 3），代码 SHA `96cf84e522914e1e5934ef27e202d4cff0c6d131`；R4/R5 CPU workflow 同样 PASS。该 CI 验证了未授权 GPU Smoke 入口退出码 2、真 R45 Reader 源码接入和测试语法，**没有实际启动 GPU Smoke**。
- **唯一下一步**：已发现原始数据，但 `M2_NORMALIZER_JSON`、`R44_DELTA_NORMALIZER`、`R44_DATASET_MANIFEST`、`R44_BASE_MODEL`、`R44_OFT_ROOT` 尚未经过当前服务器的完整路径/身份验证；先只读核验已有 R44 产物与模型位置，完成 `scripts/r45_server_cpu_gate.sh` 的完整 CPU preflight。通过后再由研究负责人单独审批一次性真实 7B GPU Smoke。GPU 吞吐与正式训练都不能提前启动。

## 2026-10-28 R45-F 现场结果：环境/挂载阻塞

- **真实服务器现场：BLOCKED**。DeepSeek 在干净隔离 worktree `/tmp/robo-r45-cpu-io-20261028` 检出固定 SHA `9c142448c83ee0ab08e8f990fb84d759e8f2afb3`；`scripts/r45_server_cpu_gate.sh` 因缺失 `M2_HDF5_ROOT` 退出码 **2**；其他必需模型/normalizer/manifest/OFT 路径变量亦未设置。没有执行真实 TRAIN HDF5/JPEG profile，GPU 未启动。
- 执行端回传的只读证据目录：`/tmp/r45-cpu-io-attempt-20261028/`（尚未从服务器直接获取，用户回传的运行声明与 GitHub CI 是不同证据层级）。
- **GitHub 代码线索**：[data/README.md](https://github.com/jho887891-sudo/robo/blob/audit/r45-e-unified-cpu-profile-20261028/data/README.md) 和 [diagnostics/r44_prepare_delta.py](https://github.com/jho887891-sudo/robo/blob/audit/r45-e-unified-cpu-profile-20261028/diagnostics/r44_prepare_delta.py) 曾使用 `/home/T7/ojh/robot_actual/data/data/hdf5`。**仅是历史候选路径，不代表当前主机上存在或者已挂载**。
- **ChatGPT 已新增** `diagnostics/r45_discover_data_root.py`：不需要预先设置环境变量，**只读**检查历史候选与手动提供的根目录下全部冻结 TRAIN/DEV 文件是否存在且非空；输出结构化 JSON。不会漫游整盘、修改挂载、接触 CUDA 或自动下载数据。该清单 Gate 即便 PASS，也仍需 HDF5 内容/hash/数据契约进一步验证。
- 优先只做路径发现（可额外经 DeepSeek 获取只读挂载信息）；**一旦唯一完整 HDF5 根目录核验存在，直接运行已有的 `diagnostics/r45_cpu_io_profile.py`，它仅依赖 HDF5 根目录与仓库的 `data/m2_splits.json`，无需先配置模型、OFT 或四份 normalizer。** 正式 R44 全套 CPU Gate 仍须全部变量具备，不能用 I/O 代理结果替代。
- 未确认真实目录或主机缺少正确挂载时，状态保持 BLOCKED；不得自动猜路径、造假样本、更改服务器挂载、复制数据或启动 GPU。

## R45-E 统一 CPU Gate 验证（2026-10-28）

- 唯一集成 PR：[**#51**](https://github.com/jho887891-sudo/robo/pull/51)，已通过 Gate 的数据根目录发现工具代码 SHA `1b85b2762220a080be6491ce106f8f7234a99e9a`，基础分支仍为 R45-D `0b35b13`。
- **GitHub CI 已实测 PASS**：[R45 Runtime CPU Gate run 40647917075](https://github.com/jho887891-sudo/robo/actions/runs/40647917075)：数据根目录发现/缺失/不覆盖测试 3/3、R44 15/15、R45 contract 4/4、运行时 6/6、合成真实 JPEG 的 HDF5 CPU profile 2/2、Toy 停止恢复 3/3，合计 **33/33**；R4、R5 CPU 工作流也 PASS。均为 GitHub CPU runner，不代表服务器真实 TRAIN 数据已经测过。
- 原 [PR #50](https://github.com/jho887891-sudo/robo/pull/50) 因被 #51 完整覆盖，已关闭未合并；历史 Git commit、测试证据保留。今后不在 #50 继续追加修复。
- **历史阶段下一步已推进**：真实 TRAIN CPU I/O proxy 已在 R45-F 回传 PASS；现在先定位剩余 R44 normalizer/manifest/7B model/OFT 文件并完成完整 CPU preflight。GPU 一律 BLOCKED。
- 真实 CPU profile 只能判断 HDF5/JPEG/缓存/Worker 的 I/O 代理瓶颈。正式模型的图像预处理、CPU→GPU、前反向仍未测；不能把 CPU 图像代理的速度当训练加速倍率。

## 2026-10-28 R45-E 交叉审计与唯一集成分支

- DeepSeek 的 R45-E 实际提交 `audit/r45-training-runtime-fixes-20261027@5da8f97c811cfe9e2d790583c68c0d9a985903e2` 已核验 GitHub 远端存在。其 R44 14/14、R45 pipeline 12/12、contract 4/4 的单项原始日志支持独立测试有通过。**但该 SHA 仍有未绑定的 DEV `r45_reader`、训练样本时间遗漏 state/label 阶段、Toy 训练中途累积批次恢复风险，不能直接称整个训练路径 VERIFIED。**
- 该 SHA 的 `evidence/r45-e-cpu-20261027/` 仅有 `pytest.log`、`unittest.log`、`r44_train_core.log`、`r45_contract.log`、`r45_pipeline.log`。其中 pytest 不可用、unittest 有三个导入错误；**GitHub 当前目录未找到所报告的 SHA256 清单**，留待原执行者补足证据，不能捏造为通过。
- 我之前直接修复的 [PR #50](https://github.com/jho887891-sudo/robo/pull/50) 已得到 GitHub CPU Gate（含真实 JPEG 之外的合成 HDF5 CPU Toy stop-resume）通过，但它和 DeepSeek R45-E 形成两条并行分支。
- 为避免重复维护：**以 [PR #51](https://github.com/jho887891-sudo/robo/pull/51) 分支 `audit/r45-e-unified-cpu-profile-20261028` 为唯一待验收集成入口。** 它从 DeepSeek E 提交出发、保留旧日志、拣选此前已验证的 TRAIN/DEV/Reader/Benchmark/恢复修复，并新增真实 TRAIN CPU profile 工具及测试。待该 PR 全部 Gate 通过，再关闭被替代的 PR #50（不删除其 commit/证据）。
- 新工具 `diagnostics/r45_cpu_io_profile.py`：按固定 SHA256 选择每任务 TRAIN 样本，比较 HDF5 每样本重开+JPEG 的 CPU Shadow 与有界 Reader；检验像素完全一致，测量冷/暖缓存、CPU host 时间、1/2/4 进程独立读解码的耗时，输出 `report.json`、`samples.jsonl`、`selected_samples.json` 和 `SHA256SUMS`。**严格不代表正式 OpenVLA image processor / GPU training 的速度。**
- 真实 TRAIN CPU profile **尚未在服务器执行，真实性能数字 UNKNOWN**；所有 GPU 实验继续 BLOCKED。当前任务是先通过 PR #51 CPU CI，再由 DeepSeek 仅在服务器使用该代码和受控 CPU 预算跑固定真实 TRAIN 样本；交回原始报告和完整 SHA。不要重复 R44/R45 已有训练设计。

## R45-F：服务器实际数据 CPU Gate（已提供命令入口，执行 UNKNOWN）

在 [PR #51](https://github.com/jho887891-sudo/robo/pull/51) 的统一代码分支 `audit/r45-e-unified-cpu-profile-20261028` 已提交 `scripts/r45_server_cpu_gate.sh`；在真实 Linux 服务器只需用**全新干净 worktree 和当时最新 PR 完整 SHA** 执行这个脚本。脚本校验固定 SHA、真实 R44 数据/normalizer/model config、R44 `--preflight-only` 与 R44/R45 CPU 契约及合成 HDF5 停止恢复，导出 `exit_codes.txt`、原始日志、SHA256SUMS。不会创建 CUDA 上下文，也不会正式训练。

运行前必须显式提供实际服务器路径（从既有可验证证据查找，不能猜测）：`M2_HDF5_ROOT`、`M2_NORMALIZER_JSON`、`R44_DELTA_NORMALIZER`、`R44_DATASET_MANIFEST`、`R44_BASE_MODEL`、`R44_OFT_ROOT`，以及 `R45_EXPECTED_SHA`。生成的 CPU 证据目录名由脚本输出。

**分工：** ChatGPT 负责 GitHub 代码修复、PR、测试与审计；DeepSeek 负责真实 Linux 服务器的干净 worktree 和 CPU 现场执行，并在获明确授权后运行 GPU 实验。

**此脚本通过之前不得启动 GPU。** 通过之后下一步是准备可执行的真实 7B GPU Smoke；现有 `diagnostics/r45_gpu_bench.py` 故意 `BLOCKED_NOT_IMPLEMENTED`，不能直接测出吞吐。此阶段的运行状态始终与 GitHub CI 状态分开记录。

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

## 五 Epoch 训练提速执行路线（当前 PLAN）

**科学目标不变**：三任务 TRAIN=108780、DEV=28268；5 个完整 Epoch，总计 543900 次 TRAIN 样本访问。保留 3 路相机、42D proprio、14D `joint_delta_v1`、OpenVLA-7B + LoRA `r=32/alpha=16/q_proj,v_proj`。正式训练样本覆盖与优化器更新次数分别报告；不得以少跑样本、改模型、只测 CPU 或只测单步冒充加速。

| 阶段 | 具体行动 | 成本 / 依赖 | Gate 与交付 |
|---|---|---|---|
| **P0: R45-E 止损修复（现在）** | 修复 TRAIN `data_timing` 未绑定、DEV `r45_reader` 未绑定；把 `sample_start` 移到数据准备之前；消除 `r45_gpu_bench.py` 的 `sleep(0.001)` 虚假 PASS；R45 假模型中断检查点游标与积累批次最小修复 | **CPU only**，不同缺陷可并行 | 实际进入 TRAIN/DEV 控制流程的 CPU stub RED→GREEN；静态字符串匹配不能单独算集成 PASS；独立审计分支、测试命令、日志、SHA |
| **P1: 数据通路 CPU 基准与保真** | 比较旧 `episode_images` 与 R45 Reader 固定样本输出；测 HDF5 open/read、JPEG、processor、CPU 数据准备、缓存命中与内存；安全测试 worker 预取和句柄生命周期，拒绝跨进程共享句柄；不改变采样顺序 | **CPU only**，可与 P0 无依赖测试并行 | 基线和优化图像/标签/特征完全符合相同契约；CPU 阶段分段时间和内存峰值留证；确认真实 TRAIN 接口实际调用优化后的 Reader |
| **P2: GPU Smoke（待明确授权）** | 真实 7B、3 相机、[1,42]、14 action tokens + STOP、bf16 前反向和一次临时 AdamW 更新；冻结基座与 LoRA 实际模块检查；峰值显存、loss/grad finite、资源保护、原始日志；不导出模型 | 需要研究负责人批准短时 GPU 窗口，独占/确认共享资源安全 | 不 OOM、接口/归一化不漂移、无其他进程受影响、显存余量满足 Gate；失败仅留证，不自动正式训练 |
| **P3: GPU 吞吐量 A/B（待单独授权）** | 用同一份冻结样本、同一模型 SHA/随机种子/优化语义，测旧 loader 和优化 loader；只在测出瓶颈后尝试兼容 SDPA/Attention、Micro-batch；同步或 CUDA Events 正确计时，区分 warm-up 与 steady；记录数据阶段、前向、反向、optimizer、峰值显存以及 end-to-end samples/sec | 建议申请**单卡最多 30 分钟总墙钟**（含加载、warm-up、收尾），独立外部超时；申请不代表授权 | 真实 forward/backward/update，绝不允许 `sleep` 占位性能作为 PASS；依实际计时判断 CPU I/O bound 还是 GPU compute/memory bound；不够样本或被截断标记 UNKNOWN |
| **P4: 模型计算进一步优化（有条件）** | 若 GPU 计算明显主导，检查支持的 SDPA/FlashAttention 路径、训练精度和 batch；若视觉塔完全冻结且可证等价，独立评估视觉特征缓存；gradient checkpointing 作为节省显存手段，可能牺牲速度；任何改变 optimizer 更新次数的方案单独标注语义差异 | CPU 兼容性验证后才申请额外少量 GPU 比较 | 数值/形状/目标一致、没有违背冻结模块与归一化约束；无收益则停止继续堆优化 |
| **P5: 重新估预算并申请正式训练** | 用真实 TRAIN steady samples/sec、全 DEV 耗时、模型载入、checkpoint I/O、资源边际估算 5 Epoch；按预算、显存、恢复回归和训练覆盖 Gate 决定是否进入正式训练 | 正式全量训练需用户**单独明确批准** | 给出可信 GPU-hours 范围、检查点/STOP/恢复计划、TRAIN/DEV/HOLDOUT 隔离与独立评估；未批准不得启动 |

### 如何判断优化方向

- **CPU/I/O 瓶颈**：数据准备占比较高，先 HDF5 句柄复用/解码多进程/预取、减少重复读取；避免把无命中收益的 256 MB 缓存当作已验证提速。
- **GPU 计算瓶颈**：forward/backward 占大头，优先兼容 Attention kernel，之后测有效 micro-batch 与梯度累积；不要仅凭每步更短就认定总训练时间更短。
- **显存瓶颈**：先测峰值及余量，再评估 batch 或 checkpointing；`set_per_process_memory_fraction` 只限 PyTorch 分配器，不等于独占/硬显存隔离。
- **GPU Smoke 先于性能测试**，没有真实模型通过单步烟测不得进入大规模基准。
- **相同优化语义下**以完整样本/墙钟秒为首要比较指标；把 warm-up、预处理、加载、DEV、checkpoint 分开。若 Batch 导致优化器更新次数与学习率调度变化，另列非等价对照。
- 老实验的 **~605 GPU-hours** 是用其他训练路径的旧单步耗时外推，**不是当前 R45 的实测结论**。真实数据缺失时总预算必须为 UNKNOWN。

预计正式训练计算：
`train_seconds = 543900 / measured_end_to_end_train_samples_per_second`；
还应加入 5 轮 DEV、模型加载/恢复、checkpoint/文件校验、可观测的资源开销与审定安全余量。以上没有实测吞吐前只是公式，不是承诺。

**持久化原则**：每个阶段只更新本文件的状态、SHA、唯一下一步；完整原始日志放 evidence。当前唯一下一步为 R45 正式训练器的真实数据契约 CPU preflight（找齐已有 normalizer、manifest、7B 模型与 OFT）；未经研究负责人同意不得自动运行 GPU Smoke、GPU Benchmark 或正式 5 Epoch。

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
