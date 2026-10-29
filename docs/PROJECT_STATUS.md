# RoboDojo / OpenVLA 项目状态（唯一日常交接入口）

> 状态快照：2026-10-29。此文档是**项目索引**，不是无需核验的最新运行结果。
> 仓库：[`jho887891-sudo/robo`](https://github.com/jho887891-sudo/robo)。
> 每次接手先读取本文件及根目录 `AGENTS.md`，随后通过 GitHub 获取分支最新完整 SHA；以实际代码、原始日志和可复算证据为准。
> 只讨论此仓库的 RoboDojo/OpenVLA 项目，其他机器人/CAD/羽毛球项目不混入。

## R46 CPU-only Smoke 安全准备（2026-10-29，最新状态）

- **实现分支**：`codex/r46-smoke-supervisor-20261029`，最终 CPU Gate 代码 SHA `b7b32790ece97c560b1164eb96b472aa26c9dbde`；参数桥接修复 `a9f2618cd9ec2707a71dde43e07096c99ce9c3ea`，PID 与审批时间边界修复 `beca55e633c27fea152d3db4aac0eb9ab2ba2c2a`。基于 PR #51 head `ba563d0f8c638b089f588027dd0af41ff290e1f4`。草稿 [PR #53](https://github.com/jho887891-sudo/robo/pull/53)，目标分支 `audit/r45-e-unified-cpu-profile-20261028`。
- **VERIFIED Linux CPU Gate**：最终 SHA 的干净 Linux worktree 上，12/12 命令 exit 0，93/93 R44/R45/R46 测试通过，无跳过。Python 3.10.12、NumPy 1.26.4、Torch 2.5.1+cu124、Transformers 4.46.3、PEFT 0.13.2、h5py 3.16.0、OpenCV 4.13.0。证据见 [`linux-final-b7b3279`](../evidence/r46-cpu-20261029/linux-final-b7b3279/)；SHA256SUMS 已校验。
- **真实样本计划**：固定 5 个真实 push_T TRAIN episode（1 warm-up + 4 measured），CPU reader/processor 输出 `[1,18,224,224]`、42D 特征、14 action token + STOP、HDF5 SHA 与严格 Manifest 一致；计划 SHA `67458ab3c6b23518c8616ec229ff62ed9af51d3d6bc817d54b21cdd109937d4f`。证据：`sample-freeze-final/`。该结果不是 GPU smoke 或吞吐验证。
- **IMPLEMENTED**：外部审批校验、父进程 1800 秒硬上限、Linux 自有 session 信号清理、GPU UUID/资源独占准入、CUDA 前 OpenVLA 模型文件与索引重哈希、固定样本 runner 及 CPU 回归已实现。发现的 PDEATHSIG 与实际模型哈希缺口均保留 RED 日志并修复后通过。
- **STOP_WAIT_GPU_APPROVAL**：现有审批 JSON 仍是 `template_only=true/approved=false`；没有 CUDA context、GPU Smoke/benchmark、正式训练或部署。7B 权重加载、forward/backward、显存及 GPU throughput 仍 UNKNOWN。R45 stable benchmark 仍 `BLOCKED_NOT_IMPLEMENTED`；R48 `NOT_RUN`。
- **PR #53 review 修复**：移除了 wrapper 参数到 `R44.parse_args()` 的冲突；修复运行后当前 GPU PID 误判；审批需保证剩余窗口足够覆盖完整请求时长。三个修复均有 RED→GREEN CPU 回归，并已在最终 Linux CPU Gate 中通过。GPU Smoke 仍未运行，也没有 GPU 授权。
- **交付**：完整报告见 [`evidence/r46-cpu-20261029/AUDIT_REPORT.md`](../evidence/r46-cpu-20261029/AUDIT_REPORT.md)。此状态由本分支更新，未改 main。

## R45-MNP 正式收官历史摘要（2026-10-29，证据 `705e186d`）

- **E01 最终收官已在独立 evidence 分支入库**：[`evidence/r45-mnp-cpu-20261029@705e186d551c01490e45bfb642b3f56da68240fb`](https://github.com/jho887891-sudo/robo/tree/705e186d551c01490e45bfb642b3f56da68240fb/evidence/r45-mnp-evidence-20261029)。GitHub ref 已核对。新 `FINAL_REPORT.md` 与 `stage_status.json`（schema `v2`）明确覆盖旧中途快照，但保留 Git 历史。状态：`S00=PASS`, `M01=PASS`, `P01=PASS_CPU_GATE_ONLY`, `E01=PASS`；OpenVLA CPU 接口 `PASS_CPU_ONLY`；`N01` OFT 历史溯源 Gate 已移除；`N02=PARTIAL_PASS_PROVENANCE_UNKNOWN`。**未经授权没有 CUDA、真实 7B GPU 模型加载或训练**。
- **E01 脱敏与校验审计**：`redaction_manifest.json`（v2）现明确记录原始 Manifest SHA `a220bd45e49c623c3a9f77dcf7460c95f8305909c05122e2d2a0556bff522ead` → 发布版本 SHA `d031308b3c9615a375d6d5c6008983a03fd110865a4cf5591e1f46a42eed6a84`，原始报告 `ccf6df41…` → 发布报告 `c4755b02…`，原始与发布各有独立 SHA256SUMS；发布端记录 `sha256sum -c exit=0`。**ChatGPT 直接对照公开 Manifest 的前一提交和本提交：300/300 文件 key 相同、300/300 嵌入的 HDF5 SHA 字段相同**；未直接读取服务器上未脱敏的原始 HDF5 或 Manifest 字节，因此不冒称独立验真全部私有原件。
- **OpenCV/NumPy 真实样本核验**：`cpu-gate-rerun-20261029/opencv-numpy-real-sample/stdout.json`、`exit_code=0`：NumPy 1.26.4 / OpenCV 4.13.0 对 `push_T/episode_0000000.hdf5` t=2 三相机 JPEG 解码得到 3 个 RGB、contiguous `uint8`、`480×640×3`；仅证明该具体样本路径可用，不代表全数据或模型训练已验证。OpenCV distribution 元数据仍要求 NumPy>=2，风险记录保留。
- **GPU 方案是计划，不是已实现的五样本执行入口**：新 [`GPU_SMOKE_PLAN.md`](https://github.com/jho887891-sudo/robo/blob/705e186d551c01490e45bfb642b3f56da68240fb/evidence/r45-mnp-evidence-20261029/GPU_SMOKE_PLAN.md) 提议固定一个 TRAIN 任务五样本（预热 1 + 测量 4）、真实 7B 权重加载、LoRA/42D 前向反向、30 分钟**独立父进程硬停止**和待批准的 42 GiB free/48 GiB total 门槛。现有 `diagnostics/r45_gpu_bench.py` 仍明确 `BLOCKED_NOT_IMPLEMENTED`。
- **新增源码审计发现：不要从零假设没有真实 Smoke**。`diagnostics/r44_gpu_smoke.py` **已经实现**标准 OpenVLA-7B+LoRA+42D 的**单个可丢弃真实 GPU optimizer step**，包括 R45 三相机读取、权重哈希与显存日志；但**从未被本次 GPU 实测**，不满足新的 5 样本及严格 30 分钟进程级硬截止。其 `--max-wall-seconds` 只在部分阶段主动检查，没有父进程 timeout+SIGKILL；现有 GPU 资源检查与计划的新独占门槛也需统一。历史的代码实现 ≠ GPU Smoke PASS。
- **后续授权状态**：Issue #52 所要求的 R46 CPU-only 安全准备已经完成，详细结果见上方 R46 状态与审计报告。真实 GPU smoke 仍需研究负责人另行明确授权；正式 5 Epoch 训练没有授权。

> 该节为最新状态，以下旧 R45-MNP 的中途 BLOCKED 和旧版 OFT 工单只保留历史语境，不是当前待办。

## R45-MNP 正式 CPU Gate 已完成（2026-10-29，现场证据审计 PASS）

- **证据固定提交**：[`evidence/r45-mnp-cpu-20261029@d37da9023319a887a8f0cba4a5ccb1fa242dcbcc`](https://github.com/jho887891-sudo/robo/tree/d37da9023319a887a8f0cba4a5ccb1fa242dcbcc/evidence/r45-mnp-evidence-20261029/cpu-gate-rerun-20261029)。**执行源码完整 SHA**：`d5be4de94ea37f1d5a715d5d47b7ebb16e07b097`（从 clean detached worktree 的 `git_head.txt` / `code_sha.txt` 读取）；本次新提交是 evidence，并非训练源码变更。
- **正式 Gate 实际 PASS**：新证据 `gate.exit_code=0`、`gate.stdout` 有 `STATUS=PASS_CPU_GATE_ONLY`，`gate/exit_codes.txt` 六项均为 0（`packages`、`r44_real_data_preflight`、`r45_runtime`、`r45_toy_resume`、`r44_delta`、`r45_contract`），各测试原始 stderr 分别显示 **9/9、3/3、15/15、4/4，全计31/31**；preflight JSON `CPU_CONTRACTS_PASS`、重新核验全部 300 HDF5、TRAIN=108780、DEV=28268，`dataset_manifest_sha256=a220bd45e49c623c3a9f77dcf7460c95f8305909c05122e2d2a0556bff522ead` 对应服务器**未脱敏原始** Manifest，非 GitHub 发布脱敏版本。
- **环境统一、标准 OpenVLA 接口静态验证**：同一 Python 3.10.12、NumPy 1.26.4、h5py 3.16.0、Torch 2.5.1+cu124、Transformers 4.46.3、PEFT 0.13.2；本地 OpenVLA `model_type=openvla`、`PrismaticProcessor` 构造、`OpenVLAForActionPrediction` 动态类导入与 ARX 14D/14D/25 通过；`configuration_prismatic.py`、`modeling_prismatic.py`、`processing_prismatic.py` 的 runtime SHA 与冻结仓库参考逐项一致，**没有加载 7B 权重**。`--oft-root` 在固定旧训练脚本中仅作为 Prismatic 运行源码路径；**当前模型明确是标准 OpenVLA-7B + LoRA + 42D projector，不是 OFT**；不必调查五个候选的历史训练身份。
- **静态模型身份边界**：上一证据提交 `dc4fd308` 的 `model_identity.json` 已列出 3 个 safetensors 的 SHA 与索引覆盖，但历史上游权重来源仍 `FILE_IDENTITY_VERIFIED_PROVENANCE_UNCHECKED`。CPU Gate 没有权重装载/前向、CUDA 上下文、GPU Smoke、显存与 samples/s、optimizer step 或完整 5 Epoch 训练，**不得升格为 GPU_READY 或 TRAINING_VERIFIED**。CPU preflight 中的时间预算是估算，非实测。
- **依赖残留风险**：OpenCV 4.13 可在 NumPy 1.26.4 环境下导入，但包元数据要求 NumPy >=2；静态导入成功不等于所有图像解码和实际模型训练稳定。这一风险在后续独立 GPU Smoke 方案内核查，不改变当前 CPU Gate PASS。
- **证据记录待整理，不阻断已确证的 Gate PASS**：`cpu-gate-rerun-20261029/` 有单独的 SHA256SUMS、原始/发布件脱敏记录与服务端校验 exit 0。旧根目录 `FINAL_REPORT.md`、`stage_status.json`、若干 `status.txt` 仍描述前几次 BLOCKED，**属于历史快照，不能覆盖本次正式 Gate PASS**；旧根 `redaction_manifest.json` 对早期发布脱敏 Manifest 的原件/发布件映射仍不完整，应补充可追溯说明而非改写旧运行结果。服务器上完整原始 Manifest 仍应保留，公开版含路径占位符，不能作为训练器输入。
- **下一步只在单独批准后进行**：先完成证据汇总状态修正和后续 GPU Smoke 规范（权重实际加载/前向、显存峰值、分辨率/相机、速度与 OpenCV/NumPy 兼容性、可靠停止），再由负责人单独授权真实 GPU 运行。**未授权 GPU，正式 5 Epoch 训练仍 BLOCKED**，且 `main` 不变。

> 本节为当前 R45-MNP CPU Gate 的**最新权威进展**，覆盖以下旧小节“CPU Gate BLOCKED / 尚缺数据 Manifest / OFT 历史溯源”的过时状态，旧小节保留以记录当时的证据快照。

## R45-MNP 最新现场证据交叉审计（2026-10-29，覆盖下方旧待办状态）

**最新 evidence 分支 SHA：[`dc4fd308ae050988c5ab3c93120fa6f8524cf019`](https://github.com/jho887891-sudo/robo/tree/dc4fd308ae050988c5ab3c93120fa6f8524cf019/evidence/r45-mnp-evidence-20261029)**；执行固定代码 SHA `d5be4de94ea37f1d5a715d5d47b7ebb16e07b097`（PR #51）。已直接读取 `FINAL_REPORT.md`、`stage_status.json`、M01 的 `report.json`/严格 Manifest、N02 模型身份报告和 P01 的原始 traceback；**未独立读取/重算服务器上 300 个 HDF5 原始字节**。

- **M01 已有真实完成报告，按执行端证据为 PASS**：`CPU_MANIFEST_REATTEST_PASS`、300 HDF5 文件记录、三个任务各 100 回合、TRAIN 108780 / DEV 28268、NumPy 1.26.4/h5py 3.16.0、冻结和重算 delta payload 均 `9f7c4f00fc2945825cc892493c21127b5e38d4f83a212bf91f7cf410776ba459`；**不能继续沿用下面旧快照中的“M01 未执行/失败”**。正式 Gate 需要的是服务器保留的**原始**严格 Manifest，不是 GitHub 已替换路径的脱敏版。
- **N02 已有静态权重 SHA**：`model_identity.json` 记录 3 片 safetensors 的完整文件名、大小、SHA256，索引覆盖；状态明确为 `FILE_IDENTITY_VERIFIED_PROVENANCE_UNCHECKED`，不表示历史来源、7B CUDA 模型加载或运行成功。**模型目标为标准 OpenVLA-7B + LoRA + 42D projector，不是 OFT 模型。历史五个 OFT 候选来源选择不再是必要 Gate。**
- **P01 实际运行但未通过**：`commands/p01.exit_code=130`，`gate/r45-f-cpu/exit_codes.txt` 只有 `packages=0`；Python 的 `KeyboardInterrupt` 发生在 `training/r44_train.py:355` 重新 SHA256 读取 HDF5 时，**不能断言文件损坏，也不能把它当作完整 CPU Gate PASS**。运行 Gate 的系统 Python 依赖为 NumPy **2.2.6**、Torch 2.5.1+cu124、Transformers 4.46.3、PEFT 0.13.2；之前通过 M01 的隔离 Python 是 NumPy **1.26.4**，但不具备完整训练依赖。必须选用单一满足冻结数值与运行时依赖的 CPU 环境，并在执行前确认 `which python3`、`python3 -c 'import numpy,torch,transformers,peft; ...'`。
- **源码兼容性仍待验证**：冻结 `training/r44_train.py` 与 `scripts/r45_server_cpu_gate.sh` 仍强制 `--oft-root` / `R44_OFT_ROOT` 并从其中导入 `prismatic` 模块。这是代码运行依赖问题，不是要求选 OFT 模型，也不能用无来源的占位目录绕过。当前 Gate 在这个路径被实际验证之前即被中断；运行兼容性 UNKNOWN。
- **证据完整性待修正，不重跑 34+ GB HDF5**：发布的 `r44_dataset_manifest.json` 有 **300 个 `<REDACTED_LOCAL_PATH>`**，其 `manifest/manifest-reattest/SHA256SUMS` 所记原件 SHA（Manifest `a220bd45…`、report `ccf6df41…`）与仓库根 `SHA256SUMS` 中发布文件的 SHA（`d031308b…`、`c4755b02…`）不同；`redaction_manifest.json` 的 `files=[]` 没有记录原件→发布件映射。旧 `manifest/status.txt`、`model/status.txt`、`gate/status.txt`、`environment/status.txt` 保留首次阻塞快照，与新版 `stage_status.json` 不一致，必须标为历史快照或发布单独补充说明。**执行端报告 sha256sum -c exit 0 不等于确认脱敏文件与服务器原件字节一致**。
- **唯一下一步**：①补充证据版本和原件/发布件 SHA 对照（不修改原始证据）；②在不覆盖既有环境的前提下确认能同时满足 NumPy 1.26.4 和 Torch/Transformers/PEFT 的实际 CPU 解释器；③确认标准 OpenVLA 需要的 `prismatic` 模块导入接口及现有参数处理，必要时独立代码修复+CPU 回归；④只重跑尚未完成的 `scripts/r45_server_cpu_gate.sh`（真实脚本会再次扫描全部 HDF5，不能伪称省去该检查），提交原始 exit/log，再复核。**未授权 GPU，正式训练仍 BLOCKED**。

> 此小节替代下文旧 R45-L、旧 R45-MNP SPEC 与 30 秒摘要中关于“尚无严格 Manifest/模型权重 SHA、必须确定 OFT 历史唯一来源”的过时任务描述；旧文保留历史记录，不代表当前动作。

## R45-K GitHub 原始证据已入库与交叉审计（2026-10-28，VERIFIED）

- **PUBLISHED / GitHub HEAD VERIFIED**：独立分支 [`evidence/r45ij-quantile-20261028`](https://github.com/jho887891-sudo/robo/tree/evidence/r45ij-quantile-20261028) HEAD = `246cac50021f26f4d0a192025c9128479fecd413`，GitHub commit 直接返回 39 个新增文件（其中 1 个为根 SHA256SUMS），`SHA256SUMS` 记录 **38 个路径、无重复**。DeepSeek 报告 push exit 0、服务端 sha256sum 38/38 退出码 0；GitHub 远端 ref/commit 已由 ChatGPT 直接读取。不能仅由 GitHub API 独立证明之前本地 git push 的进程退出码。
- **SHA256 独立复算**：ChatGPT 从 GitHub 读取文件原文字节对应的 UTF-8 内容，独立计算根 `SHA256SUMS` 的 SHA = `86ca1b9c938df3cd56869a75fc89079b9b659f39625202c6691bf284759a88cb`，与执行端相同；另抽检并复算 **7/7** 关键文件：`evidence_index.json`、`redaction_manifest.json`、R45-I/J `report.json`、历史 `normalizer.json`、脱敏 `environment.txt` 和根清单自身，均与根清单匹配。**这不是 ChatGPT 独立复算 38/38，更不是服务器原件全部逐字节已被直接核验**。
- **原始内容一致**：R45-I/J 原始报告都返回 `BLOCKED_DELTA_MISMATCH`、R45-I/J 代码 SHA 分别为 `e91c30ef31caa7e3fa37f5ada616a23ee77964ed` 和 `675ffe99fac7f57584bfbcfc23756755eaad02ab`，报告 NumPy 2.2.6、300 回合、TRAIN 108780 / DEV 28268、冻结 normalizer 文件 SHA `795052b0e6716d374488296b52befdbb012bd1840a7ccca57dfbf4df9e07383f`。R45-J 数值指纹三种候选路径各 `84/84` 精确匹配 frozen q01/q99；**多种候选，不存在唯一历史计算路径证明**。
- **历史边界核验**：历史 R44 `report.json` 状态 `BLOCKED_BUDGET`、`gpu_hours_available=0`、未训练；其 `training_config.json` 是旧 `LoRA alpha=32` / `epochs_min=2`，不是当前 R44 冻结 `alpha=16` / 5 个完整 Epoch 配置。历史 `environment.txt.redacted` 是唯一注明脱敏的文件，原始文件 SHA 与提交 SHA 分别记录在 `redaction_manifest.json`，保留 Python 3.10.12 / NumPy 1.26.4；**不能将脱敏后的 SHA 说成原始文件 SHA**。
- **仍明确缺失**：历史正式 preflight 未记录运行源码 SHA，候选历史 worktree SHA `fd0a7b87813dd55bb330a0f722a8ba6f69cbf5e3` 不是运行身份链；历史 NumPy 1.26.4 在**同一真实 TRAIN 数据上按原始 `np.quantile(float32)` 入口的复算**尚未执行；缺失正式严格 manifest、OFT 正式根、模型权重完整 SHA 和正式 CPU Gate 实测。**训练仍 `BLOCKED_DELTA_MISMATCH`，无 GPU 授权**。
- **唯一下一步 R45-L**：隔离 Linux CPU 环境固定 `numpy==1.26.4`，不改变当前全局 2.2.6，使用原始 frozen 14D TRAIN 同入口运行 `r45_diagnose_delta_fast.py`（不要重复全 HDF5/JPEG 34.5 GB 哈希）；验证 `np.quantile(float32)` 是否完整复现冻结 payload SHA `9f7c4f00fc2945825cc892493c21127b5e38d4f83a212bf91f7cf410776ba459`，保存原始证据并**安全推送独立 GitHub evidence 分支**。匹配仅为数值路径兼容性证据；历史运行源码身份无法凭数值匹配自动证明。仍不生成正式 Manifest / Gate，也不运行 GPU。

## R45-L GitHub 数值复现已发布（2026-10-29，已读原始报告）

- [`evidence/r45l-numpy1264-repro-20261029@6e55782c3defa30363249e84b7abe7e90460c4c1`](https://github.com/jho887891-sudo/robo/tree/6e55782c3defa30363249e84b7abe7e90460c4c1/evidence/r45l-numpy1264-repro-20261029)：GitHub 远端 HEAD 与证据提交 SHA 一致。ChatGPT 直接读取 `profile/report.json`、`execution/diagnosis.exit_code` 与 `environment/runtime_versions.txt`，并非只凭执行端摘要。
- 真实 CPU 隔离环境：Python 3.10.12、NumPy **1.26.4**、h5py 3.16.0；诊断进程退出码 **0**；300 回合，TRAIN **108780** / DEV **28268**。原始报告中 `difference.status=EXACT_MATCH`，`changed_field_count=0`，冻结与重算 payload SHA 都为 `9f7c4f00fc2945825cc892493c21127b5e38d4f83a212bf91f7cf410776ba459`；顶层诊断状态为 `DELTA_ONLY_COMPATIBLE_NOT_FULL_ATTESTED`。
- **已证明**：在这个真实环境、同一数据与代码入口下，NumPy 1.26.4 能数值复现冻结归一化器。**未证明**：历史运行时源码 SHA 的独立身份链；并未获得严格全量 HDF5 Manifest、完成正式 CPU Gate、运行 GPU 训练。
- **R45-F CPU 代理性能证据现已入库并直接复算统计**：独立分支 [`evidence/r45f-cpu-profile-20261029@f89142ce42dc7432ea9eb87e3126138f74b57bd9`](https://github.com/jho887891-sudo/robo/tree/f89142ce42dc7432ea9eb87e3126138f74b57bd9/evidence/r45f-cpu-profile-20261029) 包含 240 行未脱敏 `samples.jsonl`、120 个固定采样点、脱敏 `report.json.redacted`、执行记录、根 `SHA256SUMS` 与 `redaction_manifest.json`。GitHub 远端 HEAD 已核对；ChatGPT 直接解析 240 行重新计算旧路径中位 **44.039588887 ms**、R45 **20.676928340 ms**；均值 **51.219876512 / 24.601895649 ms**，P95 **77.563808 / 47.433375 ms**，均与提交报告一致。配对 240 次中 R45 更快 **197**、更慢 **43**；缓存命中 **0/240**，执行端 profile exit **0**。这只是 **CPU 恒等图像变换代理**，原路径始终先于 R45 测量，存在系统缓存/顺序偏差风险；不能声称正式 7B GPU 训练提速 2.13 倍。报告中路径字段已经脱敏，`redaction_manifest` 保留原始和提交版的不同哈希，ChatGPT 不拥有未脱敏服务器原始 report。
- **下一步只做尚未完成的工作**：R45-M 在隔离 NumPy 1.26.4 环境完成 300 HDF5 严格 SHA + Manifest；继续核查 OFT 唯一源码根与 7B 模型权重完整 SHA，正式 CPU Gate 记录原始退出码和日志。R45-F 既有 CPU 代理报告不再要求重复提交或重跑；后续真实 GPU 样本/秒 A/B 需要研究负责人另行授权，严禁将 CPU 代理时延外推为正式训练时间。

## R45-MNP 一次性 CPU 证据闭环 SPEC（2026-10-29，已提交、未执行）

- **唯一当前执行入口**：[R45_MNP_ONE_SHOT_CPU_EVIDENCE_SPEC.md @ 4569b507ab0f03891b883ae55f60f1752ba1c6f3](https://github.com/jho887891-sudo/robo/blob/4569b507ab0f03891b883ae55f60f1752ba1c6f3/docs/R45_MNP_ONE_SHOT_CPU_EVIDENCE_SPEC.md)。执行端锁定**执行时 PR #51 最新 40 位 SHA**，通过一个独立 CPU worktree 接单，一次性完成 M 严格 300 HDF5 Manifest、N 五个 OFT 候选静态身份/来源审计及 7B 模型分片流式 SHA、P 前置全部满足才执行正式 `scripts/r45_server_cpu_gate.sh`，最后将全部阶段证据安全推送同一**新的独立 GitHub evidence 分支**。
- **已证实但不重复**：R45-L NumPy 1.26.4 delta payload 精确匹配；R45-F 240 行 CPU 代理时延复算。**不是整体 GPU 吞吐/训练提速证明**。
- **必须真实报告的阻塞**：R45-M 需要对全部 HDF5 重新稳定哈希；当前 CPU Gate 也会再哈希一次，不能宣称只有一次 34.5 GB I/O。OFT 候选 5 个只有结构检查，不自动唯一选用；没有运行来源绑定/负责人明确批准就 `BLOCKED_OFT_PROVENANCE`。模型三权重分片现存的完整 SHA 仍 UNKNOWN，不能凭 config.json 判 PASS。M 或 N 失败时继续剩余独立 CPU 审计，但不能运行 P；不能伪造 exit0。GPU/训练不授权。
- **最终交付口径**：每阶段 PASS/BLOCKED/NOT_RUN、原始 command/stdout/stderr/退出码、固定 source SHA、SHA256SUMS、脱敏原件与发布件不同 SHA、GitHub evidence URL + 完整 commit SHA，一次性回传。只有 M+N+P+证据发布均通过才标记 `CPU_READY_FOR_SEPARATE_GPU_APPROVAL`；任何结果都不得自动启动 GPU。

## 30 秒状态摘要

**当前研发位置：R45-L 在隔离 NumPy 1.26.4 环境中真实复现冻结 delta normalizer（exit 0、差异字段 0、payload SHA 完全匹配）；R45-F 的真实 TRAIN 三相机 CPU 代理性能原始逐样本数据现已提交 GitHub（evidence SHA `f89142ce42dc7432ea9eb87e3126138f74b57bd9`），ChatGPT 独立复算 240 行，旧/R45 中位 44.040/20.677 ms，一致；但不是训练加速证明。严格 300 HDF5 Manifest、OFT 唯一源码身份、7B 权重 SHA 和正式 CPU Gate 仍未通过，GPU 未授权，训练 BLOCKED。下一步 R45-M 完整数据契约核验，后续仅在授权后测试真实 7B GPU 吞吐。**

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

## R45-I q01/q99 六数组微小差异定位（2026-10-28，当前最新）

- **执行端回传**：干净 worktree `/tmp/robo-r45i-20261028`，SHA `e91c30ef31caa7e3fa37f5ada616a23ee77964ed`；`diagnostics/r45_diagnose_delta_fast.py` 退出码 **3**、状态 `BLOCKED_DELTA_MISMATCH`。仅读 300 回合的 14D state/action 元信息、TRAIN 108780 / DEV 28268 行，未哈希整个 HDF5、未生成正式 Manifest、未接触 GPU。服务器原始报告 `/tmp/r45i-delta-fast-20261028T105947Z/report.json` 的 SHA256 `611d717e9fa1c22d389575f48730300c439ec77885b221314fcdac9c757c78aa`；它目前仅由执行者回传，尚未直接取得字节独立复算。
- **精确差异**：冻结 delta normalizer 文件 SHA256 `795052b0e6716d374488296b52befdbb012bd1840a7ccca57dfbf4df9e07383f`（与 R45-G 一致）；冻结 payload SHA256 `9f7c4f00fc2945825cc892493c21127b5e38d4f83a212bf91f7cf410776ba459`，当前 TRAIN 重算 SHA256 `0ff5872eec929bf4af770840b61a0f39789937c7c23ff0157cf059dd24cf36cf`。仅 `push_T`、`stack_bowls`、`sweep_blocks` 的 `q01` 和 `q99` 六个数组不同，差异维度分别为 (3/14,12/14)、(4/14,10/14)、(1/14,11/14)。最大绝对误差分别 `7.45e-9/2.533e-7`、`3.725e-9/4.098e-8`、`7.45e-9/3.353e-7`。其余 `target_mean/std`、`zero_fraction`、`n_rows`、`fit`、split SHA 均相同。
- **机制判断**：这种只发生在分位数的末位差异，值得优先审计 `np.quantile(float32)` 在不同 NumPy 版本/插值计算精度下的舍入；**这只是优先验证假说，不是已经证明的根因**。严格 payload SHA 必须相同，不能直接 `np.allclose` 放行。
- **GitHub 直接补充**：`diagnostics/r45_quantile_fingerprint.py` 为同一 TRAIN 14D 行的 `q01/q99` 同时计算当前 NumPy float32、NumPy float64、手写 float32 插值、float32 加权和、手写 float64 插值五种纯诊断候选。`diagnostics/r45_diagnose_delta_fast.py --quantile-forensics` 在**一次轻量读取中**写各任务、各维度候选值与冻结值的精确匹配/差异，不生成 Manifest、不能覆盖 normalizer、不能证明历史版本来源。
- **唯一下一步**：执行端固定 PR #51 最新 SHA，在新干净 worktree **只运行一次 fast delta --quantile-forensics**；回传 `report.json`、`SHA256SUMS`、NumPy 版本与每种候选算法逐任务 `exact_float32_match_count`。如有精确匹配候选，仍要核验历史 numpy/脚本源码身份才可确定根因；如无匹配，进一步查数据/其他运算实现，不做正式 CPU Gate，更不启动 GPU。
- **GitHub CPU CI PASS 49/49**：[R45 Runtime CPU Gate run 40708257285](https://github.com/jho887891-sudo/robo/actions/runs/40708257285)，测试代码 SHA `6568c0625188d303fae4303df63cd329cd5dd304`。新分位数 fingerprint 测试 3/3、之前 46/46 回归全部 PASS；R4/R5 workflow 同样 PASS。尚未在真实 TRAIN 上运行候选对照，真实冻结 q 的算法来源 UNKNOWN。

## R45-H 全量核验失败与轻量诊断修复（2026-10-28，当前最新）

- **服务器原始回传**：固定 `e0d82671d9d461faf463a39f754ac97c9a0dd817` 的干净 worktree `/tmp/robo-r45h-20261028` 运行 `diagnostics/r45_reattest_r44_manifest.py`，**退出码 3，BLOCKED_DELTA_MISMATCH**。原始 `/tmp/r45h-execution-20261028T092913Z/` 保存 command/stdout/stderr/exit code/SHA256SUMS。进程累计读约 **34.5 GB**；因严格 SHA 不同，工具拒绝创建目标目录与正式 manifest。完成扫描不等于正式数据契约全部通过，**不要把 HDF5 全部哈希判成独立 VERIFIED**。没有运行 OFT、模型 shard 哈希、正式 CPU Gate、CUDA/GPU。
- **差异来源尚 UNKNOWN**：可能来自历史冻结归一化器或其生成代码版本、数据变更、浮点统计/量化细节；只有冻结/recomputed 整体 SHA 不同，无法确认任何一项。不得改变当前冻结 `joint_delta_v1`、`alpha=16`、归一化器或绕过严格 hash。
- **直接修改 PR #51 的诊断代码**：新增 `diagnostics/r45_delta_payload_diff.py`，逐字段比较 `fit`、`q01/q99`、`target_mean`、`target_std`、`zero_fraction`、`n_rows` 等，并保留精确数值/索引、SHA，**不引入任何“差不多相等就放行”容差**。
- 新增 `diagnostics/r45_diagnose_delta_fast.py`：只读 300 回合 `state/action` 与任务元信息，先得到冻结/重算 delta 数值差异，**不重复全量读取 34.5 GB 图像或计算整个 HDF5 SHA**；即使 SHA 匹配也不会生成严格 manifest，更不会代表正式 CPU Gate PASS。任何差异继续 `BLOCKED_DELTA_MISMATCH`。
- 更新 `diagnostics/r45_reattest_r44_manifest.py`：如果后续确需完整扫描，即使 delta 不匹配，也会在全新 `--out` 目录写 `failure_report.json`、`hdf5_inventory_diagnostic_only.json` 和 `SHA256SUMS`，保留原始逐文件 SHA 和字段差异；**不会写出可用于正式训练的 `r44_dataset_manifest.json`**。
- **唯一下一步**：由真实 Linux 执行端在最新 PR #51 的干净 worktree **先运行 fast delta 诊断**（同一 frozen split/42D/delta 文件），回传具体字段差异、完整 SHA 和证据。若 mismatch，分析差异根源，不再重新读 34.5 GB；若精确 SHA 通过，才考虑全量 R44 manifest 和 OFT 版本核验。GPU 继续 BLOCKED。
- **最新 GitHub CPU Gate**：[run 40666654200](https://github.com/jho887891-sudo/robo/actions/runs/40666654200)，代码 SHA `5d112eb6eceb839682580e18647bc3031d1e6378`，**46/46 PASS**：发现工具 3、严格 Manifest/OFT/差异报告 7、快速 14D delta 3、运行时 9、R44 15、R45 4、JPEG proxy 2、CPU toy resume 3；R4/R5 CI 亦通过。只有合成/CPU 源码验证，不代表真实服务器差异已定位。

## R45-G 现场阻塞与 CPU-only 修复（2026-10-28，当前）

**现场结论：`BLOCKED_MISSING_R44_ARTIFACTS`，正式 CPU Gate 未执行，无退出码，未触 CUDA。** 固定 SHA `b13efad6a9fd8d546aa5b26035c600486c78fb70`，工作区 `/tmp/robo-r45g-preflight-20261028`；执行日志 `/tmp/r45g-server-preflight-20261028T081500Z/`（服务器路径，原始字节尚未直接获取，以下是现场回传）。

- **VERIFIED（执行端）**：42D 三任务归一化器 `data/m2_task_normalizer_3task.json`，文件 SHA256 `7c404f08…e4d13`；历史 `/tmp/r44-preflight-20261026T065550Z/normalizer.json` 为 `joint_delta_v1`、三任务、TRAIN split 绑定，payload SHA256 `9f7c4f00…ba459`、文件 SHA256 `795052b0…7383f`。历史 `BLOCKED_BUDGET` 是预算未授权，不等于数据契约失败；历史 LoRA alpha=32 不代表现 R44 alpha=16，**不得覆盖本轮冻结配置**。
- **BLOCKED**：现场只有 875 字节旧 `dataset_manifest.json`，不符合 `r44-dataset-manifest/v1` 严格 SHA/HDF5 文件列表契约；OFT 源码目录发现 5 个候选，无历史运行链可唯一证明。**UNKNOWN**：候选 `/home/T7/ojh/robot_actual/openvla_offline_bundle/models/openvla-7b` 的三个 safetensors 权重分片完整 SHA，模型目录结构存在不等于模型身份已核验。
- **GitHub 已实施**：新增 `diagnostics/r45_reattest_r44_manifest.py`，复用已有 `r44_prepare_delta.scan` 全量 HDF5 状态/动作/三相机/任务指令/稳定 SHA256 核验；要求 300 回合，TRAIN 108780、DEV 28268，重算 TRAIN delta payload SHA 与**已有冻结 normalizer 完全相同**，否则拒绝；不会替换旧归一化器。仅成功时在**全新目录**创建 `r44_dataset_manifest.json`（明确标注为新核验产物，非找回历史 manifest）、`report.json`、`SHA256SUMS`。
- **GitHub 已实施**：新增 `diagnostics/r45_audit_oft_candidates.py`，对执行端明确提供的 5 个 OFT 目录只读报告关键 `prismatic` 源文件 SHA、Git HEAD 与历史默认路径是否一致，**绝不自动挑选**。若不能证明历史来源或得到研究负责人对候选身份确认，仍须 BLOCKED。
- **GitHub CPU CI PASS 41/41**：[run 40659435953](https://github.com/jho887891-sudo/robo/actions/runs/40659435953)，代码 SHA `24edd76e4270230887f17c804b12e0648f44581e`；新增 strict manifest/OFT 测试 5/5，原有 36/36 回归不变，R4/R5 CPU 也 PASS。报告新增代码 SHA、脚本 SHA、Python/numpy/h5py 版本，便于独立核验。**服务器真实 300 个 HDF5 完整稳定哈希重核验尚未运行，不能宣称正式 R44 Gate PASS**。
- **唯一下一步**：服务器在 PR #51 最新干净 worktree 中运行这两个 CPU 审计入口，补齐严格新清单和 OFT 版本来源；再完成已有的 `scripts/r45_server_cpu_gate.sh`。失败只回传 BLOCKED 和原始证据，不生成替代数据、训练模型或自启 GPU。

## R45-F 真实 TRAIN CPU I/O 现场复核报告（2026-10-28，最新）

- **执行端报告：PASS，退出码 0**。固定执行 SHA `5ddddc1b4b926cb9ee467c1fa7305aff935e2314`；服务器隔离 worktree `/tmp/robo-r45f-cpu-real-20261028` 干净，未修改仓库/数据、未接触 GPU。**2026-10-29 更新：原始逐样本 `samples.jsonl` 已发布 GitHub，ChatGPT 直接读取并独立复算 240 行时延统计，与脱敏报告相同；未脱敏原始 report 仍仅在服务器，故不得将脱敏副本声明为原始字节。**
- **数据定位**：`/home/T7/ojh/robot_actual/data/data/hdf5`；执行端确认位于 `/dev/vdb2` 的 `fuseblk` 已挂载目录；TRAIN **240/240**、DEV **60/60** 文件存在且非空；真实 TRAIN 建索引 **108,780** 样本。
- **CPU profile 固定样本**：三任务各 40，共 120 个 TRAIN 样本，重复两次 **240 次**；旧逐次重开 HDF5 路径和 R45 Reader 的三相机**恒等 CPU 图像变换代理**张量相同。此检查**不包含生产 resize/normalize，也不等价于完整训练数据契约预检**。
- **读图时延（报告值）**：旧路径中位数 **44.040 ms/样本**，R45 Reader 中位数 **20.677 ms/样本**，中位比值约 **2.13×**（两条独立统计量的比值，不是严格配对提速）；HDF5 JPEG 字节读取均值 **1.259 ms**，JPEG 解码与代理变换 **13.303 ms**，CPU tensor 准备 **7.488 ms**。
- **缓存与 Worker（报告值）**：256 MiB 解码图像缓存命中 **0/240**，120 张三相机张量约 **331,776,000 B**，大于缓存上限 **268,435,456 B**；只读 1/2/4 worker 的解码代理吞吐 **53.6/123.5/157.2 样本/s**（含启动）。该 Worker 任务**没有接入正式训练器**，不宣称生产训练可以达到对应数值。
- **真实现场证据**：服务器原件 `/tmp/r45f-cpu-execution-20261028/`；GitHub 已发布安全脱敏副本及未改动逐样本数据 [R45-F evidence @ f89142ce](https://github.com/jho887891-sudo/robo/tree/f89142ce42dc7432ea9eb87e3126138f74b57bd9/evidence/r45f-cpu-profile-20261029)，包含根 SHA256SUMS、原件/发布件哈希映射与 profile exit 0。独立验证范围为 240 行统计复算和 GitHub 文件可读，尚非完整端到端 GPU 性能验证。
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

**持久化原则**：每个阶段只更新本文件的状态、SHA、唯一下一步；完整原始日志放 evidence。当前唯一执行任务为按 docs/R45_MNP_ONE_SHOT_CPU_EVIDENCE_SPEC.md 一次性完成严格数据 Manifest、OFT/7B 身份和在前置均通过时正式 CPU Gate，并安全推送 GitHub evidence；R45-L/R45-F 已结束，不重复运行；GPU/训练未授权；未经研究负责人同意不得自动运行 GPU Smoke、GPU Benchmark 或正式 5 Epoch。

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
## R46 初始实施快照（已被上方最终 CPU Gate 状态覆盖）

> 以下记录是在 Linux POSIX Gate 和最后两项安全修复之前写的中间快照；“being finalized”及旧样本计划 SHA 不代表最终状态。最终结果、精确提交 SHA 与当前证据以本文开头 R46 节为准。

- **Code scope:** isolated branch `codex/r46-smoke-supervisor-20261029`, based on PR #51 source `ba563d0f8c638b089f588027dd0af41ff290e1f4`. R44 training entrypoint and model target were not changed. Added the standard-template approval validator, POSIX parent supervisor, real OpenVLA five-sample child path, pure sample contract, and CPU tests.
- **Frozen model contract:** standard OpenVLA-7B + frozen base, LoRA r=32 / alpha=16 on q_proj/v_proj, 42D projector, three cameras in baseline order, `joint_delta_v1`, five TRAIN samples (one warm-up + four measured), exactly five disposable optimizer steps, no checkpoint/model export.
- **Real TRAIN sample set:** `config/r46_smoke_samples.json`, plan SHA `7497fd7ab39a69cbb4de55391dc29f954efb785abd34537a7096a908eacbb417`; generated on the Linux CPU host from the original strict manifest and split. Five unique `push_T` TRAIN episodes; actual HDF5 SHA, RGB decode, production processor output `(1,18,224,224)`, finite 42D feature and legal 14-token target verified. CPU freeze log reports `cuda_initialized=false`.
- **CPU verification:** Windows pure unit tests passed; Linux server POSIX supervisor tests and R44/R45 regression suite are being finalized with `CUDA_VISIBLE_DEVICES=`. The final log directory and exact code SHA will be linked after the current run exits.
- **Performance evidence:** R46 smoke runner is implemented but not run. No 7B weights were loaded and no GPU timings, memory, or throughput were collected. R45 `r45_gpu_bench.py` remains `BLOCKED_NOT_IMPLEMENTED`.
- **Authorization boundary:** The supplied `r47-gpu-approval-template/v1` is only a template (`approved=false`). R47 is `STOP_WAIT_GPU_APPROVAL`; no CUDA/GPU Smoke, benchmark, formal training, or deployment was started. R48 is `NOT_RUN` because it requires real R47 GPU evidence.
- **Approval identity field:** when a research lead completes `approved_model_sha_list`, it must bind the full static OpenVLA identity file list (config, processor/tokenizer metadata, index, and all three weight shards); the runtime rehashes every listed file and cross-checks the index before creating CUDA context.
