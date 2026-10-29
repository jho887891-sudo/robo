# R45-MNP：一次性 CPU 证据闭环 SPEC

**状态：SPEC / 未执行；版本：v1（2026-10-29）**

**目标**：由 DeepSeek 在真实 Linux 服务器一次接单，按固定 Git SHA 在一个隔离执行 worktree 中完成：**R45-M 严格数据 Manifest 重核验、R45-N OFT/7B 模型身份确认、R45-P 正式 CPU Gate**；收集全套证据并推送**一个新的 GitHub evidence 分支**。用户不需要逐关转发输出。ChatGPT 直接通过 GitHub 审计并更新 PR #51。

**范围**：仅 CPU、只读原始数据/权重/源码；允许在新目录生成报告与 Manifest。**不包含 GPU Smoke、GPU benchmark、正式训练、推理部署、修改 main、重训、调整 LoRA 超参数或训练提速结论。** 即使全部 PASS，也只表示 `CPU_READY_FOR_SEPARATE_GPU_APPROVAL`。

## 0. 不重做的已完成部分与冻结契约

- R45-K 已在 GitHub 入库；R45-L NumPy 1.26.4 真实 TRAIN 归一化器数值复现 `EXACT_MATCH`、退出码 0：证据 [R45-L](https://github.com/jho887891-sudo/robo/tree/6e55782c3defa30363249e84b7abe7e90460c4c1/evidence/r45l-numpy1264-repro-20261029)。
- R45-F 240 行原始 CPU 图像代理计时已发布并独立复算：中位数旧 `44.040 ms` / R45 `20.677 ms`；不能等同 7B 训练提速：[R45-F](https://github.com/jho887891-sudo/robo/tree/f89142ce42dc7432ea9eb87e3126138f74b57bd9/evidence/r45f-cpu-profile-20261029)。**不得重跑**这些实验代替当前任务。
- 固定：三任务 `push_T`、`stack_bowls`、`sweep_blocks`；`TRAIN=108780`、`DEV=28268`；300 回合（各任务 TRAIN 80、DEV 20）；3 相机、42D proprio、14D `joint_delta_v1`，`A_t=S_(t+1)`；冻结 split SHA256 `0c0c65eb1aeae17a177ea5cef9c868de25442d55649c3b87f45b0583ac515218`。
- 冻结 feature normalizer 文件 SHA256 `7c404f08e32fe3ead3b2bc17eca97f9106aff74107fd173d03cf8adfca4e4d13`；delta normalizer 文件 SHA256 `795052b0e6716d374488296b52befdbb012bd1840a7ccca57dfbf4df9e07383f`，内部 payload SHA256 `9f7c4f00fc2945825cc892493c21127b5e38d4f83a212bf91f7cf410776ba459`。
- 正式计划仍为冻结 OpenVLA-7B + LoRA `r=32, alpha=16, q_proj/v_proj` + 可训练 42D projector、5 个**完整** Epoch；旧历史 `alpha=32/min_epochs=2` 记录不得覆盖当前配置。
- repo：`jho887891-sudo/robo`，当前集成 PR：[#51](https://github.com/jho887891-sudo/robo/pull/51)，**执行前重新查询 PR HEAD 的完整 40 位 SHA**。SPEC 所在 commit SHA 只表明文本版本；真正执行源码 SHA 以执行时已核实的 PR HEAD 为准，所有报告必须记录该值。

## 1. 一次执行环境及保护

1. 首先阅读 `AGENTS.md`、`docs/PROJECT_STATUS.md` 和本 SPEC；`git fetch` 后锁定 PR #51 HEAD，建立全新 `git worktree add --detach`，在运行前后记录 `git rev-parse HEAD`、`git status --porcelain`。只在 detached 隔离 worktree 执行；**不允许更改 main、PR 代码、老 worktree、旧 evidence、数据、权重或缓存**。工具确有缺陷时报告源码位置和复现，不在本工单偷换算法/修改冻结 normalizer。
2. 记录 UTC、主机匿名标识、OS/Python/NumPy/h5py/Torch/Transformers/PEFT 版本、`sys.executable`、`pip freeze`、环境路径与源脚本 SHA256。准备/复用**隔离 CPU 运行环境**，要求 `numpy==1.26.4`、`h5py==3.16.0`，且能调用现有真实 CPU Gate 所需模块；不得替换系统 NumPy、禁止创建 CUDA context。所有命令环境设置 `CUDA_VISIBLE_DEVICES=""`、`NVIDIA_VISIBLE_DEVICES=none`、`PYTHONDONTWRITEBYTECODE=1`，再核验调用 `python3` 确实指向隔离环境。
3. 在**worktree 外**新建唯一的执行目录，例如 `/tmp/r45-mnp-one-shot-<UTC>/`，保存每项 command、stdout、stderr、真实退出码、SHA 清单、报告与最终 summary。任何目标文件/目录已存在都换全新名字，不删改、不覆盖。记录磁盘余量；容量不足则 `BLOCKED_DISK`，不尝试自动清理历史文件。
4. 以已存在的真实服务器证据逐一核对路径，**不可把历史路径直接当作现行可用路径**。可先核验下列已知候选，但必须检查文件身份/大小/哈希：HDF5 `/home/T7/ojh/robot_actual/data/data/hdf5`，42D `data/m2_task_normalizer_3task.json`，delta `/tmp/r44-preflight-20261026T065550Z/normalizer.json`，7B 模型 `/home/T7/ojh/robot_actual/openvla_offline_bundle/models/openvla-7b`。OFT 实际五个候选完整路径仍**未知**，从已有服务器 R45-G 记录与目录清单列出，不可编造。任何缺件写 `missing_inputs.json`，不能用替代件悄悄顶替。
5. **并行原则**：M（长时全量 HDF5 读取）与 N（OFT/模型只读身份核验）独立，资源允许可并行；禁止两个流程并发修改同一目录。某一项 BLOCKED 不终止其他不依赖的 CPU 审计。P 只在严格前置门槛都满足时执行。整个任务**一次交付所有已完成结果与剩余缺口**，无须用户反复贴日志。

## 2. R45-M：完整 HDF5 + 冻结 Manifest

**入口**：`diagnostics/r45_reattest_r44_manifest.py`。只用隔离 NumPy 1.26.4；其函数复用 `r44_prepare_delta.scan`，不是快速统计版。建议从仓库根执行：

```bash
python3 diagnostics/r45_reattest_r44_manifest.py \
  --splits data/m2_splits.json \
  --hdf5-root "$M2_HDF5_ROOT" \
  --feature-normalizer "$M2_NORMALIZER_JSON" \
  --delta-normalizer "$R44_DELTA_NORMALIZER" \
  --out "$OUT/manifest-reattest"
```

- **M PASS** 仅当真实退出码 `0`、`report.status=CPU_MANIFEST_REATTEST_PASS`、`r44_dataset_manifest.json` 的 `schema=r44-dataset-manifest/v1`、`files` 精确覆盖 300 回合、TRAIN 108780/DEV 28268、全部 HDF5 稳定内容哈希及三相机/14D/指令检查通过、重算 payload SHA 与冻结值**逐字节精确相等**；split 与两份 normalizer 文件 SHA 与第 0 节一致。证据：新 Manifest、`report.json`、`SHA256SUMS`、原始 command/stdout/stderr/exit 与执行脚本 SHA。
- **M BLOCKED**：源文件缺失/哈希不一致、冻结分位数不能精确匹配、300 个回合不完整、数据读入错误、脚本异常或产物写入失败。不得 `allclose` 放行、重拟合并替换冻结归一化器、以 875 字节旧 `dataset_manifest.json` 冒充严格 Manifest。若代码生成 `failure_report.json` 与 `hdf5_inventory_diagnostic_only.json`，只能当诊断资料，**不准交给正式 Gate**。
- 该扫描真实读取约 34.5 GB，**不要额外再运行一次相同全量扫描**。注意正式 CPU Gate 当前还会对 300 个 HDF5 重新哈希以验证新 Manifest，属于第二次独立校验；**不得谎称整个工单只做一次 34.5 GB 读取**。若 I/O、容量、时限不足，保留可复核失败结果并停止依赖的 P。

## 3. R45-N：OFT 唯一身份 + OpenVLA-7B 权重

### N1. OFT 源码

**入口**：`diagnostics/r45_audit_oft_candidates.py`，从服务器只读事实取出**全部五个真实候选根目录**，逐个传 `--candidate`，输出全新 `oft_candidates.json`：

```bash
python3 diagnostics/r45_audit_oft_candidates.py \
  --candidate "$OFT_CANDIDATE_1" --candidate "$OFT_CANDIDATE_2" \
  --candidate "$OFT_CANDIDATE_3" --candidate "$OFT_CANDIDATE_4" \
  --candidate "$OFT_CANDIDATE_5" \
  --out "$OUT/oft_candidates.json"
```

检查每个候选的目录存在、必需 `prismatic/extern/hf/{configuration_prismatic,processing_prismatic,modeling_prismatic}.py` 的 SHA256、Git HEAD、软链接解析、现有历史日志/锁定文件的明确来源引用。**脚本即使 exit 0，状态也是 `CANDIDATES_NEED_PROVENANCE`，并不意味着已选中。**

**N1 PASS** 必须另有可引用的先前实际执行日志/冻结来源标识，能唯一绑定具体候选及其源码 SHA，**或者研究负责人已明确批准的具体来源**；输出 `oft_selection.json`（source evidence、完整 SHA/文件哈希、选择理由、歧义核对）。如果五个候选都可用但无法唯一归属，就标记 `BLOCKED_OFT_PROVENANCE`、`selected=null`。不得因“目录存在 / 历史默认路径 / 能导入”而选一个。

### N2. 7B 模型静态身份（无需导入模型/CUDA）

- 优先定位现场真实本地模型目录；核验 `config.json`、`model.safetensors.index.json`（若存在）以及其 `weight_map` 所列全部权重分片。用**流式 SHA256**逐文件计算全部 `*.safetensors` 的文件大小、SHA256、读取前后 `stat` 身份。记录额外未被 index 引用的分片、缺件、symlink、weight_map 不一致；按当前已知三分片配置核对，实际数量不同必须解释，**不能猜或强行声明三片**。
- 对 `config.json`、模型索引、使用到的 processor/tokenizer 相关文件、`code/openvla_dynamic_modules/modeling_prismatic.py` 及对应本地实际动态文件（若能静态定位）计算文件 SHA。**只记录文件哈希、大小、版本、源引用和路径映射，不上传原始权重、模型文件或任何密钥**。
- 明确记录该模型与计划用作 R44 基座的来源证据（过去训练/下载 manifest、已有可信日志或批准的固定模型标识）；**完整权重 SHA 自身证明“现在的文件身份”，不自动证明“历史训练使用的是这些字节”**。
- 输出 `model_identity.json` 和 `model_sha256.txt`，要求所有 index 中引用的分片都有稳定内容哈希、无遗漏。若源无法确定，标 `BLOCKED_MODEL_PROVENANCE`；若哈希读取异常/缺片，标 `BLOCKED_MODEL_IDENTITY`。不可使用虚构的官方 SHA 作比对。

## 4. R45-P：正式 CPU Gate（仅当前置全过时执行）

**运行前强制检查**：M PASS、新 Manifest **原件**在服务器且未改写；N1 OFT 已唯一绑定来源；N2 模型所有分片核验完成且来源说明足够；隔离解释器为 NumPy 1.26.4；完整固定 git SHA、干净 worktree；六个环境变量有真实可核验文件/目录。

```bash
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=none
export PYTHONDONTWRITEBYTECODE=1
export R45_EXPECTED_SHA="$(git rev-parse HEAD)"
export R44_DATASET_MANIFEST="$OUT/manifest-reattest/r44_dataset_manifest.json"
export R45_EVIDENCE_ROOT="$OUT"
# 其余变量须由已核验的真实路径提供，不能猜测：
# M2_HDF5_ROOT, M2_NORMALIZER_JSON, R44_DELTA_NORMALIZER
# R44_BASE_MODEL, R44_OFT_ROOT
bash scripts/r45_server_cpu_gate.sh > "$OUT/cpu_gate.stdout" 2> "$OUT/cpu_gate.stderr"
```

- 记录外层命令**真实退出码**，原脚本打印的 `R45_F_EVIDENCE` 目录及其中 `code_sha.txt`、`exit_codes.txt`、`r44_real_data_preflight.stdout/stderr`、五组 CPU 回归/契约日志、`packages.json`、`SHA256SUMS`。检查子项 `packages`、`r44_real_data_preflight`、`r45_runtime`、`r45_toy_resume`、`r44_delta`、`r45_contract` 的退出码**全部为 0**；真实 preflight JSON `status=CPU_CONTRACTS_PASS`、样本数/三个 SHA 与冻结契约一致。
- **P PASS** 必须满足脚本 exit `0` + `STATUS=PASS_CPU_GATE_ONLY` + 全部子项 exit 0 + 前置文件哈希身份保持一致。它**不证明** CUDA/GPU Smoke、模型加载、训练峰值显存、端到端吞吐或 5 Epoch 成功。`r44_train --preflight-only` 会在模型加载/权重分片检查之前返回，因此 N2 的静态分片哈希不可省略。
- 如果前置条件不齐，**不要调用** Gate；记录 `P=NOT_RUN_BLOCKED_PREREQUISITE` 与具体阻塞项，**不能编造 exit code 0**。如果 Gate exit 非 0，按原错误记录 `BLOCKED_CPU_GATE`，不要通过换参数、删检查、造文件取得 PASS。

## 5. 一次性 GitHub 证据交付（无论 PASS/BLOCKED）

执行端在上述 CPU 审计结束后，**无论哪一项失败，都要提交已产生的真实小型证据**：

- 用另一个干净的 **evidence worktree/分支**（建议 `evidence/r45-mnp-cpu-20261029-<UTC>`，不复用既有分支）发布 `evidence/r45-mnp-cpu-<UTC>/`：`execution_plan.json`、`commands/` 原始命令与各阶段 exit、`environment/` 版本、`manifest/` 报告与安全可公开的 manifest 副本、`oft/` 候选审计、`model/` 静态 SHA 清单与 identity、`gate/` 原始日志、`stage_status.json`、`missing_inputs.json`、`redaction_manifest.json`、根 `SHA256SUMS`。
- 服务器**完整未脱敏证据和可供 Gate 使用的严格 Manifest 留在只读保护的新目录**；GitHub 仓库是 PUBLIC，**禁止**提交 34.5 GB 原始 HDF5、JPEG、模型权重、checkpoint、token、秘钥、内部主机/隐私路径、含凭据的远端 URL。若需脱敏，提交文件以 `.redacted` 命名，分别记录**原件 SHA 与发布件 SHA**、遮蔽字段和处理方式；绝不将脱敏文件用作服务器 Gate 输入或冒充原始字节。
- 发布根 `SHA256SUMS` 和 `evidence_index.json`，逐文件列来源路径、原始 SHA、GitHub 文件路径与发布 SHA、是否修改/脱敏、固定代码 SHA、命令、退出码。执行端必须实际运行 `sha256sum -c`，记录其退出码；`git status`、`git commit`、`git push -u origin` 记录成功或真实错误。**不能只把证据放在 /tmp；不能“推送失败但声称已上 GitHub”**。
- 建议将 evidence 分支 URL 和完整 40 位 SHA 评论到 **PR #51**。最终只回传**一次总报告**：三阶段分别 PASS/BLOCKED/NOT_RUN、各真实退出码、冻结文件 SHA 与新 Manifest SHA、OFT 来源证明或歧义、全部模型分片 SHA、正式 CPU Gate 子项与日志、GitHub evidence HEAD、根清单 SHA、数据/模型未改、GPU 未使用。
- 按 `docs/PROJECT_STATUS.md` 既有维护规则，**由 ChatGPT** 读取 GitHub 固定 evidence commit 并交叉审计后更新唯一状态文档。DeepSeek 不得自行修改 PR #51 的冻结代码或 `main`。

## 6. 最终验收表与失败策略

| Gate | PASS 必需 | 否则 |
|---|---|---|
| M / DATA | 新严格 `r44-dataset-manifest/v1`；300 个 HDF5 稳定哈希、TRAIN/DEV、指令、14D 与 42D、delta payload 完全匹配；exit 0 | `BLOCKED_DATA_*`，保留诊断，P 不运行 |
| N1 / OFT | 五候选审计 + 唯一可追溯来源或明确批准；源码完整哈希 | `BLOCKED_OFT_PROVENANCE`，P 不运行 |
| N2 / 7B | 所有 shard/index/配置文件 SHA、大小和身份稳定，来源可解释 | `BLOCKED_MODEL_IDENTITY/PROVENANCE`，P 不运行 |
| P / CPU | 真实 `r45_server_cpu_gate.sh` exit 0，全部子 Gate exit 0，preflight CPU 契约一致 | `NOT_RUN_BLOCKED_PREREQUISITE` 或 `BLOCKED_CPU_GATE` |
| E / GitHub | 唯一 evidence 分支、原始/脱敏关系、GitHub HEAD、SHA256SUMS 校验 exit 0、push exit 0 | `BLOCKED_PUBLICATION/PUSH`，不得声称已交付 |

整体只在 **M + N1 + N2 + P + E 均 PASS** 时写 `CPU_READY_FOR_SEPARATE_GPU_APPROVAL`。否则一次性提交部分证据、准确给出未完成项和唯一后续动作；不以补丁重算伪造历史来源，不因任一阶段失败而丢弃其他独立阶段的结果。

**严格停止线**：本 SPEC 执行完毕立即停止；**没有用户单独授权不得启动 GPU Smoke、GPU Benchmark、正式 5 Epoch、机器人部署，且不得把 CPU 代理 2.13× 当作 GPU 训练提速结论。**
