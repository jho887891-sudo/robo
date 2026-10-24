# R27：Codex/DeepSeek 最小远程只读工单

## 分工

**ChatGPT：** 设计诊断、审计R24/R25、阅读源码、写脚本、检查输出并裁决H1/H2/H3/H4、决定实验预算与Gate。

**Codex（优先）或DeepSeek：** 仅运行以下命令，不用自行分析模型、不修改训练器/架构、不给出因果结论。若远程服务器没有此仓库或权限，直接报告 `BLOCKED` 并贴原始错误，禁止猜测路径或自行重训。

目标：在服务器的真实M2 full-state checkpoint上提取**只读、可复算**的训练曝光与身份信息。截止条件：导出两个JSON及其SHA256或者留证BLOCKED。0 optimizer steps。

## 0. 保护范围

- 不运行任何带 `train`、`optimizer.step`、`--target-global-step` 的命令。
- 不启动真实7B/CUDA/IsaacSim/机器人控制、600步闭环或新训练。
- 不修改main，不cherry-pick到main，不清理/覆盖 checkpoint、HDF5、WAL、原R24证据或旧FAIL。
- 原始输出在新临时目录，如 `/tmp/r27-m2-check-<session>`，如果已存在，另起新的路径；不要覆盖旧文件。
- 只允许读取 `MANIFEST.json`、`sampler.json`、`trainer_state.json`、normalizer、split、stage evidence、HDF5索引。不要提交模型权重、大型HDF5或个人隐私配置。

## 1. 获取固定工具

在原服务器repo checkout的独立工作目录中，使用以下固定SHA对应的分支（**提交后从PR #37核验HEAD SHA**）：

`ops/r27-m2-readonly-evidence-collector`

此分支继承R25 A的 `tools/r24_seen_exposure_audit.py` 和R27的 `tools/r27_m2_checkpoint_inventory.py`。

如果本地工作目录不干净，不要checkout覆盖。另建干净工作树或用固定分支的新临时clone。若仓库与远端网络不可连接，保持BLOCKED并返回错误。

## 2. 定位真实checkpoint与stage文件，只读

优先核查历史候选：

`runs/m2_two_2000_20261021T123314Z/checkpoints/step2000`

这是候选相对路径，**不是已经证明它在当前服务器存在**。正确路径必须以实际 `MANIFEST.json`、`trainer_state.json`、`sampler.json` 验证，不得以目录名称作为真实性依据。

在该checkpoint所属训练运行目录寻找 `stage_evidence.json`。不得新建或修改stage证据，不得把其他训练运行的stage文件强行配对。

示例（先设置真实绝对路径）：

```bash
export CKPT="/REAL/PATH/TO/M2/CHECKPOINT/step2000"
export STAGE="/REAL/PATH/TO/M2/RUN/stage_evidence.json"
export OUTDIR="$(mktemp -d -t r27-m2-check-XXXXXXXX)"
test -f "$CKPT/MANIFEST.json" || exit 10
test -f "$CKPT/sampler.json" || exit 11
test -f "$STAGE" || exit 12
```

## 3. 固定CPU命令A：全状态身份核验

```bash
python tools/r27_m2_checkpoint_inventory.py \
  --checkpoint "$CKPT" \
  --stage-evidence "$STAGE" \
  --out "$OUTDIR/m2_inventory.json" \
  > "$OUTDIR/m2_inventory.stdout.log" \
  2> "$OUTDIR/m2_inventory.stderr.log"
```

验证 `m2_inventory.json` 的 `optimizer_steps_executed_by_this_tool == 0`、
`attested.manifest_file_hashes_verified == true` 和 `attested.stage_crosscheck_verified == true`。
输出中的 `actual_unique_gradient_exposed_frames` 必须是 `UNKNOWN_REQUIRES_FROZEN_FRAME_MAPPING`，不可改成2000。

任何hash、run_id、task-id、sampler.position、stage token哈希不一致，立刻停**该线**，保留stderr，不允许自动绕过。

## 4. 固定CPU命令B：真实训练帧曝光重建

只在步骤A成功后执行；R25 A工具必须与对应训练数据的HDF5和冻结split同时存在。**它不执行7B优化或推理**：

```bash
python tools/r24_seen_exposure_audit.py \
  --checkpoint "$CKPT" \
  --out "$OUTDIR/m2_seen_exposure.json" \
  > "$OUTDIR/m2_seen_exposure.stdout.log" \
  2> "$OUTDIR/m2_seen_exposure.stderr.log"
```

如果远端HDF5默认目录不同，先用源码 `training/m2_multitask_data.py` 确认DATASET ROOT，再显式加 `--hdf5-root /ACTUAL/DATASET/ROOT`。必须让sampler内部frames_sha256/order_hash检查通过，不能修改原sampler或偷偷减小帧池。

输出要保留每任务 `selected_train_seen` / `selected_train_unseen` / `selected_dev_seen_BAD_IF_NONZERO` 和 `unique_gradient_exposed_frames`。后者才是源码可证明的已更新训练帧数（在数据身份与任务slot mapping真实核验通过时）。

如失败，输出 `BLOCKED`，并指出HDF5 missing、normalizer hash/split mismatch、sampler frames_sha256不匹配、环境缺依赖中的具体一种。

## 5. 交付，不需要研究判断

运行结束后只导出这两份JSON及4份原始stdout/stderr日志。用以下命令生成哈希：

```bash
cd "$OUTDIR"
sha256sum *.json *.log > SHA256SUMS.txt
cat SHA256SUMS.txt
```

在新的独立 evidence 分支（不能改写原训练证据）提交**小型JSON/日志/SHA256SUMS**。如果不允许提交日志中的敏感路径，先只报告SHA与成功/失败标记，原始日志留在授权存储，不粘贴个人密钥。

交付格式只有：

```text
STATUS: VERIFIED / BLOCKED
GIT SHA:
CHECKPOINT MANIFEST SHA256:
STAGE EVIDENCE SHA256:
SAMPLER POS:
DISTINCT SCHEDULE SLOTS:
ACTUAL DISTINCT TRAIN-SEEN FRAMES: <integer or UNKNOWN>
R24 136 TRAIN SELECTED -> SEEN COUNT:
DEV CONTAMINATION COUNT:
OUTPUT LOG DIRECTORY:
EVIDENCE SHA256SUMS:
COMMAND EXIT CODES:
OPTIMIZER STEPS ADDED: 0
```

不要擅自做统计推断、改训练器、换动作头、跑推理或申请GPU。**ChatGPT收到证据后负责独立重算、科研结论及下一步。**

## 6. 远程环境常见故障处理

- `Codex CLI not installed`：这只表示远程Codex CLI未安装，不能因此认定checkpoint不存在。能通过已安装Python和SSH命令运行时，直接运行上述脚本即可。
- `Permission denied`：停止，不执行提权和覆盖操作。
- `No such file or directory`：核查真实来源路径；找不到即BLOCKED，不造一份假的metadata。
- CUDA不可用：不影响此次任务（只做CPU+HDF5索引）。
- 计算超时：保留运行时间与错误，只暂停受影响命令，不扩大预算。

**验收DoD：** 双JSON对照真实冻结来源通过、退出码为0、输出由SHA256锁定、没有新增优化步。否则按BLOCKED交付，不允许凭一条“done”宣布R25 H1/H3结案。
