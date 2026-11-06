# R51：两任务随机采样与数据流水线优化（正式修订）

状态：正式训练目标已变更；GPU 实验和正式训练仍须独立授权。历史三任务数据、配置、检查点和结果保持原样。

## 冻结训练契约

- 新训练任务仅 `push_T`、`sweep_blocks`；`stack_bowls` 不进入新训练或 DEV。
- 基座为标准 OpenVLA-7B，冻结 Backbone；LoRA `r=32, alpha=16`，仅 `q_proj/v_proj`；42D 状态投影器、Head/Left Wrist/Right Wrist 三相机、14D `joint_delta_v1` 动作目标和模型结构不变。
- 原三任务训练作为历史对照 A，保留其全部原始证据。两任务训练必须写入独立配置、输出目录和检查点命名空间。
- 两任务 TRAIN/DEV 仅从冻结 split 中筛选；DEV 不参与采样或归一化拟合。记录完整 split、Manifest、normalizer 与模型身份 SHA256。

## 采样与 Epoch 定义

现场冻结索引给出两任务 TRAIN `72,922`（push_T `25,313`、sweep_blocks `47,609`），DEV `19,552`（`6,691`、`12,861`）。每个 Epoch 固定执行 `N_draws = N_two_task_TRAIN = 72,922` 次抽样。默认研究有放回随机抽样，允许同一 Epoch 重复样本；另设全量无放回随机打乱基线。每轮种子由固定基础种子及 Epoch 编号确定。每轮记录抽样序列 SHA256、唯一样本数、重复数与重复率、两任务分布、有效 Batch、优化器更新数和学习率调度。重复访问不计作新增数据。比较 5、10、20 Epoch；同一对照使用相同 Batch Size 与每轮优化器更新数。末尾批次和断点恢复必须保持抽样序列与已提交游标一致。

## G：Sampling-aware Data Pipeline Optimization

### G1 任务权重

比较按两任务 TRAIN 样本原始比例抽样和 `push_T:sweep_blocks=50:50`。记录实际抽样比例。

### G2 抽样池

比较全量无放回、全量有放回、500/1000/2000/5000 样本 Active Pool 有放回、完整两任务 Active Pool。池从 TRAIN 以固定种子选取，保存样本 ID 与 SHA256；小池只是实验候选，必须检查 DEV 损失和逐任务成功率，不默认进入正式训练。

### G3 缓存联合实验

比较无缓存、RGB uint8 缓存、JPEG 字节缓存；容量为 128/256/512 MiB。报告所有 Worker 的累计容量、实际 RSS、命中率、淘汰数、解码节省时间及端到端吞吐。只有真实完整流水线吞吐提高且数据一致的缓存才保留。

### G4 Epoch 比较

对 5/10/20 Epoch 记录 Train Loss、DEV Loss、两任务成功率、Unique Samples、Duplicate Ratio、Cache Hit Rate、samples/sec、Optimizer Steps/sec、GPU Data Wait Time。不得只凭 Train Loss 下降认定模型改善。

## 六项技术与执行阶段

| 阶段 | 技术 | Gate |
|---|---|---|
| P00 | 固定两任务样本、Processor、硬件和计时口径 | 可复现、输入一致、SHA256 与原始日志 |
| P01 | B Multi-Worker + Prefetch | 样本顺序/标签一致，三次重复，CPU/RAM 可控 |
| P02 | E Batch Tensor Optimization | 最终模型输入一致，实测局部及完整流水线耗时 |
| P03 | G 两任务随机采样与缓存 | 抽样、覆盖、恢复、DEV 隔离及缓存 Gate |
| P04 | C GPU Image Preprocessing | Processor 数值等价，真实 GPU 吞吐和显存 |
| P05 | D Offline Preprocessing | 离线原型与五轮成本回收分析；当前不实施全量转换 |
| P06 | F DALI/NVJPEG | 三相机解码一致、显存竞争和整体吞吐 |
| P07 | 单项消融及组合 | 相同样本、Batch、硬件，三次重复 |
| P08 | 选择生产流水线 | 训练器接入、回滚、CPU/GPU 证据及负责人确认 |

P04、P06 的真实 GPU 执行须另行批准。D 按先前指示不实施离线图像转换；本规范保留其成本评估位置，P05 不能虚报为完成。

C 的纯 tensor bicubic 路径已在真实样本上证实与冻结 PIL Processor 不逐位一致，因此不得进入生产训练。保留 PIL resize/crop 的混合路径仅通过 CPU 逐位对照，须另做 GPU 数值和端到端吞吐 Gate。F 的 DALI 适配器在当前环境缺依赖，不能宣称真实 nvJPEG 解码已运行或已提速。

CPU-only P03 的 120-draw 短窗口第二遍重复了完全相同的抽样序列，缓存命中率和吞吐不代表完整两任务 Epoch。100 样本 D 原型仅核验 BF16 输入序列化与空间成本，不是全量离线转换。C、F 与组合方案的生产选择必须等待真实 GPU、DEV 及相同优化器步数的证据。

## 对照组

| 组 | 数据 | 采样 |
|---|---|---|
| A | 原三任务 | 历史完整遍历 |
| B | 两任务 | 完整无放回遍历 |
| C | 两任务 | 全量有放回 |
| D | 两任务 | 有限 Active Pool 有放回 |
| E | 两任务 | 随机抽样加最佳缓存 |
| F | 两任务 | 随机抽样加最佳流水线组合 |

分别比较相同 Optimizer Steps 的训练吞吐、相同步数的模型效果，以及相近 DEV 表现下的总墙钟时间。减少任务数导致的时间变化单独报告，不归因于图像处理。

## 数据和归一化 Gate

必须验证冻结 Manifest 对两任务子集及 DEV 子集的映射、42D Feature Normalizer 的任务独立选择、`joint_delta_v1` 的任务独立归一化、任务 prompt 编码以及 Reader 对 `stack_bowls` 的排除。核对每任务 TRAIN 行数和 normalizer fit provenance。现有三任务 normalizer 在完整冻结 split 上通过校验，push_T / sweep_blocks 块独立存储且 TRAIN 行数匹配；Feature 文件 SHA256 `7c404f08e32fe3ead3b2bc17eca97f9106aff74107fd173d03cf8adfca4e4d13`，Delta 文件 SHA256 `795052b0e6716d374488296b52befdbb012bd1840a7ccca57dfbf4df9e07383f`。可选择这两个块而保留完整历史文件身份；正式训练器的两任务 Manifest、prompt 和恢复契约仍需单独 Gate。若必须重新拟合，只用两任务 TRAIN 生成独立版本，保存来源与 SHA，不覆盖三任务文件。任一契约不符时标记 BLOCKED，停止受影响的训练与 GPU 运行。

## 交付边界

所有阶段分别记录代码 SHA、命令、环境、固定样本、原始 stdout/stderr、退出码、指标和 SHA256SUMS。区分 CPU 数据准备、GPU 图像处理、模型训练吞吐与五 Epoch 总成本。未经独立批准不启动 GPU Smoke、GPU Benchmark、正式训练或机器人部署。
