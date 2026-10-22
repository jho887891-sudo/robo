# R16 KV 首次分叉：根因源码审计与最小判别实验（2026-10-22）

执行源 feature/r15-true-continuation@9a973206a0fe71e44c65dfdbed34b925842a1234。仅对GitHub证据独立审计，无远端GPU执行权限，不把代码层推测冒充真实算子诊断。

## VERIFIED_SOURCE / EXECUTOR_EVIDENCE
- mission/r16_three_way.json: 同一M2 step2000真实7B，8组训练参考和FULL Serving输出均一致（8/8），训练参考和增量KV不一致（0/8）。首token k=0的256-bin logits严格相同；从k=1才开始分叉。32组数据均有不同动作bin，最大logits差31.390625。缓存4.05倍加速但不能部署。
- FULL路径通过完整vla模型、带三相机/42D/labels，内部BOS之后插入768视觉+1 proprio embedding，再由语言模型因果forward。缓存prefill用同样路径，k0相等；cached k1却直接调用model.language_model(input_ids=单一动作token,past_key_values=cache,use_cache=True)而非前面完整多模态包装路径。
- 可直接观察的缓存元数据BUG：r42_deploy_server.py prefill_cached把n_past设置为len(text_ids)，不是实际multimodal长度 len(text_ids)+769。然而step_cached并没有使用这个变量传给语言模型，因此该bug目前本身不能解释输出分歧。禁止仅把n_past改成+769就宣称修复！
- evidence/r8_vla_42d_smoke_20261020/logs/attempt1_kv_cache_assertion.log记载transformers=4.40.2，模型要求4.40.1。直接审阅transformers v4.40.2的LlamaModel.forward：DynamicCache会从past_key_values.get_seq_length()推算cache_position/position_ids。因此不显式传position_ids或attention_mask不必然导致错误，先检查实际KV和Attention implementation。
- 训练pure-autoreg patch禁用先前action token的zeroing，Full Serving与训练参考8/8一致，旧OFT action zeroing不应被直接归咎于增量差异。

## HYPOTHESIS（尚未通过真实7B张量对照证明）
H1: 实际KV只有文本长度，未含769多模态prefix -> RoPE错位。
H2: 实际KV长度正确，但增量attention mask/cache_position/attention backend与full attention路径不同。
H3: direct language_model跳过PEFT/Prismatic wrapper运行时hook，导致LoRA或模型行为差异。
H4: HF Transformers 4.40.2具体LM cache或attention实现与非cache不等价；先在相同language_model纯文本前缀检验。
H5: BF16/SDPA/Flash差异；k1高达数个到几十logit差通常不能无证据简单当作微小数值误差。

## 新提交的可执行源文件
tools/r16_kv_first_divergence.py：一次加载真实7B checkpoint，一条冻结DEV帧，从FULL的真实k0 argmax固定首token，使所有k1实验同条件。
逐项对照：FULL no-cache vs FULL use_cache=True；读实际cache.get_seq_length和cache格式；direct旧增量 vs 显式full-length attention mask+position_ids/cache_position；通过Prismatic 1-token branch vs 直呼LM；纯语言LM文本full vs cache。所有路径重新prefill避免因KV原地更新互相污染。输出每路径256-bin maxabs/p99、argmax、实际KV长度、例外和固定SHA，zero optimizer、不能覆盖历史。

## 判别停止规则
- 若实际cache length不等于len(prompt)+769，先查多模态prefill/cached写入具体层。
- 显式mask+position恢复k1一致，优先修cache_position/attention contract，但还要过32帧x14token完整等价。
- Prismatic cached branch恢复而直LM失败，定位wrapper/LoRA注入和runtime hooks。
- 纯文本LM也失败，优先查HF4.40.2/attn backend/cache API/PEFT，不要把问题归咎视觉。
- 若都失败，逐层hidden_state trace，在k1同输入强制teacher token对照，找第一个不同层的q/k/v、RoPE、attention mask和LoRA输出。不能靠重新训练解决。
- 如果k1通过但k2失败，继续固定每一步真实teacher token并查第一个失配层。禁止被自回归不同token产生的后续级联差异误导。

## DoD、资源和保护
建议单GPU零步30分钟以内，GPU/IsaacSim重型任务排他；确保M2现有作业归属，不能强停他人进程。由远程DeepSeek用真实checkpoint跑脚本、补来源SHA和每层KV shape/日志/版本，并做有界修复与32x14合法bin + SAFE14D等价Gate。只有14/14完整验证成功才可切换缓存部署；耗时必须包含真实GPU job wall。保护main、HDF5、权重、WAL、历史证据。

**当前独立审计结论**：已定位到k1增量LM路径，尚未有运行时张量可以区分H1–H5；根因不是已经VERIFIED。新脚本就是找出具体机制的下一次最小实验，而不是再下发泛泛的“调研原因”。
