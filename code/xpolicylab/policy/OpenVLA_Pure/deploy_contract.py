#!/usr/bin/env python3
"""部署契约：任务指令如何被固定、以及当前检查点的训练域声明。

背景（第一轮审查问题 3）
------------------------
修复前 model.py 里，任务指令有两个来源：
  * config 里的 prompt（deploy.yml: prompt: "Push the T-shaped block ..."）
  * 观测里的 instruction（_encode_obs_pure 把它塞进 observation["instruction"]）
实际上 infer() 只用 config 里的 self.instr 去构造 prompt_ids，
observation["instruction"] 从来没有被读过 —— 也就是说观测指令被静默忽略，
而日志与报告里看不出这一点，容易被误读成“策略收到了任务指令”。

这个模块把该行为变成显式契约，并给出可检查的返回值：
  * instruction 固定取自 config（instruction_source="config"，默认）；
  * 观测指令若与固定指令不同，返回一条 warning（每次只打印一次）；
  * 检查点训练域声明（TRAINED_TASKS），非训练任务上必须附带免责声明。

纯 python，无 torch 依赖，可直接单元测试。
"""

# 该检查点只见过这些任务的数据（push_T，80 个训练回合，6000 步）。
TRAINED_TASKS = ("push_T",)
TRAINED_DATASET = "push_T / ARX X5 dual-arm / 80 train episodes (split80.json) / 6000 steps"

# 训练时使用的唯一指令（训练脚本 INSTR 常量，一字不差）。
TRAINED_INSTRUCTION = (
    "Push the T-shaped block to align it precisely with the gray T-shaped pad."
)

CROSS_TASK_NOTICE = (
    "检查点只在 push_T 上微调（%s）。在其它任务上的失败只能说明它没有该任务的能力，"
    "不能作为跨任务泛化结论，也不能反过来证明 push_T 上一定可执行。" % TRAINED_DATASET
)

IGNORED_OBS_INSTRUCTION_WARNING = (
    "观测中提供的任务指令被忽略：检查点使用固定训练指令。"
    "（要改用观测指令需显式配置 instruction_source=observation，且未经验证）"
)


def resolve_instruction(config_prompt=None, task_name=None, obs_instruction=None,
                        instruction_source="config"):
    """决定本帧实际送进 LLM 的任务指令。

    返回 dict：
      instruction     实际使用的指令
      source          "config" / "observation"
      obs_instruction 观测给出的指令（可能为 None）
      obs_ignored     观测指令是否被丢弃
      trained_task    task_name 是否属于检查点训练过的任务
      warnings        list[str]
    """
    warnings = []
    base = (config_prompt or "").strip() or TRAINED_INSTRUCTION

    used, source = base, "config"
    if instruction_source == "observation":
        if obs_instruction and obs_instruction.strip():
            used, source = obs_instruction.strip(), "observation"
            warnings.append(
                "instruction_source=observation：当前检查点从未在该指令上训练过，结果不可作为泛化证据。"
            )
        else:
            warnings.append("instruction_source=observation 但观测未给指令，回退到固定训练指令。")

    obs = (obs_instruction or "").strip() or None
    obs_ignored = False
    if obs is not None and source == "config" and obs != used:
        obs_ignored = True
        warnings.append(IGNORED_OBS_INSTRUCTION_WARNING)

    trained_task = (task_name is None) or (task_name in TRAINED_TASKS)
    if not trained_task:
        warnings.append(CROSS_TASK_NOTICE)

    return {
        "instruction": used,
        "source": source,
        "obs_instruction": obs,
        "obs_ignored": obs_ignored,
        "trained_task": trained_task,
        "warnings": warnings,
    }


def prompt_text(instruction):
    """把指令包成训练时使用的同一句 prompt（训练脚本与部署必须一致）。"""
    return "What action should the robot take to " + instruction + "?"
