#!/usr/bin/env python3
"""部署契约单元测试：任务指令是否被固定、观测指令是否被忽略、训练域声明。

运行：python -m unittest tests.test_deploy_contract -v
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# 允许两种布局：仓库内（code/xpolicylab/policy/OpenVLA_Pure）与扁平拷贝（同目录）
for _p in (HERE, ROOT, os.path.join(ROOT, "code", "xpolicylab", "policy", "OpenVLA_Pure")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import deploy_contract as DC

PUSHT = DC.TRAINED_INSTRUCTION
STACK = "Stack the three bowls on top of each other."


class TestInstructionPinning(unittest.TestCase):
    def test_config_prompt_wins(self):
        r = DC.resolve_instruction(config_prompt=PUSHT, task_name="push_T",
                                   obs_instruction=PUSHT)
        self.assertEqual(r["instruction"], PUSHT)
        self.assertEqual(r["source"], "config")
        self.assertFalse(r["obs_ignored"])
        self.assertTrue(r["trained_task"])
        self.assertEqual(r["warnings"], [])

    def test_observation_instruction_is_ignored(self):
        r = DC.resolve_instruction(config_prompt=PUSHT, task_name="push_T",
                                   obs_instruction=STACK)
        self.assertEqual(r["instruction"], PUSHT)   # 仍然用固定指令
        self.assertTrue(r["obs_ignored"])
        self.assertTrue(any("被忽略" in w for w in r["warnings"]))

    def test_missing_config_prompt_falls_back_to_trained(self):
        r = DC.resolve_instruction(config_prompt=None, task_name="push_T")
        self.assertEqual(r["instruction"], DC.TRAINED_INSTRUCTION)

    def test_observation_source_is_opt_in_and_warned(self):
        r = DC.resolve_instruction(config_prompt=PUSHT, task_name="push_T",
                                   obs_instruction=STACK, instruction_source="observation")
        self.assertEqual(r["instruction"], STACK)
        self.assertEqual(r["source"], "observation")
        self.assertTrue(any("不可作为泛化证据" in w for w in r["warnings"]))

    def test_observation_source_without_instruction(self):
        r = DC.resolve_instruction(config_prompt=PUSHT, task_name="push_T",
                                   obs_instruction=None, instruction_source="observation")
        self.assertEqual(r["instruction"], PUSHT)
        self.assertTrue(r["warnings"])


class TestTrainedDomain(unittest.TestCase):
    def test_other_tasks_get_cross_task_notice(self):
        for task in ("stack_bowls", "hang_mugs"):
            r = DC.resolve_instruction(config_prompt=PUSHT, task_name=task)
            self.assertFalse(r["trained_task"])
            self.assertIn(DC.CROSS_TASK_NOTICE, r["warnings"])

    def test_trained_task_list(self):
        self.assertEqual(DC.TRAINED_TASKS, ("push_T",))

    def test_prompt_text_matches_training(self):
        self.assertEqual(
            DC.prompt_text(PUSHT),
            "What action should the robot take to " + PUSHT + "?")


if __name__ == "__main__":
    unittest.main(verbosity=2)
