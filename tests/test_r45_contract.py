#!/usr/bin/env python3
import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.r45_contract import R45Config, ContractError


class TestR45Contract(unittest.TestCase):
    def test_pinned_defaults(self):
        c = R45Config()
        self.assertEqual(c.target_contract, "joint_delta_v1")
        self.assertEqual(c.action_dim, 14)
        self.assertEqual(c.proprio_dim, 42)
        self.assertEqual(c.lora_r, 32)
        self.assertEqual(c.lora_alpha, 16)
        self.assertEqual(c.cameras, ("cam_head", "cam_left_wrist", "cam_right_wrist"))

    def test_builds_42d_state_velocity_acceleration(self):
        c = R45Config()
        now = np.arange(14, dtype=np.float32)
        velocity = np.ones(14, dtype=np.float32)
        acceleration = np.full(14, 2.0, dtype=np.float32)
        out = c.transform(now, velocity, acceleration)
        self.assertEqual(out.shape, (42,))
        np.testing.assert_array_equal(out, np.concatenate([now, velocity, acceleration]))

    def test_rejects_wrong_width_and_unproven_alignment(self):
        with self.assertRaises(ContractError):
            R45Config(proprio_dim=28)
        with self.assertRaises(ContractError):
            R45Config(history_layout="[S_t,S_(t-1)]")
        with self.assertRaises(ContractError):
            R45Config(timestamp_alignment="unknown")

    def test_target_and_lora_are_not_silently_changed(self):
        with self.assertRaises(ContractError):
            R45Config(target_contract="velocity_residual")
        with self.assertRaises(ContractError):
            R45Config(lora_r=16)
        with self.assertRaises(ContractError):
            R45Config(lora_alpha=32)
        self.assertEqual(R45Config(lora_targets="q_proj/v_proj").lora_targets, "q_proj/v_proj")


if __name__ == "__main__":
    unittest.main(verbosity=2)

