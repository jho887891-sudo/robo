#!/usr/bin/env python3
"""CPU contract tests for absolute / residual targets (no RoboDojo/GPU needed)."""
import importlib.util
import pathlib
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILE = ROOT / "code/xpolicylab/policy/OpenVLA_Pure/action_target.py"
spec = importlib.util.spec_from_file_location("action_target", FILE)
AT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AT)


class ActionTargetContractTests(unittest.TestCase):
    def setUp(self):
        self.state = np.array([0.1, -0.3, 0.25, 0.8, -1, 0, 1, 0, .2, -.1, 0, 1, -.5, 0], np.float32)
        self.action = self.state + np.array([.01, -.01, 0, 0, .003, 0, 0, 0, -.02, 0, 0, 0, 0, 0], np.float32)

    def test_absolute_roundtrip(self):
        x = AT.training_target(self.state, self.action, AT.ABSOLUTE)
        np.testing.assert_array_equal(AT.joint_command(self.state, x, AT.ABSOLUTE), self.action)

    def test_residual_roundtrip(self):
        x = AT.training_target(self.state, self.action, AT.RESIDUAL)
        np.testing.assert_allclose(AT.joint_command(self.state, x, AT.RESIDUAL),
                                   self.action, rtol=1e-6, atol=2e-7)

    def test_identity_delta_zero(self):
        v = AT.training_target(self.state, self.state, AT.RESIDUAL)
        np.testing.assert_array_equal(v, np.zeros(14, np.float32))
        np.testing.assert_array_equal(AT.joint_command(self.state, v, AT.RESIDUAL), self.state)

    def test_batch_roundtrip(self):
        st = np.stack([self.state, -self.state])
        act = np.stack([self.action, -self.action])
        for mode in AT.TARGET_MODES:
            with self.subTest(mode=mode):
                v = AT.training_target(st, act, mode)
                np.testing.assert_allclose(AT.joint_command(st, v, mode),
                                           act, rtol=1e-6, atol=2e-7)

    def test_invalid_mode_fail_closed(self):
        with self.assertRaises(ValueError):
            AT.training_target(self.state, self.action, "delta_bug")

    def test_wrong_shape_fail_closed(self):
        with self.assertRaises(ValueError):
            AT.training_target(self.state[:-1], self.action)

    def test_nonfinite_inputs_rejected(self):
        x = self.action.copy(); x[2] = np.nan
        with self.assertRaises(ValueError):
            AT.training_target(self.state, x)

    def test_metadata_legacy_absolute_is_allowed(self):
        self.assertEqual(AT.resolve_checkpoint_target({"q01": []}, AT.ABSOLUTE), AT.ABSOLUTE)

    def test_residual_requires_declared_mode(self):
        with self.assertRaises(ValueError):
            AT.resolve_checkpoint_target({}, AT.RESIDUAL)

    def test_mismatched_metadata_rejected(self):
        with self.assertRaises(ValueError):
            AT.resolve_checkpoint_target({"action_target_mode": "residual"}, AT.ABSOLUTE)

    def test_validate_quantiles(self):
        lo = np.zeros(14, np.float32); hi = np.ones(14, np.float32)
        a, b = AT.validate_stats(lo, hi)
        self.assertEqual(len(a), 14)
        with self.assertRaises(ValueError):
            AT.validate_stats(hi, lo)

    def test_checkpoint_hash_binding_and_tampering(self):
        with tempfile.TemporaryDirectory() as root:
            d = pathlib.Path(root); ad = d / "lora"
            ad.mkdir()
            w = ad / "adapter_model.safetensors"
            pp = d / "proprio.pt"
            w.write_bytes(b"adapter-synthetic-test-bytes")
            pp.write_bytes(b"projector-synthetic-test-bytes")
            m = AT.record_checkpoint_hashes({"action_target_mode": AT.RESIDUAL},
                                            str(ad), str(pp))
            self.assertTrue(AT.verify_checkpoint_hashes(m, str(ad), str(pp), AT.RESIDUAL))
            w.write_bytes(b"modified adapter")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                AT.verify_checkpoint_hashes(m, str(ad), str(pp), AT.RESIDUAL)

    def test_residual_without_hashes_rejected(self):
        with self.assertRaises(ValueError):
            AT.verify_checkpoint_hashes({}, "dummy", "dummy", AT.RESIDUAL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
