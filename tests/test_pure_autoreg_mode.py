#!/usr/bin/env python3
"""CPU-only contract tests for the isolated pure AR action-input behavior."""
import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "code/xpolicylab/policy/OpenVLA_Pure/pure_autoreg_mode.py"
spec = importlib.util.spec_from_file_location("pure_autoreg_mode", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class FakeCore:
    def _process_action_masks(self, labels):
        # Emulate OFT: action positions are masked.
        return labels != -100


class TestPureAutoreg(unittest.TestCase):
    def test_patch_preserves_action_embeddings(self):
        x = FakeCore()
        labels = torch.tensor([[-100, -100, 31744, 31745, 31999, 2]])
        old = x._process_action_masks(labels)
        self.assertTrue(bool(old[0, 2]))
        self.assertEqual(mod.enable_pure_autoreg(x)._process_action_masks(labels).sum().item(), 0)
        self.assertEqual(x._pure_autoreg_input_mode, True)

    def test_model_wide_side_effects_absent(self):
        x, y = FakeCore(), FakeCore()
        mod.enable_pure_autoreg(x)
        labels = torch.tensor([[-100, 31744, 2]])
        self.assertEqual(x._process_action_masks(labels).sum().item(), 0)
        self.assertEqual(y._process_action_masks(labels).sum().item(), 2)

    def test_preserve_embedding_and_gradient(self):
        x = mod.enable_pure_autoreg(FakeCore())
        labels = torch.tensor([[-100, 31744, 31745, 2]])
        embeddings = torch.nn.Parameter(torch.randn(1, 4, 6))
        keep = ~x._process_action_masks(labels)
        used = embeddings * keep.unsqueeze(-1)
        self.assertTrue(torch.equal(embeddings, used))
        used.sum().backward()
        self.assertTrue(torch.all(embeddings.grad[:, 1:3] != 0))

    def test_invalid_model_fails(self):
        with self.assertRaises(TypeError):
            mod.enable_pure_autoreg(object())

    def test_labels_required(self):
        x = mod.enable_pure_autoreg(FakeCore())
        with self.assertRaises(ValueError):
            x._process_action_masks(None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
