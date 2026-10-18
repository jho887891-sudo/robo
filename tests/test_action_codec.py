#!/usr/bin/env python3
"""action_codec 的边界单元测试（纯 CPU、只依赖 numpy）。

覆盖第一轮审查要求的用例：
  * -1 / 0 / +1 三个端点；
  * 分位数裁剪（raw 超过 q99 / 低于 q01）；
  * 夹爪维度 0 / 1；
  * 归一化动作的所有可能值都落在合法动作 token 窗口内，且绝不产生 PAD；
  * 旧编码（MODE_LEGACY）的两个缺陷被显式断言（作为回归证据保留）；
  * 编解码满射 / 单射、量化误差上界。

运行：
    python -m unittest tests.test_action_codec -v
或
    python tests/test_action_codec.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# 允许两种布局：仓库内（code/xpolicylab/policy/OpenVLA_Pure）与扁平拷贝（同目录）
for _p in (HERE, ROOT, os.path.join(ROOT, "code", "xpolicylab", "policy", "OpenVLA_Pure")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import action_codec as AC


class TestEndpoints(unittest.TestCase):
    def test_minus_one(self):
        idx = AC.encode_indices(np.array([-1.0], dtype=np.float32))
        self.assertEqual(int(idx[0]), 0)
        self.assertEqual(int(AC.token_ids(idx)[0]), 31744)
        self.assertAlmostEqual(float(AC.decode_indices(idx)[0]), -1.0, places=6)

    def test_zero(self):
        idx = AC.encode_indices(np.array([0.0], dtype=np.float32))
        self.assertEqual(int(idx[0]), 127)
        self.assertEqual(int(AC.token_ids(idx)[0]), 31871)
        # 0 不在格点上，误差必须落在量化上界内
        err = abs(float(AC.decode_indices(idx)[0]))
        self.assertLessEqual(err, AC.MAX_QUANT_ERROR + 1e-7)

    def test_plus_one(self):
        idx = AC.encode_indices(np.array([1.0], dtype=np.float32))
        self.assertEqual(int(idx[0]), 255)
        tid = int(AC.token_ids(idx)[0])
        self.assertEqual(tid, 31999)
        self.assertNotEqual(tid, AC.PAD_TOKEN_ID)          # 关键：+1 不是 PAD
        self.assertAlmostEqual(float(AC.decode_indices(idx)[0]), 1.0, places=6)

    def test_boundary_table(self):
        # v, 期望 idx, 期望 token
        cases = [
            (-1.0, 0, 31744),
            (-1.0 + 2.0 / 255.0, 1, 31745),
            (0.0, 127, 31871),
            (2.0 / 255.0, 128, 31872),
            (1.0 - 2.0 / 255.0, 254, 31998),
            (1.0, 255, 31999),
        ]
        for v, want_idx, want_tok in cases:
            idx = AC.encode_indices(np.array([v], dtype=np.float32))
            self.assertEqual(int(idx[0]), want_idx, "v=%r" % v)
            self.assertEqual(int(AC.token_ids(idx)[0]), want_tok, "v=%r" % v)


class TestLegalWindow(unittest.TestCase):
    def test_all_values_map_to_legal_tokens(self):
        v = np.linspace(-1.0, 1.0, 200001, dtype=np.float32)
        idx = AC.encode_indices(v)
        tid = AC.token_ids(idx)
        self.assertTrue((tid >= AC.LEGAL_TOKEN_MIN).all())
        self.assertTrue((tid <= AC.LEGAL_TOKEN_MAX).all())
        self.assertFalse((tid == AC.PAD_TOKEN_ID).any())
        self.assertFalse((tid == AC.STOP_INDEX).any())
        # 校验函数不得抛异常
        self.assertEqual(AC.validate_token_ids(tid)[2], tid.size)

    def test_random_values_map_to_legal_tokens(self):
        rng = np.random.default_rng(0)
        v = rng.uniform(-1.0, 1.0, 200000).astype(np.float32)
        tid = AC.token_ids(AC.encode_indices(v))
        AC.assert_legal_token_ids(tid)

    def test_no_pad_even_after_clipping_extremes(self):
        v = np.array([-5.0, -1.0, 1.0, 5.0, np.inf, -np.inf], dtype=np.float32)
        tid = AC.token_ids(AC.encode_indices(v))
        self.assertNotIn(AC.PAD_TOKEN_ID, tid.tolist())
        AC.assert_legal_token_ids(tid)

    def test_all_256_indices_used(self):
        v = np.linspace(-1.0, 1.0, 200001, dtype=np.float32)
        idx = AC.encode_indices(v)
        self.assertEqual(len(np.unique(idx)), 256)
        self.assertTrue((idx >= 0).all() and (idx <= 255).all())

    def test_validate_rejects_illegal(self):
        for bad in (AC.PAD_TOKEN_ID, AC.STOP_INDEX, 31743, 32001, 0):
            with self.assertRaises(ValueError):
                AC.validate_token_ids(np.array([bad]))


class TestQuantization(unittest.TestCase):
    def test_roundtrip_error_bound(self):
        rng = np.random.default_rng(1)
        v = rng.uniform(-1.0, 1.0, 200000).astype(np.float32)
        _, dec, err = AC.roundtrip(v)
        self.assertLessEqual(float(err.max()), AC.MAX_QUANT_ERROR + 1e-7)
        self.assertLessEqual(float(err.mean()), AC.MAX_QUANT_ERROR / 2 + 1e-7)

    def test_encode_decode_idempotent(self):
        idx = np.arange(256)
        v = AC.decode_indices(idx)
        self.assertTrue((AC.encode_indices(v) == idx).all())

    def test_monotonic(self):
        v = np.linspace(-1.0, 1.0, 5000, dtype=np.float32)
        idx = AC.encode_indices(v)
        self.assertTrue((np.diff(idx) >= 0).all())

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            AC.encode_indices(np.array([0.0]), mode="nope")


class TestQuantileClipping(unittest.TestCase):
    def setUp(self):
        # 两维：第 0 维关节 -1..1，第 1 维夹爪 0..1
        self.q01 = np.array([-1.0, 0.0], dtype=np.float32)
        self.q99 = np.array([1.0, 1.0], dtype=np.float32)

    def test_raw_above_q99(self):
        raw = np.array([1.5, 1.5], dtype=np.float32)     # 远超分位数上界
        tid = AC.token_ids_for_actions(raw, self.q01, self.q99)
        self.assertEqual(tid.tolist(), [31999, 31999])
        AC.assert_legal_token_ids(tid)
        back = AC.actions_from_token_ids(tid, self.q01, self.q99)
        self.assertAlmostEqual(float(back[0]), 1.0, places=5)

    def test_raw_below_q01(self):
        raw = np.array([-9.0, -9.0], dtype=np.float32)
        tid = AC.token_ids_for_actions(raw, self.q01, self.q99)
        self.assertEqual(tid.tolist(), [31744, 31744])
        back = AC.actions_from_token_ids(tid, self.q01, self.q99)
        self.assertAlmostEqual(float(back[0]), -1.0, places=5)
        self.assertAlmostEqual(float(back[1]), 0.0, places=5)

    def test_gripper_zero_one(self):
        # 夹爪原始值 0 与 1 必须能精确往返，且用的是合法 token
        raw = np.array([0.0, 0.0], dtype=np.float32)
        tid0 = AC.token_ids_for_actions(raw, self.q01, self.q99)
        raw1 = np.array([0.0, 1.0], dtype=np.float32)
        tid1 = AC.token_ids_for_actions(raw1, self.q01, self.q99)
        self.assertEqual(int(tid0[1]), 31744)             # 0 -> idx 0
        self.assertEqual(int(tid1[1]), 31999)             # 1 -> idx 255
        self.assertNotEqual(int(tid1[1]), AC.PAD_TOKEN_ID)
        back0 = AC.actions_from_token_ids(tid0, self.q01, self.q99)
        back1 = AC.actions_from_token_ids(tid1, self.q01, self.q99)
        self.assertAlmostEqual(float(back0[1]), 0.0, places=5)
        self.assertAlmostEqual(float(back1[1]), 1.0, places=5)

    def test_gripper_all_values_legal(self):
        raw = np.stack([np.zeros(1001), np.linspace(0.0, 1.0, 1001)], axis=1).astype(np.float32)
        tid = AC.token_ids_for_actions(raw, self.q01, self.q99)
        AC.assert_legal_token_ids(tid)
        self.assertNotIn(AC.PAD_TOKEN_ID, tid.tolist())


class TestLegacyDefectsDocumented(unittest.TestCase):
    """旧编码的两个缺陷必须被记录下来（回归证据，不是期望行为）。"""

    def test_legacy_plus_one_yields_pad(self):
        idx = AC.encode_indices(np.array([1.0], dtype=np.float32), mode=AC.MODE_LEGACY)
        self.assertEqual(int(idx[0]), 256)
        tid = AC.ACTION_TOKEN_BEGIN_IDX + 1 + int(idx[0])
        self.assertEqual(tid, AC.PAD_TOKEN_ID)
        with self.assertRaises(ValueError):
            AC.validate_token_ids(np.array([tid]))

    def test_legacy_minus_one_never_hits_window_start(self):
        idx = AC.encode_indices(np.array([-1.0], dtype=np.float32), mode=AC.MODE_LEGACY)
        self.assertEqual(int(idx[0]), 1)
        tid = AC.ACTION_TOKEN_BEGIN_IDX + 1 + int(idx[0])
        self.assertEqual(tid, 31745)                       # 窗口首个 token 31744 不可达
        self.assertGreater(tid, AC.LEGAL_TOKEN_MIN)

    def test_legacy_range_exceeds_window(self):
        v = np.linspace(-1.0, 1.0, 200001, dtype=np.float32)
        tid = AC.ACTION_TOKEN_BEGIN_IDX + 1 + AC.encode_indices(v, mode=AC.MODE_LEGACY)
        self.assertEqual(int(tid.max()), AC.PAD_TOKEN_ID)
        self.assertGreater(int(tid.max()), AC.LEGAL_TOKEN_MAX)
        self.assertGreaterEqual(int(tid.min()), AC.LEGAL_TOKEN_MIN)


if __name__ == "__main__":
    unittest.main(verbosity=2)
