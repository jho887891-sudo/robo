#!/usr/bin/env python3
"""真实数据集成测试：训练标签必须全部落在合法动作 token 窗口内，不得出现 PAD。

用真实的 push_T 数据集 + 训练时落盘的 q01/q99（training/run_meta_6000.json）复算
旧编码与新编码产出的 label token id，并断言：
  * 新编码（MODE_FIXED）的 label 全部在 31744..31999，PAD 数为 0；
  * 旧编码（MODE_LEGACY）确实产生了 PAD 标签 —— 作为缺陷的实测证据。

数据集不在仓库内，因此缺少数据时自动 skip（skip 不算通过）。
环境变量：
  PUSHT_HDF5   数据集目录（默认 /home/dgut/pusht_runtime/hdf5_push_T/arx_x5/data）
  PUSHT_SPLIT  80/20 划分文件（默认 data/split80.json）
  PUSHT_META   q01/q99 来源（默认 training/run_meta_6000.json）

运行：python -m unittest tests.test_label_dataset -v
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT, os.path.join(ROOT, "code", "xpolicylab", "policy", "OpenVLA_Pure")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import action_codec as AC

HDF5_DIR = os.environ.get("PUSHT_HDF5", "/home/dgut/pusht_runtime/hdf5_push_T/arx_x5/data")
SPLIT = os.environ.get("PUSHT_SPLIT", os.path.join(ROOT, "data", "split80.json"))
META = os.environ.get("PUSHT_META", os.path.join(ROOT, "training", "run_meta_6000.json"))


def _c14(h, g):
    return np.concatenate([
        h[g + "/left_arm_joint_states"][:], h[g + "/left_ee_joint_states"][:],
        h[g + "/right_arm_joint_states"][:], h[g + "/right_ee_joint_states"][:],
    ], axis=1).astype(np.float32)


def _load_actions(episodes, limit=None):
    import h5py
    out = []
    for e in episodes[:limit] if limit else episodes:
        p = os.path.join(HDF5_DIR, e)
        if not os.path.exists(p):
            return None
        with h5py.File(p, "r") as h:
            out.append(_c14(h, "action"))
    return np.concatenate(out, 0) if out else None


@unittest.skipUnless(os.path.isdir(HDF5_DIR), "push_T dataset not present (set PUSHT_HDF5)")
class TestLabelsOnRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(META) as fh:
            meta = json.load(fh)
        cls.q01 = np.asarray(meta["q01"], dtype=np.float32)
        cls.q99 = np.asarray(meta["q99"], dtype=np.float32)
        with open(SPLIT) as fh:
            split = json.load(fh)
        # 20 个留出回合足够，且与离线评测用的是同一批样本
        A = _load_actions(split["val"], limit=20)
        cls.A = A
        cls.norm = AC.normalize(A, cls.q01, cls.q99)

    def test_fixed_labels_are_all_legal(self):
        idx = AC.encode_indices(self.norm, mode=AC.MODE_FIXED)
        tid = AC.token_ids(idx)
        lo, hi, n = AC.validate_token_ids(tid)
        self.assertEqual(n, self.A.size)
        self.assertGreaterEqual(lo, AC.LEGAL_TOKEN_MIN)
        self.assertLessEqual(hi, AC.LEGAL_TOKEN_MAX)
        self.assertEqual(int((tid == AC.PAD_TOKEN_ID).sum()), 0)
        print("\n[fixed]  labels=%d  token range=[%d,%d]  PAD=%d"
              % (n, lo, hi, int((tid == AC.PAD_TOKEN_ID).sum())))

    def test_legacy_labels_contain_pad(self):
        idx = AC.encode_indices(self.norm, mode=AC.MODE_LEGACY)
        tid = AC.ACTION_TOKEN_BEGIN_IDX + 1 + idx
        n_pad = int((tid == AC.PAD_TOKEN_ID).sum())
        frac = 100.0 * n_pad / tid.size
        frames = int((tid == AC.PAD_TOKEN_ID).reshape(-1, 14).any(axis=1).sum())
        nf = tid.size // 14
        print("\n[legacy] PAD labels=%d/%d = %.2f%%   frames with >=1 PAD = %d/%d = %.2f%%"
              % (n_pad, tid.size, frac, frames, nf, 100.0 * frames / nf))
        self.assertGreater(n_pad, 0, "旧编码应当产生 PAD 标签（缺陷证据）")
        # 记录本轮实测值，便于与报告数字对照
        self.assertAlmostEqual(frac, 12.23, delta=2.0)
        self.assertAlmostEqual(100.0 * frames / nf, 97.70, delta=4.0)

    def test_no_pad_at_any_normalized_value(self):
        v = np.linspace(-1.0, 1.0, 100001, dtype=np.float32)
        tid = AC.token_ids(AC.encode_indices(v, mode=AC.MODE_FIXED))
        self.assertNotIn(AC.PAD_TOKEN_ID, tid.tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
