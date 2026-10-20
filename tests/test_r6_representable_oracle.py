#!/usr/bin/env python3
"""Synthetic-only CPU tests for the ACTUAL best-bin representation bound."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

source=Path(__file__).resolve().parents[1]/"diagnostics/r6_representable_oracle.py"
spec=importlib.util.spec_from_file_location("r6_representable_oracle",source)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class TestRepresentableBound(unittest.TestCase):
    def test_best_grid_can_beat_floor_encoder(self):
        lo=np.zeros(14,np.float32)
        hi=np.ones(14,np.float32)
        y=np.full((4,14),0.9,np.float32)
        r=M.summarize_oracle(y,lo,hi)
        self.assertLess(r["nearest_representable_lower_bound_L1"],r["floor_encoder_oracle_L1"])
        self.assertGreater(r["fraction_entries_requiring_different_bin"],0)

    def test_nearest_matches_exhaustive_grid(self):
        rng=np.random.default_rng(19)
        lo=np.full(14,-0.4,np.float32)
        hi=np.full(14,0.6,np.float32)
        y=rng.uniform(-0.55,0.75,(41,14)).astype(np.float32)
        ids,p=M.representable_nearest(y,lo,hi)
        all_ids=np.arange(256,dtype=np.int64)[:,None]*np.ones((1,14),np.int64)
        candidates=M.decode_grid(all_ids,lo,hi)
        exhaustive=np.abs(candidates[None,:,:]-y[:,None,:]).min(axis=1)
        np.testing.assert_allclose(np.abs(p-y),exhaustive,atol=1e-7,rtol=1e-6)
        self.assertTrue(((ids>=0)&(ids<256)).all())

    def test_extreme_clipping(self):
        lo=np.full(14,-1.,np.float32)
        hi=np.full(14,1.,np.float32)
        y=np.vstack((np.full(14,12),np.full(14,-14))).astype(np.float32)
        ids,p=M.representable_nearest(y,lo,hi)
        np.testing.assert_array_equal(ids[0],255)
        np.testing.assert_array_equal(ids[1],0)

    def test_invalid_statistics(self):
        with self.assertRaises(ValueError):
            M.summarize_oracle(np.ones((2,14),np.float32),np.ones(14),np.zeros(14))

    def test_legacy_floor_vs_best_per_dim(self):
        lo=np.full(14,-1.,np.float32)
        hi=np.full(14,1.,np.float32)
        y=np.zeros((2,14),np.float32)
        report=M.summarize_oracle(y,lo,hi)
        self.assertLessEqual(report["nearest_representable_lower_bound_L1"],
                             report["floor_encoder_oracle_L1"]+1e-8)
        self.assertEqual(report["n_frames"],2)


if __name__=="__main__":
    unittest.main(verbosity=2)
