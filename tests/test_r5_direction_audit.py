#!/usr/bin/env python3
"""CPU checks of the R5 post-hoc direction metrics; no model or HDF5 required."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("r5_direction_audit", HERE / "diagnostics/r5_direction_audit.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class TestR5DirectionMetrics(unittest.TestCase):
    def test_zero_variance_correlation_is_undefined(self):
        self.assertIsNone(M.pearson_or_none([1., 1., 1.], [0., 1., 2.]))
        self.assertIsNone(M.pearson_or_none([0., 1., 2.], [1., 1., 1.]))

    def test_position_corr_is_not_delta_corr(self):
        cfg = {"action_target_mode": "absolute", "q01": [0.] * 14, "q99": [2.] * 14,
               "val_samples": [["e1", 1], ["e2", 2], ["e3", 3]]}
        e = {"gen_tokens_per_sample": [[100]*14, [120]*14, [140]*14],
             "gt_tokens_per_sample": [[100]*14, [130]*14, [150]*14]}
        r = M.direction_report(cfg, e, threshold=0.002)
        self.assertEqual(r["correlation_quantity"], "absolute_position")
        self.assertIsNone(r["active_gt_sign_accuracy"])
        self.assertEqual(r["undefined_corr_dims"], [])

    def test_decode_token_endpoints(self):
        lo = [-1.] * 14
        hi = [1.] * 14
        vals = M.decode_tokens([[0]*14, [255]*14], lo, hi)
        np.testing.assert_allclose(vals[0], -1.)
        np.testing.assert_allclose(vals[1], 1.)

    def test_threshold_excludes_weak_quantized_motion(self):
        cfg = {"action_target_mode": "residual",
               "q01": [-1.] * 14, "q99": [1.] * 14,
               "val_samples": [["v", 0], ["v", 1], ["v", 2]]}
        gt = [[127]*14, [150]*14, [100]*14]
        pred = [[127]*14, [100]*14, [100]*14]
        r = M.direction_report(cfg, {"gt_tokens_per_sample": gt,
                                   "gen_tokens_per_sample": pred}, threshold=0.02)
        self.assertEqual(r["active_gt_dim_frames"], 28)
        self.assertAlmostEqual(r["active_gt_sign_accuracy"], 0.5)
        self.assertEqual(r["correlation_quantity"], "quantized_residual_increment")

    def test_invalid_token_range_refused(self):
        with self.assertRaises(ValueError):
            M.decode_tokens([[256]*14], [-1.]*14, [1.]*14)

    def test_committed_r5_evidence_reproduces_neg_corr_and_sign_change(self):
        report = M.audit(HERE / "evidence/r5_ab_run_20261019T103523Z", threshold=.002)
        absolute = report["modes"]["absolute"]
        residual = report["modes"]["residual"]
        self.assertAlmostEqual(absolute["mean_corr_defined_dims_only"], .6962647, places=4)
        self.assertEqual(absolute["correlation_quantity"], "absolute_position")
        self.assertAlmostEqual(residual["mean_corr_defined_dims_only"], -.1943720, places=4)
        self.assertEqual(residual["undefined_corr_dims"], [13])
        self.assertEqual(residual["active_gt_dim_frames"], 292)
        self.assertAlmostEqual(residual["active_gt_sign_accuracy"], .5479452, places=4)
        self.assertEqual(residual["all_dim_frames"], 840)
        self.assertTrue(max(residual["quantized_target_bin_widths_physical_units"]) < .001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
