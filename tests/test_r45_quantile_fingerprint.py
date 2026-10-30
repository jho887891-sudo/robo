#!/usr/bin/env python3
"""Pure CPU synthetic tests of numerical quantile fingerprints."""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.r45_quantile_fingerprint import (
    fingerprint_quantiles, quantile_candidates)


class TestR45QuantileFingerprint(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.rows = rng.normal(0.0, 0.1, (1001, 14)).astype(np.float32)

    def test_distinct_interpolation_paths_and_native_exact_identity(self):
        for q in (0.01, 0.99):
            got = quantile_candidates(self.rows, q)
            self.assertEqual(set(got), {
                "numpy_current_float32_input", "numpy_float64_input",
                "manual_float32_lerp", "manual_float32_weighted_sum",
                "manual_float64_lerp",
            })
            for name, vec in got.items():
                with self.subTest(q=q, name=name):
                    self.assertEqual(vec.dtype, np.dtype("float32"))
                    self.assertEqual(vec.shape, (14,))
                    self.assertTrue(np.isfinite(vec).all())
            native = np.quantile(self.rows, q, axis=0).astype(np.float32)
            np.testing.assert_array_equal(got["numpy_current_float32_input"], native)

    def test_one_ulp_drift_stays_blocked_no_auto_tolerance(self):
        q01 = np.quantile(self.rows, .01, axis=0).astype(np.float32)
        q99 = np.quantile(self.rows, .99, axis=0).astype(np.float32)
        q99[3] = np.nextafter(q99[3], np.float32(np.inf), dtype=np.float32)
        report = fingerprint_quantiles(
            {"push_T": self.rows},
            {"tasks": {"push_T": {"q01": q01.tolist(), "q99": q99.tolist()}}},
        )
        current01 = report["tasks"]["push_T"]["q01"]["candidates"][
            "numpy_current_float32_input"]
        current99 = report["tasks"]["push_T"]["q99"]["candidates"][
            "numpy_current_float32_input"]
        self.assertTrue(current01["exact_float32_all_equal"])
        self.assertFalse(current99["exact_float32_all_equal"])
        self.assertEqual(current99["differing_indices"], [3])
        self.assertGreater(current99["max_abs_diff"], 0)

    def test_input_rejects_malformed_rows_and_missing_frozen_task(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            quantile_candidates(np.full((10, 14), np.nan), .99)
        with self.assertRaisesRegex(ValueError, "task missing"):
            fingerprint_quantiles({"push_T": self.rows}, {"tasks": {}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
