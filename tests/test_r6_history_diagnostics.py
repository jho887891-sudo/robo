#!/usr/bin/env python3
"""R6: Phase-1 diagnostic maths on synthetic data (CPU, numpy only)."""
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("diagnostics", "training"):
    sys.path.insert(0, os.path.join(REPO, sub))
import r6_align as AL       # noqa: E402
import r6_history_value as HV   # noqa: E402
import r6_quant_oracle as QO    # noqa: E402
import r6_metrics as M      # noqa: E402


def synthetic_episode(n=40, seed=0, gain=0.7):
    """Actions are the NEXT state (the real push_T relation), motions are smooth."""
    rng = np.random.default_rng(seed)
    vel = rng.normal(0, 0.01, size=(n, 14))
    vel = np.cumsum(vel, axis=0) * 0.1 + rng.normal(0, 0.01, size=(n, 14))
    state = np.cumsum(vel, axis=0)
    action = np.vstack([state[1:], state[-1:]])
    return state, action


class TestAlignment(unittest.TestCase):
    def test_e1_zero_when_action_is_next_state(self):
        s, a = synthetic_episode()
        e = AL.frame_errors(s, a)
        self.assertEqual(float(np.abs(e["e1_next_state"]).max()), 0.0)
        self.assertGreater(float(e["e2_current_state"].mean()), 0.0)
        self.assertGreater(float(e["e3_extrapolated"].mean()), 0.0)

    def test_shift_scan_finds_offset_one(self):
        s, a = synthetic_episode()
        scan = AL.shift_scan(s, a)
        self.assertEqual(min(scan, key=scan.get), "1")
        self.assertLess(scan["1"], scan["0"])
        self.assertLess(scan["1"], scan["2"])

    def test_leak_test(self):
        s, a = synthetic_episode()
        out = AL.leak_test(s, a)
        self.assertTrue(out["e3_unchanged_when_future_scrambled"])
        self.assertTrue(out["e1_changed_when_future_scrambled"])
        self.assertAlmostEqual(out["corr_future_motion_vs_expert_delta"], 1.0, places=6)

    def test_structural_checks(self):
        s, a = synthetic_episode()
        st = AL.structural_checks(s, a, frequency=25)
        self.assertEqual(st["frames"], len(s))
        self.assertEqual(st["dt_seconds"], 1.0 / 25)
        self.assertTrue(st["last_frame_finite"])

    def test_duplicate_frame_detection(self):
        s, a = synthetic_episode(6)
        s[3] = s[2]
        st = AL.structural_checks(s, a, frequency=25)
        self.assertGreaterEqual(st["identical_consecutive_state_frames"], 1)


class TestHistoryValue(unittest.TestCase):
    def test_corr_one_when_delta_equals_past_motion(self):
        # a world where the expert keeps its velocity: Delta_t == V_t exactly
        rng = np.random.default_rng(11)
        s = np.cumsum(rng.normal(0, 0.01, size=(30, 14)), axis=0)
        a = s.copy()
        a[1:] = s[1:] + (s[1:] - s[:-1])
        ser = HV.episode_series(s, a)
        np.testing.assert_allclose(ser["D"], ser["V"], atol=1e-12)
        corr = HV._corr(ser["V"][:, 0], ser["D"][:, 0])
        self.assertAlmostEqual(corr, 1.0, places=6)

    def test_zero_variance_dim_is_excluded_not_zeroed(self):
        V = np.zeros((10, 3)); V[:, 0] = np.arange(10)
        D = V.copy()
        out = HV.per_dim_corr(V, D)
        self.assertIsNone(out[1])
        self.assertIsNone(out[2])
        self.assertAlmostEqual(out[0], 1.0, places=6)

    def test_sign_block_counts(self):
        V = np.array([[0.01], [-0.01], [0.0], [0.002]])
        D = np.array([[0.01], [0.01], [0.0], [0.003]])
        rows = HV.sign_block(V, D, thresholds=(0.0, 0.005, 1e-9))
        by_thr = {(r["threshold"], r["basis"]): r for r in rows}
        self.assertEqual(by_thr[(0.0, "expert_delta")]["n"], 4)
        self.assertEqual(by_thr[(0.005, "expert_delta")]["n"], 2)
        self.assertEqual(by_thr[(1e-9, "past_motion")]["n"], 3)
        self.assertAlmostEqual(by_thr[(1e-9, "past_motion")]["sign_agreement"], 2.0 / 3.0, places=6)

    def test_fit_linear_recovers_scale(self):
        rng = np.random.default_rng(3)
        V = rng.normal(size=(200, 4))
        D = 0.5 * V + 0.01
        fit = HV.fit_linear(V, D)
        np.testing.assert_allclose(fit["slope"], np.full(4, 0.5), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(fit["intercept"], np.full(4, 0.01), rtol=1e-6, atol=1e-6)

    def test_fit_linear_constant_dim_falls_back_to_mean(self):
        V = np.zeros((5, 2)); V[:, 0] = np.arange(5)
        D = np.ones((5, 2))
        fit = HV.fit_linear(V, D)
        self.assertEqual(fit["slope"][1], 0.0)
        self.assertAlmostEqual(fit["intercept"][1], 1.0, places=6)


class TestQuantizationOracle(unittest.TestCase):
    def setUp(self):
        self.q01 = np.full(14, -0.05, np.float32)
        self.q99 = np.full(14, 0.05, np.float32)

    def test_roundtrip_error_bounded_by_step(self):
        tgt = np.zeros((1, 14), np.float32)
        q = QO.quantize(tgt, self.q01, self.q99)
        step = float(q["step"][0])
        self.assertAlmostEqual(step, 0.1 / 255, places=9)
        # the codec floors onto the grid, so 0.0 (a non-grid value) keeps at most one step
        self.assertGreater(float(np.abs(q["rec"] - tgt).max()), 0.0)
        self.assertLessEqual(float(np.abs(q["rec"] - tgt).max()), step + 1e-12)

    def test_clipping_is_reported_separately(self):
        tgt = np.full((2, 14), 0.5, np.float32)      # far outside q99
        q = QO.quantize(tgt, self.q01, self.q99)
        self.assertAlmostEqual(float(q["clipped_mask"].mean()), 1.0)
        np.testing.assert_allclose(q["inv"], np.full((2, 14), 0.05), atol=1e-6)
        blk = QO.block(tgt, tgt, self.q01, self.q99, "x")
        self.assertAlmostEqual(blk["clipping_L1"], 0.45, places=6)
        self.assertLess(blk["rounding_L1"], blk["clipping_L1"])

    def test_block_decomposition_and_direction(self):
        rng = np.random.default_rng(0)
        gt = rng.uniform(-0.02, 0.02, size=(500, 14)).astype(np.float32)
        blk = QO.block(gt, gt, self.q01, self.q99, "oracle")
        self.assertLessEqual(blk["quantized_L1"], blk["rounding_L1"] + blk["clipping_L1"] + 1e-9)
        self.assertEqual(blk["clipped_entry_fraction"], 0.0)
        for row in blk["direction"]:
            if row["n"] and row["threshold"] >= 5e-3:
                # well above the grid resolution the sign cannot flip
                self.assertEqual(row["quantized_sign_flip_rate"], 0.0)
            if row["n"] and row["threshold"] == 0.0:
                # entries at |delta| ~ 0 may flip: that is reported, never hidden
                self.assertLessEqual(row["quantized_sign_flip_rate"], 0.02)
        self.assertEqual(blk["per_arm_quantized_L1"]["left"] > 0.0, True)

    def test_perfect_predictor_error_is_the_floor(self):
        gt = np.full((4, 14), 0.0123, np.float32)
        blk = QO.block(gt, gt, self.q01, self.q99, "oracle")
        self.assertGreater(blk["quantized_L1"], 0.0)
        self.assertLess(blk["quantized_L1"], 0.1 / 255)


class TestDirectionMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.state = rng.normal(size=(8, 14)).astype(np.float32)
        self.prev = self.state - rng.normal(0, 0.01, size=(8, 14)).astype(np.float32)
        self.cv = self.state + (self.state - self.prev)
        self.gt = self.cv.copy()
        self.eps = ["e%d" % (i % 2) for i in range(8)]

    def test_perfect_predictor(self):
        out = M.summarize_direction(self.gt, self.gt, self.state, self.prev, self.eps)
        self.assertAlmostEqual(out["delta_correlation"]["pooled"], 1.0, places=6)
        self.assertEqual(out["episodes_beating_constant_velocity"], out["episode_count"])
        row = next(r for r in out["sign_agreement"] if r["basis"] == "expert_delta" and r["threshold"] == 1e-3)
        self.assertAlmostEqual(row["sign_agreement"], 1.0)

    def test_hold_state_predictor_is_reference_free(self):
        out = M.summarize_direction(self.state, self.gt, self.state, self.prev, self.eps)
        self.assertIsNone(out["delta_correlation"]["pooled"])
        self.assertEqual(out["episodes_beating_constant_velocity"], 0)
        self.assertEqual(len(out["delta_correlation"]["undefined_dims"]), 14)

    def test_threshold_counts_shrink(self):
        out = M.summarize_direction(self.cv * 1.5, self.gt, self.state, self.prev, self.eps)
        rows = [r for r in out["sign_agreement"] if r["basis"] == "expert_delta"]
        ns = [r["n"] for r in rows]
        self.assertEqual(ns, sorted(ns, reverse=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
