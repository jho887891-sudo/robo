#!/usr/bin/env python3
"""R7 unit tests: state-dynamics metrics, targets, splits and the MLP trainer (CPU)."""
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("training", "diagnostics", "code/xpolicylab/policy/OpenVLA_Pure"):
    sys.path.insert(0, os.path.join(REPO, sub))
import r7_metrics as M       # noqa: E402
import r7_state_mlp as T     # noqa: E402


def fake_blob(n=40, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.normal(0, 0.01, size=(n, 14)).astype(np.float32)
    S = rng.normal(0, 1, size=(n, 14)).astype(np.float32)
    R = 0.5 * V
    A = S + V + R
    return {"S": S, "A": A, "V": V, "Acc": 0.25 * V,
            "episode": np.array(["e%d" % (i % 4) for i in range(n)]),
            "frame": np.arange(n) + 2}


class TestTargetsAndFeatures(unittest.TestCase):
    def test_R_is_delta_minus_velocity(self):
        b = fake_blob()
        X, R = T.build_xy(b, "SV", "R")
        np.testing.assert_allclose(R, (b["A"] - b["S"]) - b["V"], atol=0)
        self.assertEqual(X.shape, (len(b["S"]), 28))

    def test_delta_target_is_delta(self):
        b = fake_blob()
        _, R = T.build_xy(b, "SV", "delta")
        np.testing.assert_allclose(R, b["A"] - b["S"], atol=0)

    def test_feature_shapes(self):
        b = fake_blob()
        self.assertEqual(T.build_xy(b, "S", "R")[0].shape[1], 14)
        self.assertEqual(T.build_xy(b, "V", "R")[0].shape[1], 14)
        self.assertEqual(T.build_xy(b, "SVAcc", "R")[0].shape[1], 42)
        with self.assertRaises(ValueError):
            T.build_xy(b, "nope", "R")

    def test_features_do_not_use_the_label(self):
        b = fake_blob()
        X1, _ = T.build_xy(b, "SVAcc", "R")
        b2 = dict(b); b2["A"] = b["A"] + 5.0          # change only the label
        X2, _ = T.build_xy(b2, "SVAcc", "R")
        np.testing.assert_array_equal(X1, X2)


class TestSplits(unittest.TestCase):
    def test_episode_split_is_disjoint_and_deterministic(self):
        eps = ["ep%02d" % i for i in range(80)]
        tr, dv = T.split_episodes(eps, n_dev=16)
        self.assertEqual(len(dv), 16)
        self.assertFalse(set(tr) & set(dv))
        self.assertEqual(set(tr) | set(dv), set(eps))
        tr2, dv2 = T.split_episodes(eps, n_dev=16)
        self.assertEqual((tr, dv), (tr2, dv2))


class TestMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        n = 30
        self.S = rng.normal(size=(n, 14)) * 0.1
        self.V = rng.normal(size=(n, 14)) * 0.01
        self.A = self.S + self.V                        # perfect constant-velocity data
        self.eps = np.array(["e%d" % (i % 3) for i in range(n)])

    def test_constant_velocity_is_exact(self):
        ev = M.evaluate(self.S + self.V, self.A, self.S, self.V, self.eps)
        self.assertAlmostEqual(ev["L1"], 0.0, places=12)
        self.assertGreater(ev["identity_L1"], 0.0)
        self.assertEqual(ev["episodes_beating_cv"], 0)

    def test_direction_metrics_on_perfect_prediction(self):
        ev = M.evaluate(self.A, self.A, self.S, self.V, self.eps)
        self.assertAlmostEqual(ev["direction"]["corr_delta_pooled"], 1.0, places=6)
        row = ev["direction"]["sign_agreement_at_1e-3"]
        self.assertTrue(row is None or row == 1.0)

    def test_correction_metrics(self):
        R = np.zeros((10, 14)); R[:, 0] = 0.01
        out = M.evaluate_correction(R, R)
        self.assertAlmostEqual(out["L1"], 0.0, places=12)
        self.assertAlmostEqual(out["sign_agreement"], 1.0, places=12)
        out2 = M.evaluate_correction(np.zeros((10, 14)), R)
        self.assertLess(out2["R2_variance_explained"], 1e-9)
        self.assertAlmostEqual(out2["relative_to_zero"], 1.0, places=6)

    def test_per_episode_present_and_shapes(self):
        ev = M.evaluate(self.S + self.V, self.A, self.S, self.V, self.eps)
        self.assertEqual(len(ev["per_episode"]), 3)
        self.assertEqual(len(ev["per_dim_L1"]), 14)
        self.assertEqual(len(ev["per_arm_L1"]), 2)

    def test_rejects_shape_and_length_mismatch(self):
        with self.assertRaises(ValueError):
            M.evaluate(self.S[:, :13], self.A, self.S, self.V, self.eps)
        with self.assertRaises(ValueError):
            M.evaluate(self.S, self.A, self.S, self.V, self.eps[:-1])

    def test_nan_rejected(self):
        bad = self.S.copy(); bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            M.evaluate(bad, self.A, self.S, self.V, self.eps)


class TestTrainer(unittest.TestCase):
    def test_mlp_learns_a_linear_correction(self):
        import torch
        rng = np.random.default_rng(5)
        n = 800
        V = rng.normal(0, 1, size=(n, 14)).astype(np.float32)
        S = rng.normal(0, 1, size=(n, 14)).astype(np.float32)
        X = np.concatenate([S, V], axis=1)
        R = (0.3 * V - 0.2 * S).astype(np.float32)
        net, stats = T.train_mlp(X[:600], R[:600], X[600:], R[600:], seed=0,
                                 hidden=(32,), epochs=60, lr=3e-3, batch=128,
                                 patience=60, torch=torch)
        pred = T.predict(net, X[600:], stats, torch)
        zero = float(np.abs(R[600:]).mean())
        got = float(np.abs(pred - R[600:]).mean())
        self.assertLess(got, 0.6 * zero, "MLP must beat the zero-correction baseline")
        self.assertTrue(np.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
