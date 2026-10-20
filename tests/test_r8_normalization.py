#!/usr/bin/env python3
"""R8 normalization regression tests (CPU).

Each test locks one property that the audit asked for: the legacy transform must stay
bit-compatible with the production contract, the signed transform must map zero motion
to zero and keep the sign, and the real-HDF5 information loss must be measured (not
assumed).
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("code/xpolicylab/policy/OpenVLA_Pure", "diagnostics", "training"):
    sys.path.insert(0, os.path.join(REPO, sub))
import r8_contract as C            # noqa: E402
import r8_normalization as N       # noqa: E402
import r6_data as D                # noqa: E402

HDF5 = os.environ.get("PUSHT_HDF5", "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
SPLIT = os.environ.get("PUSHT_SPLIT", "/home/dgut/pusht_runtime/split80.json")
BOUNDS = "/home/dgut/pusht_runtime/push_T_bounds.npz"

PMIN = np.array([0.0, -1.2, -0.5, -1.0, -0.2, -0.9, 0.0, -1.0, -0.3, 0.0, -1.2, -0.4, -0.8, 0.0],
                np.float32)
PMAX = np.array([1.0, 0.5, 2.2, 0.1, 0.4, 0.5, 1.0, 1.0, 2.2, 1.8, 0.1, 0.35, 0.33, 1.0],
                np.float32)


def rnd(rows=300, scale=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, scale, (rows, 14)).astype(np.float32),
            rng.normal(0, scale / 3, (rows, 14)).astype(np.float32))


class TestLegacyParity(unittest.TestCase):
    def test_legacy_matches_production_contract(self):
        v, a = rnd()
        s = np.random.default_rng(1).normal(0, 0.5, (len(v), 14)).astype(np.float32)
        contract = N.NormalizationContract(N.LEGACY_POSITION, PMIN, PMAX)
        got = contract.transform(s, v, a)
        want = np.stack([C.build_proprio(s[i], s[i], None, PMIN, PMAX) for i in range(len(s))])
        # the production path builds [s, v, acc] from frames; compare the motion slots only
        want_v = np.stack([C.normalize_block(v[i], PMIN, PMAX) for i in range(len(v))])
        want_a = np.stack([C.normalize_block(a[i], PMIN, PMAX) for i in range(len(a))])
        np.testing.assert_allclose(got[:, 14:28], want_v, atol=1e-6)
        np.testing.assert_allclose(got[:, 28:], want_a, atol=1e-6)
        np.testing.assert_allclose(got[:, :14], want[:, :14], atol=1e-6)

    def test_legacy_zero_motion_is_not_zero_where_bounds_exclude_zero(self):
        contract = N.NormalizationContract(N.LEGACY_POSITION, PMIN, PMAX)
        offsets = contract.zero_motion_offsets()
        self.assertFalse(offsets["is_zero"])
        self.assertTrue(any(abs(x) > 1e-6 for x in offsets["velocity"]))


class TestSignedMotion(unittest.TestCase):
    def test_zero_motion_maps_to_zero(self):
        v, a = rnd()
        contract = N.NormalizationContract.fit(v, v, a, PMIN, PMAX, N.SIGNED_MOTION, 0.99)
        zero = np.zeros((1, 14), np.float32)
        out = contract.transform(np.zeros((1, 14), np.float32), zero, zero)
        np.testing.assert_allclose(out[0, 14:28], 0.0, atol=1e-7)
        np.testing.assert_allclose(out[0, 28:], 0.0, atol=1e-7)

    def test_sign_and_magnitude_preserved_inside_range(self):
        contract = N.NormalizationContract(N.SIGNED_MOTION, PMIN, PMAX,
                                          np.full(14, 0.1, np.float32),
                                          np.full(14, 0.05, np.float32), 0.99)
        v = np.array([[0.05] * 14, [-0.05] * 14, [0.0] * 14], np.float32)
        out = contract.transform(np.zeros((3, 14), np.float32), v, np.zeros((3, 14), np.float32))
        self.assertTrue(np.all(out[0, 14:28] > 0))
        self.assertTrue(np.all(out[1, 14:28] < 0))
        np.testing.assert_allclose(out[2, 14:28], 0.0, atol=1e-7)

    def test_bound_counterexample_from_the_audit(self):
        """bounds [0,1]: legacy maps -0.01 and 0.00 to the same -1; signed does not."""
        lo = np.zeros(14, np.float32); hi = np.ones(14, np.float32)
        legacy = N.NormalizationContract(N.LEGACY_POSITION, lo, hi)
        signed = N.NormalizationContract(N.SIGNED_MOTION, lo, hi,
                                         np.full(14, 0.02, np.float32),
                                         np.full(14, 0.02, np.float32), 0.99)
        v = np.array([[-0.01] * 14, [0.0] * 14, [0.005] * 14], np.float32)
        L = legacy.transform(np.zeros((3, 14), np.float32), v, np.zeros((3, 14), np.float32))[:, 14:28]
        S = signed.transform(np.zeros((3, 14), np.float32), v, np.zeros((3, 14), np.float32))[:, 14:28]
        np.testing.assert_allclose(L[0], -1.0, atol=1e-7)     # negative motion clipped
        np.testing.assert_allclose(L[1], -1.0, atol=1e-7)     # zero motion also -1
        self.assertLess(S[0, 0], 0.0)                          # sign preserved
        np.testing.assert_allclose(S[1], 0.0, atol=1e-7)
        self.assertLess(S[0, 0], S[1, 0])

    def test_values_are_bounded_and_finite(self):
        contract = N.NormalizationContract(N.SIGNED_MOTION, PMIN, PMAX,
                                          np.full(14, 0.1, np.float32),
                                          np.full(14, 0.05, np.float32), 0.99)
        big = np.full((4, 14), 10.0, np.float32)
        out = contract.transform(np.zeros((4, 14), np.float32), big, -big)
        self.assertTrue(np.isfinite(out).all())
        self.assertLessEqual(np.abs(out).max(), 1.0 + 1e-6)

    def test_no_hidden_state_future_independence(self):
        v, a = rnd()
        contract = N.NormalizationContract.fit(v, v, a, PMIN, PMAX, N.SIGNED_MOTION, 0.99)
        s = np.zeros((1, 14), np.float32)
        one = contract.transform(s, v[:1], a[:1])
        two = contract.transform(s, v[-1:], a[-1:])
        again = contract.transform(s, v[:1], a[:1])
        np.testing.assert_array_equal(one, again)
        self.assertFalse(np.array_equal(one, two))


class TestScaleFitting(unittest.TestCase):
    def test_scales_are_train_only_quantiles(self):
        v, a = rnd(rows=500, scale=0.07, seed=3)
        sv, sa = N.fit_scales(v, a, 0.99)
        np.testing.assert_allclose(sv, np.maximum(np.quantile(np.abs(v), 0.99, axis=0), 1e-6), rtol=1e-6)
        np.testing.assert_allclose(sa, np.maximum(np.quantile(np.abs(a), 0.99, axis=0), 1e-6), rtol=1e-6)

    def test_fit_rejects_too_few_rows_and_bad_quantile(self):
        v, a = rnd(rows=50)
        with self.assertRaises(ValueError):
            N.fit_scales(v, a, 0.99)
        v2, a2 = rnd(rows=200)
        with self.assertRaises(ValueError):
            N.fit_scales(v2, a2, 1.5)

    def test_missing_scales_rejected(self):
        with self.assertRaises(ValueError):
            N.NormalizationContract(N.SIGNED_MOTION, PMIN, PMAX)


class TestMetadata(unittest.TestCase):
    def test_roundtrip(self):
        v, a = rnd()
        c = N.NormalizationContract.fit(v, v, a, PMIN, PMAX, N.SIGNED_MOTION, 0.99)
        meta = c.to_metadata()
        back = N.NormalizationContract.from_metadata(meta)
        self.assertEqual(back.mode, N.SIGNED_MOTION)
        np.testing.assert_allclose(back.v_scale, c.v_scale)
        x = c.transform(np.zeros((2, 14), np.float32), v[:2], a[:2])
        y = back.transform(np.zeros((2, 14), np.float32), v[:2], a[:2])
        np.testing.assert_array_equal(x, y)

    def test_fail_closed(self):
        with self.assertRaises(ValueError):
            N.NormalizationContract.from_metadata({})
        meta = N.NormalizationContract(N.LEGACY_POSITION, PMIN, PMAX).to_metadata()
        meta["normalization_mode"] = "bogus"
        with self.assertRaises(ValueError):
            N.NormalizationContract.from_metadata(meta)
        meta2 = N.NormalizationContract(N.SIGNED_MOTION, PMIN, PMAX,
                                        np.full(14, 0.1, np.float32), None, 0.99) if False else None
        with self.assertRaises(ValueError):
            N.NormalizationContract.from_metadata(
                {"normalization_mode": "signed_motion", "pmin": PMIN.tolist(), "pmax": PMAX.tolist()})


@unittest.skipUnless(os.path.isdir(HDF5) and os.path.isfile(SPLIT) and os.path.isfile(BOUNDS),
                     "real HDF5 / bounds unavailable")
class TestRealDataLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b = np.load(BOUNDS)
        cls.pmin, cls.pmax = b["pmin"].astype(np.float32), b["pmax"].astype(np.float32)
        split = D.load_split(SPLIT)
        eps = split["train"][:24]
        vs, accs = [], []
        for name in eps:
            ep = D.load_episode(name, HDF5)
            s = np.asarray(ep["state"], np.float32)
            if len(s) < 5:
                continue
            t = np.arange(2, len(s) - 1)
            v = s[t] - s[t - 1]
            vs.append(v); accs.append(v - (s[t - 1] - s[t - 2]))
        cls.V = np.concatenate(vs)
        cls.A = np.concatenate(accs)
        cls.legacy = N.NormalizationContract(N.LEGACY_POSITION, cls.pmin, cls.pmax)
        cls.signed = N.NormalizationContract.fit(cls.V, cls.V, cls.A, cls.pmin, cls.pmax,
                                                 N.SIGNED_MOTION, 0.99)

    def test_legacy_clips_about_a_third_of_motion_entries(self):
        out = self.legacy.transform(np.zeros_like(self.V), self.V, self.A)
        floor = float(np.mean(out[:, 14:] <= -1.0 + 1e-7))
        self.assertGreater(floor, 0.25)
        self.assertLess(floor, 0.40)

    def test_signed_clipping_is_only_the_quantile_tail(self):
        out = self.signed.transform(np.zeros_like(self.V), self.V, self.A)
        floor = float(np.mean(out[:, 14:] <= -1.0 + 1e-7))
        self.assertLess(floor, 0.02)

    def test_five_dimensions_lose_negative_magnitude_under_legacy(self):
        out = self.legacy.transform(np.zeros_like(self.V), self.V, self.A)[:, 14:28]
        annihilated = []
        for d in range(14):
            neg = out[self.V[:, d] < -1e-8, d]
            if neg.size and np.all(neg <= -1.0 + 1e-7):
                annihilated.append(d)
        self.assertEqual(sorted(annihilated), [2, 6, 8, 9, 13])

    def test_signed_preserves_sign_on_real_data(self):
        v, a = self.V[:2000], self.A[:2000]
        res = self.signed.sign_preserved(v, a)
        self.assertTrue(res["velocity"] and res["acceleration"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
