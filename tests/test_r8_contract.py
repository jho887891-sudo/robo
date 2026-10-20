#!/usr/bin/env python3
"""R8 contract tests: 42D three-frame proprio + velocity-residual target (CPU).

Real HDF5 is used where the claim is about data; synthetic arrays where it is about
arithmetic. Every test states the property it protects.
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("code/xpolicylab/policy/OpenVLA_Pure", "diagnostics", "training"):
    sys.path.insert(0, os.path.join(REPO, sub))
import r8_contract as C      # noqa: E402
import action_codec as AC    # noqa: E402
import action_target as AT   # noqa: E402
import r6_data as D          # noqa: E402

PMIN = np.linspace(-2.0, -0.5, 14).astype(np.float32)
PMAX = np.linspace(2.0, 0.5, 14).astype(np.float32)
HDF5 = os.environ.get("PUSHT_HDF5", "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
SPLIT = os.environ.get("PUSHT_SPLIT", "/home/dgut/pusht_runtime/split80.json")


def synth_episode(n=30, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 0.01, size=(n, 14)).astype(np.float32)
    S = np.cumsum(v, axis=0).astype(np.float32)
    A = np.vstack([S[1:], S[-1:]]).astype(np.float32)
    return S, A


class TestProprioShape(unittest.TestCase):
    def test_dim_42_and_layout(self):
        S, _ = synth_episode()
        p = C.build_proprio(S[5], S[4], S[3], PMIN, PMAX)
        self.assertEqual(p.shape, (42,))
        now, v, acc = p[:14], p[14:28], p[28:]
        np.testing.assert_allclose(now, C.normalize_block(S[5], PMIN, PMAX), atol=1e-6)
        np.testing.assert_allclose(v, C.normalize_block(S[5] - S[4], PMIN, PMAX), atol=1e-6)
        np.testing.assert_allclose(acc, C.normalize_block(S[5] - 2 * S[4] + S[3], PMIN, PMAX), atol=1e-6)

    def test_bounded_and_finite(self):
        S, _ = synth_episode()
        p = C.build_proprio(S[10], S[9], S[8], PMIN, PMAX)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue((p >= -1.0).all() and (p <= 1.0).all())

    def test_requires_previous_state(self):
        S, _ = synth_episode()
        with self.assertRaises(ValueError):
            C.build_proprio(S[3], None, None, PMIN, PMAX)


class TestBoundaryRules(unittest.TestCase):
    def test_boundary_flags(self):
        self.assertTrue(C.boundary_flags(0)["velocity_defined"] is False)
        self.assertTrue(C.boundary_flags(1)["velocity_defined"] is True)
        self.assertTrue(C.boundary_flags(1)["acceleration_defined"] is False)
        self.assertTrue(C.boundary_flags(2)["acceleration_defined"] is True)
        with self.assertRaises(ValueError):
            C.boundary_flags(-1)

    def test_frame_two_has_zero_acceleration(self):
        S, _ = synth_episode()
        p = C.build_proprio(S[2], S[1], None, PMIN, PMAX)
        np.testing.assert_allclose(p[28:], C.normalize_block(np.zeros(14, np.float32), PMIN, PMAX))

    def test_buffer_reproduces_offline_vectors(self):
        S, _ = synth_episode()
        buf = C.HistoryBuffer()
        for t in range(6):
            online = buf.observe(S[t], PMIN, PMAX)
            if t == 0:
                offline = C.build_proprio(S[0], S[0], None, PMIN, PMAX)
                self.assertEqual(C.boundary_flags(0)["velocity_defined"], False)
            elif t == 1:
                offline = C.build_proprio(S[1], S[0], None, PMIN, PMAX)
            else:
                offline = C.build_proprio(S[t], S[t - 1], S[t - 2], PMIN, PMAX)
            np.testing.assert_allclose(online, offline, atol=1e-6,
                                       err_msg="online/offline mismatch at t=%d" % t)

    def test_reset_restores_cold_start(self):
        S, _ = synth_episode()
        buf = C.HistoryBuffer()
        for t in range(4):
            buf.observe(S[t], PMIN, PMAX)
        buf.reset()
        first = buf.observe(S[10], PMIN, PMAX)
        fresh = C.HistoryBuffer().observe(S[10], PMIN, PMAX)
        np.testing.assert_array_equal(first, fresh)

    def test_env_isolation(self):
        S, _ = synth_episode()
        buf = C.HistoryBuffer()
        buf.observe(S[0], PMIN, PMAX, env_idx=0)
        other = buf.observe(S[5], PMIN, PMAX, env_idx=1)
        zero = C.normalize_block(np.zeros(14, np.float32), PMIN, PMAX)
        np.testing.assert_allclose(other[14:28], zero)   # fresh env: V = 0
        np.testing.assert_allclose(other[28:], zero)     # and Acc = 0


class TestNoFutureLeak(unittest.TestCase):
    def test_future_frames_do_not_change_input_or_target(self):
        S, A = synth_episode(12)
        t = 6
        base = C.build_proprio(S[t], S[t - 1], S[t - 2], PMIN, PMAX)
        v = S[t] - S[t - 1]
        base_t = C.training_target(S[t], A[t], v)
        S2, A2 = S.copy(), A.copy()
        S2[t + 1:] += 5.0
        A2[t + 1:] -= 5.0
        new = C.build_proprio(S2[t], S2[t - 1], S2[t - 2], PMIN, PMAX)
        new_t = C.training_target(S2[t], A2[t], v)
        np.testing.assert_array_equal(base, new)
        np.testing.assert_array_equal(base_t, new_t)

    def test_buffer_never_uses_predictions(self):
        S, _ = synth_episode(6)
        buf = C.HistoryBuffer()
        out = [buf.observe(S[t], PMIN, PMAX) for t in range(4)]
        # after three pushes the third slot must equal the observed S[t-2]
        np.testing.assert_allclose(out[3][28:],
                                   C.normalize_block(S[3] - 2 * S[2] + S[1], PMIN, PMAX), atol=1e-6)


class TestVelocityResidualTarget(unittest.TestCase):
    def test_target_equals_delta_minus_velocity(self):
        S, A = synth_episode()
        v = S[4] - S[3]
        r = C.training_target(S[4], A[4], v)
        np.testing.assert_allclose(r, (A[4] - S[4]) - v, atol=1e-6)
        np.testing.assert_allclose(r, AT.training_target(S[4], A[4], AT.RESIDUAL) - v, atol=1e-6)

    def test_command_inverts_target_exactly(self):
        S, A = synth_episode()
        v = S[7] - S[6]
        r = C.training_target(S[7], A[7], v)
        cmd = C.command(S[7], v, r)
        np.testing.assert_allclose(cmd, A[7], atol=1e-6)

    def test_modes_are_distinct(self):
        S, A = synth_episode()
        v = S[3] - S[2]
        r8 = C.training_target(S[3], A[3], v, C.VELOCITY_RESIDUAL)
        r5 = C.training_target(S[3], A[3], v, C.POSITION_RESIDUAL)
        self.assertFalse(np.allclose(r8, r5))

    def test_unknown_mode_rejected(self):
        S, A = synth_episode()
        with self.assertRaises(ValueError):
            C.training_target(S[0], A[0], S[0] - S[0], "delta")


class TestCodecRoundTrip(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.q01 = np.full(14, -0.05, np.float32)
        self.q99 = np.full(14, 0.05, np.float32)
        self.R = rng.uniform(-0.03, 0.03, size=(200, 14)).astype(np.float32)

    def _nearest(self, target):
        levels = AC.decode_indices(np.arange(256), mode=AC.MODE_FIXED)
        grid = np.stack([AC.denormalize(levels, np.float32(self.q01[d]), np.float32(self.q99[d]))
                         for d in range(14)], axis=0)
        return np.abs(target[:, None, :] - grid.T[None, :, :]).min(axis=1)

    def test_floor_encoding_bounded_by_one_step(self):
        idx = AC.encode_indices(AC.normalize(self.R, self.q01, self.q99), mode=AC.MODE_FIXED)
        dec = AC.denormalize(AC.decode_indices(idx, mode=AC.MODE_FIXED), self.q01, self.q99)
        step = (np.asarray(self.q99) - np.asarray(self.q01)) / 255.0
        self.assertTrue((np.abs(dec - self.R) <= step[None, :] + 1e-6).all())

    def test_nearest_bin_is_the_lower_bound(self):
        err = self._nearest(self.R).mean()
        idx = AC.encode_indices(AC.normalize(self.R, self.q01, self.q99), mode=AC.MODE_FIXED)
        dec = AC.denormalize(AC.decode_indices(idx, mode=AC.MODE_FIXED), self.q01, self.q99)
        self.assertLessEqual(err, np.abs(dec - self.R).mean() + 1e-9)

    def test_command_after_decoding_stays_finite(self):
        S, A = synth_episode()
        v = S[5] - S[4]
        r = C.training_target(S[5], A[5], v)
        idx = AC.encode_indices(AC.normalize(r, self.q01, self.q99), mode=AC.MODE_FIXED)
        dec = AC.denormalize(AC.decode_indices(idx, mode=AC.MODE_FIXED), self.q01, self.q99)
        cmd = C.command(S[5], v, dec)
        self.assertTrue(np.isfinite(cmd).all())
        self.assertLess(np.abs(cmd - A[5]).mean(), np.abs(S[5] + v - A[5]).mean() + 1e-3)


class TestCheckpointContract(unittest.TestCase):
    def test_legacy_14_and_28_still_resolve(self):
        self.assertEqual(C.resolve_checkpoint({})["kind"], "legacy")
        self.assertEqual(C.resolve_checkpoint({})["proprio_dim"], 14)
        self.assertEqual(C.resolve_checkpoint({"proprio_dim": 28,
                                               "proprio_history_mode": "true"})["proprio_dim"], 28)

    def test_r8_requires_full_metadata(self):
        good = C.checkpoint_fields(q01=np.zeros(14), q99=np.ones(14))
        self.assertEqual(C.resolve_checkpoint(good, 42)["kind"], "r8")
        broken = dict(good); broken.pop("proprio_dim")
        with self.assertRaises(ValueError):
            C.resolve_checkpoint(broken, 42)
        wrong_target = dict(good); wrong_target["target_representation"] = C.POSITION_RESIDUAL
        with self.assertRaises(ValueError):
            C.resolve_checkpoint(wrong_target, 42)
        missing_q = dict(good); missing_q.pop("q01")
        with self.assertRaises(ValueError):
            C.resolve_checkpoint(missing_q, 42)

    def test_projector_width_mismatch_rejected(self):
        good = C.checkpoint_fields(q01=np.zeros(14), q99=np.ones(14))
        with self.assertRaises(ValueError):
            C.resolve_checkpoint(good, 28)

    def test_unsupported_legacy_dim_rejected(self):
        with self.assertRaises(ValueError):
            C.resolve_checkpoint({"proprio_dim": 42})

    def test_checklist_covers_required_items(self):
        items = C.integration_checklist()
        self.assertGreaterEqual(len(items), 10)
        names = " ".join(i["item"] for i in items)
        for key in ("42D", "q01/q99", "256-bin", "KV-cache", "metadata",
                    "first/second frame", "time alignment"):
            self.assertIn(key, names)


@unittest.skipUnless(os.path.isdir(HDF5) and os.path.isfile(SPLIT), "real HDF5 unavailable")
class TestRealHDF5Batch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        split = D.load_split(SPLIT)
        cls.split = split
        cls.episode = split["val"][0]
        ep = D.load_episode(cls.episode, HDF5)
        cls.S, cls.A = ep["state"].astype(np.float32), ep["action"].astype(np.float32)
        nb = np.load("/home/dgut/pusht_runtime/push_T_bounds.npz")
        cls.pmin, cls.pmax = nb["pmin"].astype(np.float32), nb["pmax"].astype(np.float32)

    def test_minimal_three_frame_batch(self):
        for t in (2, 3, 4):
            p = C.build_proprio(self.S[t], self.S[t - 1], self.S[t - 2], self.pmin, self.pmax)
            self.assertEqual(p.shape, (42,))
            self.assertTrue(np.isfinite(p).all())
            v = self.S[t] - self.S[t - 1]
            r = C.training_target(self.S[t], self.A[t], v)
            self.assertEqual(r.shape, (14,))
            self.assertTrue(np.isfinite(r).all())

    def test_quantiles_from_train_only_on_R(self):
        train = self.split["train"][:20]
        Ss, As, Vs = [], [], []
        for name in train:
            ep = D.load_episode(name, HDF5)
            t = D.valid_frames(len(ep["state"]))
            t = t[t >= 2]
            Ss.append(ep["state"][t]); As.append(ep["action"][t])
            Vs.append(ep["state"][t] - ep["state"][t - 1])
        q01, q99 = C.velocity_residual_quantiles(np.concatenate(Ss), np.concatenate(As),
                                                 np.concatenate(Vs))
        self.assertEqual(q01.shape, (14,))
        self.assertTrue((q99 > q01).all())

    def test_no_leak_on_real_episode(self):
        t = 50
        base = C.build_proprio(self.S[t], self.S[t - 1], self.S[t - 2], self.pmin, self.pmax)
        S2 = self.S.copy(); S2[t + 1:] += 10.0
        now = C.build_proprio(S2[t], S2[t - 1], S2[t - 2], self.pmin, self.pmax)
        np.testing.assert_array_equal(base, now)

    def test_deploy_buffer_matches_training_vectors_on_real_episode(self):
        buf = C.HistoryBuffer()
        for t in range(40, 46):
            online = buf.observe(self.S[t], self.pmin, self.pmax)
            offline = (C.build_proprio(self.S[t], self.S[t], None, self.pmin, self.pmax) if t == 40
                       else C.build_proprio(self.S[t], self.S[t - 1], None, self.pmin, self.pmax)
                       if t == 41 else
                       C.build_proprio(self.S[t], self.S[t - 1], self.S[t - 2], self.pmin, self.pmax))
            np.testing.assert_allclose(online, offline, atol=1e-5)

    def test_action_target_module_unchanged(self):
        # R2-R6 contract must not be affected by R8
        self.assertEqual(AT.ABSOLUTE, "absolute")
        self.assertEqual(AT.RESIDUAL, "residual")
        np.testing.assert_allclose(AT.training_target(self.S[3], self.A[3], AT.ABSOLUTE),
                                   self.A[3], atol=0)
        np.testing.assert_allclose(AT.training_target(self.S[3], self.A[3], AT.RESIDUAL),
                                   self.A[3] - self.S[3], atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
