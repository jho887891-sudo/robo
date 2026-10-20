#!/usr/bin/env python3
"""R6: 28D history-proprio input contract (CPU, numpy only)."""
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))
import proprio_history as PH  # noqa: E402

PMIN = np.linspace(-2.0, -0.5, 14).astype(np.float32)
PMAX = np.linspace(2.0, 0.5, 14).astype(np.float32)
CUR = np.linspace(-1.0, 1.0, 14).astype(np.float32)
PREV = np.linspace(1.0, -1.0, 14).astype(np.float32)


class TestDimAndModes(unittest.TestCase):
    def test_dims(self):
        self.assertEqual(PH.proprio_dim(PH.NONE), 14)
        self.assertEqual(PH.proprio_dim(PH.DUPLICATE), 28)
        self.assertEqual(PH.proprio_dim(PH.TRUE), 28)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            PH.proprio_dim("previous_action")

    def test_modes_tuple_is_the_contract(self):
        self.assertEqual(set(PH.MODES), {"none", "duplicate", "true"})


class TestBuild(unittest.TestCase):
    def test_none_mode_matches_training_normalization(self):
        got = PH.build(CUR, PREV, PMIN, PMAX, PH.NONE)
        want = np.clip(2.0 * (CUR - PMIN) / (PMAX - PMIN + 1e-8) - 1.0, -1.0, 1.0).astype(np.float32)
        self.assertEqual(got.shape, (14,))
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-7)

    def test_true_mode_layout_is_current_then_previous(self):
        got = PH.build(CUR, PREV, PMIN, PMAX, PH.TRUE)
        self.assertEqual(got.shape, (28,))
        np.testing.assert_allclose(got[:14], PH.build(CUR, None, PMIN, PMAX, PH.NONE), atol=1e-7)
        np.testing.assert_allclose(got[14:], PH.build(PREV, None, PMIN, PMAX, PH.NONE), atol=1e-7)

    def test_duplicate_mode_cannot_see_history(self):
        a = PH.build(CUR, PREV, PMIN, PMAX, PH.DUPLICATE)
        b = PH.build(CUR, PREV + 0.37, PMIN, PMAX, PH.DUPLICATE)
        c = PH.build(CUR, None, PMIN, PMAX, PH.DUPLICATE)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(a[:14], a[14:])

    def test_true_mode_does_see_history(self):
        a = PH.build(CUR, PREV, PMIN, PMAX, PH.TRUE)
        b = PH.build(CUR, PREV + 0.1, PMIN, PMAX, PH.TRUE)
        self.assertGreater(float(np.abs(a - b).max()), 0.0)
        self.assertGreater(float(np.abs(a[14:] - a[:14]).mean()), 0.0)

    def test_first_frame_policy(self):
        a = PH.build(CUR, None, PMIN, PMAX, PH.TRUE)
        # cold start: slot 1 is filled with the SAME normalized current state (zero velocity)
        self.assertTrue(np.array_equal(a[14:], a[:14]))
        self.assertEqual(PH.first_frame_policy()["implied_velocity"], 0.0)

    def test_normalization_is_bounded(self):
        huge = np.full(14, 1e6, dtype=np.float32)
        got = PH.build(huge, -huge, PMIN, PMAX, PH.TRUE)
        self.assertTrue(np.all(got <= 1.0) and np.all(got >= -1.0))

    def test_rejects_nonfinite_and_bad_bounds(self):
        bad = CUR.copy(); bad[3] = np.nan
        with self.assertRaises(ValueError):
            PH.build(bad, PREV, PMIN, PMAX, PH.TRUE)
        with self.assertRaises(ValueError):
            PH.build(CUR, PREV, PMAX, PMIN, PH.TRUE)
        with self.assertRaises(ValueError):
            PH.build(CUR[:13], PREV, PMIN, PMAX, PH.TRUE)


class TestHistoryBuffer(unittest.TestCase):
    def test_buffer_uses_observed_previous_state(self):
        buf = PH.HistoryBuffer(PH.TRUE)
        first = buf.observe(CUR, PMIN, PMAX)
        np.testing.assert_array_equal(first[:14], first[14:])          # cold start: zero velocity
        buf.push(CUR, 0)
        second = buf.observe(CUR + 0.2, PMIN, PMAX)
        want_prev = PH.build(CUR, None, PMIN, PMAX, PH.NONE)
        np.testing.assert_allclose(second[14:], want_prev, atol=1e-6)

    def test_reset_drops_history(self):
        buf = PH.HistoryBuffer(PH.TRUE)
        buf.push(CUR, 0); buf.push(CUR + 1.0, 0)
        buf.reset(0)
        out = buf.observe(CUR + 2.0, PMIN, PMAX)
        np.testing.assert_array_equal(out[:14], out[14:])

    def test_env_isolation(self):
        buf = PH.HistoryBuffer(PH.TRUE)
        buf.push(CUR, 0)
        other = buf.observe(CUR + 0.5, PMIN, PMAX, env_idx=1)
        np.testing.assert_array_equal(other[:14], other[14:])

    def test_duplicate_buffer_matches_duplicate_build(self):
        buf = PH.HistoryBuffer(PH.DUPLICATE)
        buf.push(CUR, 0)
        out = buf.observe(CUR + 0.3, PMIN, PMAX)
        np.testing.assert_array_equal(out, PH.build(CUR + 0.3, CUR, PMIN, PMAX, PH.DUPLICATE))


class TestCheckpointMetadata(unittest.TestCase):
    def test_legacy_checkpoint_resolves_to_none(self):
        self.assertEqual(PH.resolve_mode({}), PH.NONE)
        PH.verify_projector_width({}, 14)

    def test_28d_checkpoint_requires_explicit_metadata(self):
        with self.assertRaises(ValueError):
            PH.verify_projector_width({}, 28)
        meta = {"proprio_dim": 28}
        with self.assertRaises(ValueError):
            PH.resolve_mode(meta)                     # mode missing -> inconsistent

    def test_inconsistent_metadata_rejected(self):
        with self.assertRaises(ValueError):
            PH.resolve_mode({"proprio_dim": 28, "proprio_history_mode": PH.NONE})
        with self.assertRaises(ValueError):
            PH.resolve_mode({"proprio_dim": 14, "proprio_history_mode": PH.TRUE})

    def test_checkpoint_fields_roundtrip(self):
        for mode, dim in ((PH.NONE, 14), (PH.DUPLICATE, 28), (PH.TRUE, 28)):
            f = PH.checkpoint_fields(mode)
            self.assertEqual(f["proprio_dim"], dim)
            self.assertEqual(PH.resolve_mode(f), mode)
            PH.verify_projector_width(f, dim, mode)

    def test_mode_mismatch_between_checkpoint_and_deployment(self):
        f = PH.checkpoint_fields(PH.TRUE)
        with self.assertRaises(ValueError):
            PH.verify_projector_width(f, 28, PH.DUPLICATE)

    def test_time_alignment_report(self):
        self.assertEqual(PH.time_alignment_report(CUR, CUR, 0)["policy"],
                         "previous_state := current_state")
        with self.assertRaises(ValueError):
            PH.time_alignment_report(CUR, PREV, 0)
        self.assertEqual(PH.time_alignment_report(CUR, PREV, 7)["frame"], 7)
        with self.assertRaises(ValueError):
            PH.time_alignment_report(CUR, None, 7)


class TestDeploymentIntegration(unittest.TestCase):
    """The deployed policy must not hard-code the 14D interface any more."""

    def setUp(self):
        self.path = os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure", "model.py")
        self.src = open(self.path).read()

    def test_policy_resolves_proprio_interface_from_metadata(self):
        self.assertIn("proprio_history as PH", self.src)
        self.assertIn("PH.resolve_mode(meta)", self.src)

    def test_projector_width_is_not_hardcoded_to_14(self):
        self.assertIn("ProprioProjector(proprio_dim=self.proprio_dim", self.src)
        self.assertNotIn("ProprioProjector(proprio_dim=14", self.src)

    def test_inference_uses_the_history_buffer(self):
        self.assertIn("self.proprio_history.observe(", self.src)
        self.assertIn("self.proprio_history.reset()", self.src)

    def test_checkpoint_width_mismatch_is_refused_before_loading(self):
        self.assertIn("PH.verify_projector_width(meta, width, self.proprio_history_mode)", self.src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
