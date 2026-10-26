#!/usr/bin/env python3
"""CPU tests for the joint_delta_v1 contract, epoch sampler and offline decoder."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "training"))

import r44_delta_v1 as D  # noqa: E402


def fake_deltas(n=500, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for task, scale in (("push_T", 0.02), ("stack_bowls", 0.05)):
        d = rng.normal(0, scale, (n, 14)).astype(np.float32)
        d[:, 6] = rng.uniform(-0.5, 0.5, n)
        d[:, 13] = rng.uniform(-0.5, 0.5, n)
        out[task] = d
    return out


class TestTarget(unittest.TestCase):
    def test_delta_is_action_minus_state(self):
        s = np.arange(14, dtype=np.float32)
        a = s + 0.5
        np.testing.assert_allclose(D.delta_target(s, a), np.full(14, 0.5))

    def test_step_contract_requires_bit_exact(self):
        s = np.zeros(14, np.float32)
        s2 = np.ones(14, np.float32)
        D.verify_step_contract(s, s2.copy(), s2)
        with self.assertRaises(D.R44ContractError):
            D.verify_step_contract(s, s2 + 1e-6, s2)

    def test_non_finite_rejected(self):
        s = np.zeros(14, np.float32)
        s[3] = np.nan
        with self.assertRaises(D.R44ContractError):
            D.delta_target(s, np.zeros(14, np.float32))


class TestNormalizer(unittest.TestCase):
    def test_fit_is_train_only_and_hashed(self):
        payload = D.fit_normalizer(fake_deltas())
        self.assertEqual(payload["target"], D.CONTRACT_ID)
        self.assertEqual(payload["fit"]["source"], "TRAIN only")
        self.assertEqual(payload["sha256"], D.payload_sha256(payload))
        D.validate_normalizer(payload)

    def test_wrong_schema_and_token_window_are_rejected_even_with_recomputed_hash(self):
        bad_schema = D.fit_normalizer(fake_deltas())
        bad_schema["schema"] = "unknown/v9"
        bad_schema["sha256"] = D.payload_sha256(bad_schema)
        with self.assertRaisesRegex(D.R44ContractError, "schema"):
            D.validate_normalizer(bad_schema)

        bad_window = D.fit_normalizer(fake_deltas())
        bad_window["token_window"] = [31743, 31999]
        bad_window["sha256"] = D.payload_sha256(bad_window)
        with self.assertRaisesRegex(D.R44ContractError, "token window"):
            D.validate_normalizer(bad_window)

    def test_legacy_residual_payload_is_refused(self):
        payload = D.fit_normalizer(fake_deltas())
        legacy = dict(payload)
        legacy["target"] = "velocity_residual"
        with self.assertRaises(D.R44ContractError):
            D.validate_normalizer(legacy)
        legacy2 = dict(payload)
        legacy2["tasks"] = {k: dict(v) for k, v in payload["tasks"].items()}
        legacy2["tasks"]["push_T"]["q01"] = [0.0] * 14
        with self.assertRaises(D.R44ContractError):
            D.validate_normalizer(legacy2)

    def test_tampering_breaks_the_hash(self):
        payload = D.fit_normalizer(fake_deltas())
        payload["tasks"]["push_T"]["q99"][0] += 1e-3
        with self.assertRaises(D.R44ContractError):
            D.validate_normalizer(payload)

    def test_refuse_legacy_checkpoint_metadata(self):
        with self.assertRaises(D.R44ContractError):
            D.refuse_legacy({"target_representation": "velocity_residual"})
        self.assertTrue(D.refuse_legacy({"target_representation": D.CONTRACT_ID}))

    def test_degenerate_dimension_is_rejected(self):
        d = fake_deltas()
        d["push_T"][:, 5] = 0.0
        with self.assertRaises(D.R44ContractError):
            D.fit_normalizer(d)

    def test_too_few_rows_rejected(self):
        with self.assertRaises(D.R44ContractError):
            D.fit_normalizer({"push_T": np.zeros((10, 14), np.float32)})


class TestCodec(unittest.TestCase):
    def test_roundtrip_and_legal_tokens(self):
        payload = D.fit_normalizer(fake_deltas())
        q01, q99 = D.normalizer_for_task(payload, "push_T")
        span = q99 - q01
        d = fake_deltas()["push_T"][:200]
        worst_norm = 0.0
        for row in d:
            idx, tok = D.encode_delta(row, "push_T", payload)
            self.assertEqual(idx.shape, (14,))
            self.assertTrue((tok >= D.LEGAL_TOKEN_MIN).all())
            self.assertTrue((tok <= D.LEGAL_TOKEN_MAX).all())
            back = D.decode_delta(idx, "push_T", payload)
            # only in-range values can be represented at all; out-of-range values are clipped
            # by the [-1,1] normalisation (measured separately in the preflight report)
            inside = (row >= q01) & (row <= q99)
            if inside.any():
                worst_norm = max(worst_norm, float(
                    np.max(np.abs(back - row)[inside] / span[inside])))
        self.assertLessEqual(worst_norm, 0.5 * D.MAX_NORM_QUANT_ERROR + 1e-4)

    def test_out_of_range_values_are_clipped_not_extrapolated(self):
        payload = D.fit_normalizer(fake_deltas())
        q01, q99 = D.normalizer_for_task(payload, "push_T")
        beyond = (q99 + 10.0 * (q99 - q01)).astype(np.float32)
        idx, _tok = D.encode_delta(beyond, "push_T", payload)
        back = D.decode_delta(idx, "push_T", payload)
        np.testing.assert_allclose(back, q99, rtol=0, atol=1e-5)

    def test_recover_action_adds_state_not_velocity(self):
        payload = D.fit_normalizer(fake_deltas())
        s = np.linspace(-0.2, 0.2, 14).astype(np.float32)
        a = s + 0.01
        err, idx, tok = D.roundtrip_error(s, a, "push_T", payload)
        self.assertEqual(err.shape, (14,))
        a_hat = D.recover_action(s, idx, "push_T", payload)
        np.testing.assert_allclose(a_hat, s + D.decode_delta(idx, "push_T", payload))


class TestSampler(unittest.TestCase):
    def test_epoch_is_a_complete_permutation(self):
        s = D.EpochSampler(1000, 0, 3)
        self.assertEqual(len(s.order), 1000)
        self.assertEqual(len(set(s.order.tolist())), 1000)
        cov = s.coverage()
        self.assertTrue(cov["complete"])
        self.assertEqual((cov["min_visits"], cov["max_visits"]), (1, 1))

    def test_deterministic_and_epoch_dependent(self):
        a = D.EpochSampler(500, 7, 0)
        b = D.EpochSampler(500, 7, 0)
        c = D.EpochSampler(500, 7, 1)
        self.assertEqual(a.order_hash(), b.order_hash())
        self.assertNotEqual(a.order_hash(), c.order_hash())
        d = D.EpochSampler(500, 8, 0)
        self.assertNotEqual(a.order_hash(), d.order_hash())


if __name__ == "__main__":
    unittest.main(verbosity=2)
