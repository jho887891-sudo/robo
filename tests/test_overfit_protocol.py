#!/usr/bin/env python3
"""No-GPU, no-HDF5 tests for R3 experiment split hygiene and prediction logs."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "training" / "overfit_protocol.py"
spec = importlib.util.spec_from_file_location("overfit_protocol", PATH)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class TestTrainOnlySampling(unittest.TestCase):
    def setUp(self):
        self.split = {"train": ["ep03", "ep01", "ep02"], "val": ["ep99", "ep90"]}
        self.lengths = {"ep01": 101, "ep02": 61, "ep03": 71}

    def test_samples_from_train_never_validation(self):
        xs = P.select_train_samples(self.split, self.lengths, n=4)
        self.assertEqual(xs, [["ep01", 20], ["ep01", 60], ["ep02", 20], ["ep03", 20]])
        self.assertFalse({x[0] for x in xs}.intersection(self.split["val"]))

    def test_repeatable_selection(self):
        a = P.select_train_samples(self.split, self.lengths, n=4)
        b = P.select_train_samples(self.split, self.lengths, n=4)
        self.assertEqual(a, b)

    def test_reject_overlapping_splits(self):
        split = {"train": ["ep01", "ep99"], "val": ["ep99"]}
        with self.assertRaisesRegex(ValueError, "leakage"):
            P.select_train_samples(split, {"ep01": 100, "ep99": 100}, n=1)

    def test_short_episodes_do_not_create_out_of_range_frame(self):
        split = {"train": ["short", "long"], "val": []}
        result = P.select_train_samples(split, {"short": 2, "long": 30}, n=1)
        self.assertEqual(result, [["long", 20]])

    def test_insufficient_data_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "not enough"):
            P.select_train_samples(self.split, self.lengths, n=100)

    def test_bad_arguments_rejected(self):
        for kwargs in ({"n": 0}, {"start_frame": -1}, {"stride": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    P.select_train_samples(self.split, self.lengths, **kwargs)


class TestGeneratedTokenLogs(unittest.TestCase):
    def test_log_uses_prediction_not_gt(self):
        predicted = [3] * 14
        gt = [7] * 14
        logged = P.validated_generation_indices(predicted)
        self.assertEqual(logged, predicted)
        self.assertNotEqual(logged, gt)

    def test_invalid_prediction_length_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 14"):
            P.validated_generation_indices([0] * 13)

    def test_invalid_prediction_index_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            P.validated_generation_indices([0] * 13 + [256])


if __name__ == "__main__":
    unittest.main(verbosity=2)
