#!/usr/bin/env python3
"""CPU-only R5 distribution/validation sampling and diagnostics tests."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "training"


def load(name):
    f = ROOT / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load("r5_sampling")
M = load("r5_metrics")


class R5SamplingTests(unittest.TestCase):
    def setUp(self):
        self.split = {"train": ["B", "A", "C"], "val": ["V", "W"]}
        self.rows = []
        for e in self.split["train"]:
            states = np.zeros((145, 14), np.float32)
            actions = states.copy()
            for t in range(145):
                actions[t, :7] = ((t % 4) / 100.0) + (0.01 if e == "B" else 0)
                actions[t, 7:] = ((t % 7) / 100.0) + (0.01 if e == "C" else 0)
            self.rows.extend(S.candidate_frames(e, states, actions, stride=20))

    def test_episode_coverage_deterministic(self):
        v = S.diversified_manifest(self.split, self.rows, n=12, seed=42)
        self.assertEqual({x[0] for x in v}, {"A", "B", "C"})
        self.assertEqual(len({(x[0], x[1]) for x in v}), 12)
        self.assertEqual(v, S.diversified_manifest(self.split, self.rows, n=12, seed=42))

    def test_refuses_validation_candidate(self):
        with self.assertRaisesRegex(ValueError, "non-training"):
            S.diversified_manifest(self.split, self.rows + [("V", 3, .1, 0)], n=2)

    def test_refuses_train_val_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            S.diversified_manifest({"train": ["A", "V"], "val": ["V"]}, self.rows, n=1)

    def test_candidate_frame_excludes_end(self):
        a = np.zeros((100, 14), np.float32)
        rows = S.candidate_frames("A", a, a, stride=20)
        self.assertEqual([row[1] for row in rows], [20, 40, 60, 80])

    def test_summarize_activity(self):
        s = S.summarize_manifest([("A", 2, 0.1, 0), ("B", 1, 0, .1),
                                  ("C", 4, .1, .1), ("C", 5, 0, 0)])
        self.assertEqual(s["activity_counts"],
                         {"left_only": 1, "right_only": 1, "both_active": 1, "neither_active": 1})

    def test_validation_spread_all_episodes(self):
        val_lengths = {"V": 110, "W": 80}
        v = S.spread_validation_frames(self.split, val_lengths, n=6, seed=1)
        self.assertEqual(len(v), 6)
        self.assertEqual(len({tuple(x) for x in v}), 6)
        self.assertEqual([sum(1 for e, t in v if e == target) for target in ["V", "W"]], [3, 3])
        self.assertEqual(v, S.spread_validation_frames(self.split, val_lengths, n=6, seed=1))

    def test_reject_insufficient_val_coverage(self):
        with self.assertRaisesRegex(ValueError, "every held-out"):
            S.spread_validation_frames(self.split, {"V": 80, "W": 80}, n=1)


class R5MetricsTests(unittest.TestCase):
    def test_per_episode_macro_distinct_from_frame_mean(self):
        a = np.zeros((3, 14), np.float32)
        p = a.copy()
        p[:2] += 1.0
        result = M.summarize_prediction(p, a, a, a,
                                        ["episodeA", "episodeA", "episodeB"], [[100]*14]*3)
        self.assertAlmostEqual(result["episode_macro_L1"]["model"], 0.5)
        self.assertEqual(result["evaluation_episode_count"], 2)
        self.assertEqual(result["per_dim_generated_token_diversity"][0]["unique_tokens"], 1)
        self.assertEqual(result["unique_generated_token_rows"], 1)

    def test_velocity_baseline(self):
        s = np.array([[.10]*14, [.20]*14], np.float32)
        old = np.array([[.00]*14, [.10]*14], np.float32)
        gt = s + .10
        r = M.summarize_prediction(s, gt, s, old, ["e1", "e2"], [[120]*14]*2)
        self.assertLess(r["episode_macro_L1"]["constant_velocity"], 1e-6)
        self.assertAlmostEqual(r["episode_macro_L1"]["hold_state"], 0.1, places=5)

    def test_right_arm_detects_error(self):
        gt = np.zeros((2,14), np.float32)
        p = gt.copy(); p[:, 7:] += 0.1
        r = M.summarize_prediction(p, gt, gt, gt, ["e1", "e2"], [[123]*14]*2)
        self.assertAlmostEqual(r["per_arm_physical_L1"]["left"]["model"], 0.0)
        self.assertAlmostEqual(r["per_arm_physical_L1"]["right"]["model"], 0.1, places=6)

    def test_reject_missing_pred_tokens(self):
        x = np.zeros((2, 14), np.float32)
        with self.assertRaisesRegex(ValueError, "generated action tokens"):
            M.summarize_prediction(x, x, x, x, ["e1", "e2"], [[127]*14])

    def test_median_delta_baseline(self):
        x = np.zeros((2,14), np.float32)
        gt = x + .03
        r = M.summarize_prediction(x, gt, x, x, ["e1", "e2"], [[127]*14]*2,
                                   median_delta=np.ones(14, np.float32)*.03)
        self.assertLess(r["episode_macro_L1"]["train_median_delta"], 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
