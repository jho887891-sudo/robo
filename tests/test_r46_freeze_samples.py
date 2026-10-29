from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diagnostics.r46_freeze_samples import ordered_train_episodes, sample_time_index  # noqa: E402


class FreezeSampleSelectionTests(unittest.TestCase):
    def test_episode_order_is_deterministic_and_unique(self):
        source = [f"episode_{i:07d}.hdf5" for i in range(20)]
        self.assertEqual(ordered_train_episodes(source), ordered_train_episodes(list(reversed(source))))
        self.assertEqual(len(set(ordered_train_episodes(source))), len(source))

    def test_sample_time_is_deterministic_and_inside_train_history_range(self):
        for frames in (4, 25, 480):
            t = sample_time_index("episode_0000003.hdf5", frames)
            self.assertEqual(t, sample_time_index("episode_0000003.hdf5", frames))
            self.assertGreaterEqual(t, 2)
            self.assertLess(t, frames - 1)

    def test_too_short_episode_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "at least four"):
            sample_time_index("episode_0000003.hdf5", 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
