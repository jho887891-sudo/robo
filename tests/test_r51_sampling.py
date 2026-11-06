"""R51 two-task draw contract without CUDA or model loading."""
import unittest

from training.r51_sampling import draw_epoch


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.entries = [("push_T", i) for i in range(7)] + [
            ("sweep_blocks", i) for i in range(13)]

    def test_replacement_is_reproducible_and_epoch_length_fixed(self):
        first, summary = draw_epoch(self.entries, base_seed=51, epoch=2,
                                    mode="replacement", task_mix="balanced",
                                    pool_size=6)
        again, same = draw_epoch(self.entries, base_seed=51, epoch=2,
                                 mode="replacement", task_mix="balanced",
                                 pool_size=6)
        self.assertEqual(first, again)
        self.assertEqual(summary, same)
        self.assertEqual(len(first), len(self.entries))
        self.assertEqual(sum(summary["task_draws"].values()), len(first))
        self.assertLessEqual(summary["unique_samples"], 6)
        next_epoch, next_summary = draw_epoch(self.entries, base_seed=51, epoch=3,
                                              mode="replacement", pool_size=6)
        self.assertNotEqual(first, next_epoch)
        self.assertEqual(summary["pool_sha256"], next_summary["pool_sha256"])

    def test_no_replacement_visits_every_entry_once(self):
        draws, summary = draw_epoch(self.entries, base_seed=51, epoch=0,
                                    mode="without_replacement")
        self.assertEqual(sorted(draws), list(range(len(self.entries))))
        self.assertEqual(summary["duplicate_draws"], 0)

    def test_third_task_rejected(self):
        with self.assertRaises(ValueError):
            draw_epoch(self.entries + [("stack_bowls", 0)], base_seed=51, epoch=0)


if __name__ == "__main__":
    unittest.main()
