"""CPU Gate for two-task plan derivation and replacement resume cursor."""
import unittest

from training.r51_formal_plan import (epoch_draws, two_task_plan,
                                      validate_resume_cursor)


def fake_full():
    counts = {"push_T": (25313, 6691), "stack_bowls": (35858, 8716),
              "sweep_blocks": (47609, 12861)}
    train = [(task, f"train_{i}", 2) for task, (n, _) in counts.items()
             for i in range(n)]
    dev = [(task, f"dev_{i}", 2) for task, (_, n) in counts.items()
           for i in range(n)]
    paths = {f"{task}/episode": task for task in counts}
    return {"n_train": len(train), "n_dev": len(dev),
            "train_samples": train, "dev_samples": dev,
            "split_sha256": "split", "dataset_manifest_sha256": "manifest",
            "feature_normalizer_sha256": "feature", "delta_normalizer_sha256": "delta",
            "feature_payload": {"tasks": {}},
            "delta_payload": {"tasks": {t: {"n_rows": n}
                                        for t, (n, _) in counts.items()}},
            "episode_paths": paths,
            "manifest_files": {k: {} for k in paths},
            "file_stats_before": {k: () for k in paths},
            "instruction_sha256": {k: k for k in counts},
            "train_samples_sha256": "full-train", "dev_samples_sha256": "full-dev"}


class FormalPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = fake_full()

    def test_two_task_subset_preserves_source_and_dev_isolation(self):
        plan = two_task_plan(self.full)
        self.assertEqual((plan["n_train"], plan["n_dev"]), (72922, 19552))
        self.assertEqual(set(plan["tasks"]), {"push_T", "sweep_blocks"})
        self.assertNotIn("stack_bowls", {row[0] for row in plan["train_samples"]})
        self.assertNotIn("stack_bowls", {row[0] for row in plan["dev_samples"]})
        self.assertEqual(plan["source_train_samples_sha256"], "full-train")
        self.assertEqual(len(self.full["train_samples"]), 108780)

    def test_replacement_resume_uses_draw_cursor(self):
        plan = two_task_plan(self.full)
        draws, summary = epoch_draws(plan, seed=51, epoch=2, pool_size=500)
        self.assertEqual((len(draws), summary["draws"]), (72922, 72922))
        self.assertLessEqual(summary["unique_samples"], 500)
        identity = validate_resume_cursor(plan, draws, cursor=731,
                                          committed_steps=2 * 72922 + 731, epoch=2)
        self.assertEqual(len(identity), 64)
        with self.assertRaises(ValueError):
            validate_resume_cursor(plan, draws, cursor=731,
                                   committed_steps=2 * 72922 + 730, epoch=2)

    def test_contract_drift_rejected(self):
        altered = dict(self.full)
        altered["delta_payload"] = {"tasks": {**self.full["delta_payload"]["tasks"],
                                              "push_T": {"n_rows": 1}}}
        with self.assertRaises(ValueError):
            two_task_plan(altered)

    def test_formal_entry_uses_r51_draws_and_fails_unsafe_flags_before_cuda(self):
        from training import r44_train
        plan = two_task_plan(self.full)
        args = r44_train.parse_args(["--out", "/tmp/r51-test", "--r51-two-task",
                                     "--r51-active-pool", "500"])
        draws = r44_train.epoch_order(args, plan, 1, None)
        summary = r44_train.r51_epoch_summary(args, plan, 1, draws)
        self.assertEqual(len(draws), 72922)
        self.assertEqual(summary["optimizer_steps"], 72922)
        self.assertLessEqual(summary["unique_samples"], 500)
        self.assertEqual(summary["effective_batch"], 1)
        with self.assertRaises(SystemExit):
            r44_train.main(["--out", "/tmp/r51-test", "--r51-two-task",
                            "--epochs", "6"])
        with self.assertRaises(SystemExit):
            r44_train.main(["--out", "/tmp/r51-test", "--r51-two-task",
                            "--r51-active-pool", "500", "--r51-sampling",
                            "without_replacement"])

    def test_formal_r44_default_preserves_no_replacement_order(self):
        from training import r44_train
        from training.r44_delta_v1 import EpochSampler
        from types import SimpleNamespace
        args = r44_train.parse_args(["--out", "/tmp/r44-test"])
        order = r44_train.epoch_order(args, {"n_train": 20}, 0,
                                      SimpleNamespace(EpochSampler=EpochSampler))
        self.assertEqual(sorted(order.tolist()), list(range(20)))

    def test_r51_entry_forwards_explicit_mode_once(self):
        from training import r51_train
        from unittest.mock import patch
        with patch.object(r51_train.r44_train, "main", return_value=0) as run:
            self.assertEqual(r51_train.main(["--out", "/tmp/r51-test"]), 0)
        run.assert_called_once_with(["--r51-two-task", "--out", "/tmp/r51-test"])
        with self.assertRaises(SystemExit):
            r51_train.main(["--r51-two-task"])


if __name__ == "__main__":
    unittest.main()
