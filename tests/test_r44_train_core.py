import unittest

from training.r44_train import build_label_block, estimate_budget_hours


class R44TrainerCoreTests(unittest.TestCase):
    def test_label_block_masks_prompt_and_supervises_14_tokens_plus_stop(self):
        prompt = [1, 29871, 42]
        actions = list(range(31744, 31758))
        result = build_label_block(prompt, actions, stop_token_id=2)
        self.assertEqual(result["input_ids"], prompt + actions + [2])
        self.assertEqual(result["labels"], [-100] * len(prompt) + actions + [2])
        self.assertEqual(result["n_prompt"], len(prompt))
        self.assertEqual(len(result["action_tokens"]), 14)

    def test_label_block_rejects_wrong_action_length(self):
        with self.assertRaisesRegex(ValueError, "expected 14"):
            build_label_block([1, 2], [31744, 31745])

    def test_label_block_rejects_illegal_token(self):
        tokens = [31744] * 14
        tokens[5] = 32000
        with self.assertRaisesRegex(ValueError, "legal token window"):
            build_label_block([1, 2], tokens)

    def test_budget_includes_train_dev_load_and_margin(self):
        result = estimate_budget_hours(
            n_train=100, n_dev=20, epochs=2,
            seconds_per_train_sample=3.0, seconds_per_dev_sample=0.5,
            model_load_seconds=10.0, margin_fraction=0.1,
        )
        self.assertEqual(result["optimizer_steps_max"], 200)
        self.assertAlmostEqual(result["train_hours_estimate"], 600 / 3600)
        self.assertAlmostEqual(result["dev_hours_estimate"], 10 / 3600)
        self.assertAlmostEqual(result["base_total_hours_estimate"], 620 / 3600)
        self.assertAlmostEqual(result["required_budget_hours"], 682 / 3600)
        self.assertTrue(result["is_estimate_not_measurement"])

    def test_budget_rejects_invalid_counts(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            estimate_budget_hours(0, 20, 2, 1.0, 1.0, 1.0, 0.0)

    def test_budget_rejects_negative_margin(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            estimate_budget_hours(10, 10, 1, 1.0, 1.0, 1.0, -0.1)


if __name__ == "__main__":
    unittest.main()
