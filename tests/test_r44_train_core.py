import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training.r44_train import (
    LORA_ALPHA, LORA_R, LORA_SCALE, LORA_TARGET_MODULES,
    build_label_block, estimate_budget_hours, parse_args, task_episode_names,
    episode_images, sha256_file, stat_identity, validate_resume_compatibility,
    verify_files_unchanged,
)


class R44TrainerCoreTests(unittest.TestCase):
    def test_task_episode_names_preserves_task_qualification(self):
        class FakeM2:
            @staticmethod
            def split_qualified(key):
                if not isinstance(key, str) or key.count("/") != 1:
                    raise ValueError("invalid qualified key")
                return tuple(key.split("/"))

        self.assertEqual(
            task_episode_names("push_T", ["push_T/episode_0000000.hdf5"], FakeM2),
            ["episode_0000000.hdf5"],
        )
        with self.assertRaisesRegex(ValueError, "belongs to"):
            task_episode_names("push_T", ["stack_bowls/episode_0000000.hdf5"], FakeM2)

    def test_frozen_lora_hyperparameters_match_project_contract(self):
        self.assertEqual(LORA_R, 32)
        self.assertEqual(LORA_ALPHA, 16)
        self.assertEqual(LORA_SCALE, 0.5)
        self.assertEqual(LORA_TARGET_MODULES, ("q_proj", "v_proj"))

    def test_multitask_image_loader_uses_frozen_three_camera_order(self):
        import sys
        import types
        import numpy as np
        from unittest.mock import patch

        accessed = []

        class FakeDataset:
            def __len__(self):
                return 3
            def __getitem__(self, index):
                return b"jpeg-bytes"

        class FakeFile:
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def __getitem__(self, key):
                accessed.append(key)
                return FakeDataset()

        class FakeTensor:
            def __init__(self, shape):
                self.shape = tuple(shape)
            def unsqueeze(self, dim):
                shape = list(self.shape)
                shape.insert(dim, 1)
                return FakeTensor(shape)
            def to(self, *_args):
                return self

        fake_torch = types.SimpleNamespace(
            bfloat16="bf16",
            cat=lambda tensors, dim: FakeTensor(
                (len(tensors),) + tuple(tensors[0].shape[1:])
            ),
        )
        fake_cv2 = types.SimpleNamespace(
            IMREAD_COLOR=1,
            imdecode=lambda _buf, _flag: np.zeros((4, 4, 3), dtype=np.uint8),
        )
        fake_h5py = types.SimpleNamespace(File=lambda _path, _mode: FakeFile())
        fake_pil = types.ModuleType("PIL")
        fake_pil.Image = types.SimpleNamespace(fromarray=lambda image: image)

        class FakeImageProcessor:
            def apply_transform(self, _image):
                return FakeTensor((1, 3, 224, 224))

        proc = types.SimpleNamespace(image_processor=FakeImageProcessor())
        fake_m2 = types.SimpleNamespace(
            CAMERA_ORDER=("cam_head", "cam_left_wrist", "cam_right_wrist")
        )
        with patch.dict(sys.modules, {
            "torch": fake_torch, "cv2": fake_cv2, "h5py": fake_h5py, "PIL": fake_pil,
        }):
            result = episode_images(proc, "/tmp/fake-task/episode.hdf5", 1, 3,
                                    (None, None, None, fake_m2))
        self.assertEqual(accessed, [
            "vision/cam_head/colors",
            "vision/cam_left_wrist/colors",
            "vision/cam_right_wrist/colors",
        ])
        self.assertEqual(result.shape, (1, 3, 3, 224, 224))

    def test_formal_defaults_are_five_epochs_and_early_stopping_disabled(self):
        args = parse_args(["--out", "/tmp/r44-test-output"])
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.early_stopping_patience, 0)
        self.assertEqual(args.resource_mode, "time-window")
        self.assertFalse(args.formal_training_authorized)

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
        self.assertAlmostEqual(result["dev_hours_estimate"], 20 / 3600)
        self.assertAlmostEqual(result["base_total_hours_estimate"], 630 / 3600)
        self.assertAlmostEqual(result["required_budget_hours"], 693 / 3600)
        self.assertTrue(result["is_estimate_not_measurement"])

    def test_budget_rejects_invalid_counts(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            estimate_budget_hours(0, 20, 2, 1.0, 1.0, 1.0, 0.0)

    def test_budget_rejects_nonfinite_timing_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            estimate_budget_hours(10, 10, 1, float("nan"), 1.0, 1.0, 0.0)

    def test_verify_files_unchanged_detects_content_change_with_same_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.hdf5"
            path.write_bytes(b"AAAA")
            original_stat = path.stat()
            key = "push_T/episode_0000000.hdf5"
            plan = {
                "episode_paths": {key: path},
                "file_stats_before": {key: stat_identity(path)},
                "manifest_files": {key: {"sha256": sha256_file(path)}},
            }
            path.write_bytes(b"BBBB")
            os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            self.assertEqual(stat_identity(path), plan["file_stats_before"][key])
            with self.assertRaisesRegex(RuntimeError, "content SHA256 changed"):
                verify_files_unchanged(plan)

    def test_resume_contract_pins_source_data_and_schedule(self):
        args = SimpleNamespace(
            epochs=5, min_epochs=2, seed=0, gradient_checkpointing=False,
            early_stopping_patience=0,
        )
        plan = {
            "train_samples_sha256": "train", "dev_samples_sha256": "dev",
            "split_sha256": "split", "feature_normalizer_sha256": "feature",
            "delta_normalizer_sha256": "delta", "dataset_manifest_sha256": "manifest",
        }
        base = {"weights": "sha"}
        model = {
            "modeling_sha256": "model", "lora_config": {"r": 32, "lora_alpha": 16},
            "target_module_count": 2, "trainable_lora_parameter_count": 100,
            "projector_parameter_count": 200, "prompt_text_sha256": {"push_T": "prompt"},
            "prompt_token_ids_sha256": {"push_T": "tokens"},
        }
        scheduler = {"total_steps": 500, "warmup_steps": 5}
        meta = {
            "schema": "r44-run-meta/v1", "source_sha": "commit", "epochs_max": 5,
            "min_epochs": 2, "seed": 0, "gradient_checkpointing": False,
            "target_representation": "joint_delta_v1", **plan,
            "base_model_identity": base, "scheduler": scheduler,
            "early_stopping_patience": 0, "model_info": model,
        }
        validate_resume_compatibility(meta, args, plan, base, model, scheduler, "commit")
        args.epochs = 4
        with self.assertRaisesRegex(ValueError, "epochs_max"):
            validate_resume_compatibility(meta, args, plan, base, model, scheduler, "commit")

    def test_budget_rejects_negative_margin(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            estimate_budget_hours(10, 10, 1, 1.0, 1.0, 1.0, -0.1)


if __name__ == "__main__":
    unittest.main()
