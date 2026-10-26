import contextlib
import hashlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from training.r44_train import (
    LORA_ALPHA, LORA_R, LORA_SCALE, LORA_TARGET_MODULES,
    build_label_block, estimate_budget_hours, parse_args, task_episode_names,
    episode_images, sha256_file, stat_identity, validate_resume_compatibility,
    verify_files_unchanged, verify_exact_action_next_state,
    verify_delta_normalizer_matches_train, TASKS, should_early_stop,
    validate_preflight_report, validate_gpu_smoke_report,
    budget_remaining_seconds, load_budget_ledger, persist_budget_ledger,
    run_artifact_manifest_path, write_run_artifact_manifest,
)
from training import r44_delta_v1 as D
from diagnostics.r44_gpu_smoke import main as gpu_smoke_main


class R44TrainerCoreTests(unittest.TestCase):
    def test_exact_transition_contract_accepts_exact_next_state(self):
        import numpy as np
        rng = np.random.default_rng(7)
        state = rng.normal(size=(8, 14)).astype(np.float32)
        action = rng.normal(size=(8, 14)).astype(np.float32)
        action[:-1] = state[1:]
        result = verify_exact_action_next_state(
            state, action, task="push_T", episode="episode_0000000.hdf5"
        )
        self.assertTrue(result["exact"])
        self.assertEqual(result["transitions"], 7)
        self.assertEqual(result["max_abs_diff"], 0.0)

    def test_exact_transition_contract_rejects_tiny_nonzero_difference(self):
        import numpy as np
        state = np.zeros((4, 14), dtype=np.float32)
        action = np.zeros((4, 14), dtype=np.float32)
        action[:-1] = state[1:]
        action[0, 0] = np.nextafter(np.float32(0.0), np.float32(1.0))
        with self.assertRaisesRegex(ValueError, "exact A\\[t\\] == S\\[t\\+1\\] contract failed"):
            verify_exact_action_next_state(
                state, action, task="push_T", episode="episode_0000000.hdf5"
            )

    def test_delta_normalizer_must_reproduce_from_exact_train_rows(self):
        import numpy as np
        rng = np.random.default_rng(44)
        deltas = {
            task: rng.normal(loc=i * 0.2, scale=0.5, size=(400, 14)).astype(np.float32)
            for i, task in enumerate(TASKS)
        }
        payload = D.fit_normalizer(deltas, split_sha256="split-sha")
        got = verify_delta_normalizer_matches_train(deltas, payload, "split-sha", D)
        self.assertEqual(got["sha256"], payload["sha256"])
        shifted = {task: values + np.float32(0.1) for task, values in deltas.items()}
        with self.assertRaisesRegex(ValueError, "does not reproduce from the exact frozen TRAIN rows"):
            verify_delta_normalizer_matches_train(shifted, payload, "split-sha", D)

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

    def test_gpu_smoke_requires_explicit_authorization_before_any_runtime_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "smoke-output"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                rc = gpu_smoke_main(["--out", str(out)])
            self.assertEqual(rc, 2)
            self.assertIn("BLOCKED_AUTHORIZATION", stream.getvalue())
            self.assertFalse(out.exists())

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
        self.assertEqual(args.min_epochs, 5)
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
            epochs=5, min_epochs=5, seed=0, gradient_checkpointing=False,
            early_stopping_patience=0, checkpoint_every_steps=5000,
            gpu_budget_hours=2.0, runtime_stop_margin_seconds=600.0,
            budget_accounting_every_steps=50, resource_check_every=10,
            min_free_gpu_gb=12.0, expected_peak_gpu_memory_gb=24.0,
            expected_gpu_uuid="GPU-test", resource_mode="time-window",
            max_process_memory_gb=None,
        )
        plan = {
            "train_samples_sha256": "train", "dev_samples_sha256": "dev",
            "split_sha256": "split", "feature_normalizer_sha256": "feature",
            "delta_normalizer_sha256": "delta", "dataset_manifest_sha256": "manifest",
            "preflight_report_sha256": "preflight", "gpu_smoke_report_sha256": "smoke",
            "approved_gpu_budget_hours": 2.0, "runtime_stop_margin_seconds": 600.0,
            "budget_accounting_every_steps": 50, "resource_check_every": 10,
            "min_free_gpu_gb": 12.0, "expected_peak_gpu_memory_gb": 24.0,
            "expected_gpu_uuid": "GPU-test", "resource_mode": "time-window",
            "max_process_memory_gb": None,
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
            "min_epochs": 5, "seed": 0, "gradient_checkpointing": False,
            "target_representation": "joint_delta_v1", **plan,
            "base_model_identity": base, "scheduler": scheduler,
            "early_stopping_patience": 0, "checkpoint_every_steps": 5000, "model_info": model,
        }
        validate_resume_compatibility(meta, args, plan, base, model, scheduler, "commit")
        args.epochs = 4
        with self.assertRaisesRegex(ValueError, "epochs_max"):
            validate_resume_compatibility(meta, args, plan, base, model, scheduler, "commit")

    def test_run_artifact_manifest_hashes_logs_and_preserves_prior_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "train_metrics.jsonl").write_text('{"global_step":1}\\n', encoding="utf-8")
            (out / "checkpoints").mkdir()
            (out / "checkpoints" / "state.pt").write_bytes(b"checkpoint-bytes")
            first_path = run_artifact_manifest_path(out, "STOPPED_SAFE", 1)
            write_run_artifact_manifest(out, first_path, source_sha="commit-sha",
                                        status="STOPPED_SAFE", global_step=1)
            first = json.loads(first_path.read_text(encoding="utf-8"))
            self.assertTrue(first["manifest_excludes_itself"])
            self.assertIn("train_metrics.jsonl", first["files"])
            self.assertEqual(first["files"]["checkpoints/state.pt"]["sha256"],
                             hashlib.sha256(b"checkpoint-bytes").hexdigest())
            self.assertNotIn(first_path.name, first["files"])
            (out / "train_metrics.jsonl").write_text('{"global_step":2}\\n', encoding="utf-8")
            second_path = run_artifact_manifest_path(out, "COMPLETED", 2)
            write_run_artifact_manifest(out, second_path, source_sha="commit-sha",
                                        status="COMPLETED", global_step=2)
            second = json.loads(second_path.read_text(encoding="utf-8"))
            self.assertIn(first_path.name, second["files"])
            self.assertIn("checkpoints/state.pt", second["files"])
            self.assertNotIn(second_path.name, second["files"])

    def test_cumulative_budget_ledger_does_not_reset_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            initial = load_budget_ledger(out, "commit-sha", 2.0, resume=False)
            self.assertEqual(initial["cumulative_gpu_wall_seconds"], 0.0)
            start = time.monotonic() - 2.0
            persisted = persist_budget_ledger(
                out, source_sha="commit-sha", approved_gpu_budget_hours=2.0,
                base_elapsed_seconds=3600.0, process_start_monotonic=start,
                global_step=50, reason="test",
            )
            self.assertGreaterEqual(persisted, 3601.9)
            resumed = load_budget_ledger(out, "commit-sha", 2.0, resume=True)
            self.assertAlmostEqual(resumed["cumulative_gpu_wall_seconds"], persisted, places=3)
            with self.assertRaisesRegex(ValueError, "budget changed across resume"):
                load_budget_ledger(out, "commit-sha", 3.0, resume=True)
            self.assertAlmostEqual(budget_remaining_seconds(2.0, 3600.0, 600.0), 3000.0)
            with self.assertRaisesRegex(ValueError, "budget_ledger.json"):
                load_budget_ledger(out / "missing", "commit-sha", 2.0, resume=True)

    def test_patience_zero_disables_early_stopping(self):
        self.assertFalse(should_early_stop(5, 0, 5, 100))
        self.assertFalse(should_early_stop(3, 2, 2, 2))
        self.assertTrue(should_early_stop(3, 2, 3, 2))
        self.assertFalse(should_early_stop(3, 2, 3, 1))

    def _make_preflight_bundle(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        normalizer = root / "normalizer.json"
        normalizer.write_text(json.dumps({"test": "normalizer"}), encoding="utf-8")
        dataset_manifest = root / "r44_dataset_manifest.json"
        dataset_manifest.write_text(json.dumps({"schema": "r44-dataset-manifest/v1"}), encoding="utf-8")
        plan = {
            "split_sha256": "split-sha",
            "feature_normalizer_sha256": "feature-sha",
            "delta_normalizer_sha256": sha256_file(normalizer),
            "dataset_manifest_sha256": sha256_file(dataset_manifest),
            "delta_payload": {"sha256": "payload-sha"},
            "train_samples_sha256": "train-sha",
            "dev_samples_sha256": "dev-sha",
            "n_train": 10,
            "n_dev": 4,
            "manifest_files": {"push_T/episode_0000000.hdf5": {"sha256": "file-sha"}},
        }
        coverage = {
            "n_train": 10, "base_seed": 0,
            "epochs": [
                {"epoch": epoch, "complete": True, "total": 10, "unique_seen": 10,
                 "min_visits": 1, "max_visits": 1}
                for epoch in range(5)
            ],
        }
        (root / "sample_coverage.json").write_text(json.dumps(coverage), encoding="utf-8")
        config = {
            "target_representation": "joint_delta_v1",
            "lora": {"r": 32, "lora_alpha": 16, "scale": 0.5, "lora_dropout": 0.0,
                     "target_modules": ["q_proj", "v_proj"]},
            "epochs_max": 5, "epochs_min": 5, "early_stopping_patience": 0,
            "n_train_samples": 10, "normalizer_sha256": "payload-sha",
            "optimizer": {"lr": 5e-4, "projector_lr_scale": 0.1, "warmup_fraction": 0.01},
            "sampler": {"base_seed": 0}, "checkpoint_every_steps": 5000,
        }
        (root / "training_config.json").write_text(json.dumps(config), encoding="utf-8")
        dataset_hashes = {
            "splits_sha256": "split-sha",
            "normalizer_file_sha256": sha256_file(normalizer),
            "episode_files": {"push_T": [{"episode": "episode_0000000.hdf5", "sha256": "file-sha"}]},
        }
        (root / "dataset_hashes.json").write_text(json.dumps(dataset_hashes), encoding="utf-8")
        budget = {
            "n_train_samples": 10, "n_dev_samples": 4, "max_epochs": 5,
            "max_optimizer_steps": 50, "required_gpu_hours": 1.0,
        }
        (root / "budget.json").write_text(json.dumps(budget), encoding="utf-8")
        required_passes = [
            "source_git_sha_available",
            "all_train_dev_episodes_14d_and_three_camera",
            "action_equals_next_state_bit_exact",
            "all_hdf5_files_hashed_stably",
            "instruction_text_matches_frozen_task_contract",
            "normalizer_fitted_on_train_only",
            "codec_roundtrip_within_half_codec_step_for_representable_values",
            "epoch_complete_no_replacement_5_epochs",
            "feature_normalizer_matches_frozen_split",
        ]
        checks = []
        for name in required_passes:
            detail = {}
            if name == "normalizer_fitted_on_train_only":
                detail = {"sha256": "payload-sha"}
            elif name == "epoch_complete_no_replacement_5_epochs":
                detail = {"epochs": 5, "unique_seen": 10}
            checks.append({"name": name, "result": "PASS", "detail": detail})
        checks.append({"name": "gpu_budget_covers_the_approved_plan", "result": "FAIL",
                       "detail": {"required_gpu_hours": 1.0}})
        report = {
            "schema": "r44-preflight/v1", "source_sha": "commit-sha",
            "status": "BLOCKED_BUDGET", "splits_sha256": "split-sha",
            "scope": {"training_started": False, "gpu_used": False,
                      "checkpoints_written": False, "simulator_or_robot": False},
            "checks": checks, "failed_contract_checks": [], "budget": budget,
        }
        report_path = root / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        names = ["report.json", "normalizer.json", "r44_dataset_manifest.json",
                 "sample_coverage.json", "training_config.json", "dataset_hashes.json", "budget.json"]
        (root / "SHA256SUMS").write_text(
            "".join(f"{sha256_file(root / name)}  {name}\n" for name in sorted(names)),
            encoding="utf-8",
        )
        return report_path, plan

    def test_preflight_report_requires_same_sha_and_integrity_checked_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path, plan = self._make_preflight_bundle(Path(tmp) / "preflight")
            info = validate_preflight_report(
                report_path, plan, source_sha="commit-sha", expected_epochs=5,
                expected_budget={"required_budget_hours": 1.0},
            )
            self.assertEqual(info["status"], "BLOCKED_BUDGET")
            self.assertEqual(info["sha256"], sha256_file(report_path))
            with self.assertRaisesRegex(ValueError, "source SHA"):
                validate_preflight_report(report_path, plan, source_sha="other-commit")
            with (report_path.parent / "training_config.json").open("a", encoding="utf-8") as fh:
                fh.write(" ")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                validate_preflight_report(report_path, plan, source_sha="commit-sha")

    def test_gpu_smoke_report_requires_memory_and_checkpoint_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path, plan = self._make_preflight_bundle(Path(tmp) / "preflight")
            preflight_info = validate_preflight_report(
                report_path, plan, source_sha="commit-sha", expected_budget=None,
            )
            args = SimpleNamespace(
                resource_mode="time-window", gradient_checkpointing=False,
                include_task_token=False, seed=0, lr=5e-4, projector_lr_multiplier=0.1,
                warmup_fraction=0.01, expected_peak_gpu_memory_gb=24.0,
                max_process_memory_gb=None, expected_gpu_uuid="GPU-test",
                min_free_gpu_gb=12.0, checkpoint_every_steps=5000, epochs=5,
            )
            expected_lora = {"r": 32, "lora_alpha": 16, "scale": 0.5,
                             "target_modules": ["q_proj", "v_proj"], "dropout": 0.0}
            checkpoint_bytes = 1024
            checkpoint_count = 7
            smoke = {
                "schema": "r44-gpu-smoke/v1", "status": "PASS_SMOKE",
                "source_sha": "commit-sha", "cuda_context_created": True,
                "formal_training": False, "weights_exported": False, "checkpoint_exported": False,
                "optimizer_steps_executed": 1, "resource_mode": "time-window",
                "gradient_checkpointing": False, "include_task_token": False, "seed": 0,
                "lr": 5e-4, "projector_lr_multiplier": 0.1, "warmup_fraction": 0.01,
                "expected_peak_gpu_memory_gb": 24.0, "min_free_gpu_gb": 12.0,
                "max_process_memory_gb": None,
                "gpu_before": {"uuid": "GPU-test", "memory_free_gb": 40.0},
                "cpu_contracts": {
                    "train_samples": 10, "dev_samples": 4,
                    "train_samples_sha256": "train-sha", "dev_samples_sha256": "dev-sha",
                    "split_sha256": "split-sha", "feature_normalizer_sha256": "feature-sha",
                    "delta_normalizer_sha256": plan["delta_normalizer_sha256"],
                    "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
                    "preflight_report_sha256": preflight_info["sha256"],
                },
                "model_info": {"lora_config": expected_lora, "runtime_contract": {"ok": True}},
                "peak_within_expected_peak": True, "peak_within_allocator_cap": None,
                "memory_budget_verdict": "PASS_WITHIN_EXPECTED_PEAK",
                "global_free_memory_after_step_gb": 20.0,
                "checkpoint_size_status": "MEASURED", "checkpoint_state_bytes": checkpoint_bytes,
                "checkpoint_serialization_mode": "in-memory only; no checkpoint file written",
                "checkpoint_every_steps": 5000, "epochs_max": 5,
                "checkpoint_count_estimate": checkpoint_count,
                "estimated_checkpoint_storage_gb": checkpoint_bytes * checkpoint_count / (1024**3),
                "measured_peak_allocated_gb": 23.0, "measured_peak_reserved_gb": 23.5,
                "trainable_lora_parameter_count": 100, "projector_parameter_count": 200,
                "scheduler_info": {
                    "total_steps": 50, "warmup_steps": 1, "decay": "linear_to_zero",
                    "warmup_fraction": 0.01, "lr_lora": 5e-4, "lr_projector": 5e-5,
                },
                "base_model_identity": {"weights": "sha"},
            }
            smoke_path = Path(tmp) / "smoke" / "report.json"
            smoke_path.parent.mkdir(parents=True)
            smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
            (smoke_path.parent / "SHA256SUMS.json").write_text(
                json.dumps({"report.json": sha256_file(smoke_path)}), encoding="utf-8"
            )
            result = validate_gpu_smoke_report(
                smoke_path, plan, preflight_info, args, source_sha="commit-sha",
                expected_base_identity={"weights": "sha"},
            )
            self.assertEqual(result["checkpoint_state_bytes"], checkpoint_bytes)
            self.assertEqual(result["checkpoint_count_estimate"], checkpoint_count)
            smoke["peak_within_expected_peak"] = False
            smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
            (smoke_path.parent / "SHA256SUMS.json").write_text(
                json.dumps({"report.json": sha256_file(smoke_path)}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "peak allocated memory"):
                validate_gpu_smoke_report(
                    smoke_path, plan, preflight_info, args, source_sha="commit-sha",
                    expected_base_identity={"weights": "sha"},
                )

    def test_budget_rejects_negative_margin(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            estimate_budget_hours(10, 10, 1, 1.0, 1.0, 1.0, -0.1)


if __name__ == "__main__":
    unittest.main()
