import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diagnostics.r46_smoke_contract import (  # noqa: E402
    CAMERA_ORDER,
    FROZEN_CONTRACT,
    SELECTION_SOURCE_SHA,
    SamplePlanError,
    canonical_plan_sha256,
    validate_smoke_report,
    validate_sample_plan,
)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(root):
    train = [f"push_T/episode_{i:07d}.hdf5" for i in range(5)]
    dev = ["push_T/episode_0000009.hdf5"]
    split_doc = {"splits": {
        "push_T": {"train": train, "dev": dev},
        "stack_bowls": {"train": ["stack_bowls/episode_0000000.hdf5"],
                        "dev": ["stack_bowls/episode_0000001.hdf5"]},
        "sweep_blocks": {"train": ["sweep_blocks/episode_0000000.hdf5"],
                         "dev": ["sweep_blocks/episode_0000001.hdf5"]},
    }}
    split_sha = digest("split bytes")
    manifest_sha = digest("manifest bytes")
    files = {}
    samples = []
    folder = root / "push_T" / "arx_x5" / "data"
    folder.mkdir(parents=True)
    for index, episode_key in enumerate(train):
        episode = episode_key.split("/")[1]
        payload = f"fake-hdf5-{index}".encode()
        (folder / episode).write_bytes(payload)
        file_sha = hashlib.sha256(payload).hexdigest()
        files[episode_key] = {"sha256": file_sha}
        samples.append({
            "order_index": index,
            "role": "warmup" if index == 0 else "measured",
            "task": "push_T",
            "episode": episode,
            "t": index + 2,
            "hdf5_sha256": file_sha,
            "camera_order": list(CAMERA_ORDER),
            "image_shape": [480, 640, 3],
            "image_dtype": "uint8",
            "image_colorspace": "RGB",
            "processor_output_shape": list(FROZEN_CONTRACT["processor_output_shape"]),
            "proprio_shape": [1, 42],
            "action_token_ids": [31744 + i for i in range(14)],
            "stop_token_id": 2,
            "prompt_labels_ignored": True,
        })
    manifest = {"files": files}
    plan = {
        "schema": "r46-smoke-samples/v1",
        "source_sha": SELECTION_SOURCE_SHA,
        "selection_tool_sha256": digest("selection tool source"),
        "seed": 0,
        "task": "push_T",
        "sample_count": 5,
        "split_sha256": split_sha,
        "manifest_sha256": manifest_sha,
        "contract": copy.deepcopy(FROZEN_CONTRACT),
        "samples": samples,
    }
    plan["sample_plan_sha256"] = canonical_plan_sha256(plan)
    return plan, split_doc, manifest, split_sha, manifest_sha


class SamplePlanTests(unittest.TestCase):
    def validate(self, plan, split_doc, manifest, split_sha, manifest_sha, root, **kwargs):
        return validate_sample_plan(
            plan,
            split_doc=split_doc,
            manifest=manifest,
            split_sha256=split_sha,
            manifest_sha256=manifest_sha,
            hdf5_root=root,
            verify_files=False,
            **kwargs,
        )

    def test_accepts_exactly_one_warmup_and_four_distinct_train_episodes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan, splits, manifest, split_sha, manifest_sha = fixture(root)
            result = self.validate(plan, splits, manifest, split_sha, manifest_sha, root,
                                   source_sha=SELECTION_SOURCE_SHA)
            self.assertEqual(result["status"], "CPU_SAMPLE_CONTRACT_PASS")
            self.assertEqual(result["unique_episodes"], 5)
            self.assertEqual(result["roles"], ["warmup"] + ["measured"] * 4)

    def test_rejects_dev_sample_and_duplicate_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan, splits, manifest, split_sha, manifest_sha = fixture(root)
            plan["samples"][2]["episode"] = "episode_0000009.hdf5"
            plan["sample_plan_sha256"] = canonical_plan_sha256(plan)
            with self.assertRaisesRegex(SamplePlanError, "not in TRAIN"):
                self.validate(plan, splits, manifest, split_sha, manifest_sha, root)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan, splits, manifest, split_sha, manifest_sha = fixture(root)
            plan["samples"][1]["episode"] = plan["samples"][0]["episode"]
            plan["samples"][1]["t"] = plan["samples"][0]["t"]
            plan["sample_plan_sha256"] = canonical_plan_sha256(plan)
            with self.assertRaisesRegex(SamplePlanError, "duplicate"):
                self.validate(plan, splits, manifest, split_sha, manifest_sha, root)

    def test_rejects_changed_source_contract_camera_order_and_label_shape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan, splits, manifest, split_sha, manifest_sha = fixture(root)
            with self.assertRaisesRegex(SamplePlanError, "source SHA"):
                self.validate(plan, splits, manifest, split_sha, manifest_sha, root,
                              source_sha="f" * 40)
            plan["contract"]["lora"]["alpha"] = 32
            plan["sample_plan_sha256"] = canonical_plan_sha256(plan)
            with self.assertRaisesRegex(SamplePlanError, "frozen OpenVLA"):
                self.validate(plan, splits, manifest, split_sha, manifest_sha, root)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan, splits, manifest, split_sha, manifest_sha = fixture(root)
            plan["samples"][0]["camera_order"].reverse()
            plan["sample_plan_sha256"] = canonical_plan_sha256(plan)
            with self.assertRaisesRegex(SamplePlanError, "camera order"):
                self.validate(plan, splits, manifest, split_sha, manifest_sha, root)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan, splits, manifest, split_sha, manifest_sha = fixture(root)
            plan["samples"][0]["action_token_ids"].pop()
            plan["sample_plan_sha256"] = canonical_plan_sha256(plan)
            with self.assertRaisesRegex(SamplePlanError, "14 legal"):
                self.validate(plan, splits, manifest, split_sha, manifest_sha, root)

    def test_rejects_manifest_file_sha_split_sha_and_plan_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan, splits, manifest, split_sha, manifest_sha = fixture(root)
            with self.assertRaisesRegex(SamplePlanError, "split SHA"):
                self.validate(plan, splits, manifest, "0" * 64, manifest_sha, root)
            with self.assertRaisesRegex(SamplePlanError, "manifest SHA"):
                self.validate(plan, splits, manifest, split_sha, "0" * 64, root)
            manifest["files"]["push_T/episode_0000000.hdf5"]["sha256"] = "0" * 64
            plan["samples"][0]["hdf5_sha256"] = "0" * 64
            plan["sample_plan_sha256"] = canonical_plan_sha256(plan)
            with self.assertRaisesRegex(SamplePlanError, "actual HDF5 SHA"):
                validate_sample_plan(plan, split_doc=splits, manifest=manifest,
                                     split_sha256=split_sha, manifest_sha256=manifest_sha,
                                     hdf5_root=root, verify_files=True)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan, splits, manifest, split_sha, manifest_sha = fixture(root)
            plan["samples"][0]["t"] += 1
            with self.assertRaisesRegex(SamplePlanError, "plan SHA256"):
                self.validate(plan, splits, manifest, split_sha, manifest_sha, root)


class SmokeReportSchemaTests(unittest.TestCase):
    def fixture_report(self):
        latency = {key: 0.01 for key in (
            "hdf5_read_s", "jpeg_preprocess_s", "state_label_s", "host_to_gpu_s",
            "forward_s", "backward_s", "optimizer_s", "end_to_end_s",
        )}
        memory = {"peak_allocated_gib": 1.0, "peak_reserved_gib": 1.5}
        return {
            "schema": "r46-openvla-five-sample-smoke/v1", "status": "PASS_SMOKE",
            "optimizer_steps_executed": 5, "warmup_count": 1, "measured_count": 4,
            "throughput_scope": "preliminary_smoke_only_not_stable_training_throughput",
            "smoke_interval_samples_per_second_excludes_model_load_and_warmup": 1.0,
            "throughput_excludes_model_load": True, "throughput_excludes_warmup_sample": True,
            "weights_exported": False, "checkpoint_exported": False,
            "optimizer_state_exported": False, "formal_training": False,
            "samples": [
                {"role": "warmup" if index == 0 else "measured", "exit_status": "PASS_STEP",
                 "latency": dict(latency), "memory": dict(memory), "loss": 1.0, "grad_norm": 0.5}
                for index in range(5)
            ],
        }

    def test_accepts_synthetic_cpu_fixture_with_five_step_scope(self):
        # This validates JSON structure only; it never runs or claims a GPU smoke.
        checked = validate_smoke_report(self.fixture_report())
        self.assertEqual(checked["sample_count"], 5)
        self.assertEqual(checked["status"], "CPU_REPORT_SCHEMA_VALID")

    def test_rejects_incomplete_or_semantically_expanded_smoke_report(self):
        report = self.fixture_report()
        report["optimizer_steps_executed"] = 4
        with self.assertRaisesRegex(SamplePlanError, "exactly five"):
            validate_smoke_report(report)
        report = self.fixture_report()
        report["formal_training"] = True
        with self.assertRaisesRegex(SamplePlanError, "disposable smoke"):
            validate_smoke_report(report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
