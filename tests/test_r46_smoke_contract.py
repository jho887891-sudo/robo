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


if __name__ == "__main__":
    unittest.main(verbosity=2)
