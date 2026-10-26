import hashlib
import tempfile
import unittest
from pathlib import Path

from training.r44_preflight_contract import (
    final_preflight_status,
    make_training_manifest,
    sha256_file_stable,
)


class R44PreflightContractTests(unittest.TestCase):
    def test_contract_failure_cannot_be_overridden_by_budget(self):
        status, failures = final_preflight_status(
            [
                {"name": "hdf5_contract", "result": "FAIL"},
                {"name": "gpu_budget_covers_the_approved_plan", "result": "PASS"},
            ],
            budget_ok=True,
        )
        self.assertEqual(status, "FAIL_PREFLIGHT")
        self.assertEqual([x["name"] for x in failures], ["hdf5_contract"])

    def test_budget_block_is_distinct_from_data_failure(self):
        status, failures = final_preflight_status(
            [{"name": "gpu_budget_covers_the_approved_plan", "result": "FAIL"}],
            budget_ok=False,
        )
        self.assertEqual(status, "BLOCKED_BUDGET")
        self.assertEqual(failures, [])

    def test_ready_requires_all_contract_checks_and_budget(self):
        status, failures = final_preflight_status(
            [{"name": "hdf5_contract", "result": "PASS"},
             {"name": "feature_normalizer", "result": "PASS"},
             {"name": "gpu_budget_covers_the_approved_plan", "result": "PASS"}],
            budget_ok=True,
        )
        self.assertEqual(status, "READY")
        self.assertEqual(failures, [])

    def test_stable_file_hash_contains_content_sha_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.hdf5"
            path.write_bytes(b"immutable-test-bytes")
            record = sha256_file_stable(path)
            self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record["size_bytes"], len(b"immutable-test-bytes"))
            self.assertEqual(record["path"], str(path.resolve()))

    def test_training_manifest_pins_split_normalizers_files_and_instruction(self):
        text = "move the red block"
        ins_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        manifest = make_training_manifest(
            split_sha256="split-sha",
            feature_normalizer_sha256="feature-sha",
            delta_normalizer_sha256="delta-file-sha",
            delta_normalizer_payload_sha256="delta-payload-sha",
            files={"push_T/ep0.hdf5": {
                "path": "/data/push_T/ep0.hdf5", "sha256": "file-sha", "size_bytes": 10
            }},
            instructions={"push_T": {"text": text, "sha256": ins_sha}},
        )
        self.assertEqual(manifest["schema"], "r44-dataset-manifest/v1")
        self.assertEqual(manifest["split_sha256"], "split-sha")
        self.assertEqual(manifest["files"]["push_T/ep0.hdf5"]["sha256"], "file-sha")
        self.assertEqual(manifest["instructions"]["push_T"]["sha256"], ins_sha)

    def test_training_manifest_rejects_instruction_hash_tampering(self):
        with self.assertRaisesRegex(ValueError, "instruction text/hash mismatch"):
            make_training_manifest(
                split_sha256="split-sha",
                feature_normalizer_sha256="feature-sha",
                delta_normalizer_sha256="delta-file-sha",
                delta_normalizer_payload_sha256="delta-payload-sha",
                files={"push_T/ep0.hdf5": {
                    "path": "/data/push_T/ep0.hdf5", "sha256": "file-sha", "size_bytes": 10
                }},
                instructions={"push_T": {"text": "move block", "sha256": "bad"}},
            )


if __name__ == "__main__":
    unittest.main()
