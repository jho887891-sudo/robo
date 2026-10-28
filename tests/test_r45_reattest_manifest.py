#!/usr/bin/env python3
"""R45-G CPU-only contract re-attestation and OFT source discovery tests."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.r45_reattest_r44_manifest import validated_manifest
from diagnostics.r45_audit_oft_candidates import inspect_candidate, REQUIRED


class TestR44ManifestReattestation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.splits = self.root / "splits.json"
        self.feature = self.root / "42d.json"
        self.delta = self.root / "delta.json"
        self.hdf5 = self.root / "data"
        self.hdf5.mkdir()
        tasks = ("push_T", "stack_bowls", "sweep_blocks")
        split = {"splits": {t: {
            "train": [f"{t}/episode_{i:07d}.hdf5" for i in range(80)],
            "dev": [f"{t}/episode_{i:07d}.hdf5" for i in range(80, 100)],
        } for t in tasks}}
        self.splits.write_text(json.dumps(split))
        self.feature.write_text("existing frozen 42D bytes, not a new fit")
        frozen = {"fit": {"split_sha256": hashlib.sha256(self.splits.read_bytes()).hexdigest()},
                  "tasks": {t: {"n_rows": 36260} for t in tasks},
                  "sha256": "frozen-payload-hash"}
        self.delta.write_text(json.dumps(frozen))
        import m2_multitask_data as M2
        self.instructions = {t: {
            "text": M2.TASK_PROMPTS[t],
            "sha256": hashlib.sha256(M2.TASK_PROMPTS[t].encode()).hexdigest(),
        } for t in tasks}
        files = {}
        blocks = {}
        for task in tasks:
            for key in split["splits"][task]["train"] + split["splits"][task]["dev"]:
                name = Path(key).name
                files[f"{task}/{name}"] = {
                    "path": str(self.hdf5 / task / "arx_x5" / "data" / name),
                    "size_bytes": 100, "sha256": "frozen-fixture-hash",
                }
            blocks[task] = {
                "excluded": [], "train_episodes": 80, "dev_episodes": 20,
                "train_samples": 36260, "dev_samples": 9422 if task != "sweep_blocks" else 9424,
            }
        self.fake_audit = {
            "tasks": blocks, "files": files, "instructions": self.instructions,
            "hash_errors": [], "contract_violations": [], "instruction_violations": [],
            "max_abs_action_vs_next_state": 0.0,
            "totals": {"train_samples": 108780, "dev_samples": 28268},
        }
        self.deltas = {t: [] for t in tasks}

    def tearDown(self):
        self.temp.cleanup()

    def build(self, delta_hash="frozen-payload-hash", audit=None):
        import r44_delta_v1 as D
        import m2_task_normalizer as TN
        with mock.patch.object(TN, "load_payload", return_value={}), \
             mock.patch.object(D, "validate_normalizer", return_value=True), \
             mock.patch.object(D, "fit_normalizer", return_value={"sha256": delta_hash}):
            return validated_manifest(
                splits_path=self.splits, root=self.hdf5, feature_path=self.feature,
                delta_path=self.delta,
                scan_fn=lambda *_: (self.fake_audit if audit is None else audit, self.deltas))

    def test_binds_existing_normalizer_file_without_refitting_it(self):
        original = self.delta.read_bytes()
        manifest, report = self.build()
        self.assertEqual(manifest["schema"], "r44-dataset-manifest/v1")
        self.assertEqual(len(manifest["files"]), 300)
        self.assertEqual(manifest["delta_normalizer_sha256"],
                         hashlib.sha256(original).hexdigest())
        self.assertEqual(manifest["delta_normalizer_payload_sha256"], "frozen-payload-hash")
        self.assertEqual(report["status"], "CPU_MANIFEST_REATTEST_PASS")
        self.assertFalse(report["gpu_smoke_authorized"])
        self.assertEqual(self.delta.read_bytes(), original)

    def test_refuses_delta_reconstruction_drift(self):
        with self.assertRaisesRegex(ValueError, "BLOCKED_DELTA_MISMATCH"):
            self.build(delta_hash="different-recomputed-fit")

    def test_refuses_missing_episode_and_instruction_mismatch(self):
        audit = dict(self.fake_audit)
        audit["files"] = dict(self.fake_audit["files"])
        audit["files"].pop(next(iter(audit["files"])))
        with self.assertRaisesRegex(ValueError, "exactly the frozen 300"):
            self.build(audit=audit)
        audit = dict(self.fake_audit)
        audit["instruction_violations"] = [{"episode": "tampered"}]
        with self.assertRaisesRegex(ValueError, "instruction mismatch"):
            self.build(audit=audit)

    def test_refuses_overwrite_untouched_evidence(self):
        out = self.root / "historical"
        out.mkdir()
        marker = out / "KEEP"
        marker.write_text("protected")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "diagnostics/r45_reattest_r44_manifest.py"),
             "--hdf5-root", str(self.hdf5),
             "--feature-normalizer", str(self.feature),
             "--delta-normalizer", str(self.delta),
             "--out", str(out)],
            capture_output=True, text=True, timeout=15,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        self.assertEqual(proc.returncode, 2)
        self.assertIn("BLOCKED_OUTPUT_EXISTS", proc.stderr)
        self.assertEqual(marker.read_text(), "protected")


class TestOFTStaticFingerprints(unittest.TestCase):
    def test_only_viable_static_source_is_reported_never_autoselected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "oft"
            root.mkdir()
            self.assertFalse(inspect_candidate(str(root))["structurally_viable"])
            for entry in REQUIRED:
                target = root / entry
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("module source")
            report = inspect_candidate(str(root))
            self.assertTrue(report["structurally_viable"])
            self.assertEqual(len(report["required"]), 3)
            evidence = Path(d) / "audit.json"
            result = subprocess.run(
                [sys.executable, str(ROOT / "diagnostics/r45_audit_oft_candidates.py"),
                 "--candidate", str(root), "--out", str(evidence)],
                capture_output=True, text=True, timeout=15,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            self.assertEqual(result.returncode, 0, result.stderr)
            obj = json.loads(evidence.read_text())
            self.assertEqual(obj["status"], "CANDIDATES_NEED_PROVENANCE")
            self.assertIsNone(obj["selected"])
            second = subprocess.run(
                [sys.executable, str(ROOT / "diagnostics/r45_audit_oft_candidates.py"),
                 "--candidate", str(root), "--out", str(evidence)],
                capture_output=True, text=True, timeout=15)
            self.assertEqual(second.returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
