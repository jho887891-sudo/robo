#!/usr/bin/env python3
"""Read-only dataset-discovery gate tests, stdlib only, no GPU or real data."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "diagnostics/r45_discover_data_root.py"


class TestR45DataRootDiscovery(unittest.TestCase):
    def setUp(self):
        self.ctx = tempfile.TemporaryDirectory()
        self.base = Path(self.ctx.name)
        self.splits = self.base / "splits.json"
        self.splits.write_text(json.dumps({"splits": {
            "push_T": {"train": ["push_T/episode_train.hdf5"],
                       "dev": ["push_T/episode_dev.hdf5"]},
            "stack_bowls": {"train": ["stack_bowls/episode_train.hdf5"],
                            "dev": ["stack_bowls/episode_dev.hdf5"]},
            "sweep_blocks": {"train": ["sweep_blocks/episode_train.hdf5"],
                             "dev": ["sweep_blocks/episode_dev.hdf5"]},
        }}), encoding="utf-8")
        self.root = self.base / "hdf5"

    def tearDown(self):
        self.ctx.cleanup()

    def put_files(self, all_files=True):
        manifest = json.loads(self.splits.read_text(encoding="utf-8"))["splits"]
        for task, groups in manifest.items():
            for split, files in groups.items():
                for file in files:
                    if not all_files and task == "sweep_blocks" and split == "dev":
                        continue
                    episode = self.root / task / "arx_x5" / "data" / Path(file).name
                    episode.parent.mkdir(parents=True, exist_ok=True)
                    episode.write_bytes(b"synthetic placeholder, presence only")

    def discover(self, *extra):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--splits", str(self.splits),
             "--root-candidate", str(self.root), *extra],
            capture_output=True, text=True, timeout=15,
            env={**os.environ, "M2_HDF5_ROOT": "", "CUDA_VISIBLE_DEVICES": ""},
        )

    def test_missing_path_is_blocked_without_side_effects(self):
        result = self.discover()
        self.assertEqual(result.returncode, 2)
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], "BLOCKED_NO_COMPLETE_DATASET")
        self.assertIsNone(data["chosen_root"])
        self.assertFalse(self.root.exists())

    def test_partial_path_is_not_mistaken_for_complete_dataset(self):
        self.put_files(all_files=False)
        result = self.discover()
        self.assertEqual(result.returncode, 2)
        record = next(x for x in json.loads(result.stdout)["candidates"]
                      if x["candidate"] == str(self.root))
        self.assertEqual(record["missing_count"], 1)
        self.assertEqual(record["present_train"], 3)
        self.assertFalse(record["complete"])

    def test_complete_inventory_is_discovered_but_not_trusted_as_hdf5(self):
        self.put_files(all_files=True)
        report_path = self.base / "discovery.json"
        result = self.discover("--out", str(report_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(report_path.read_text())
        self.assertEqual(data["status"], "DISCOVERED_CANDIDATE_NEEDS_HDF5_GATE")
        self.assertEqual(data["chosen_root"], str(self.root))
        self.assertEqual(data["candidates"][-1]["present_train"], 3)
        self.assertEqual(data["candidates"][-1]["present_dev"], 3)
        self.assertTrue(data["splits_sha256"])
        # Cannot overwrite archival evidence.
        repeated = self.discover("--out", str(report_path))
        self.assertEqual(repeated.returncode, 2)
        self.assertIn("BLOCKED_OUTPUT_EXISTS", repeated.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
