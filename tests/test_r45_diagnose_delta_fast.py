#!/usr/bin/env python3
"""Only CPU/stdlib fixtures for fast delta discrepancy diagnosis."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.r45_diagnose_delta_fast import scan_delta_only, diagnose


class TestFastDeltaDiagnostics(unittest.TestCase):
    def test_all_300_episodes_and_split_rows_without_full_hdf5_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = ("push_T", "stack_bowls", "sweep_blocks")
            split = {}
            for task in tasks:
                split[task] = {
                    "train": [f"{task}/episode_{i:04d}.hdf5" for i in range(80)],
                    "dev": [f"{task}/episode_{i:04d}.hdf5" for i in range(80, 100)],
                }
            calls = []
            def read_episode(path):
                import m2_multitask_data as M2
                task = path.parent.parent.parent.name
                part = "train" if int(path.stem.split("_")[-1]) < 80 else "dev"
                # 60 TRAIN samples have one extra row (108780 total),
                # eight DEV samples have one extra row (28268 total).
                index = int(path.stem.split("_")[-1])
                extra = (task == "push_T" and index < 60) if part == "train" else (
                    task == "push_T" and index in range(80, 88))
                T = (453 if part == "train" else 471) + 3 + int(extra)
                state = np.arange(T, dtype=np.float64).reshape(-1, 1) * np.ones((1, 14))
                action = np.concatenate([state[1:], state[-1:]], axis=0)
                calls.append(str(path))
                return {"state": state, "action": action,
                        "instruction": M2.TASK_PROMPTS[task],
                        "cameras": ["cam_head", "cam_left_wrist", "cam_right_wrist"]}
            for task in tasks:
                for rel in split[task]["train"] + split[task]["dev"]:
                    path = root / task / "arx_x5" / "data" / Path(rel).name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"x")
            deltas, inventory, totals = scan_delta_only(split, root, read_episode=read_episode)
            self.assertEqual(len(inventory), 300)
            self.assertEqual(totals["train_samples"], 108780)
            self.assertEqual(totals["dev_samples"], 28268)
            self.assertEqual(len(calls), 300)
            self.assertEqual(sum(len(x) for x in deltas.values()), 108780)
            self.assertTrue(all("sha256" not in row for row in inventory))

    def test_fast_payload_mismatch_is_blocked_and_reported_without_manifest(self):
        import r44_delta_v1 as D
        import m2_task_normalizer as TN
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.json"
            feature = root / "feature.json"
            delta = root / "delta.json"
            splits = {"splits": {t: {"train": [f"{t}/e0.hdf5"],
                                     "dev": [f"{t}/e1.hdf5"]}
                                 for t in ("push_T", "stack_bowls", "sweep_blocks")}}
            split.write_text(json.dumps(splits))
            feature.write_text("frozen")
            sha = hashlib.sha256(split.read_bytes()).hexdigest()
            frozen = {"sha256": "old", "fit": {"split_sha256": sha},
                      "tasks": {"push_T": {"target_mean": [0.0] * 14}}}
            delta.write_text(json.dumps(frozen))
            results = ({"push_T": np.zeros((2,14))},
                       [{"episode": "push_T/e0.hdf5", "size_bytes": 1}] * 300,
                       {"train_samples": 108780, "dev_samples": 28268})
            with mock.patch.object(TN, "load_payload", return_value=True), \
                 mock.patch.object(D, "validate_normalizer", return_value=True), \
                 mock.patch.object(D, "fit_normalizer",
                                   return_value={"sha256":"new","fit":{"split_sha256":sha},
                                                 "tasks":{"push_T":{"target_mean":[0.1]*14}}}):
                report = diagnose(split, root, feature, delta,
                                  scan_fn=lambda *_: results)
            self.assertEqual(report["status"], "BLOCKED_DELTA_MISMATCH")
            self.assertFalse(report["formal_manifest_created"])
            self.assertFalse(report["full_hdf5_sha256_checked"])
            self.assertEqual(report["difference"]["changed_fields"][0]["field"],
                             "tasks.push_T.target_mean")

    def test_nonexistent_output_does_not_override_existing_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            (output / "KEEP").write_text("frozen")
            result = subprocess.run(
                [sys.executable, str(ROOT / "diagnostics/r45_diagnose_delta_fast.py"),
                 "--out", str(output), "--hdf5-root", str(output),
                 "--feature-normalizer", str(output / "missing"),
                 "--delta-normalizer", str(output / "missing")],
                capture_output=True, text=True, timeout=15,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            self.assertEqual(result.returncode, 2)
            self.assertIn("BLOCKED_OUTPUT_EXISTS", result.stderr)
            self.assertEqual((output / "KEEP").read_text(), "frozen")


if __name__ == "__main__":
    unittest.main(verbosity=2)
