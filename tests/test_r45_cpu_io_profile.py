#!/usr/bin/env python3
"""Synthetic JPEG/HDF5 CPU profile CLI regression. No CUDA or model weights."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class TestR45CpuIoProfile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        folder = self.root / "hdf5" / "push_T" / "arx_x5" / "data"
        folder.mkdir(parents=True)
        length = 8
        path = folder / "episode_0.hdf5"
        with h5py.File(path, "w") as f:
            for key, dim in [
                ("left_arm_joint_states", 6), ("left_ee_joint_states", 1),
                ("right_arm_joint_states", 6), ("right_ee_joint_states", 1),
            ]:
                f.create_dataset("state/" + key, data=np.zeros((length, dim)))
            for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                ds = f.create_dataset("vision/" + camera + "/colors", (length,),
                                      dtype=h5py.vlen_dtype(np.dtype("uint8")))
                for t in range(length):
                    bgr = np.full((8, 10, 3), t + 1, dtype=np.uint8)
                    success, encoded = cv2.imencode(".jpg", bgr)
                    self.assertTrue(success)
                    ds[t] = np.frombuffer(encoded.tobytes(), dtype=np.uint8)
        self.split = self.root / "splits.json"
        self.split.write_text(json.dumps({"splits": {"push_T": {
            "train": ["push_T/episode_0.hdf5"], "dev": []
        }}}))

    def tearDown(self):
        self.tmp.cleanup()

    def run_benchmark(self, output, *extra):
        cmd = [sys.executable, str(ROOT / "diagnostics/r45_cpu_io_profile.py"),
               "--splits", str(self.split),
               "--hdf5-root", str(self.root / "hdf5"),
               "--out", str(output), "--per-task", "2",
               "--repeats", "2", "--workers", "1,2"] + list(extra)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=90,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})

    def test_real_jpeg_bytes_parity_repeated_cache_worker_results(self):
        out = self.root / "audit"
        result = self.run_benchmark(out)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(report["status"], "CPU_IO_PROXY_PASS")
        self.assertEqual(report["index_n"], 5)
        self.assertEqual(report["selected_n"], 2)
        self.assertEqual(report["cache_hits"], 2)
        self.assertEqual([v["workers"] for v in report["worker_runs"]], [1, 2])
        self.assertTrue((out / "SHA256SUMS").exists())
        self.assertEqual(len((out / "samples.jsonl").read_text().splitlines()), 4)
        out2 = self.root / "audit2"
        repeat = self.run_benchmark(out2)
        self.assertEqual(repeat.returncode, 0, repeat.stderr[-2000:])
        manifest2 = json.loads((out2 / "report.json").read_text())
        self.assertEqual(report["sample_manifest_sha256"], manifest2["sample_manifest_sha256"])

    def test_existing_output_fail_closed_no_overwrite(self):
        out = self.root / "exists"
        out.mkdir()
        marker = out / "KEEP"
        marker.write_text("safe")
        result = self.run_benchmark(out)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("BLOCKED_CPU_IO_PROFILE", result.stderr)
        self.assertEqual(marker.read_text(), "safe")


if __name__ == "__main__":
    unittest.main(verbosity=2)
