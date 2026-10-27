#!/usr/bin/env python3
"""CPU-only synthetic HDF5 end-to-end regression for the R45 toy control-plane trainer.

Never instantiates the real OpenVLA model or any CUDA context.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import r44_delta_v1 as D  # noqa: E402


class TestToyStopResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        data = self.root / "data" / "push_T" / "arx_x5" / "data"
        data.mkdir(parents=True)
        self.hdf5 = data / "episode_0000000.hdf5"
        length = 13  # yields exactly 10 TRAIN samples, t=2...11
        with h5py.File(self.hdf5, "w") as h:
            for cat in ("state", "action"):
                for key, dim in zip(
                    ("left_arm_joint_states", "left_ee_joint_states",
                     "right_arm_joint_states", "right_ee_joint_states"),
                    (6, 1, 6, 1),
                ):
                    arr = np.arange(length * dim, dtype=np.float64).reshape(length, dim)
                    h.create_dataset(f"{cat}/{key}", data=arr * 0.001)
            for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                # Only the index contract checks these paths for the toy CPU route.
                h.create_dataset(f"vision/{cam}/colors", data=np.zeros((length, 3), dtype=np.uint8))

        self.splits = self.root / "splits.json"
        self.splits.write_text(json.dumps({"splits": {
            "push_T": {"train": ["push_T/episode_0000000.hdf5"],
                       "dev": ["push_T/episode_0000000.hdf5"]}
        }}), encoding="utf-8")
        self.normalizer = self.root / "normalizer.json"
        d = np.random.default_rng(0).normal(0, 0.02, (300, 14)).astype(np.float32)
        self.normalizer.write_text(json.dumps(D.fit_normalizer({"push_T": d})), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def run_toy(self, output, more=(), batch=2, accum=2):
        cmd = [
            sys.executable, str(ROOT / "training" / "r45_train.py"),
            "--model", "fake",
            "--splits", str(self.splits), "--hdf5-root", str(self.root / "data"),
            "--normalizer", str(self.normalizer), "--out", str(output),
            "--epochs", "2", "--base-seed", "13",
            "--micro-batch", str(batch), "--grad-accum", str(accum),
            "--checkpoint-every-frac", "0.5",
        ] + list(more)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    def checkpoint(self, out, pattern):
        matches = list((out / "checkpoints").glob(pattern))
        self.assertEqual(len(matches), 1, matches)
        return matches[0]

    def test_resume_at_mid_epoch_matches_uninterrupted_optimizer_and_weights(self):
        baseline = self.root / "baseline"
        stopped = self.root / "stopped"
        resumed = self.root / "resumed"
        b = self.run_toy(baseline)
        self.assertEqual(b.returncode, 0, b.stderr[-3000:])
        t = self.run_toy(stopped, ["--max-steps", "1"])
        self.assertEqual(t.returncode, 0, t.stderr[-3000:])
        stopck = self.checkpoint(stopped, "*_stopped")
        meta = json.loads((stopck / "meta.json").read_text())
        self.assertEqual(meta["cursor"], 4)
        self.assertEqual(meta["global_step"], 1)
        resume = self.run_toy(resumed, ["--resume", str(stopck)])
        self.assertEqual(resume.returncode, 0, resume.stderr[-3000:])
        bstate = torch.load(self.checkpoint(baseline, "epoch002_*_epoch_end") / "state.pt",
                            map_location="cpu", weights_only=False)
        rstate = torch.load(self.checkpoint(resumed, "epoch002_*_epoch_end") / "state.pt",
                            map_location="cpu", weights_only=False)
        self.assertEqual(bstate["meta"]["global_step"], 6)  # 3 updates/epoch x 2
        self.assertEqual(rstate["meta"]["global_step"], 6)
        for name, value in bstate["model"].items():
            torch.testing.assert_close(value, rstate["model"][name], rtol=0, atol=0)
        self.assertEqual(json.loads((resumed / "run_summary.json").read_text())["samples_seen"], 20)

    def test_tail_batch_produces_optimizer_step_and_valid_checkpoint(self):
        out = self.root / "tail"
        r = self.run_toy(out, batch=3, accum=2)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        summary = json.loads((out / "run_summary.json").read_text())
        self.assertEqual(summary["n_train"], 10)
        self.assertEqual(summary["samples_seen"], 20)
        self.assertEqual(summary["global_step"], 4)  # ceil(10/6) * 2
        self.checkpoint(out, "epoch002_*_epoch_end")

    def test_resume_rejects_changed_normalizer(self):
        stopped = self.root / "stopped"
        rc = self.run_toy(stopped, ["--max-steps", "1"])
        self.assertEqual(rc.returncode, 0, rc.stderr[-3000:])
        stopck = self.checkpoint(stopped, "*_stopped")
        payload = json.loads(self.normalizer.read_text())
        payload["created_by_auditor"] = "different normalizer identity"
        self.normalizer.write_text(json.dumps(payload))
        denied = self.run_toy(self.root / "rejected", ["--resume", str(stopck)])
        self.assertNotEqual(denied.returncode, 0)
        self.assertIn("BLOCKED", denied.stderr + denied.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
