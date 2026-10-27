#!/usr/bin/env python3
"""R45 CPU tests: data pipeline, coverage, checkpoint/resume, budget guard."""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))

import r45_data as DATA            # noqa: E402
import r45_gpu_bench as BENCH      # noqa: E402

SPLITS = os.path.join(ROOT, "data", "m2_splits.json")
HDF5 = "/home/T7/ojh/robot_actual/data/data/hdf5"


def small_splits(n=2, task="push_T"):
    doc = json.load(open(SPLITS))
    s = {task: {"train": doc["splits"][task]["train"][:n],
                "dev": doc["splits"][task]["dev"][:n]}}
    return s


class TestIndex(unittest.TestCase):
    def test_index_covers_the_frozen_window_exactly_once(self):
        s = small_splits()
        entries = DATA.build_index(s, HDF5, "train")
        import h5py
        for task, path, t in entries:
            with h5py.File(path, "r") as h:
                T = int(h["state"]["left_arm_joint_states"].shape[0])
            self.assertTrue(2 <= t <= T - 2, (task, t, T))
        per_ep = {}
        for task, path, t in entries:
            per_ep.setdefault(path, []).append(t)
        import h5py
        for path, ts in per_ep.items():
            with h5py.File(path, "r") as h:
                T = int(h["state"]["left_arm_joint_states"].shape[0])
            self.assertEqual(sorted(ts), list(range(2, T - 1)))
            self.assertEqual(len(ts), len(set(ts)))

    def test_dev_entries_are_disjoint_from_train(self):
        doc = json.load(open(SPLITS))
        tr = DATA.build_index(doc["splits"], HDF5, "train")
        dv = DATA.build_index(doc["splits"], HDF5, "dev")
        tr_eps = {(p, t) for _task, p, t in tr}
        dv_eps = {(p, t) for _task, p, t in dv}
        self.assertFalse(tr_eps & dv_eps)


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.entries = DATA.build_index(small_splits(), HDF5, "train")[:40]

    def test_item_contract(self):
        ds = DATA.R45Dataset(self.entries, decode=True)
        it = ds[0]
        self.assertEqual(it["state"].shape, (14,))
        self.assertEqual(it["action"].shape, (14,))
        self.assertEqual(it["history"].shape, (3, 14))
        np.testing.assert_allclose(it["delta"], it["action"] - it["state"])
        self.assertEqual(set(it["images"]), set(DATA.CAMERAS))
        for cam in DATA.CAMERAS:
            self.assertEqual(it["images"][cam].dtype, np.uint8)
            self.assertEqual(it["images"][cam].shape[2], 3)

    def test_single_worker_equals_multi_worker(self):
        import torch
        ds = DATA.R45Dataset(self.entries, decode=True, cache_bytes=0)
        single = [ds[i] for i in range(8)]
        loader = torch.utils.data.DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=2,
            worker_init_fn=DATA.worker_init)
        multi = []
        for batch in loader:
            multi.append({k: (v[0].numpy() if torch.is_tensor(v) else v[0])
                          for k, v in batch.items() if k != "images"})
        for a, b in zip(single, multi):
            self.assertEqual(a["index"], int(b["index"]))
            self.assertEqual(a["t"], int(b["t"]))
            np.testing.assert_array_equal(a["state"], np.asarray(b["state"]))

    def test_cache_is_bounded(self):
        ds = DATA.R45Dataset(self.entries, decode=True, cache_bytes=1)
        for i in range(5):
            ds[i]
        self.assertEqual(len(ds._cache), 0)
        self.assertGreater(ds.stats["reads"], 0)

    def test_order_is_deterministic_and_epoch_dependent(self):
        a = DATA.EpochOrder(100, 0, 0)
        b = DATA.EpochOrder(100, 0, 0)
        c = DATA.EpochOrder(100, 0, 1)
        self.assertEqual(a.hash(), b.hash())
        self.assertNotEqual(a.hash(), c.hash())
        self.assertEqual(sorted(a.order.tolist()), list(range(100)))


class TestBudgetGuard(unittest.TestCase):
    def test_guard_expires(self):
        g = BENCH.BudgetGuard(0.05)
        self.assertFalse(g.expired())
        import time
        time.sleep(0.08)
        self.assertTrue(g.expired())
        self.assertGreaterEqual(g.remaining(), 0.0)

    def test_bench_refuses_without_approval(self):
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(ROOT, "diagnostics", "r45_gpu_bench.py"),
                            "--out", "/tmp/r45-bench-should-not-exist"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertIn("BLOCKED", r.stderr)
        self.assertFalse(os.path.exists("/tmp/r45-bench-should-not-exist"))


class TestTrainerCheckpointResume(unittest.TestCase):
    """Runs the real trainer entry point on CPU with the toy model."""

    @classmethod
    def setUpClass(cls):
        import r44_delta_v1 as D
        cls.tmp = "/tmp/r45-trainer-test"
        os.makedirs(cls.tmp, exist_ok=True)
        doc = json.load(open(SPLITS))
        cls.splits = os.path.join(cls.tmp, "splits.json")
        json.dump({"splits": {t: {"train": doc["splits"][t]["train"][:1],
                                  "dev": doc["splits"][t]["dev"][:1]}
                              for t in list(doc["splits"])[:1]}}, open(cls.splits, "w"))
        rng = np.random.default_rng(0)
        payload = D.fit_normalizer({"push_T": rng.normal(0, 0.02, (300, 14)).astype(np.float32)})
        cls.norm = os.path.join(cls.tmp, "normalizer.json")
        json.dump(payload, open(cls.norm, "w"))

    def _cmd(self, out, extra=()):
        return [sys.executable, os.path.join(ROOT, "training", "r45_train.py"),
                "--model", "fake", "--out", out, "--splits", self.splits,
                "--hdf5-root", HDF5, "--normalizer", self.norm,
                "--limit-samples", "24", "--micro-batch", "2", "--epochs", "2",
                "--checkpoint-every-frac", "0.5"] + list(extra)

    def _run(self, out, extra=()):
        import subprocess
        return subprocess.run(self._cmd(out, extra), capture_output=True, text=True, timeout=600)

    def _first_epoch_end(self, out):
        ck = sorted(os.listdir(os.path.join(out, "checkpoints")))
        ends = [c for c in ck if "epoch_end" in c]
        self.assertTrue(ends, "no epoch_end checkpoint in %s" % (ck,))
        return os.path.join(out, "checkpoints", ends[0])

    def test_end_to_end_then_resume_from_epoch_boundary(self):
        out = os.path.join(self.tmp, "run1")
        r = self._run(out)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        summary = json.load(open(os.path.join(out, "run_summary.json")))
        self.assertIsNone(summary["stop_reason"])
        self.assertEqual(summary["global_step"], 24)     # 12 optimizer steps/epoch x 2 epochs
        ckpt = self._first_epoch_end(out)
        meta = json.load(open(os.path.join(ckpt, "meta.json")))
        self.assertEqual(meta["phase"], "epoch_boundary")
        self.assertEqual(meta["cursor"], 0)
        self.assertEqual(meta["epoch"], 1)
        r2 = self._run(os.path.join(self.tmp, "run2"), ["--resume", ckpt])
        self.assertEqual(r2.returncode, 0, r2.stderr[-2000:])
        s2 = json.load(open(os.path.join(self.tmp, "run2", "run_summary.json")))
        # global_step is cumulative: 12 (restored) + 12 (epoch 1) = 24
        self.assertEqual(s2["global_step"], 24)

    def test_max_steps_writes_a_stopped_checkpoint(self):
        out = os.path.join(self.tmp, "run5")
        r = self._run(out, ["--max-steps", "4"])
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        ck = sorted(os.listdir(os.path.join(out, "checkpoints")))
        self.assertTrue(any("stopped" in c for c in ck), ck)
        s = json.load(open(os.path.join(out, "run_summary.json")))
        self.assertEqual(s["stop_reason"], "max-steps")
        self.assertEqual(s["global_step"], 4)

    def test_resume_rejects_a_tampered_cursor(self):
        out = os.path.join(self.tmp, "run3")
        self._run(out)
        ckpt = self._first_epoch_end(out)
        meta = json.load(open(os.path.join(ckpt, "meta.json")))
        meta["cursor"] = 1
        json.dump(meta, open(os.path.join(ckpt, "meta.json"), "w"))
        r = self._run(os.path.join(self.tmp, "run4"), ["--resume", ckpt])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("BLOCKED", r.stderr + r.stdout)



class TestFormalEntryIntegration(unittest.TestCase):
    def test_formal_r44_entry_uses_r45_runtime_reader_and_phase_timing(self):
        source = open(os.path.join(ROOT, "training", "r44_train.py"), encoding="utf-8").read()
        self.assertIn("r45_data", source)
        self.assertIn("data_timing", source)
        self.assertIn("samples_per_sec", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
