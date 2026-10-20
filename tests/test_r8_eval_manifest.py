#!/usr/bin/env python3
"""R8-C tests: dataset manifest / freeze tooling (CPU, synthetic HDF5)."""
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import r8_eval_manifest as M   # noqa: E402


def write_episode(path, n=40, seed=0, instruction="Push the T-shaped block to align it."):
    import h5py
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 0.01, (n, 14)).astype(np.float32)
    state = np.cumsum(v, axis=0).astype(np.float32)
    action = np.vstack([state[1:], state[-1:]]).astype(np.float32)
    with h5py.File(path, "w") as h:
        for g, arr in (("state", state), ("action", action)):
            h.create_dataset(g + "/left_arm_joint_states", data=arr[:, :6])
            h.create_dataset(g + "/left_ee_joint_states", data=arr[:, 6:7])
            h.create_dataset(g + "/right_arm_joint_states", data=arr[:, 7:13])
            h.create_dataset(g + "/right_ee_joint_states", data=arr[:, 13:])
        h.create_dataset("instruction", data=np.bytes_(instruction))
        h.create_dataset("additional_info/frequency", data=np.int64(25))
    return path


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.data = os.path.join(self.dir, "data")
        os.makedirs(self.data)
        for i in range(3):
            write_episode(os.path.join(self.data, "episode_%07d.hdf5" % i), n=40 + 5 * i, seed=i)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_build_is_deterministic(self):
        a = M.build(self.data)
        b = M.build(self.data)
        self.assertEqual(a["dataset_sha256"], b["dataset_sha256"])
        self.assertEqual(a["n_episodes"], 3)
        self.assertEqual(a["n_unique_content"], 3)

    def test_content_hash_detects_a_single_changed_value(self):
        a = M.build(self.data)
        p = os.path.join(self.data, "episode_0000001.hdf5")
        import h5py
        with h5py.File(p, "r+") as h:
            arr = h["state/left_arm_joint_states"][:]
            arr[0, 0] += 1e-6
            h["state/left_arm_joint_states"][:] = arr
        b = M.build(self.data)
        self.assertNotEqual(a["dataset_sha256"], b["dataset_sha256"])
        changed = [x["file"] for x, y in zip(a["episodes"], b["episodes"])
                   if x["content_sha256"] != y["content_sha256"]]
        self.assertEqual(changed, ["episode_0000001.hdf5"])

    def test_duplicate_content_detected(self):
        shutil.copy(os.path.join(self.data, "episode_0000000.hdf5"),
                    os.path.join(self.data, "copy_of_ep0.hdf5"))
        m = M.build(self.data)
        self.assertEqual(m["n_episodes"], 4)
        self.assertEqual(m["n_unique_content"], 3)
        self.assertEqual(m["n_duplicate_content"], 1)

    def test_verify_reports_tampering(self):
        m = M.build(self.data)
        self.assertTrue(M.verify(m)["ok"])
        write_episode(os.path.join(self.data, "episode_0000000.hdf5"), seed=99)
        v = M.verify(m)
        self.assertFalse(v["ok"])
        self.assertTrue(any("content changed" in p for p in v["problems"]))

    def test_disjoint_detects_overlap_with_used_split(self):
        m = M.build(self.data)
        split = os.path.join(self.dir, "split.json")
        with open(split, "w") as fh:
            json.dump({"train": ["episode_0000000.hdf5"], "val": ["episode_0000002.hdf5"]}, fh)
        res = M.check_disjoint(m, split)
        self.assertFalse(res["ok"])
        self.assertEqual(res["n_overlap_with_used"], 2)
        with open(split, "w") as fh:
            json.dump({"train": ["other0.hdf5"], "val": ["other1.hdf5"]}, fh)
        self.assertTrue(M.check_disjoint(m, split)["ok"])

    def test_freeze_binds_manifest_protocol_and_sha(self):
        m = M.build(self.data)
        proto = {"gates": {"g1": "l1 < copy-V"}}
        out = os.path.join(self.dir, "frozen.json")
        frozen = M.freeze(m, proto, "deadbeef", out)
        reloaded = json.load(open(out))
        self.assertEqual(reloaded["git_sha"], "deadbeef")
        self.assertEqual(reloaded["manifest"]["dataset_sha256"], m["dataset_sha256"])
        self.assertEqual(reloaded["protocol"], proto)
        self.assertIn("frozen_utc", reloaded)
        self.assertEqual(len(reloaded["manifest_sha256"]), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
