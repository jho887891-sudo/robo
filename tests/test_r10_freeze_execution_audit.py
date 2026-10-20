#!/usr/bin/env python3
"""CPU-only independent negative controls for the REAL R10 freeze write path."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import r10_freeze_guard as G


def fake_result():
    names = ["new_%02d.hdf5" % i for i in range(25)]
    episodes = [{"file": n, "content_sha256": "content-%02d" % i,
                 "state_action_sha256": "sa-%02d" % i,
                 "length": 150, "frequency": 25,
                 "instruction": "Push the T-shaped block"} for i, n in enumerate(names)]
    split = {"pass": True, "dev": names[:5], "final": names[5:],
             "seed": "r10-dev-v1"}
    checks = {k: {"pass": True} for k in
              ("schema", "build", "cross_set_disjoint", "counts", "verify")}
    checks["split_assignment"] = split
    checks["cross_set_disjoint"]["small_set_override"] = False
    checks["counts"]["expect_total"] = 25
    return {"pass": True, "checks": checks,
            "manifest": {"episodes": episodes, "n_episodes": 25,
                         "n_unique_content": 25, "task_filter": "push"},
            "protocol_sha256": G.FROZEN_PROTOCOL_SHA256,
            "used_manifest_sha256": "old100-content-manifest-hash"}


class FreezeAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.proto = os.path.join(REPO, "protocol", "r9_eval_protocol.json")
        self.assertEqual(hashlib.sha256(open(self.proto, "rb").read()).hexdigest(),
                         G.FROZEN_PROTOCOL_SHA256)
        self.freeze = os.path.join(self.temp.name, "frozen.json")
        self.sha = "a" * 40

    def test_writes_actual_5_20_partition_into_hashed_manifest(self):
        r = G.freeze_checked(fake_result(), self.proto, self.sha, self.freeze)
        with open(self.freeze) as fh:
            saved = json.load(fh)
        part = saved["manifest"]["r10_split_assignment"]
        self.assertEqual((len(part["dev"]), len(part["final"])), (5, 20))
        self.assertEqual(saved["git_sha"], self.sha)
        digest = hashlib.sha256(json.dumps(saved["manifest"], sort_keys=True).encode()).hexdigest()
        self.assertEqual(saved["manifest_sha256"], digest)
        self.assertEqual(r, saved)

    def test_rejects_failed_gate(self):
        r = fake_result()
        r["checks"]["schema"]["pass"] = False
        with self.assertRaises(ValueError):
            G.freeze_checked(r, self.proto, self.sha, self.freeze)
        self.assertFalse(os.path.exists(self.freeze))

    def test_rejects_missing_or_short_git_sha(self):
        for sha in ("", "abc"):
            with self.assertRaises(ValueError):
                G.freeze_checked(fake_result(), self.proto, sha, self.freeze)
        self.assertFalse(os.path.exists(self.freeze))

    def test_rejects_test_mode_flag_even_if_all_checks_pass(self):
        r = fake_result()
        r["checks"]["cross_set_disjoint"]["small_set_override"] = True
        with self.assertRaises(ValueError):
            G.freeze_checked(r, self.proto, self.sha, self.freeze)

    def test_rejects_missing_gate_even_if_all_remaining_pass(self):
        r = fake_result()
        del r["checks"]["verify"]
        with self.assertRaises(ValueError):
            G.freeze_checked(r, self.proto, self.sha, self.freeze)

    def test_rejects_protocol_drift(self):
        bad_protocol = os.path.join(self.temp.name, "changed.json")
        with open(bad_protocol, "w") as fh:
            json.dump({"splits": {"dev_episodes": 5}}, fh)
        with self.assertRaises(ValueError):
            G.freeze_checked(fake_result(), bad_protocol, self.sha, self.freeze)
        self.assertFalse(os.path.exists(self.freeze))

    def test_rejects_missing_episode(self):
        r = fake_result()
        r["manifest"]["episodes"].pop()
        with self.assertRaises(ValueError):
            G.freeze_checked(r, self.proto, self.sha, self.freeze)

    def test_rejects_dev_final_overlap(self):
        r = fake_result()
        r["checks"]["split_assignment"]["final"][0] = r["checks"]["split_assignment"]["dev"][0]
        with self.assertRaises(ValueError):
            G.freeze_checked(r, self.proto, self.sha, self.freeze)

    def test_does_not_overwrite_existing_freeze(self):
        with open(self.freeze, "w") as fh:
            fh.write("immutable old record")
        with self.assertRaises(FileExistsError):
            G.freeze_checked(fake_result(), self.proto, self.sha, self.freeze)
        with open(self.freeze) as fh:
            self.assertEqual(fh.read(), "immutable old record")

    def test_cli_rejects_test_bypass_during_freeze_before_loading_data(self):
        args = [sys.executable, os.path.join(REPO, "tools", "r10_freeze_guard.py"),
                "--candidate-dir", "/nonexistent/unread",
                "--used-manifest", "/nonexistent/used.json",
                "--split", "/nonexistent/split.json",
                "--protocol", self.proto,
                "--out", os.path.join(self.temp.name, "evidence.json"),
                "--freeze-out", self.freeze,
                "--git-sha", self.sha,
                "--allow-small-set-for-tests"]
        p = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(p.returncode, 9)
        self.assertFalse(os.path.exists(self.freeze))

    def test_cli_rejects_20_episode_override_during_freeze(self):
        args = [sys.executable, os.path.join(REPO, "tools", "r10_freeze_guard.py"),
                "--candidate-dir", "/nonexistent/unread",
                "--used-manifest", "/nonexistent/used.json",
                "--split", "/nonexistent/split.json",
                "--protocol", self.proto,
                "--out", os.path.join(self.temp.name, "evidence.json"),
                "--freeze-out", self.freeze,
                "--git-sha", self.sha, "--expect-total", "20"]
        p = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(p.returncode, 9)
        self.assertFalse(os.path.exists(self.freeze))


if __name__ == "__main__":
    unittest.main()
