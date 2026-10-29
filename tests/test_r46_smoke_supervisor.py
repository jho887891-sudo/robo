import hashlib
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diagnostics.r46_smoke_supervisor import (  # noqa: E402
    ApprovalError,
    main as supervisor_main,
    probe_gpu,
    run_child,
    validate_approval,
    validate_smoke_args,
    verify_model_identity,
)


def approval_doc(now=None):
    now = now or datetime.now(timezone.utc)
    return {
        "schema": "r47-gpu-approval-template/v1",
        "template_only": False,
        "approved": True,
        "approval_id": "approval-test-001",
        "approver": "research-lead-test-fixture",
        "approved_source_sha": "a" * 40,
        "approved_gpu_uuid": "GPU-test-uuid",
        "approved_window_start_iso8601": (now - timedelta(minutes=1)).isoformat(),
        "approved_window_end_iso8601": (now + timedelta(minutes=10)).isoformat(),
        "approved_max_wall_seconds": 120,
        "approved_min_total_gib": 48,
        "approved_min_free_gib": 42,
        "require_exclusive_gpu": True,
        "approved_model_sha_list": [
            {"filename": f"model-0000{i}-of-00003.safetensors", "sha256": str(i) * 64}
            for i in (1, 2, 3)
        ],
        "approved_original_manifest_sha256": "b" * 64,
        "approved_smoke_samples_sha256": "c" * 64,
        "allowed_operation": "FIVE_SAMPLE_SMOKE_ONLY",
        "formal_training_approved": False,
        "extra_benchmark_approved": False,
    }


def expected_hashes(**overrides):
    values = {
        "approved_original_manifest_sha256": "b" * 64,
        "approved_smoke_samples_sha256": "c" * 64,
        "approved_model_sha_list": approval_doc()["approved_model_sha_list"],
    }
    values.update(overrides)
    return values


class ApprovalTests(unittest.TestCase):
    def test_accepts_complete_in_window_approval_and_freezes_contract(self):
        doc = approval_doc()
        result = validate_approval(
            doc,
            source_sha="a" * 40,
            requested_wall_seconds=90,
            expected_hashes=expected_hashes(),
            now=datetime.now(timezone.utc),
        )
        self.assertEqual(result["approval_id"], "approval-test-001")
        self.assertEqual(result["approved_gpu_uuid"], "GPU-test-uuid")

    def test_rejects_missing_approval_fields_and_wrong_sha(self):
        doc = approval_doc()
        del doc["approved_gpu_uuid"]
        with self.assertRaisesRegex(ApprovalError, "gpu_uuid"):
            validate_approval(doc, source_sha="a" * 40, requested_wall_seconds=90,
                              expected_hashes=expected_hashes(), now=datetime.now(timezone.utc))
        doc = approval_doc()
        with self.assertRaisesRegex(ApprovalError, "source SHA"):
            validate_approval(doc, source_sha="e" * 40, requested_wall_seconds=90,
                              expected_hashes=expected_hashes(), now=datetime.now(timezone.utc))

    def test_rejects_out_of_window_wrong_hash_or_nonexclusive_resource_contract(self):
        now = datetime.now(timezone.utc)
        doc = approval_doc(now)
        doc["approved_window_end_iso8601"] = (now - timedelta(seconds=1)).isoformat()
        with self.assertRaisesRegex(ApprovalError, "outside"):
            validate_approval(doc, source_sha="a" * 40, requested_wall_seconds=90,
                              expected_hashes=expected_hashes(), now=now)

        doc = approval_doc(now)
        with self.assertRaisesRegex(ApprovalError, "approved_smoke_samples_sha256"):
            validate_approval(doc, source_sha="a" * 40, requested_wall_seconds=90,
                              expected_hashes=expected_hashes(approved_smoke_samples_sha256="e" * 64), now=now)

        doc = approval_doc(now)
        doc["require_exclusive_gpu"] = False
        with self.assertRaisesRegex(ApprovalError, "exclusive"):
            validate_approval(doc, source_sha="a" * 40, requested_wall_seconds=90,
                              expected_hashes=expected_hashes(), now=now)

    def test_rejects_budget_over_1800_and_live_training_authorization(self):
        doc = approval_doc()
        doc["approved_max_wall_seconds"] = 1801
        with self.assertRaisesRegex(ApprovalError, "1800"):
            validate_approval(doc, source_sha="a" * 40, requested_wall_seconds=90,
                              expected_hashes=expected_hashes(), now=datetime.now(timezone.utc))

    def test_unmodified_published_template_is_not_an_approval(self):
        doc = approval_doc()
        doc["template_only"] = True
        doc["approved"] = False
        with self.assertRaisesRegex(ApprovalError, "completed"):
            validate_approval(doc, source_sha="a" * 40, requested_wall_seconds=90,
                              expected_hashes=expected_hashes(), now=datetime.now(timezone.utc))

    def test_only_frozen_input_paths_may_be_forwarded_to_smoke(self):
        self.assertEqual(validate_smoke_args(["--hdf5-root", "/data", "--base-model=/model"]),
                         ["--hdf5-root", "/data", "--base-model=/model"])
        for forbidden in ("--formal-training-authorized", "--epochs", "--cams", "--lr"):
            with self.subTest(forbidden=forbidden), self.assertRaisesRegex(ApprovalError, "not permitted"):
                validate_smoke_args([forbidden])

    def test_rejects_wrong_approved_model_shard_identity(self):
        doc = approval_doc()
        doc["approved_model_sha_list"][0]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ApprovalError, "model shard SHA"):
            validate_approval(doc, source_sha="a" * 40, requested_wall_seconds=90,
                              expected_hashes=expected_hashes(), now=datetime.now(timezone.utc))

    def test_rejects_unknown_approval_fields(self):
        doc = approval_doc()
        doc["gpu_budget_override"] = True
        with self.assertRaisesRegex(ApprovalError, "unsupported approval fields"):
            validate_approval(doc, source_sha="a" * 40, requested_wall_seconds=90,
                              expected_hashes=expected_hashes(), now=datetime.now(timezone.utc))


class AdmissionTests(unittest.TestCase):
    def test_gpu_probe_accepts_only_matching_uuid_memory_and_no_compute_users(self):
        outputs = [
            SimpleNamespace(returncode=0, stdout="GPU-test-uuid, 49152, 43008, 1\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
        with patch("diagnostics.r46_smoke_supervisor.subprocess.run", side_effect=outputs):
            result = probe_gpu("GPU-test-uuid", 48, 42)
        self.assertEqual(result["gpu"]["utilization_percent"], 1)
        self.assertEqual(result["active_compute_processes"], [])

    def test_gpu_probe_refuses_wrong_uuid_low_memory_and_compute_occupancy(self):
        base = SimpleNamespace(returncode=0, stdout="GPU-other, 49152, 43008, 2\n", stderr="")
        with patch("diagnostics.r46_smoke_supervisor.subprocess.run", return_value=base):
            with self.assertRaisesRegex(RuntimeError, "UUID"):
                probe_gpu("GPU-test-uuid", 48, 42)
        low = SimpleNamespace(returncode=0, stdout="GPU-test-uuid, 49152, 4096, 2\n", stderr="")
        with patch("diagnostics.r46_smoke_supervisor.subprocess.run", return_value=low):
            with self.assertRaisesRegex(RuntimeError, "memory thresholds"):
                probe_gpu("GPU-test-uuid", 48, 42)
        occupied = [
            SimpleNamespace(returncode=0, stdout="GPU-test-uuid, 49152, 43008, 2\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="1234, python\n", stderr=""),
        ]
        with patch("diagnostics.r46_smoke_supervisor.subprocess.run", side_effect=occupied):
            with self.assertRaisesRegex(RuntimeError, "compute processes exist"):
                probe_gpu("GPU-test-uuid", 48, 42)

    def test_model_identity_checks_index_and_all_three_actual_shards(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = []
            names = [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]
            for i, name in enumerate(names, 1):
                payload = f"shard-{i}".encode()
                (root / name).write_bytes(payload)
                files.append({"name": name, "sha256": hashlib.sha256(payload).hexdigest()})
            config_bytes = b'{"model_type":"openvla"}\n'
            (root / "config.json").write_bytes(config_bytes)
            files.append({"name": "config.json", "sha256": hashlib.sha256(config_bytes).hexdigest()})
            index_doc = {"weight_map": {f"weight_{i}": name for i, name in enumerate(names)}}
            index_bytes = (json.dumps(index_doc) + "\n").encode()
            (root / "model.safetensors.index.json").write_bytes(index_bytes)
            files.append({"name": "model.safetensors.index.json",
                          "sha256": hashlib.sha256(index_bytes).hexdigest()})
            identity = {"schema": "r45-static-base-identity/v1", "shard_count": 3,
                        "indexed": True, "files": files}
            self.assertEqual(len(verify_model_identity(identity, root)), 3)
            (root / names[0]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ApprovalError, "model identity SHA mismatch"):
                verify_model_identity(identity, root)
        doc = approval_doc()
        doc["formal_training_approved"] = True
        with self.assertRaisesRegex(ApprovalError, "formal training"):
            validate_approval(doc, source_sha="a" * 40, requested_wall_seconds=90,
                              expected_hashes=expected_hashes(), now=datetime.now(timezone.utc))


@unittest.skipUnless(sys.platform == "linux", "process-group hard stop is exercised on Linux")
class ParentSupervisorTests(unittest.TestCase):
    def test_parent_verifies_on_disk_model_identity_before_gpu_admission(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest, samples, identity_path = root / "manifest.json", root / "samples.json", root / "identity.json"
            manifest.write_text("{}", encoding="utf-8")
            samples.write_text("{}", encoding="utf-8")
            model_root = root / "model"
            model_root.mkdir()
            shard_names = [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]
            file_rows = []
            for index, name in enumerate(shard_names):
                payload = f"shard-{index}".encode()
                (model_root / name).write_bytes(payload)
                file_rows.append({"name": name, "sha256": hashlib.sha256(payload).hexdigest()})
            for name, payload in (("config.json", b"{}\n"),):
                (model_root / name).write_bytes(payload)
                file_rows.append({"name": name, "sha256": hashlib.sha256(payload).hexdigest()})
            index_doc = {"weight_map": {f"weight.{i}": name for i, name in enumerate(shard_names)}}
            index_bytes = (json.dumps(index_doc) + "\n").encode()
            (model_root / "model.safetensors.index.json").write_bytes(index_bytes)
            file_rows.append({"name": "model.safetensors.index.json",
                              "sha256": hashlib.sha256(index_bytes).hexdigest()})
            identity = {"schema": "r45-static-base-identity/v1", "shard_count": 3,
                        "indexed": True, "files": file_rows}
            identity_path.write_text(json.dumps(identity), encoding="utf-8")
            approval = approval_doc()
            approval["approved_model_sha_list"] = sorted(
                [{"filename": row["name"], "sha256": row["sha256"]} for row in file_rows],
                key=lambda row: row["filename"],
            )
            approval["approved_original_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            approval["approved_smoke_samples_sha256"] = hashlib.sha256(samples.read_bytes()).hexdigest()
            source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            approval["approved_source_sha"] = source_sha
            approval_path = root / "approval.json"
            approval_path.write_text(json.dumps(approval), encoding="utf-8")
            (model_root / shard_names[0]).write_bytes(b"changed after identity freeze")
            out = root / "evidence"
            argv = ["--approval-file", str(approval_path), "--source-sha", source_sha,
                    "--dataset-manifest", str(manifest), "--sample-plan", str(samples),
                    "--model-identity", str(identity_path), "--output-dir", str(out),
                    "--max-wall-seconds", "60", "--", "--base-model", str(model_root)]
            with patch("diagnostics.r46_smoke_supervisor.probe_gpu") as gpu_probe, \
                    contextlib.redirect_stderr(io.StringIO()):
                result = supervisor_main(argv)
            self.assertEqual(result, 2)
            gpu_probe.assert_not_called()
            self.assertFalse(out.exists())

    def test_normal_child_exit_is_recorded_with_logs_and_unique_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "run"
            result = run_child([sys.executable, "-c", "print('child-ok')"], out, 5)
            self.assertEqual(result["child_exit_code"], 0)
            self.assertEqual(result["stop_reason"], "CHILD_EXITED")
            self.assertIn("child-ok", (out / "stdout.log").read_text())
            self.assertTrue((out / "supervisor.jsonl").is_file())
            self.assertTrue((out / "supervisor_final.json").is_file())

    def test_child_start_failure_still_writes_final_report(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "start-fail"
            result = run_child([str(Path(td) / "does-not-exist")], out, 5)
            self.assertEqual(result["status"], "CHILD_START_FAILED")
            self.assertTrue((out / "supervisor_final.json").is_file())

    def test_normal_exit_cleans_descendants_left_in_owned_session(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "orphan-child"
            code = (
                "import subprocess,sys; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                "print(p.pid, flush=True)"
            )
            result = run_child([sys.executable, "-c", code], out, 5)
            self.assertEqual(result["stop_reason"], "CHILD_EXITED_WITH_ORPHANS_CLEANED")
            self.assertEqual(result["child_exit_code"], 0)
            self.assertIn("SIGKILL_ORPHAN_GROUP", result["signals_sent"])
            self.assertTrue(result["hard_stop_confirmed"])

    def test_gpu_admission_failure_leaves_parent_owned_blocked_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest, samples, identity = root / "manifest.json", root / "samples.json", root / "identity.json"
            manifest.write_text("{}", encoding="utf-8")
            samples.write_text("{}", encoding="utf-8")
            model_root = root / "model"
            model_root.mkdir()
            shard_names = [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]
            file_rows = []
            for index, name in enumerate(shard_names):
                payload = f"shard-{index}".encode()
                (model_root / name).write_bytes(payload)
                file_rows.append({"name": name, "sha256": hashlib.sha256(payload).hexdigest()})
            config_bytes = b"{}\n"
            (model_root / "config.json").write_bytes(config_bytes)
            file_rows.append({"name": "config.json", "sha256": hashlib.sha256(config_bytes).hexdigest()})
            index_bytes = (json.dumps({"weight_map": {f"weight.{i}": name
                                                        for i, name in enumerate(shard_names)}}) + "\n").encode()
            (model_root / "model.safetensors.index.json").write_bytes(index_bytes)
            file_rows.append({"name": "model.safetensors.index.json",
                              "sha256": hashlib.sha256(index_bytes).hexdigest()})
            identity.write_text(json.dumps({
                "schema": "r45-static-base-identity/v1", "shard_count": 3, "indexed": True,
                "files": file_rows,
            }), encoding="utf-8")
            all_model_hashes = [
                {"filename": row["name"], "sha256": row["sha256"]} for row in file_rows
            ]
            approval_doc_now = approval_doc()
            approval_doc_now["approved_model_sha_list"] = sorted(all_model_hashes, key=lambda x: x["filename"])
            source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            approval = approval_doc_now
            approval["approved_source_sha"] = source_sha
            approval["approved_original_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            approval["approved_smoke_samples_sha256"] = hashlib.sha256(samples.read_bytes()).hexdigest()
            approval_path = root / "approval.json"
            approval_path.write_text(json.dumps(approval), encoding="utf-8")
            out = root / "evidence"
            argv = ["--approval-file", str(approval_path), "--source-sha", source_sha,
                    "--dataset-manifest", str(manifest), "--sample-plan", str(samples),
                    "--model-identity", str(identity), "--output-dir", str(out),
                    "--max-wall-seconds", "60", "--", "--base-model", str(model_root)]
            with patch("diagnostics.r46_smoke_supervisor.probe_gpu",
                       side_effect=RuntimeError("simulated busy GPU")), contextlib.redirect_stderr(io.StringIO()):
                code = supervisor_main(argv)
            self.assertEqual(code, 2)
            final = json.loads((out / "supervisor_final.json").read_text())
            self.assertEqual(final["status"], "BLOCKED_GPU_ADMISSION")
            self.assertIsNone(final["child_pid"])
            self.assertFalse((out / "smoke").exists())

    def test_deadline_sends_sigint_then_sigkill_only_to_owned_process_group(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "timeout"
            child = (
                "import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); "
                "print('ready', flush=True); time.sleep(30)"
            )
            started = time.monotonic()
            result = run_child([sys.executable, "-c", child], out, 0.8,
                               poll_interval=0.02, grace_seconds=0.25)
            elapsed = time.monotonic() - started
            self.assertEqual(result["stop_reason"], "STOP_TIMEOUT_KILLED")
            self.assertEqual(result["signals_sent"], ["SIGINT", "SIGKILL"])
            self.assertLess(elapsed, 2.0)
            self.assertEqual(result["child_exit_code"], -signal.SIGKILL)
            self.assertIn("ready", (out / "stdout.log").read_text())

    def test_cooperative_sigint_is_recorded_without_sigkill(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "cooperative"
            child = (
                "import signal,sys,time; "
                "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
                "print('ready', flush=True); time.sleep(30)"
            )
            result = run_child([sys.executable, "-c", child], out, 0.8,
                               poll_interval=0.01, grace_seconds=0.25)
            self.assertEqual(result["stop_reason"], "STOP_TIMEOUT_SIGINT")
            self.assertEqual(result["signals_sent"], ["SIGINT"])
            self.assertEqual(result["child_exit_code"], 0)

    def test_timeout_does_not_signal_unrelated_process_group(self):
        with tempfile.TemporaryDirectory() as td:
            sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                       start_new_session=True)
            try:
                out = Path(td) / "isolated"
                child = "import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(30)"
                result = run_child([sys.executable, "-c", child], out, 0.8,
                                   poll_interval=0.02, grace_seconds=0.25)
                self.assertEqual(result["stop_reason"], "STOP_TIMEOUT_KILLED")
                self.assertIsNone(sibling.poll())
            finally:
                sibling.terminate()
                sibling.wait(timeout=2)

    def test_output_collision_refuses_without_overwriting_existing_files(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "existing"
            out.mkdir()
            sentinel = out / "sentinel.txt"
            sentinel.write_text("keep")
            with self.assertRaises(FileExistsError):
                run_child([sys.executable, "-c", "pass"], out, 1)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_cli_without_approval_refuses_before_gpu_probe_child_or_output(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "must_not_exist"
            source_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            result = subprocess.run(
                [sys.executable, str(ROOT / "diagnostics/r46_smoke_supervisor.py"),
                 "--approval-file", str(Path(td) / "absent-approval.json"),
                 "--source-sha", source_sha,
                 "--dataset-manifest", str(Path(td) / "manifest.json"),
                 "--sample-plan", str(Path(td) / "samples.json"),
                 "--model-identity", str(Path(td) / "model-identity.json"),
                 "--out", str(out), "--max-wall-seconds", "60"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("BLOCKED_AUTHORIZATION", result.stderr)
            self.assertFalse(out.exists())

    def test_smoke_child_direct_invocation_refuses_before_torch_import(self):
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / "torch-imported"
            torch_stub = Path(td) / "torch.py"
            torch_stub.write_text(
                "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('bad')\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = td
            result = subprocess.run(
                [sys.executable, str(ROOT / "diagnostics/r46_openvla_smoke.py"),
                 "--approval-file", "absent", "--source-sha", "a" * 40,
                 "--sample-plan", "absent", "--dataset-manifest", "absent",
                 "--model-identity", "absent", "--approved-gpu-uuid", "GPU-test",
                 "--out", str(Path(td) / "out"), "--max-wall-seconds", "60"],
                env=env, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("BLOCKED_PARENT", result.stderr)
            self.assertFalse(marker.exists())
            self.assertFalse((Path(td) / "out").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
