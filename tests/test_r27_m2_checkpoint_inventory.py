"""R27 CPU-only validation of sealed M2 checkpoint inventory.

No torch, HDF5, CUDA, checkpoint contents, or remote secrets required.
"""
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "r27_m2_checkpoint_inventory", ROOT / "tools/r27_m2_checkpoint_inventory.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build_fixture(root):
    ckpt = root / "ckpt"
    ckpt.mkdir()
    blobs = {
        "lora_adapter/adapter_model.safetensors": b"FAKE LORA, NOT A REAL MODEL",
        "proprio_projector_42.pt": b"FAKE PROJECTOR",
        "optimizer_state.pt": b"FAKE OPTIMIZER",
        "scheduler_state.pt": b"FAKE SCHEDULER",
        "rng.pt": b"FAKE RNG",
    }
    tasks = ["push_T", "stack_bowls"]
    task_ids = {"push_T": 0, "stack_bowls": 1}
    state = {
        "checkpoint_schema": "r15-full-state-v1",
        "run_id": "m2-test-only",
        "global_step": 4,
        "model_kind": "peft",
        "tasks": tasks, "task_ids": task_ids,
        "normalizer_sha256": "a"*64,
        "data_contract": {"tasks": tasks, "task_ids": task_ids,
                          "normalizer_sha256": "a"*64,
                          "split_sha256": "b"*64},
    }
    sampler = {
        "position": 4, "n_frames": 10, "perm": [9, 1, 4, 2, 3, 5, 6, 7, 8, 0],
        "order_hash": "c"*64, "frames_sha256": "d"*64,
    }
    blobs["trainer_state.json"] = json.dumps(state, sort_keys=True).encode()
    blobs["sampler.json"] = json.dumps(sampler, sort_keys=True).encode()
    for name, blob in blobs.items():
        dst = ckpt / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(blob)
    man = {
        "schema": "r15-full-state-v1", "run_id": "m2-test-only",
        "global_step": 4, "files": {name: sha(blob) for name, blob in blobs.items()}
    }
    (ckpt / "MANIFEST.json").write_text(json.dumps(man))
    stage = {
        "schema": "m2-train-evidence-v1",
        "run_id": "m2-test-only", "global_step_reached": 4,
        "tasks": tasks, "task_ids": task_ids,
        "normalizer_sha256": "a"*64,
        "prompt_token_ids_sha256": {"push_T": "e"*64, "stack_bowls": "f"*64}
    }
    stage_path = root / "stage_evidence.json"
    stage_path.write_text(json.dumps(stage))
    return ckpt, stage_path


class TestR27ReadonlyInventory(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.ckpt, self.stage = build_fixture(self.root)

    def run_inspection(self):
        return M.inspect(self.ckpt, self.stage)

    def test_good_manifest_and_sampler_report_counts_but_not_fake_seen_frames(self):
        before = (self.ckpt / "MANIFEST.json").read_bytes()
        info = self.run_inspection()
        self.assertEqual(info["checkpoint"]["global_step"], 4)
        self.assertEqual(info["sampler"]["consumed_schedule_slots"], 4)
        self.assertEqual(info["sampler"]["distinct_consumed_slot_indices"], 4)
        self.assertEqual(info["sampler"]["actual_unique_gradient_exposed_frames"],
                         "UNKNOWN_REQUIRES_FROZEN_FRAME_MAPPING")
        self.assertTrue(info["attested"]["manifest_file_hashes_verified"])
        self.assertTrue(info["attested"]["stage_crosscheck_verified"])
        self.assertEqual(info["optimizer_steps_executed_by_this_tool"], 0)
        self.assertEqual(before, (self.ckpt / "MANIFEST.json").read_bytes())

    def test_corrupt_checkpoint_file_fails_closed(self):
        (self.ckpt / "proprio_projector_42.pt").write_bytes(b"TAMPER")
        with self.assertRaisesRegex(M.InventoryError, "hash|SHA|projector"):
            self.run_inspection()

    def test_path_traversal_and_symlink_refused(self):
        man_path = self.ckpt / "MANIFEST.json"
        man = json.loads(man_path.read_text())
        man["files"]["../stage_evidence.json"] = sha(self.stage.read_bytes())
        man_path.write_text(json.dumps(man))
        with self.assertRaisesRegex(M.InventoryError, "path|relative|escape"):
            self.run_inspection()

    def test_sampler_global_step_mismatch_is_blocking(self):
        p = self.ckpt / "sampler.json"
        sam = json.loads(p.read_text())
        sam["position"] = 3
        p.write_text(json.dumps(sam))
        self._update_man_hash("sampler.json")
        with self.assertRaisesRegex(M.InventoryError, "position|step"):
            self.run_inspection()

    def test_sampler_perm_duplicates_fail_even_with_valid_hash(self):
        p = self.ckpt / "sampler.json"
        sam = json.loads(p.read_text())
        sam["perm"][2] = sam["perm"][1]
        p.write_text(json.dumps(sam))
        self._update_man_hash("sampler.json")
        with self.assertRaisesRegex(M.InventoryError, "permutation|duplicate"):
            self.run_inspection()

    def test_missing_prompt_token_evidence_fails_closed(self):
        stage = json.loads(self.stage.read_text())
        stage.pop("prompt_token_ids_sha256")
        self.stage.write_text(json.dumps(stage))
        with self.assertRaisesRegex(M.InventoryError, "prompt|token"):
            self.run_inspection()

    def test_wrong_stage_run_id_is_rejected(self):
        stage = json.loads(self.stage.read_text())
        stage["run_id"] = "DIFFERENT"
        self.stage.write_text(json.dumps(stage))
        with self.assertRaisesRegex(M.InventoryError, "run_id|stage"):
            self.run_inspection()

    def test_stage_missing_report_is_explicitly_blocked_not_verified(self):
        info = M.inspect(self.ckpt, None)
        self.assertFalse(info["attested"]["stage_crosscheck_verified"])
        self.assertEqual(info["stage"]["status"], "BLOCKED_NO_STAGE_EVIDENCE")

    def test_write_is_exclusive_and_not_inside_checkpoint(self):
        info = self.run_inspection()
        out = self.root / "proof.json"
        M.write_result_exclusive(out, info, checkpoint_dir=self.ckpt)
        self.assertTrue(out.is_file())
        with self.assertRaisesRegex(M.InventoryError, "exists|overwrite"):
            M.write_result_exclusive(out, info, checkpoint_dir=self.ckpt)
        with self.assertRaisesRegex(M.InventoryError, "checkpoint|inside"):
            M.write_result_exclusive(self.ckpt / "proof.json", info, checkpoint_dir=self.ckpt)

    def _update_man_hash(self, rel):
        path = self.ckpt / "MANIFEST.json"
        man = json.loads(path.read_text())
        man["files"][rel] = sha((self.ckpt / rel).read_bytes())
        path.write_text(json.dumps(man))


if __name__ == "__main__":
    unittest.main()
