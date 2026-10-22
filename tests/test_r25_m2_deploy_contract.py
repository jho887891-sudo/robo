"""R25 M2 deploy preflight: real two-task payload + sealed synthetic checkpoint.

TDD: intentionally written BEFORE m2_deploy_contract exists.
No HDF5, no actual weights, no external GPU credentials.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, "code", "xpolicylab", "policy", "OpenVLA_R42"))
import m2_task_normalizer as TN
import m2_train_shared as M2
import r42_adapter as R
import m2_deploy_contract as C


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True))
    return path


@pytest.fixture
def case(tmp_path):
    src = os.path.join(ROOT, "data", "m2_task_normalizer_2task.json")
    payload = json.load(open(src))
    assert sorted(payload["tasks"]) == ["push_T", "stack_bowls"]
    tasks = ["push_T", "stack_bowls"]
    normalizer = save(tmp_path / "normalizer.json", payload)
    splits = tmp_path / "splits.json"
    splits.write_bytes(open(os.path.join(ROOT, "data", "m2_splits.json"), "rb").read())

    dc = {
        "schema": M2.SCHEMA_EVIDENCE,
        "tasks": tasks, "task_ids": payload["task_id"],
        "normalizer_sha256": TN.normalizer_sha256(payload),
        "per_task_normalizer_sha256": {
            t: TN.sha256_json(payload["tasks"][t]) for t in tasks},
        "task_mix_schedule_sha256": "a" * 64,
        "split_sha256": file_sha(splits),
        "camera_order": list(R.CAMERA_ORDER),
        "gripper_dims": list(R.GRIPPER_DIMS),
        "base_dim": 14, "n42": 42,
        "alignment_version": R.ALIGNMENT_VERSION,
        "target_representation": R.TARGET,
        "action_tokens": 15,
    }
    state = {
        "checkpoint_schema": "r15-full-state-v1",
        "global_step": 2000, "run_id": "m2-2task-test",
        "model_kind": "peft", "optimizer_format_version": 1,
        "tasks": tasks, "task_ids": payload["task_id"],
        "normalizer_sha256": TN.normalizer_sha256(payload),
        "per_task_normalizer_sha256": dc["per_task_normalizer_sha256"],
        "base_model": {"config.json": "b" * 64},
        "data_contract": dc,
        "task_mix": {"schedule_sha256": "a" * 64},
    }
    ckpt = tmp_path / "checkpoint"
    files = {
        "lora_adapter/adapter_model.safetensors": b"stub weights, not loadable",
        "proprio_projector_42.pt": b"stub projector",
        "optimizer_state.pt": b"stub optimizer",
        "scheduler_state.pt": b"stub scheduler",
        "rng.pt": b"stub RNG",
        "sampler.json": b"{}",
    }
    for rel, data in files.items():
        target = ckpt / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    save(ckpt / "trainer_state.json", state)
    hashes = {rel: file_sha(ckpt / rel) for rel in
              list(files) + ["trainer_state.json"]}
    save(ckpt / "MANIFEST.json",
         {"schema": "r15-full-state-v1", "global_step": 2000,
          "run_id": state["run_id"], "files": hashes})

    prompt_text = M2.prompts_for(tasks, include_task_token=False)
    stage = {
        "schema": M2.SCHEMA_EVIDENCE,
        "global_step_reached": 2000,
        "run_id": state["run_id"],
        "tasks": tasks, "task_ids": payload["task_id"],
        "normalizer_sha256": TN.normalizer_sha256(payload),
        "split_sha256": file_sha(splits),
        "include_task_token": False, "seed": 0,
        "prompts": prompt_text,
        "prompt_sha256": {t: hashlib.sha256(prompt_text[t].encode()).hexdigest()
                          for t in tasks},
    }
    stage_path = save(tmp_path / "stage_evidence.json", stage)
    safety = {
        "schema": "r25-m2-robot-safety-v1",
        "source_kind": "robot_joint_hard_limits",
        "verified_by": "unit-test synthetic fixture, NOT a hardware attestation",
        "tasks": {
            t: {"amin": [-10.0] * 14,
                "amax": [10.0] * 14,
                "max_rate": [0.5] * 14}
            for t in tasks
        },
    }
    safety_path = save(tmp_path / "safety.json", safety)
    return {"checkpoint_dir": str(ckpt),
            "normalizer_path": str(normalizer),
            "splits_path": str(splits),
            "stage_evidence_path": str(stage_path),
            "safety_contract_path": str(safety_path),
            "task": "push_T"}


def test_real_m2_normalizer_session_metadata_is_valid(case):
    meta, att = C.build_session_metadata(**case)
    assert R.validate_metadata(meta, expect_task="push_T")
    assert meta["normalizer_sha256"] == TN.normalizer_sha256(
        json.load(open(case["normalizer_path"])))
    assert meta["task_id"] == meta["serving_task"] == "push_T"
    assert meta["tasks"] == ["push_T", "stack_bowls"]
    assert meta["normalizer"]["schema"] == "m2-task-normalizer-v1"
    assert meta["safety_contract_sha256"] == file_sha(case["safety_contract_path"])
    assert att["checkpoint_global_step"] == 2000
    assert meta["trained_prompt"] == M2.prompts_for(["push_T", "stack_bowls"])["push_T"]
    assert meta["tokenizer_prompt_parity"] == "GPU_SMOKE_REQUIRED"


def test_task_specific_42d_history_and_decoder(case):
    case["task"] = "stack_bowls"
    meta, _ = C.build_session_metadata(**case)
    ctx = C.M2SessionContract(meta, "stack_bowls")
    feat0 = ctx.observe(np.zeros(14, dtype=np.float32), timestamp=1.0)
    feat1 = ctx.observe(np.full(14, 0.001, dtype=np.float32), timestamp=2.0)
    assert feat0.shape == feat1.shape == (42,)
    assert np.isfinite(feat1).all()
    assert ctx.expected_prompt == meta["trained_prompt"]
    s = np.full(14, 0.001, np.float32)
    v = ctx.velocity()
    cmd, res, info = ctx.decode(np.full(14, 128, dtype=np.int64), s, v)
    assert cmd.shape == res.shape == (14,)
    assert cmd[6] == pytest.approx(float(s[6] + v[6]), abs=1e-7)
    assert np.isfinite(cmd).all()


def test_wrong_task_and_task_id_rejected(case):
    meta, _ = C.build_session_metadata(**case)
    with pytest.raises((ValueError, RuntimeError), match="task"):
        C.M2SessionContract(meta, "sweep_blocks")
    with pytest.raises((ValueError, RuntimeError), match="task"):
        C.M2SessionContract(meta, "stack_bowls")
    state_path = os.path.join(case["checkpoint_dir"], "trainer_state.json")
    state = json.load(open(state_path))
    state["task_ids"]["push_T"] = 1
    with open(state_path, "w") as f:
        json.dump(state, f)
    with pytest.raises((ValueError, RuntimeError), match="hash|task"):
        C.build_session_metadata(**case)


def test_changed_normalizer_block_rejected(case):
    obj = json.load(open(case["normalizer_path"]))
    obj["tasks"]["push_T"]["q99"][0] += 0.02
    with open(case["normalizer_path"], "w") as f:
        json.dump(obj, f)
    with pytest.raises((ValueError, RuntimeError), match="normalizer|sha256"):
        C.build_session_metadata(**case)


def test_missing_safety_contract_refused(case):
    case["safety_contract_path"] = str(os.path.join(
        os.path.dirname(case["normalizer_path"]), "nonexistent.json"))
    with pytest.raises((ValueError, RuntimeError, OSError), match="safety|missing"):
        C.build_session_metadata(**case)


def test_untrusted_data_bounds_as_safety_refused(case):
    obj = json.load(open(case["safety_contract_path"]))
    obj["source_kind"] = "training_quantiles"
    with open(case["safety_contract_path"], "w") as f:
        json.dump(obj, f)
    with pytest.raises((ValueError, RuntimeError), match="safety|source_kind|hard_limits"):
        C.build_session_metadata(**case)


def test_stage_training_prompt_tamper_rejected(case):
    obj = json.load(open(case["stage_evidence_path"]))
    obj["prompts"]["push_T"] = "another instruction"
    with open(case["stage_evidence_path"], "w") as f:
        json.dump(obj, f)
    with pytest.raises((ValueError, RuntimeError), match="prompt|hash"):
        C.build_session_metadata(**case)


def test_camera_contract_mismatch_rejected(case):
    state_path = os.path.join(case["checkpoint_dir"], "trainer_state.json")
    state = json.load(open(state_path))
    state["data_contract"]["camera_order"] = ["cam_head"]
    with open(state_path, "w") as f:
        json.dump(state, f)
    with pytest.raises((ValueError, RuntimeError), match="camera|hash"):
        C.build_session_metadata(**case)


def test_bad_safety_bounds_rejected(case):
    obj = json.load(open(case["safety_contract_path"]))
    obj["tasks"]["push_T"]["amin"] = [10.0]*14
    with open(case["safety_contract_path"], "w") as f:
        json.dump(obj, f)
    with pytest.raises((ValueError, RuntimeError), match="safety|limits|bounds|amax"):
        C.build_session_metadata(**case)


def test_history_or_decoder_rejects_unknown_task(case):
    meta, _ = C.build_session_metadata(**case)
    with pytest.raises((ValueError, RuntimeError), match="task"):
        C.M2SessionContract(meta, "")
    session = C.M2SessionContract(meta, "push_T")
    with pytest.raises((ValueError, RuntimeError), match="14D|14|shape"):
        session.observe(np.zeros(42, dtype=np.float32))
