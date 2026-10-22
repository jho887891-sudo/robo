#!/usr/bin/env python3
"""R25 M2 D0-A: STRICT CPU-only multi-task serving preflight.

This is not a 7B inference server and not a permission to deploy.
It reconstructs ONE task-bound session's metadata from a sealed M2
full-state checkpoint, TRAIN-only two-task normalizer, frozen dataset
split, training-stage prompt evidence and an external authoritative robot
safety contract.  Missing or inconsistent evidence always raises.

The frozen legacy R15 metadata bridge and server are never modified.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
for _sub in ("training", "code/xpolicylab/policy/OpenVLA_Pure",
             "code/xpolicylab/policy/OpenVLA_R42", "diagnostics"):
    _p = os.path.join(_REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import m2_train_shared as MT  # noqa: E402
import m2_task_normalizer as TN  # noqa: E402
import r15_resume as RS  # noqa: E402
import r42_adapter as R  # noqa: E402


SAFETY_SCHEMA = "r25-m2-robot-safety-v1"
EXPECTED_TASKS = ("push_T", "stack_bowls")


class M2DeployContractError(ValueError):
    """A missing or unproven multi-task serving contract (fail-closed)."""


def file_sha256(path):
    if not os.path.isfile(path):
        raise M2DeployContractError("missing required file: %s" % path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checked_json(path):
    file_sha256(path)
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh)
    except (ValueError, UnicodeError) as exc:
        raise M2DeployContractError("invalid JSON %s: %s" % (path, exc))
    if not isinstance(value, dict):
        raise M2DeployContractError("JSON root must be a dict: %s" % path)
    return value


def _eq(name, got, expected):
    if got != expected:
        raise M2DeployContractError(
            "%s mismatch: actual %r != frozen %r" % (name, got, expected))


def _sha(name, value):
    if not isinstance(value, str) or len(value) != 64 or any(
            c not in "0123456789abcdef" for c in value.lower()):
        raise M2DeployContractError("%s must contain sha256, got %r" % (name, value))
    return value


def _task_ids(tasks, payload):
    frozen = {t: TN.M2.TASK_IDS[t] for t in tasks}
    _eq("payload.task_id", payload.get("task_id"), frozen)
    return frozen


def verify_safety_contract(source_path, payload, tasks):
    """The external robot-limit *document* is validated, not self-certified as hardware truth.

    R24's TRAIN-derived pmin/pmax are NOT certified robot physical limits.
    Even a syntactically valid safety document needs separate human/robotic
    provenance review before authorizing control or deployment.
    """
    safety = checked_json(source_path)
    _eq("safety.schema", safety.get("schema"), SAFETY_SCHEMA)
    _eq("safety.source_kind", safety.get("source_kind"), "robot_joint_hard_limits")
    attested_by = safety.get("verified_by")
    if not isinstance(attested_by, str) or not attested_by.strip():
        raise M2DeployContractError("safety contract needs a named independent verified_by")
    choices = safety.get("tasks")
    if not isinstance(choices, dict):
        raise M2DeployContractError("safety.tasks must be a per-task mapping")
    _eq("safety.task_set", sorted(choices), sorted(tasks))
    result = {}
    for task in tasks:
        limits = choices[task]
        if not isinstance(limits, dict):
            raise M2DeployContractError("safety.%s must be a dict" % task)
        arrays = {}
        for name in ("amin", "amax", "max_rate"):
            if name not in limits:
                raise M2DeployContractError("safety.%s missing %s" % (task, name))
            try:
                arr = np.asarray(limits[name], dtype=np.float32)
            except (ValueError, TypeError) as exc:
                raise M2DeployContractError("safety.%s.%s invalid: %s" % (task, name, exc))
            if arr.shape != (14,) or not np.isfinite(arr).all():
                raise M2DeployContractError("safety.%s.%s must be finite 14D" % (task, name))
            arrays[name] = arr
        if (arrays["amax"] <= arrays["amin"]).any():
            raise M2DeployContractError("safety limits amax <= amin for task %s" % task)
        if (arrays["max_rate"] < 0).any():
            raise M2DeployContractError("safety max_rate must be non-negative for task %s" % task)
        block = payload["tasks"][task]
        if (np.allclose(arrays["amin"], block["pmin"], rtol=0, atol=1e-8)
                and np.allclose(arrays["amax"], block["pmax"], rtol=0, atol=1e-8)):
            raise M2DeployContractError(
                "safety limits for %s repeat training pmin/pmax; require independently "
                "validated robot joint limits, not training quantile bounds" % task)
        result[task] = {name: arr.tolist() for name, arr in arrays.items()}
    return result, file_sha256(source_path)


def verify_checkpoint(checkpoint_dir):
    """Verify every manifest-declared file hash without importing torch or CUDA."""
    manifest_path = os.path.join(checkpoint_dir, "MANIFEST.json")
    manifest = checked_json(manifest_path)
    _eq("checkpoint manifest schema", manifest.get("schema"), RS.CHECKPOINT_SCHEMA)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise M2DeployContractError("checkpoint manifest lacks declared files")
    for rel in RS.REQUIRED_FILES[1:] + ("lora_adapter/adapter_model.safetensors",):
        if rel not in files:
            raise M2DeployContractError("checkpoint missing mandatory manifest hash for %s" % rel)
    for rel, want in files.items():
        if os.path.isabs(rel) or ".." in rel.replace("\\", "/").split("/"):
            raise M2DeployContractError("unsafe checkpoint manifest path %r" % rel)
        _sha("checkpoint MANIFEST %s" % rel, want)
        actual = file_sha256(os.path.join(checkpoint_dir, rel))
        _eq("checkpoint artifact hash %s" % rel, actual, want)
    state = checked_json(os.path.join(checkpoint_dir, "trainer_state.json"))
    _eq("checkpoint state schema", state.get("checkpoint_schema"), RS.CHECKPOINT_SCHEMA)
    _eq("manifest vs trainer global step", state.get("global_step"), manifest.get("global_step"))
    if int(state["global_step"]) < 1:
        raise M2DeployContractError("checkpoint step must be positive")
    _eq("model_kind", state.get("model_kind"), "peft")
    _eq("optimizer_format_version", state.get("optimizer_format_version"), RS.OPTIMIZER_FORMAT_VERSION)
    return manifest, state, file_sha256(manifest_path)


def verify_train_identity(trainer, stage, payload, splits_path):
    tasks = sorted(payload["tasks"], key=lambda t: TN.M2.TASK_IDS[t])
    if len(tasks) != 2 or tasks != list(EXPECTED_TASKS):
        raise M2DeployContractError(
            "M2 D0-A scoped to the two original tasks, got %s" % tasks)
    ids = _task_ids(tasks, payload)
    _eq("trainer tasks", trainer.get("tasks"), tasks)
    _eq("trainer task_ids", trainer.get("task_ids"), ids)
    _eq("trainer normalizer sha", trainer.get("normalizer_sha256"),
        TN.normalizer_sha256(payload))
    dc = trainer.get("data_contract")
    if not isinstance(dc, dict):
        raise M2DeployContractError("trainer data_contract missing")
    frozen = {
        "schema": MT.SCHEMA_EVIDENCE,
        "tasks": tasks,
        "task_ids": ids,
        "normalizer_sha256": TN.normalizer_sha256(payload),
        "per_task_normalizer_sha256": {
            t: TN.sha256_json(payload["tasks"][t]) for t in tasks},
        "split_sha256": file_sha256(splits_path),
        "camera_order": list(R.CAMERA_ORDER),
        "gripper_dims": list(R.GRIPPER_DIMS),
        "base_dim": 14,
        "n42": 42,
        "alignment_version": R.ALIGNMENT_VERSION,
        "target_representation": R.TARGET,
        "action_tokens": R.N_ACTIONS + 1,
    }
    for name, want in frozen.items():
        _eq("trainer data_contract.%s" % name, dc.get(name), want)
    _eq("trainer.per_task_normalizer_sha256",
        trainer.get("per_task_normalizer_sha256"),
        frozen["per_task_normalizer_sha256"])
    mix = trainer.get("task_mix")
    if not isinstance(mix, dict):
        raise M2DeployContractError("trainer missing frozen task mix")
    _sha("trainer mix.schedule_sha256", mix.get("schedule_sha256"))
    _eq("trainer schedule vs data_contract",
        mix["schedule_sha256"], dc.get("task_mix_schedule_sha256"))

    _eq("stage evidence schema", stage.get("schema"), MT.SCHEMA_EVIDENCE)
    _eq("stage evidence global step", stage.get("global_step_reached"),
        trainer["global_step"])
    _eq("stage evidence run_id", stage.get("run_id"), trainer.get("run_id"))
    _eq("stage evidence tasks", stage.get("tasks"), tasks)
    _eq("stage evidence task_ids", stage.get("task_ids"), ids)
    _eq("stage evidence normalizer", stage.get("normalizer_sha256"),
        TN.normalizer_sha256(payload))
    _eq("stage evidence split", stage.get("split_sha256"),
        frozen["split_sha256"])
    if not isinstance(stage.get("include_task_token"), bool):
        raise M2DeployContractError("stage evidence include_task_token bool is required")
    trained_prompts = MT.prompts_for(
        tasks, include_task_token=stage["include_task_token"])
    _eq("stage evidence prompts", stage.get("prompts"), trained_prompts)
    prompt_digests = {t: hashlib.sha256(trained_prompts[t].encode()).hexdigest()
                      for t in tasks}
    _eq("stage evidence prompt_sha256", stage.get("prompt_sha256"), prompt_digests)
    if not isinstance(stage.get("seed"), int):
        raise M2DeployContractError("stage evidence requires exact seed")
    base = trainer.get("base_model")
    if not isinstance(base, dict):
        raise M2DeployContractError("trainer base_model attestation missing")
    _sha("base_model.config.json", base.get("config.json"))
    return tasks, frozen, trained_prompts


def build_session_metadata(checkpoint_dir, task, normalizer_path, splits_path,
                           stage_evidence_path, safety_contract_path):
    """Return (M2 session metadata, provenance). Does NOT start a robot or model."""
    if not isinstance(task, str) or task not in EXPECTED_TASKS:
        raise M2DeployContractError("task must be an explicit valid two-task name")
    manifest, trainer, manifest_sha = verify_checkpoint(checkpoint_dir)
    payload = checked_json(normalizer_path)
    try:
        TN.assert_payload_complete(payload, tasks=list(EXPECTED_TASKS))
    except (ValueError, KeyError, TypeError) as exc:
        raise M2DeployContractError("normalizer payload invalid: %s" % exc)
    stage = checked_json(stage_evidence_path)
    tasks, contract, prompts = verify_train_identity(
        trainer, stage, payload, splits_path)
    if task not in tasks:
        raise M2DeployContractError("task is not served by checkpoint")
    limits_by_task, safety_sha = verify_safety_contract(
        safety_contract_path, payload, tasks)

    freeze_sha = hashlib.sha256(R.canonical_json(
        trainer["data_contract"]).encode("utf-8")).hexdigest()
    limits = limits_by_task[task]
    meta = R.build_metadata(
        task_id=task,
        base_model_sha256=trainer["base_model"]["config.json"],
        lora_sha256=manifest["files"]["lora_adapter/adapter_model.safetensors"],
        projector_sha256=manifest["files"]["proprio_projector_42.pt"],
        frozen_contract_sha256=freeze_sha,
        split_sha256=contract["split_sha256"],
        train_steps=int(trainer["global_step"]), seed=int(stage["seed"]),
        gripper_policy={"gripper_dims": list(R.GRIPPER_DIMS),
                        "fallback": "S+V", "mode": "r42-m2-frozen"},
        safety_limits=limits,
        normalizer=payload,
        prompt_source="m2_train_shared._make_prompt",
        codec_mode="fixed")
    meta.update({
        "serving_task": task,
        "trained_prompt": prompts[task],
        "include_task_token": stage["include_task_token"],
        "tokenizer_prompt_parity": "GPU_SMOKE_REQUIRED",
        "m2_checkpoint_manifest_sha256": manifest_sha,
        "m2_training_evidence_sha256": file_sha256(stage_evidence_path),
        "m2_data_contract_sha256": freeze_sha,
        "m2_safety_contract_sha256": safety_sha,
        "safety_contract_sha256": safety_sha,
        "m2_strict_task_binding": True,
    })
    try:
        R.validate_metadata(meta, expect_task=task)
    except (ValueError, KeyError, TypeError) as exc:
        raise M2DeployContractError("invalid derived R42 metadata: %s" % exc)
    report = {
        "schema": "r25-m2-session-attestation-v1",
        "checkpoint_global_step": int(trainer["global_step"]),
        "checkpoint_manifest_sha256": manifest_sha,
        "trainer_data_contract_sha256": freeze_sha,
        "normalizer_sha256": TN.normalizer_sha256(payload),
        "normalizer_file_sha256": file_sha256(normalizer_path),
        "split_file_sha256": contract["split_sha256"],
        "training_evidence_sha256": file_sha256(stage_evidence_path),
        "safety_contract_sha256": safety_sha,
        "serving_task": task,
        "other_tasks_preserved": [t for t in tasks if t != task],
        "no_optimizer_steps": True,
        "device": "CPU_PREFLIGHT",
        "server_status": "BLOCKED_PENDING_TRAINED_TOKEN_PARITY_AND_REAL_7B_GPU_SMOKE",
        "hardware_safety_verification": "REQUIRES_INDEPENDENT_REVIEW",
        "source_checkpoint": os.path.abspath(checkpoint_dir),
    }
    return meta, report


class M2SessionContract:
    """Task-frozen, CPU-only 42D feature + 14D action decoder.

    This class makes it impossible to silently swap to another task or
    use a single-task R15 prompt. A production model wrapper MUST preserve
    this task binding and verify tokenizer prompt ids on the GPU server.
    """

    def __init__(self, metadata, task):
        if not isinstance(task, str) or not task:
            raise M2DeployContractError("task is required, never inferred from a prompt")
        try:
            R.validate_metadata(metadata, expect_task=task)
        except (ValueError, KeyError, TypeError) as exc:
            raise M2DeployContractError("task metadata invalid: %s" % exc)
        if metadata.get("task_id") != task or metadata.get("serving_task") != task:
            raise M2DeployContractError("serving task mismatch in metadata")
        if not R.is_task_normalizer(metadata["normalizer"]):
            raise M2DeployContractError("task-specific M2 session requires multi-task payload")
        if metadata.get("m2_strict_task_binding") is not True:
            raise M2DeployContractError("strict task binding absent")
        if not isinstance(metadata.get("trained_prompt"), str) or not metadata["trained_prompt"]:
            raise M2DeployContractError("trained prompt missing")
        self.metadata = metadata
        self.task = task
        self.expected_prompt = metadata["trained_prompt"]
        self.normalizer = R.normalizer_for_task(metadata["normalizer"], task)
        from r8_normalization import NormalizationContract
        self.history = R.EpisodeHistory(
            NormalizationContract.from_metadata(self.normalizer))

    def reset(self, env_idx=0):
        self.history.reset(int(env_idx))

    def observe(self, state, timestamp=None, env_idx=0):
        return self.history.observe(int(env_idx), state, timestamp=timestamp)

    def velocity(self, env_idx=0):
        return self.history.velocity(int(env_idx))

    def check_images(self, images):
        return R.validate_images(images)

    def decode(self, bins, state, velocity, prev_cmd=None, dt=None, on_unsafe="raise"):
        return R.decode_with_metadata(
            self.metadata, bins, state, velocity, prev_cmd=prev_cmd, dt=dt,
            on_unsafe=on_unsafe, task=self.task)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint-dir", "task", "normalizer-path", "splits-path",
                 "stage-evidence-path", "safety-contract-path", "out"):
        ap.add_argument("--" + flag, required=True)
    a = ap.parse_args()
    meta, report = build_session_metadata(
        a.checkpoint_dir, a.task, a.normalizer_path,
        a.splits_path, a.stage_evidence_path, a.safety_contract_path)
    if os.path.exists(a.out):
        raise SystemExit("refusing to overwrite existing metadata bundle: %s" % a.out)
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump({"metadata": meta, "attestation": report}, fh, indent=2,
                  sort_keys=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
