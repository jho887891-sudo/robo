#!/usr/bin/env python3
"""R27 read-only, stdlib-only remote M2 checkpoint evidence collector.

Designed so Codex/DeepSeek can execute ONE prescribed command on the
checkpoint-hosting machine. There are ZERO optimiser steps, CUDA imports,
HDF5 loads, file modifications under the checkpoint or model predictions.

Critically, sampler indices do NOT identify gradient-exposed frames without
the exact frozen slot map: this tool never invents a TRAIN-SEEN count. Use
tools/r24_seen_exposure_audit.py for the separate HDF5-backed mapping gate.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys


SCHEMA = "r27-m2-readonly-checkpoint-inventory-v1"
TRAINER_SCHEMA = "r15-full-state-v1"
STAGE_SCHEMA = "m2-train-evidence-v1"
MANDATORY_FILES = (
    "trainer_state.json", "sampler.json", "optimizer_state.pt",
    "scheduler_state.pt", "rng.pt", "proprio_projector_42.pt"
)


class InventoryError(ValueError):
    """Unverifiable source or unsafe output path; fail closed."""


def is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def require(condition, message):
    if not condition:
        raise InventoryError(message)


def load_json(path):
    path = pathlib.Path(path)
    require(path.is_file(), "required JSON file missing: %s" % path)
    try:
        with path.open(encoding="utf-8") as f:
            item = json.load(f)
    except (OSError, ValueError, UnicodeError) as exc:
        raise InventoryError("invalid JSON %s: %s" % (path, exc))
    require(isinstance(item, dict), "JSON root must be object: %s" % path)
    return item


def hash_file(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_relative_file(root, rel):
    """Neither external symlinks nor absolute/traversal entries are acceptable."""
    require(isinstance(rel, str) and rel != "", "manifest path must be nonempty relative str")
    pieces = rel.replace("\\", "/").split("/")
    require(not os.path.isabs(rel) and all(part not in ("", ".", "..") for part in pieces),
            "manifest unsafe relative path %r" % rel)
    root = pathlib.Path(root).resolve()
    path = root.joinpath(*pieces)
    require(os.path.commonpath((str(root), str(path.resolve()))) == str(root),
            "manifest path escapes checkpoint tree: %r" % rel)
    require(path.is_file(), "checkpoint manifest member missing: %s" % rel)
    return path


def checked_manifest_and_state(checkpoint_dir):
    ckpt = pathlib.Path(checkpoint_dir).resolve()
    require(ckpt.is_dir(), "checkpoint directory missing: %s" % ckpt)
    manpath = ckpt / "MANIFEST.json"
    manifest = load_json(manpath)
    require(manifest.get("schema") == TRAINER_SCHEMA, "MANIFEST checkpoint schema mismatch")
    files = manifest.get("files")
    require(isinstance(files, dict) and len(files) >= len(MANDATORY_FILES),
            "MANIFEST missing files map")
    require(all(x in files for x in MANDATORY_FILES),
            "checkpoint MANIFEST missing mandatory full-state files")
    require(any(x.startswith("lora_adapter/") and x.endswith(".safetensors")
                for x in files),
            "checkpoint MANIFEST missing a LoRA safetensors file")
    sizes = {}
    for rel, expected in sorted(files.items()):
        require(is_sha256(expected), "invalid manifest SHA256 for %r" % rel)
        path = safe_relative_file(ckpt, rel)
        actual = hash_file(path)
        require(actual == expected,
                "checkpoint artifact hash mismatch for %r: %s != %s"
                % (rel, actual, expected))
        sizes[rel] = path.stat().st_size
    state = load_json(ckpt / "trainer_state.json")
    require(state.get("checkpoint_schema") == TRAINER_SCHEMA,
            "trainer checkpoint schema mismatch")
    require(state.get("model_kind") == "peft",
            "M2 evidence requires PEFT full-state checkpoint")
    require(type(state.get("global_step")) is int and state["global_step"] > 0,
            "trainer invalid global_step")
    require(manifest.get("global_step") == state["global_step"],
            "checkpoint manifest vs trainer global_step mismatch")
    require(manifest.get("run_id") == state.get("run_id"),
            "checkpoint manifest vs trainer run_id mismatch")
    require(state.get("tasks") == ["push_T", "stack_bowls"],
            "only frozen M2 2-task checkpoint is allowed")
    require(state.get("task_ids") == {"push_T": 0, "stack_bowls": 1},
            "checkpoint task ID assignment drift")
    dc = state.get("data_contract")
    require(isinstance(dc, dict), "checkpoint data_contract missing")
    require(dc.get("tasks") == state["tasks"] and dc.get("task_ids") == state["task_ids"],
            "data_contract task mapping mismatch")
    require(dc.get("normalizer_sha256") == state.get("normalizer_sha256"),
            "trainer vs data_contract normalizer identity mismatch")
    require(is_sha256(dc.get("normalizer_sha256")),
            "data_contract normalizer SHA256 missing")
    require(is_sha256(dc.get("split_sha256")),
            "data_contract split SHA256 missing")
    return ckpt, manifest, state, sizes, hash_file(manpath)


def checked_sampler(ckpt, expected_step):
    sampler = load_json(ckpt / "sampler.json")
    n = sampler.get("n_frames")
    pos = sampler.get("position")
    order = sampler.get("perm")
    require(type(n) is int and n > 0, "sampler n_frames invalid")
    require(type(pos) is int and pos == expected_step,
            "sampler position/global_step mismatch")
    require(pos <= n, "sampler position greater than available slot pool")
    require(isinstance(order, list) and len(order) == n,
            "sampler permutation length mismatch")
    require(all(type(i) is int and 0 <= i < n for i in order),
            "sampler permutation contains invalid slot index")
    require(len(set(order)) == n, "sampler permutation contains duplicate index")
    require(is_sha256(sampler.get("frames_sha256"))
            and is_sha256(sampler.get("order_hash")),
            "sampler frozen frame/order digests missing")
    consumed = order[:pos]
    return {
        "position": pos,
        "frozen_sampler_slot_pool_size": n,
        "consumed_schedule_slots": len(consumed),
        "distinct_consumed_slot_indices": len(set(consumed)),
        "frames_sha256": sampler["frames_sha256"],
        "claimed_order_hash": sampler["order_hash"],
        "order_hash_source_recomputed": False,
        "actual_unique_gradient_exposed_frames":
            "UNKNOWN_REQUIRES_FROZEN_FRAME_MAPPING",
        "mapping_note": "Slot indices can include repeated frames. The original "
                        "TRAIN frame mapping requires HDF5 index/split/normalizer "
                        "and tools/r24_seen_exposure_audit.py.",
    }


def checked_stage(stage_path, state):
    if stage_path is None:
        return {"status": "BLOCKED_NO_STAGE_EVIDENCE"}, False
    stage = load_json(stage_path)
    require(stage.get("schema") == STAGE_SCHEMA,
            "stage evidence schema mismatch")
    for key in ("run_id", "tasks", "task_ids", "normalizer_sha256"):
        require(stage.get(key) == state.get(key), "stage evidence %s mismatch" % key)
    stage_step = stage.get("global_step_reached")
    require(stage_step == state["global_step"],
            "stage evidence global_step_reached mismatch")
    digests = stage.get("prompt_token_ids_sha256")
    tasks = state["tasks"]
    require(isinstance(digests, dict) and sorted(digests) == sorted(tasks),
            "stage prompt token SHA256 evidence missing or has wrong task coverage")
    require(all(is_sha256(digests[t]) for t in tasks),
            "stage prompt token SHA256 digest malformed")
    # No raw token IDs, personal keys or dataset contents emitted.
    return {
        "status": "SOURCE_FIELDS_MATCHED",
        "sha256": hash_file(stage_path),
        "global_step_reached": stage_step,
        "prompt_token_ids_sha256": {t: digests[t] for t in tasks},
        "note": "Crosschecked stage fields, NOT authenticity of the training "
                "execution against a remote GPU session or immutable WAL.",
    }, True


def inspect(checkpoint_dir, stage_evidence_path=None):
    ckpt, manifest, state, sizes, msh = checked_manifest_and_state(checkpoint_dir)
    sampler = checked_sampler(ckpt, state["global_step"])
    stage, stage_verified = checked_stage(stage_evidence_path, state)
    return {
        "schema": SCHEMA,
        "checkpoint": {
            "source_directory": str(ckpt),
            "run_id": state["run_id"],
            "global_step": state["global_step"],
            "tasks": state["tasks"],
            "task_ids": state["task_ids"],
            "model_kind": state["model_kind"],
            "manifest_sha256": msh,
            "files_verified": len(sizes),
            "file_sizes_bytes": sizes,
            "normalizer_sha256": state["data_contract"]["normalizer_sha256"],
            "split_sha256": state["data_contract"]["split_sha256"],
        },
        "sampler": sampler,
        "stage": stage,
        "attested": {
            "manifest_file_hashes_verified": True,
            "sampler_permutation_structure_verified": True,
            "sampler_source_order_hash_recomputed": False,
            "stage_crosscheck_verified": stage_verified,
            "actual_train_frame_identities_verified": False,
            "real_7b_forward_executed": False,
            "official_closed_loop_executed": False,
        },
        "optimizer_steps_executed_by_this_tool": 0,
        "interpretation": "Read-only CPU identity check. Does not establish "
                          "actual TRAIN-SEEN frame count or model closed-loop success.",
    }


def write_result_exclusive(out, result, checkpoint_dir):
    ckpt = pathlib.Path(checkpoint_dir).resolve()
    out = pathlib.Path(out).resolve()
    require(os.path.commonpath((str(ckpt), str(out))) != str(ckpt),
            "output inside checkpoint is prohibited")
    require(not out.exists(), "output already exists, refusing overwrite: %s" % out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 'x' protects against races with an existing file.
    try:
        with out.open("x", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except FileExistsError:
        raise InventoryError("output exists, refusing overwrite: %s" % out)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stage-evidence", default=None,
                   help="Stage evidence JSON; omit to explicitly mark stage BLOCKED")
    p.add_argument("--out", required=True,
                   help="New JSON location OUTSIDE checkpoint tree; never overwritten")
    args = p.parse_args(argv)
    try:
        report = inspect(args.checkpoint, args.stage_evidence)
        out = write_result_exclusive(args.out, report, args.checkpoint)
    except (InventoryError, OSError) as exc:
        print("R27_INVENTORY_BLOCKED: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "VERIFIED_FULL_STATE_CPU_INVENTORY",
        "artifact": str(out),
        "run_id": report["checkpoint"]["run_id"],
        "global_step": report["checkpoint"]["global_step"],
        "stage_status": report["stage"]["status"],
        "actual_unique_seen_frames": report["sampler"][
            "actual_unique_gradient_exposed_frames"],
        "optimizer_steps_executed": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
