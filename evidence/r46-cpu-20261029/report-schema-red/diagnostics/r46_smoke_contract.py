"""Pure-Python validation for a frozen R46 real-TRAIN sample plan."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "r46-smoke-samples/v1"
SELECTION_SOURCE_SHA = "ba563d0f8c638b089f588027dd0af41ff290e1f4"
CAMERA_ORDER = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
FROZEN_CONTRACT = {
    "model": "OpenVLA-7B",
    "lora": {"r": 32, "alpha": 16, "target_modules": ["q_proj", "v_proj"], "dropout": 0.0},
    "proprio_dim": 42,
    "action_target": "joint_delta_v1",
    "action_dim": 14,
    "stop_token_id": 2,
    "camera_order": CAMERA_ORDER,
    "processor_output_shape": [1, 18, 224, 224],
    "optimizer_steps": 5,
    "roles": ["warmup", "measured", "measured", "measured", "measured"],
}


class SamplePlanError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_plan_sha256(plan: dict) -> str:
    body = {key: value for key, value in plan.items() if key != "sample_plan_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def validate_sample_plan(plan: dict, *, split_doc: dict, manifest: dict,
                         split_sha256: str, manifest_sha256: str,
                         hdf5_root: Path, source_sha: str | None = None,
                         verify_files: bool = True) -> dict:
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise SamplePlanError(f"sample plan schema must be {SCHEMA}")
    if plan.get("sample_plan_sha256") != canonical_plan_sha256(plan):
        raise SamplePlanError("sample plan SHA256 mismatch")
    if source_sha is not None and plan.get("source_sha") != source_sha:
        raise SamplePlanError("sample plan source SHA mismatch")
    tool_sha = plan.get("selection_tool_sha256")
    if not isinstance(tool_sha, str) or len(tool_sha) != 64 \
            or any(ch not in "0123456789abcdef" for ch in tool_sha.lower()):
        raise SamplePlanError("sample selection tool SHA256 is missing or invalid")
    if plan.get("contract") != FROZEN_CONTRACT:
        raise SamplePlanError("sample plan differs from the frozen OpenVLA smoke contract")
    split_map = split_doc.get("splits") or {}
    if set(split_map) != {"push_T", "stack_bowls", "sweep_blocks"}:
        raise SamplePlanError("split must contain exactly the three frozen R44 tasks")
    for split_task, split_row in split_map.items():
        split_train, split_dev = split_row.get("train") or [], split_row.get("dev") or []
        try:
            train_eps = [entry.split("/", 1)[1] for entry in split_train
                         if entry.split("/", 1)[0] == split_task]
            dev_eps = [entry.split("/", 1)[1] for entry in split_dev
                       if entry.split("/", 1)[0] == split_task]
        except (AttributeError, IndexError) as exc:
            raise SamplePlanError(f"invalid qualified split entry for {split_task}") from exc
        if len(train_eps) != len(split_train) or len(dev_eps) != len(split_dev) \
                or len(set(train_eps)) != len(train_eps) or len(set(dev_eps)) != len(dev_eps) \
                or set(train_eps) & set(dev_eps):
            raise SamplePlanError(f"TRAIN/DEV has invalid names, duplicates or overlap for {split_task}")
    if plan.get("split_sha256") != split_sha256:
        raise SamplePlanError("sample plan split SHA does not match the frozen split file")
    task = plan.get("task")
    if task != "push_T" or task not in split_map:
        raise SamplePlanError("smoke samples must use the frozen TRAIN-only push_T task")
    train = set(split_map[task].get("train") or [])
    dev = set(split_map[task].get("dev") or [])
    if train & dev:
        raise SamplePlanError("TRAIN/DEV split overlaps")
    files = manifest.get("files") or {}
    if plan.get("manifest_sha256") != manifest_sha256:
        raise SamplePlanError("sample plan manifest SHA does not match the frozen manifest")
    rows = plan.get("samples")
    roles = FROZEN_CONTRACT["roles"]
    if not isinstance(rows, list) or len(rows) != 5:
        raise SamplePlanError("sample plan must contain exactly five entries")
    identities, episodes = set(), set()
    root = Path(hdf5_root).resolve()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("order_index") != index:
            raise SamplePlanError(f"sample {index}: order_index is invalid")
        if row.get("role") != roles[index]:
            raise SamplePlanError(f"sample {index}: role must be {roles[index]}")
        if row.get("task") != task:
            raise SamplePlanError(f"sample {index}: task differs from the frozen task")
        episode = row.get("episode")
        key = f"{task}/{episode}"
        if key not in train:
            raise SamplePlanError(f"sample {index}: episode is not in TRAIN split")
        if key in dev:
            raise SamplePlanError(f"sample {index}: DEV episode is forbidden")
        t = row.get("t")
        if not isinstance(t, int) or isinstance(t, bool) or t < 2:
            raise SamplePlanError(f"sample {index}: time index must be an integer >=2")
        identity = (task, episode, t)
        if identity in identities or episode in episodes:
            raise SamplePlanError(f"sample {index}: duplicate sample or episode")
        identities.add(identity)
        episodes.add(episode)
        if row.get("camera_order") != CAMERA_ORDER:
            raise SamplePlanError(f"sample {index}: camera order mismatch")
        if row.get("image_shape") != [480, 640, 3] or row.get("image_dtype") != "uint8" \
                or row.get("image_colorspace") != "RGB":
            raise SamplePlanError(f"sample {index}: decoded image contract mismatch")
        if row.get("processor_output_shape") != FROZEN_CONTRACT["processor_output_shape"]:
            raise SamplePlanError(f"sample {index}: production processor output shape mismatch")
        if row.get("proprio_shape") != [1, 42]:
            raise SamplePlanError(f"sample {index}: proprio input must be [1,42]")
        token_ids = row.get("action_token_ids")
        if not isinstance(token_ids, list) or len(token_ids) != 14 \
                or any(not isinstance(token, int) or token < 31744 or token > 31999
                       for token in token_ids):
            raise SamplePlanError(f"sample {index}: expected 14 legal action token IDs")
        if row.get("stop_token_id") != 2 or row.get("prompt_labels_ignored") is not True:
            raise SamplePlanError(f"sample {index}: action/STOP label contract mismatch")
        record = files.get(key)
        digest = row.get("hdf5_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or not record \
                or record.get("sha256") != digest:
            raise SamplePlanError(f"sample {index}: HDF5 SHA does not match frozen manifest")
        path = (root / task / "arx_x5" / "data" / episode).resolve()
        if root not in path.parents or not path.is_file():
            raise SamplePlanError(f"sample {index}: HDF5 path is missing or escapes data root")
        if verify_files and sha256_file(path) != digest:
            raise SamplePlanError(f"sample {index}: actual HDF5 SHA differs from manifest")
        if verify_files:
            try:
                import h5py
                with h5py.File(path, "r") as h5:
                    if t >= int(h5["state/left_arm_joint_states"].shape[0]) - 1:
                        raise SamplePlanError(f"sample {index}: time index is outside the TRAIN episode")
                    if tuple(h5["vision"].keys()) != tuple(CAMERA_ORDER):
                        raise SamplePlanError(f"sample {index}: HDF5 camera order differs from contract")
                    if any(int(h5[f"vision/{camera}/colors"].shape[0]) <= t
                           for camera in CAMERA_ORDER):
                        raise SamplePlanError(f"sample {index}: camera frame index is out of range")
            except (OSError, KeyError) as exc:
                raise SamplePlanError(f"sample {index}: cannot validate HDF5 structure: {exc}") from exc
    return {"sample_count": 5, "roles": roles, "unique_episodes": len(episodes),
            "sample_plan_sha256": plan["sample_plan_sha256"], "status": "CPU_SAMPLE_CONTRACT_PASS"}
