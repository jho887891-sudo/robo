#!/usr/bin/env python3
"""R10.2 D4: build TASK_SCOPE_V1 for the 12 generalisation tasks from THIS machine.

Read-only: parses the local task inventory and task YAMLs; never downloads, never infers
compatibility from a name. Anything not provable from local files is marked UNKNOWN.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

TASKS12 = ("stack_bowls", "push_T", "pack_objects_into_box", "fold_clothes", "hang_mugs",
           "sweep_blocks", "pour_liquid_into_cup", "make_toast", "arrange_largest_number",
           "sort_nesting_dolls_by_size", "store_laptop_and_headphones", "stack_blocks")


def inventory(repo):
    p = subprocess.run([sys.executable, "scripts/internal/task_inventory.py", "--format", "json"],
                       cwd=repo, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("task_inventory failed: %s" % p.stderr[-300:])
    return json.loads(p.stdout)


def scan_config(path):
    out = {"instruction": "UNKNOWN", "cameras": "UNKNOWN", "frequency_hz": "UNKNOWN",
           "robot": "UNKNOWN", "action_dim": "UNKNOWN", "config_sha256": "UNKNOWN"}
    if not path or not os.path.isfile(path):
        out["exists"] = False
        return out
    raw = open(path, "rb").read()
    text = raw.decode("utf-8", "replace")
    out["exists"] = True
    out["config_sha256"] = hashlib.sha256(raw).hexdigest()
    m = re.search(r"^\s*(?:instruction|prompt|task_description)\s*:\s*(.+?)\s*$", text, re.M)
    if m:
        out["instruction"] = m.group(1).strip().strip("\"''")
    m = re.search(r"^\s*frequency\s*:\s*([0-9]+)", text, re.M)
    if m:
        out["frequency_hz"] = int(m.group(1))
    cams = re.findall(r"^\s*(cam_[a-z_]+)\s*:", text, re.M)
    if cams:
        out["cameras"] = sorted(set(cams))
    m = re.search(r"^\s*(?:robot|robot_name|embodiment)\s*:\s*(\S+)", text, re.M)
    if m:
        out["robot"] = m.group(1).strip("\"'")
    m = re.search(r"^\s*(?:action_dim|dof|num_joints)\s*:\s*([0-9]+)", text, re.M)
    if m:
        out["action_dim"] = int(m.group(1))
    return out


def build(repo, data_root):
    inv = inventory(repo)
    by_name = {t["name"]: t for t in inv["tasks"]}
    rows = []
    for name in TASKS12:
        t = by_name.get(name)
        row = {"task_id": name, "in_local_inventory": bool(t),
               "dimension": t.get("dimension") if t else "UNKNOWN",
               "runnable": t.get("runnable") if t else False,
               "config": t.get("config") if t else None,
               "module": t.get("module") if t else None,
               "variant": t.get("variant") if t else None}
        cfg_path = os.path.join(repo, t["config"]) if (t and t.get("config")) else ""
        row.update(scan_config(cfg_path))
        local_dir = os.path.join(data_root, name, "arx_x5", "data")
        n_eps = len([f for f in os.listdir(local_dir) if f.endswith(".hdf5")]) \
            if os.path.isdir(local_dir) else 0
        row["local_episodes"] = n_eps
        row["data_status"] = "LOCAL_AVAILABLE" if n_eps else "UNKNOWN_NO_LOCAL_DATA"
        if n_eps:
            # Prefer facts from the REAL local episodes over guesses from config text.
            try:
                import h5py
                first = sorted(f for f in os.listdir(local_dir) if f.endswith(".hdf5"))[0]
                with h5py.File(os.path.join(local_dir, first), "r") as h:
                    ins = h["instruction"][()] if "instruction" in h else None
                    if isinstance(ins, bytes):
                        ins = ins.decode("utf-8", "replace")
                    elif hasattr(ins, "reshape") and getattr(ins, "size", 1) > 1:
                        ins = str(ins.reshape(-1)[0])
                    row["instruction"] = str(ins) if ins is not None else "UNKNOWN"
                    row["instruction_source"] = "local_hdf5"
                    if "additional_info/frequency" in h:
                        row["frequency_hz"] = int(h["additional_info/frequency"][()])
                    cams = sorted(h["vision"].keys()) if "vision" in h else []
                    if cams:
                        row["cameras"] = cams
                    row["action_dim"] = int(h["state/left_arm_joint_states"].shape[1]
                                            + h["state/left_ee_joint_states"].shape[1]
                                            + h["state/right_arm_joint_states"].shape[1]
                                            + h["state/right_ee_joint_states"].shape[1])
            except Exception as exc:  # noqa: BLE001
                row["local_data_probe_error"] = repr(exc)[:160]
        row["embodiment_adapter_needed"] = "UNKNOWN"
        rows.append(row)
    present = [r["task_id"] for r in rows if r["in_local_inventory"]]
    notes = [
        "task names are not evidence of identical dims/rate/cameras: UNKNOWN is kept where the local files do not state a value",
        "the *_random OOD variants are separate tasks and are excluded from this scope",
        "the upstream 42-task benchmark is not this project 12-task scope",
    ]
    return {"scope_version": "TASK_SCOPE_V1", "benchmark": inv.get("benchmark"),
            "inventory_counts": inv.get("counts"), "n_scope_tasks": len(TASKS12),
            "n_present_locally": len(present), "present": present,
            "absent": [t for t in TASKS12 if t not in present],
            "tasks": rows, "notes": notes}


def route(registry, task_id):
    for row in registry["tasks"]:
        if row["task_id"] == task_id:
            if row["instruction"] == "UNKNOWN":
                raise KeyError("instruction for %r is UNKNOWN locally; refusing a hardcoded prompt"
                               % task_id)
            return row["instruction"]
    raise KeyError("task %r is not in TASK_SCOPE_V1" % task_id)


def assert_stats_source(split_name):
    if str(split_name).upper() != "TRAIN":
        raise ValueError("normalisation/quantile statistics must come from TRAIN, got %r"
                         % (split_name,))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--data-root", default="/home/T7/ojh/robot_actual/data/data/hdf5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reg = build(os.path.abspath(args.repo), args.data_root)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(reg, open(args.out, "w"), indent=1)
    print("scope tasks=%d present=%d absent=%d" % (reg["n_scope_tasks"], reg["n_present_locally"],
                                                   len(reg["absent"])))
    for r in reg["tasks"]:
        print("%-30s %-14s instr=%-6s cams=%-6s hz=%-5s eps=%d" % (
            r["task_id"], r["dimension"], str(r["instruction"])[:6], str(r["cameras"])[:6],
            str(r["frequency_hz"]), r["local_episodes"]))
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
