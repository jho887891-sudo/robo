#!/usr/bin/env python3
"""R9 Gate D: read-only survey of candidate push_T evaluation data + acquisition budget.

This tool NEVER downloads, records, copies or deletes anything. It only stats files, reads
HDF5 structure/instruction of a sample, and emits a machine-readable budget so the research
lead can authorise (or refuse) an acquisition route with real numbers.
"""
import hashlib
import json
import os
import sys

BASE_DIM = 14
TARGET_TASK = "Push the T-shaped block"
CANDIDATES = [
    {"id": "local_push_T_100",
     "path": "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data",
     "expected_task": "Push the T-shaped block", "note": "the 100 episodes already used by R2-R9"},
    {"id": "robodojo_demo_arx_x5",
     "path": "/home/T7/ojh/robot_actual/RoboDojo/data/demo/arx_x5/data",
     "expected_task": "Stack the three bowls", "note": "different task, cannot serve as push_T final set"},
    {"id": "openvla_offline_bundle_stage63",
     "path": "/home/T7/ojh/robot_actual/openvla_offline_bundle/stage63_d0/sim_data/push_T",
     "expected_task": "Push the T-shaped block", "note": "declared (600,480,640,3) but 9 KB per file -> fixtures"},
    {"id": "stack_bowls",
     "path": "/home/T7/ojh/robot_actual/data/data/hdf5/stack_bowls/arx_x5/data",
     "expected_task": "Stack the three bowls", "note": "different task on this host"},
    {"id": "sweep_blocks",
     "path": "/home/T7/ojh/robot_actual/data/data/hdf5/sweep_blocks/arx_x5/data",
     "expected_task": "Sweep the blocks", "note": "different task on this host"},
]


def h5_probe(path):
    """Structure probe: which datasets exist and what the instruction says."""
    try:
        import h5py
        with h5py.File(path, "r") as h:
            keys = sorted(h.keys())
            groups = sorted(h.keys())
            has_state = "state/left_arm_joint_states" in h
            has_action = "action/left_arm_joint_states" in h
            ins = None
            if "instruction" in h:
                ins = h["instruction"][()]
                if isinstance(ins, bytes):
                    ins = ins.decode("utf-8", "replace")
                elif hasattr(ins, "shape") and getattr(ins, "shape", ()) != ():
                    first = ins.reshape(-1)[0] if ins.size else b""
                    ins = (first.decode("utf-8", "replace") if isinstance(first, bytes)
                           else str(first))
            frames = int(h["state/left_arm_joint_states"].shape[0]) if has_state else None
            vis = None
            if "vision/cam_head/colors" in h:
                vis = list(h["vision/cam_head/colors"].shape)
            return {"readable": True, "top_level": keys[:6], "groups": groups[:6],
                    "has_state_action": bool(has_state and has_action),
                    "frames": frames, "instruction": ins, "vision_shape": vis}
    except Exception as exc:  # noqa: BLE001
        return {"readable": False, "error": repr(exc)[:200]}


def survey_candidate(cand):
    path = cand["path"]
    out = {"id": cand["id"], "path": path, "note": cand["note"],
           "exists": os.path.isdir(path)}
    if not out["exists"]:
        out["verdict"] = "ABSENT"
        return out
    files = sorted(f for f in os.listdir(path) if f.endswith(".hdf5"))
    sizes = [os.path.getsize(os.path.join(path, f)) for f in files]
    out.update({"n_hdf5": len(files), "total_bytes": int(sum(sizes)),
                "median_bytes": int(sorted(sizes)[len(sizes) // 2]) if sizes else 0,
                "min_bytes": int(min(sizes)) if sizes else 0,
                "max_bytes": int(max(sizes)) if sizes else 0,
                "sample_files": files[:3]})
    probes = [h5_probe(os.path.join(path, f)) for f in files[:3]]
    out["probe"] = probes
    out["all_probe_state_action"] = all(p.get("has_state_action") for p in probes) if probes else False
    out["instructions"] = sorted({p.get("instruction") for p in probes if p.get("instruction")})
    if out["n_hdf5"] == 0:
        out["verdict"] = "EMPTY"
    elif out["median_bytes"] < 100_000:
        out["verdict"] = "PLACEHOLDER_FIXTURES"
    elif not out["all_probe_state_action"]:
        out["verdict"] = "NO_STATE_ACTION"
    else:
        ins0 = (out["instructions"][0].lower() if out["instructions"] else "")
        out["is_target_task"] = TARGET_TASK.lower() in ins0
        out["expected_task_matches"] = (cand["expected_task"].lower() in ins0)
        out["verdict"] = "USABLE_PUSH_T" if out["is_target_task"] else "DIFFERENT_TASK"
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else ""
    cands = [survey_candidate(c) for c in CANDIDATES]
    local = next((c for c in cands if c["id"] == "local_push_T_100"), {})
    per_ep = (local.get("total_bytes", 0) / local["n_hdf5"]) if local.get("n_hdf5") else 0
    options = [
        {"route": "official_increment",
         "description": "download additional official push_T episodes that do not overlap this host",
         "episodes": [20, 128], "estimated_bytes": [int(20 * per_ep), int(128 * per_ep)],
         "requires_authorization": "data download", "executed": False},
        {"route": "new_collection_expert_script",
         "description": "record 20-30 new episodes with the RoboDojo expert script (same task/robot)",
         "episodes": [20, 30], "estimated_bytes": [int(20 * per_ep), int(30 * per_ep)],
         "requires_authorization": "simulation collection", "executed": False},
        {"route": "cross_task_transfer",
         "description": "other RoboDojo tasks already on disk",
         "episodes": [20], "estimated_bytes": [int(20 * per_ep)],
         "requires_authorization": "none, but cannot be a push_T final set", "executed": False},
    ]
    res = {"candidates": cands,
           "bytes_per_episode_estimate": int(per_ep),
           "gb_per_episode_estimate": round(per_ep / 1e9, 3),
           "acquisition_options": options,
           "unused_local_push_T_episodes": 0,
           "blocker": "BLOCKED_DATA_ACQUISITION_AUTHORIZATION",
           "policy": "no download, no collection, no FINAL-set evaluation in this round"}
    print(json.dumps({k: v for k, v in res.items() if k != "candidates"}, indent=1))
    for c in cands:
        print("%-32s %-20s n=%s median=%s instruction=%r" % (
            c["id"], c.get("verdict"), c.get("n_hdf5"), c.get("median_bytes"),
            (c.get("instructions") or [None])[0]))
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=1)
        print("->", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
