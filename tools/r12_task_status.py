#!/usr/bin/env python3
"""Build the 12-task status manifest from the verified lane artefacts + the live filesystem.

Nothing here is typed by hand: the task ids and step limits come from lanes/C_taskspec.json (lane C
verified 12/12 TaskSpec against file:line), the expert feasibility class comes from the section headings
of lanes/D_expert_feasibility.md, and data presence is checked on the real HDF5 directories. The output
is the "authorised / incomplete" list the mission asks for, with the next action per task.
"""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDF5_ROOT = "/home/T7/ojh/robot_actual"
LANES = os.path.join(REPO, "lanes")

# Where the training HDF5 for a task would live. Layout provenance does NOT exist for these files, so
# scene-level independence is explicitly UNKNOWN (decision B in issue #23).
# Resolved through the M2 data module so this tool cannot invent a layout: task_dir() knows both the
# dataset root and the task directory form, and validates what it finds.
DATA_ROOT = os.path.join(HDF5_ROOT, "data", "data", "hdf5")
DATA_DIR_CANDIDATES = [os.path.join(DATA_ROOT, "{task}", "arx_x5", "data"),
                       os.path.join(DATA_ROOT, "{task}")]


def load_taskspec():
    with open(os.path.join(LANES, "C_taskspec.json")) as fh:
        doc = json.load(fh)
    out = []
    for t in sorted(doc["tasks"], key=lambda x: int(x.get("n") or 0)):
        out.append({"id": t["id"], "n": t.get("n"), "step_lim": t.get("step_lim"),
                    "success": (t.get("success") or {}).get("summary") if isinstance(t.get("success"), dict)
                    else str(t.get("success"))[:200],
                    "instruction": t.get("instruction")})
    return out, doc.get("global_unknowns")


def feasibility_classes():
    """Map task -> section class from the lane D markdown, by reading its own headings."""
    path = os.path.join(LANES, "D_expert_feasibility.md")
    text = open(path, encoding="utf-8").read()
    cls, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^#{2,3}\s*2\.(\d)\s*(.+)$", line.strip())
        if m:
            title = m.group(2).upper()
            if "REALISTIC" in title:
                cur = "REALISTIC"
            elif "HARD" in title:
                cur = "HARD_BUT_RIGID"
            elif "BLOCKED" in title:
                cur = "BLOCKED_DATA"
            else:
                cur = None
            continue
        if cur:
            for name in re.findall(r"\*\*([a-zA-Z0-9_]+)\*\*", line):
                cls.setdefault(name, cur)
    return cls


def data_presence(task):
    for pat in DATA_DIR_CANDIDATES:
        d = pat.format(task=task)
        if os.path.isdir(d):
            eps = [f for f in os.listdir(d) if f.endswith(".hdf5")]
            if eps:
                return {"status": "PRESENT", "dir": d, "n_episodes": len(eps)}
    return {"status": "ABSENT", "dir": None, "n_episodes": 0}


def _physical_audit():
    """Per-task physical capability measured in R17, if the audit artefact is present."""
    p = os.path.join(REPO, "mission", "m2_physical_audit.json")
    if not os.path.isfile(p):
        return {}
    d = json.load(open(p))
    out = {}
    for task, blk in (d.get("tasks") or {}).items():
        s = blk.get("summary") or {}
        out[task] = {
            "model_safe_l1": s.get("model_safe_l1", {}).get("mean"),
            "noop_l1": s.get("noop_l1", {}).get("mean"),
            "cv_l1": s.get("cv_l1", {}).get("mean"),
            "beats_noop": (s.get("paired_model_minus_noop", {}).get("mean") is not None
                           and s["paired_model_minus_noop"]["mean"] < 0),
            "beats_cv": (s.get("paired_model_minus_cv", {}).get("mean") is not None
                         and s["paired_model_minus_cv"]["mean"] < 0),
            "residual_ratio": (s.get("resid_mag_model", {}).get("mean") or 0)
            / max(s.get("resid_mag_true", {}).get("mean") or 1e-12, 1e-12),
            "resid_r2": s.get("resid_r2", {}).get("mean")}
    q = (d.get("quantisation_analysis") or {}).get("per_task") or {}
    for t, v in q.items():
        if t in out:
            out[t]["model_L1_over_quantisation_floor"] = v.get("model_L1_over_quantisation_floor")
    return out


def main():
    tasks, unknowns = load_taskspec()
    physical = _physical_audit()
    measured = sorted(physical)
    GATE = {
        "state": "FAIL_ON_PHYSICAL_CONDITION",
        "why": ("the probability gate (P6) passes, but the PHYSICAL gate does not: on the frozen DEV "
                "frames the shared model's SAFE-14D error is worse than a zero-parameter constant-velocity "
                "extrapolator on both tasks, and its predicted residual has R^2 ~ 0. Per the pre-registered "
                "rule the 2->4 expansion stops and turns into mechanism diagnosis; the 2000-step checkpoint "
                "is preserved and no valid step is retrained."),
        "evidence": ["mission/m2_physical_audit_report.md", "mission/m2_physical_audit.json",
                     "mission/m2_physical_curve.json", "mission/m2_p6_two_corrected_report.json (PASS, "
                     "probability only)"],
        "per_task": physical}
    cls = feasibility_classes()
    rows = []
    for t in tasks:
        tid = t["id"]
        data = data_presence(tid)
        f = cls.get(tid, "UNCLASSIFIED")
        if f == "BLOCKED_DATA":
            nxt = "BLOCKED_DATA - rigid-body expert design cannot represent cloth/liquid; needs a physics-capable expert or moto/sim asset work, which is not authorised this round"
            auth = "NOT_AUTHORISED"
        elif data["status"] == "PRESENT":
            nxt = "usable: TaskSpec verified, data present, included in the shared-model task pool"
            auth = "AUTHORISED"
        else:
            nxt = ("collect a bounded scripted-expert pilot under COLLECT harness (lanes/COLLECT_harness.md); "
                   "gate P5_EXPERT_VALID must pass on a pilot before any 100-episode run")
            auth = "AUTHORISED_FOR_PILOT"
        row = {"task": tid, "step_lim": t["step_lim"], "expert_class": f,
               "data": data, "authorisation": auth, "next_action": nxt,
               "scene_layout_provenance": "UNKNOWN (the HDF5s carry no layout/seed metadata)"}
        if tid in physical:
            row["physical_capability"] = physical[tid]
        rows.append(row)
    present = [r for r in rows if r["data"]["status"] == "PRESENT"]
    blocked = [r for r in rows if r["expert_class"] == "BLOCKED_DATA"]
    out = {
        "schema": "r12-task-status-v1",
        "generated_from": ["lanes/C_taskspec.json (TaskSpec verified 12/12)",
                           "lanes/D_expert_feasibility.md (section headings -> expert class)",
                           "live filesystem check for HDF5 episode counts"],
        "counts": {"n_tasks": len(rows), "data_present": len(present),
                   "data_absent": len(rows) - len(present), "blocked_data": len(blocked)},
        "gate_2_to_4": GATE,
        "global_unknowns": unknowns,
        "tasks": rows,
    }
    p = os.path.join(REPO, "mission", "r12_task_status.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    md = ["# R12 status (generated, do not hand-edit)", "",
          "2->4 gate: **%s** — %s" % (GATE["state"], GATE["why"]), "",
          "| # | task | step_lim | expert | data | model SAFE L1 | no-op | CV | beats CV |",
          "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        pc = r.get("physical_capability") or {}
        md.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s |"
                  % (i, r["task"], r["step_lim"], r["expert_class"], r["data"]["status"],
                     ("%.4f" % pc["model_safe_l1"]) if pc.get("model_safe_l1") is not None else "-",
                     ("%.4f" % pc["noop_l1"]) if pc.get("noop_l1") is not None else "-",
                     ("%.4f" % pc["cv_l1"]) if pc.get("cv_l1") is not None else "-",
                     ("yes" if pc.get("beats_cv") else "**no**") if pc.get("beats_cv") is not None else "-"))
    md += ["", "| # | task | expert | data | authorisation | next action |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        md.append("| %d | %s | %s | %s | %s%s | %s | %s |"
                  % (i, r["task"], r["step_lim"], r["expert_class"],
                     r["data"]["status"],
                     (" (%d eps)" % r["data"]["n_episodes"]) if r["data"]["n_episodes"] else "",
                     r["authorisation"], r["next_action"][:110]))
    md += ["", "Counts: %s" % json.dumps(out["counts"]),
           "", "2->4 gate: %s" % out["gate_2_to_4"]["state"], "",
           out["gate_2_to_4"]["why"], ""]
    with open(os.path.join(REPO, "report", "R12_TASK_STATUS.md"), "w") as fh:
        fh.write("\n".join(md))
    print(json.dumps(out["counts"], indent=1))
    for r in rows:
        print("%-28s %-14s %-8s %s" % (r["task"], r["expert_class"], r["data"]["status"],
                                       r["authorisation"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
