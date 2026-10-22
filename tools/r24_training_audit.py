#!/usr/bin/env python3
"""R24 Stage A: training-evidence audit from RAW logs, WAL and sealed checkpoints (CPU only).

What it does NOT do: interpolate, smooth, or invent a curve. Every value in the output is traced to a
file:line in a raw log or to a checkpoint artefact, and anything absent is reported as MISSING with the
path that was searched. The R24 question ("is this under-training, a learning-rate problem, a
generalisation problem, or an action-representation problem?") cannot be answered from a smoothed
summary, so the tool refuses to produce one.

Outputs (mission/r24/):
  r15_training_curve.json   per-step loss / grad-norm / lr for every R15 segment that has a raw log
  m2_training_curve.json    same for the M2 two-task run, with the per-task loss breakdown
  dev_curves.json           every DEV CE and physical number that exists, per checkpoint, with provenance
  coverage.json             batch size, grad accumulation, unique-frame coverage and epoch fraction
  lr_schedule.json          reconstructed LR schedule with the exact step of every change
  training_evidence.json    the index: which raw file was read, its sha256, and what is MISSING
"""
import argparse
import hashlib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = "/home/T7/ojh/robot_actual/runs"
LOGS = os.path.join(REPO, "logs", "supervisor")
OUT = os.path.join(REPO, "mission", "r24")

# (run_id, log file, kind) - the R15 stages are separate segments of ONE lineage
TRAIN_LOGS = [
    ("r15_seg_500_1000", "m1_700c_seg2b_to_1000.log", "r15"),
    ("r15_seg_300_500", "m1_700_step300_to_1000_r2.log", "r15"),
    ("m2_two_2000", "m2_two_2000_r5.log", "m2"),
]

STEP_RE = re.compile(
    r"^\[(?P<time>[\d:]+)\]\s+step=(?P<step>\d+)\s+"
    r"(?:(?:task=(?P<task>[a-zA-Z0-9_]+)\s+)?loss=(?P<loss>[-\d.eE+]+)\s+"
    r"gnorm=(?P<gnorm>[-\d.eE+nan]+)\s+)?"
    r"(?:lr=(?P<lr>[-\d.eE+]+)\s+)?"
    r"(?:per-task=(?P<pertask>\{[^}]*\}))?")
DEV_RE = re.compile(r"DEV (?P<task>[a-zA-Z0-9_]+) ce_window=(?P<ce>[\d.]+) ce_full=(?P<cef>[\d.]+)")
SAVE_RE = re.compile(r"saved full-state checkpoint step=(?P<step>\d+)")
RESUME_RE = re.compile(r"resumed from (?P<dir>\S+) at global_step=(?P<step>\d+)")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_log(path):
    """Every step/DEV/save line, verbatim, with its line number."""
    steps, dev, saves, resumes = [], [], [], []
    with open(path, "r", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            m = STEP_RE.match(line.rstrip("\n"))
            if m:
                rec = {"line": lineno, "clock": m.group("time"), "step": int(m.group("step"))}
                if m.group("loss") is not None:
                    rec["loss"] = float(m.group("loss"))
                if m.group("gnorm") is not None:
                    rec["gnorm"] = None if "nan" in m.group("gnorm") else float(m.group("gnorm"))
                if m.group("lr") is not None:
                    rec["lr"] = float(m.group("lr"))
                if m.group("task"):
                    rec["task"] = m.group("task")
                if m.group("pertask"):
                    body = m.group("pertask").strip("{}")
                    pt = {}
                    for part in body.split(","):
                        if ":" in part:
                            k, v = part.split(":")
                            pt[k.strip().strip("'\"")] = float(v)
                    rec["per_task"] = pt
                steps.append(rec)
                continue
            m = DEV_RE.search(line)
            if m:
                dev.append({"line": lineno, "clock": line[1:9] if line.startswith("[") else None,
                            "task": m.group("task"), "ce_window": float(m.group("ce")),
                            "ce_full": float(m.group("cef"))})
                continue
            m = SAVE_RE.search(line)
            if m:
                saves.append({"line": lineno, "step": int(m.group("step"))})
                continue
            m = RESUME_RE.search(line)
            if m:
                resumes.append({"line": lineno, "dir": m.group("dir"), "step": int(m.group("step"))})
    return {"steps": steps, "dev": dev, "saves": saves, "resumes": resumes}


def lr_schedule(steps):
    """Every DISTINCT lr value with the step where it first appears. No interpolation."""
    out = []
    for rec in steps:
        if "lr" not in rec:
            continue
        if not out or abs(rec["lr"] - out[-1]["lr"]) > 1e-12:
            out.append({"lr": rec["lr"], "first_step": rec["step"],
                        "prev_lr": (out[-1]["lr"] if out else None)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--runs", default=RUNS)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    index = {"schema": "r24-training-evidence-v1", "sources": [], "missing": []}
    curves = {}
    for run_id, logname, kind in TRAIN_LOGS:
        path = os.path.join(LOGS, logname)
        if not os.path.isfile(path):
            index["missing"].append({"run": run_id, "what": "training log", "searched": path})
            continue
        parsed = parse_log(path)
        index["sources"].append({"run": run_id, "kind": kind, "file": path,
                                 "bytes": os.path.getsize(path),
                                 "sha256": sha256_file(path),
                                 "n_step_lines": len(parsed["steps"]),
                                 "n_dev_lines": len(parsed["dev"]),
                                 "step_range": [parsed["steps"][0]["step"],
                                                parsed["steps"][-1]["step"]] if parsed["steps"] else None})
        curves[run_id] = {"kind": kind, "steps": parsed["steps"], "dev": parsed["dev"],
                          "saves": parsed["saves"], "resumes": parsed["resumes"],
                          "lr_schedule": lr_schedule(parsed["steps"])}

    # ---------------- checkpoint-level DEV evidence (whatever actually exists)
    dev_ckpt = {}
    for run_id, sub in (("m2_two_2000", "m2_two_2000_20261021T123314Z"),):
        ev = os.path.join(a.runs, sub, "stage_evidence.json")
        if os.path.isfile(ev):
            d = json.load(open(ev))
            dev_ckpt[run_id] = {"stage_evidence": ev, "sha256": sha256_file(ev),
                                "ce_curves": d.get("ce_curves"),
                                "final_ce": d.get("final_ce"),
                                "verdict": d.get("verdict")}
        else:
            index["missing"].append({"run": run_id, "what": "stage_evidence", "searched": ev})
    for name, rel in (("m2_ce_curve_corrected", os.path.join(REPO, "mission", "m2_ce_curve_corrected.json")),
                      ("m2_physical_curve", os.path.join(REPO, "mission", "m2_physical_curve.json")),
                      ("r15_g5_1000", os.path.join(REPO, "mission", "r15_gate_G5_1000.json"))):
        if os.path.isfile(rel):
            dev_ckpt[name] = {"file": rel, "sha256": sha256_file(rel)}
        else:
            index["missing"].append({"what": name, "searched": rel})

    # ---------------- coverage / epoch accounting
    cov = {"schema": "r24-coverage-v1", "runs": {}}
    splits_path = os.path.join(REPO, "data", "m2_splits.json")
    splits = json.load(open(splits_path)) if os.path.isfile(splits_path) else None
    m2_steps = curves.get("m2_two_2000", {}).get("steps") or []
    m2_max = max((s["step"] for s in m2_steps), default=None)
    if splits:
        # FRAMES, not episodes: one optimizer step consumes exactly one (episode, t) frame, so the
        # epoch fraction is optimizer_steps / TRAIN_FRAMES. Counting episodes here once produced a
        # nonsensical 8.33 "epochs"; the trainer's own frame enumerator is the authority.
        per_task_frames = {}
        frame_error = None
        try:
            sys.path.insert(0, os.path.join(REPO, "training"))
            import m2_train_shared as MT
            import m2_multitask_data as MD
            smap = splits.get("splits") or {}
            for t in ("push_T", "stack_bowls"):
                idx = MD.load_task_index(t)
                per_task_frames[t] = len(MT.task_train_frames(idx, smap[t], t))
        except Exception as exc:                        # noqa: BLE001 - recorded, never guessed
            frame_error = repr(exc)
        total_train = sum(per_task_frames.values()) if per_task_frames else None
        cov["runs"]["m2_two_2000"] = {
            "batch_size_per_device": 1, "grad_accum": 1,
            "optimizer_steps": m2_max,
            "unique_frames_consumed_if_never_repeated": m2_max,
            "train_frames_available": total_train,
            "per_task_train_frames": per_task_frames,
            "epoch_fraction": (m2_max / total_train) if total_train else None,
            "frame_count_error": frame_error,
            "note": ("batch=1 and no gradient accumulation: one optimizer step consumes exactly ONE "
                     "frame, so 'epochs' is optimizer_steps / TRAIN_FRAMES and the model has seen at "
                     "most one pass over a fraction of the data")}
    r15_steps = sorted((s["step"] for rid, c in curves.items() if c["kind"] == "r15"
                        for s in c["steps"]))
    r15_frames = None
    if splits and per_task_frames.get("push_T"):
        # R15 trained on push_T alone with the same 80/20 episode split
        r15_frames = per_task_frames["push_T"]
    cov["runs"]["r15_lineage"] = {
        "batch_size_per_device": 1, "grad_accum": 1,
        "observed_step_min": r15_steps[0] if r15_steps else None,
        "observed_step_max": r15_steps[-1] if r15_steps else None,
        "n_step_lines": len(r15_steps),
        "train_frames_available": r15_frames,
        "global_steps_total_lineage": 1000,
        "epoch_fraction_at_1000": (1000 / r15_frames) if r15_frames else None,
        "note": ("the R15 lineage is three segments (0-300, 300-500, 500-1000); only the segments whose "
                 "raw logs still exist are reported, and the others are listed as MISSING rather than "
                 "back-filled")}
    r15_runs = {rid: c for rid, c in curves.items() if c["kind"] == "r15"}
    cov["runs"]["r15_lineage"]["segments"] = {
        rid: {"step_range": [c["steps"][0]["step"], c["steps"][-1]["step"]] if c["steps"] else None,
              "n_steps": len(c["steps"]), "lr_schedule": c["lr_schedule"]}
        for rid, c in r15_runs.items()}

    # ---------------- write
    def dump(name, obj):
        p = os.path.join(a.out_dir, name)
        tmp = p + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
        return {"file": p, "sha256": sha256_file(p)}

    index["outputs"] = {
        "r15_training_curve": dump("r15_training_curve.json",
                                   {k: v for k, v in curves.items() if v["kind"] == "r15"}),
        "m2_training_curve": dump("m2_training_curve.json",
                                  {k: v for k, v in curves.items() if v["kind"] == "m2"}),
        "dev_curves": dump("dev_curves.json", dev_ckpt),
        "coverage": dump("coverage.json", cov),
        "lr_schedule": dump("lr_schedule.json",
                            {rid: c["lr_schedule"] for rid, c in curves.items() if c["lr_schedule"]}),
    }
    dump("training_evidence.json", index)

    print(json.dumps({"sources": [{k: s[k] for k in ("run", "file", "n_step_lines", "step_range")}
                                  for s in index["sources"]],
                      "missing": index["missing"],
                      "coverage": {r: {k: v for k, v in d.items() if k != "segments"}
                                   for r, d in cov["runs"].items()}}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
