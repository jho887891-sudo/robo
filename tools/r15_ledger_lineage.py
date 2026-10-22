#!/usr/bin/env python3
"""Honest accounting across the R15 WAL: executed steps vs DISTINCT LINEAGE steps.

The WAL books real executed optimizer steps, which is correct for GPU cost accounting but wrong for
"how many steps did the model lineage actually take": when a run is stopped and restarted from an
older checkpoint (exactly what happened after the resume bug), the same global_step is executed and
booked twice, or three times. Audit lane H found 1014 step_commit events covering only 963 distinct
global_step values, with global_step 301 booked three times.

This tool does not modify the WAL (it is immutable evidence). It derives, from the WAL alone:
  * executed steps and booked GPU seconds (the cost truth);
  * distinct global_step values covered (the lineage truth);
  * every global_step booked more than once, with the runs that booked it;
  * per-run wall clock minus booked step seconds (unbilled load/save/inference overhead);
  * the resume chain that the current lineage actually descends from.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

DEFAULT_WAL = os.path.join(os.environ.get("R15_LEDGER_DIR",
                                          "/home/T7/ojh/robot_actual/r15_ledger"), "events.jsonl")


def load(path):
    evs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                evs.append(json.loads(line))
    return evs


def report(events):
    runs = defaultdict(dict)
    step_events = defaultdict(list)
    booked = Counter()
    for ev in events:
        rid = ev.get("run_id")
        kind = ev.get("kind")
        p = ev.get("payload") or {}
        if kind == "job_begin":
            runs[rid].update({k: p.get(k) for k in
                              ("start_global_step", "target_global_step", "steps_requested",
                               "resume_from", "gpu_estimate_s")})
            runs[rid].setdefault("begin_utc", ev.get("utc"))
        elif kind == "job_end":
            runs[rid].update({k: p.get(k) for k in
                              ("global_step", "verdict", "wall_clock_s",
                               "formal_steps_booked_here", "status")})
            runs[rid].setdefault("end_utc", ev.get("utc"))
            runs[rid]["exit"] = p.get("exit")
        elif kind in ("step_commit", "abort"):
            gs = p.get("global_step")
            step_events[rid].append({"global_step": gs, "utc": ev.get("utc"),
                                     "gpu_seconds": float(ev.get("gpu_seconds") or 0.0),
                                     "formal_steps": int(ev.get("formal_steps") or 0)})
            if kind == "step_commit" and gs is not None:
                booked[int(gs)] += 1

    executed = sum(int(ev.get("formal_steps") or 0) for ev in events)
    gpu = round(sum(float(ev.get("gpu_seconds") or 0.0) for ev in events), 3)
    dup = {gs: n for gs, n in sorted(booked.items()) if n > 1}
    duplicates = []
    for gs, n in dup.items():
        who = [rid for rid, lst in step_events.items()
               if any(s["global_step"] == gs for s in lst)]
        duplicates.append({"global_step": gs, "times_booked": n, "run_ids": who})

    overhead = []
    for rid, rec in runs.items():
        step_secs = round(sum(s["gpu_seconds"] for s in step_events.get(rid, [])), 1)
        wall = rec.get("wall_clock_s")
        overhead.append({"run_id": rid, "wall_clock_s": wall, "booked_step_seconds": step_secs,
                         "unbooked_seconds": (round(float(wall) - step_secs, 1)
                                              if wall is not None else None),
                         "steps_executed": sum(s["formal_steps"] for s in step_events.get(rid, [])),
                         "verdict": rec.get("verdict") or rec.get("status"),
                         "resume_from": rec.get("resume_from"),
                         "begin_utc": rec.get("begin_utc"), "end_utc": rec.get("end_utc")})
    overhead.sort(key=lambda r: r["begin_utc"] or "")

    return {
        "wal_path": None,
        "n_events": len(events),
        "executed_formal_steps": executed,
        "distinct_global_steps_booked": len(booked),
        "re_executed_steps": executed - len(booked),
        "gpu_seconds_booked": gpu,
        "duplicate_global_steps": duplicates,
        "unbooked_overhead_seconds": sum(r["unbooked_seconds"] or 0.0 for r in overhead),
        "runs": overhead,
        "note": ("executed_formal_steps is the real cost of training; distinct_global_steps_booked "
                 "is the model lineage. A stage whose restart resumed from an older checkpoint "
                 "inflates the former without advancing the latter."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wal", default=DEFAULT_WAL)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = report(load(a.wal))
    rep["wal_path"] = a.wal
    if a.out:
        d = os.path.dirname(os.path.abspath(a.out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(rep, fh, indent=1)
    print("events                        %d" % rep["n_events"])
    print("executed formal steps         %d" % rep["executed_formal_steps"])
    print("distinct global steps booked  %d" % rep["distinct_global_steps_booked"])
    print("re-executed (wasted) steps    %d" % rep["re_executed_steps"])
    print("gpu seconds booked            %.1f" % rep["gpu_seconds_booked"])
    print("unbooked overhead seconds     %.1f" % rep["unbooked_overhead_seconds"])
    print("global steps booked >1x       %d" % len(rep["duplicate_global_steps"]))
    for d in rep["duplicate_global_steps"][:10]:
        print("   step %-5d x%d  %s" % (d["global_step"], d["times_booked"], d["run_ids"]))
    print("runs:")
    for r in rep["runs"]:
        print("   %-34s %-10s steps=%-5s wall=%-8s booked=%-8s unbooked=%s"
              % (r["run_id"], r["verdict"], r["steps_executed"], r["wall_clock_s"],
                 r["booked_step_seconds"], r["unbooked_seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
