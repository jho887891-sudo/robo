#!/usr/bin/env python3
"""Append-only correction of the M2 GPU booking (issue #26 item 3).

The first M2 job booked gpu_seconds=0.0 at job_end while its wall clock was 7742.6 s: the 7B load, the
DEV counterfactual CE (8 frames x 2 tasks x 2 prompts of real forward) and every full-state save were
never charged anywhere. The per-step events booked only the step loop.

The WAL is immutable, so this does not rewrite anything: it appends ONE correction event whose GPU
seconds are the difference between the real job wall (measured from the WAL timestamps themselves) and
what those runs already booked. Idempotent - a second run does nothing.
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "training"))
import r15_ledger as L  # noqa: E402

CORRECTION_RUN_ID = "m2-wall-correction-r5"
KINDS_STEP = ("step_commit",)


def parse_utc(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def main():
    led = L.Ledger()
    with led:
        evs = led.read_events()
        if any(e["run_id"] == CORRECTION_RUN_ID for e in evs):
            print("%s already present - idempotent, nothing to do" % CORRECTION_RUN_ID)
            return 0
        # group M2 events by run
        begins, ends = {}, {}
        booked = {}
        for e in evs:
            b = (e.get("payload") or {}).get("bucket") or e.get("bucket")
            if b != "TWO_TASK_PILOT":
                continue
            rid = e["run_id"]
            if e["kind"] == "job_begin":
                begins[rid] = e["utc"]
            elif e["kind"] == "job_end":
                ends[rid] = e["utc"]
            if e["kind"] in KINDS_STEP:
                booked[rid] = booked.get(rid, 0.0) + float(e.get("gpu_seconds") or 0.0)
        per_run, missing_total = {}, 0.0
        for rid, t0 in begins.items():
            if rid not in ends:
                per_run[rid] = {"status": "UNFINISHED_NO_JOB_END"}
                continue
            wall = parse_utc(ends[rid]) - parse_utc(t0)
            bk = float(booked.get(rid, 0.0))
            miss = max(0.0, wall - bk)
            per_run[rid] = {"wall_s": round(wall, 1), "booked_step_s": round(bk, 1),
                            "unbooked_s": round(miss, 1),
                            "begin_utc": t0, "end_utc": ends[rid]}
            missing_total += miss
        tot = led.totals()
        before = tot["gpu_seconds_used"]
        ev = led.append(
            CORRECTION_RUN_ID, "job_end_correction", bucket="TWO_TASK_PILOT",
            payload={
                "why": ("M2 job_end booked gpu_seconds=0.0; the 7B load, DEV counterfactual CE and "
                        "full-state saves of these runs were never charged. The correction books "
                        "exactly the gap between the real WAL wall clock and the per-step seconds "
                        "already booked, so nothing is double counted."),
                "per_run": per_run,
                "counts_toward_total": True,
                "no_double_charge": True,
                "gpu_seconds_before": round(before, 3),
                "gpu_seconds_after": round(before + missing_total, 3),
                "reference_issue": "https://github.com/jho887891-sudo/robota/issues/26"},
            gpu_seconds=missing_total, formal_steps=0)
        after = led.totals()
        print(json.dumps({"correction_seq": ev["seq"], "unbooked_s": round(missing_total, 1),
                          "gpu_before": round(before, 1), "gpu_after": round(after["gpu_seconds_used"], 1),
                          "per_run": per_run}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
