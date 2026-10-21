#!/usr/bin/env python3
"""Reversal + correctly scoped rebooking of the M2 wall correction.

The first correction (m2-wall-correction-r5) summed BOTH the supervisor wrapper runs (sup:*) and the
inner trainer runs. A supervisor wrapper with count_gpu=false does not own GPU time - it is the same
wall clock as its inner program - so that double counted the M2 occupancy: the ledger went from
24183.9 s to 34121.6 s (+9937.7) when the true unbilled amount is ~2055 s.

The WAL is append-only, so nothing is rewritten: this appends (a) a reversal of exactly the amount the
first correction over-booked and (b) a rebooking that counts INNER runs only, with the one crashed
run (which has a job_begin but no job_end) bounded by its supervisor wrapper's wall.
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "training"))
import r15_ledger as L  # noqa: E402

REVERSAL_ID = "m2-wall-correction-r5-reversal"
REBOOK_ID = "m2-wall-correction-r5-rebook"
FIRST = "m2-wall-correction-r5"
MAX_BOUND_S = 600.0


def ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def main():
    led = L.Ledger()
    with led:
        evs = led.read_events()
        by_id = {e["run_id"]: e for e in evs}
        if REVERSAL_ID in by_id and REBOOK_ID in by_id:
            print("reversal+rebook already present - idempotent")
            return 0
        first = by_id.get(FIRST)
        if first is None:
            print("first correction missing; refusing to guess")
            return 2
        first_amount = float(first.get("gpu_seconds") or 0.0)
        if REVERSAL_ID not in by_id:
            led.append(REVERSAL_ID, "job_end_correction_reversal", bucket="TWO_TASK_PILOT",
                       payload={"reverses": FIRST,
                                "why": ("that correction summed supervisor wrapper runs (sup:*) as "
                                        "well as their inner trainer runs. A wrapper with "
                                        "count_gpu=false is the SAME wall clock as its inner "
                                        "program, so the M2 GPU occupancy was booked twice. This "
                                        "event reverses the whole amount so the rebooking below "
                                        "can state the correct figure."),
                                "reversed_gpu_seconds": first_amount,
                                "counts_toward_total": True},
                       gpu_seconds=-first_amount, formal_steps=0)
            print("reversal appended: %.1f s" % -first_amount)
        # ---- recompute using INNER runs only
        begins, ends, booked, wrapper = {}, {}, {}, []
        for e in evs:
            rid = e["run_id"]
            b = e.get("bucket")
            if b != "TWO_TASK_PILOT":
                continue
            if rid.startswith("sup:"):
                if e["kind"] == "job_begin":
                    begins.setdefault("__wrapper__", []).append((rid, e["utc"]))
                elif e["kind"] == "job_end":
                    wrapper.append((rid, e["utc"]))
                continue
            if e["kind"] == "job_begin":
                begins[rid] = e["utc"]
            elif e["kind"] == "job_end":
                ends[rid] = e["utc"]
            elif e["kind"] in ("step_commit", "abort"):
                booked[rid] = booked.get(rid, 0.0) + float(e.get("gpu_seconds") or 0.0)
        wbeg = {rid: u for rid, u in begins.get("__wrapper__", [])}
        wend = dict(wrapper)
        per_run, total = {}, 0.0
        for rid, t0 in begins.items():
            if rid == "__wrapper__":
                continue
            if rid in ends:
                wall = ts(ends[rid]) - ts(t0)
                miss = max(0.0, wall - float(booked.get(rid, 0.0)))
                per_run[rid] = {"kind": "completed", "wall_s": round(wall, 1),
                                "booked_step_s": round(float(booked.get(rid, 0.0)), 1),
                                "unbooked_s": round(miss, 1)}
            else:
                # crashed run: bound its occupancy by the wrapper whose interval contains its start
                bound = 0.0
                for wrid, wb in wbeg.items():
                    if wrid not in wend:
                        continue
                    if ts(wb) <= ts(t0) <= ts(wend[wrid]):
                        bound = min(MAX_BOUND_S, ts(wend[wrid]) - ts(t0))
                        break
                miss = max(0.0, bound - float(booked.get(rid, 0.0)))
                per_run[rid] = {"kind": "crashed_no_job_end", "bounded_by_wrapper_s": round(bound, 1),
                                "booked_step_s": round(float(booked.get(rid, 0.0)), 1),
                                "unbooked_s": round(miss, 1)}
            total += miss
        before = led.totals()["gpu_seconds_used"]
        led.append(REBOOK_ID, "job_end_correction", bucket="TWO_TASK_PILOT",
                   payload={"why": ("M2 GPU wall for the runs that actually touched the device: 7B "
                                    "load, DEV counterfactual CE (8 frames x 2 tasks x 2 prompts of "
                                    "real forward) and full-state saves were unbooked because "
                                    "job_end wrote gpu_seconds=0.0. Supervisor wrappers are excluded "
                                    "on purpose: they do not own a second interval of GPU time."),
                            "per_run": per_run, "counts_toward_total": True,
                            "no_double_charge": True,
                            "gpu_seconds_before": round(before, 1),
                            "gpu_seconds_after": round(before + total, 1)},
                   gpu_seconds=total, formal_steps=0)
        after = led.totals()
        print(json.dumps({"rebook_s": round(total, 1),
                          "gpu_before": round(before, 1),
                          "gpu_after": round(after["gpu_seconds_used"], 1),
                          "tier_bucket_s": round(after["buckets"]["TWO_TASK_PILOT"]["used_s"], 1),
                          "per_run": per_run}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
