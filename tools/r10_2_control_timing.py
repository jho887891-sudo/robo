#!/usr/bin/env python3
"""R10.2 D5: control-timeline accounting for a 42D policy (no real model run).

Without a real 7B loop every latency is NOT_MEASURED. This module only provides the
bookkeeping: per-stage timestamps, budget checks against the 25 Hz (40 ms) frame, and
p50/p95/p99 aggregation once real measurements exist.
"""
import json
import sys

FRAME_MS = 40.0          # 25 Hz physical control period observed in the demo data
ACTION_HOLD_MS = 100.0   # legacy deployment held one action ~0.93 s; recorded, not assumed
STAGES = ("capture", "queue", "prefill", "decode14", "sim_step", "hold")


def percentile(values, q):
    if not values:
        raise ValueError("no samples")
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def stage_timestamps(t0_ms, stage_durations_ms):
    missing = [s for s in STAGES if s not in stage_durations_ms]
    if missing:
        raise ValueError("stage durations missing: %s" % missing)
    out = {}
    t = float(t0_ms)
    for s in STAGES:
        out[s] = {"start_ms": t, "duration_ms": float(stage_durations_ms[s])}
        t += float(stage_durations_ms[s])
    out["total_ms"] = t - float(t0_ms)
    return out


def budget_check(total_ms, frame_ms=FRAME_MS):
    return {"total_ms": float(total_ms), "frame_ms": float(frame_ms),
            "fits_frame": bool(total_ms <= frame_ms),
            "overrun_ms": max(0.0, float(total_ms) - float(frame_ms)),
            "note": "fits_frame=false never implies the robot is unsafe: the action is held"}


def summarize(samples, label="latency"):
    if not samples:
        return {"label": label, "status": "NOT_MEASURED", "n": 0,
                "note": "no real run in this round; nothing is claimed about timing"}
    return {"label": label, "status": "MEASURED", "n": len(samples),
            "p50_ms": percentile(samples, 0.50), "p95_ms": percentile(samples, 0.95),
            "p99_ms": percentile(samples, 0.99), "min_ms": min(samples), "max_ms": max(samples)}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else ""
    demo = stage_timestamps(0.0, {"capture": 8.0, "queue": 2.0, "prefill": 25.0,
                                  "decode14": 30.0, "sim_step": 5.0, "hold": 40.0})
    res = {"frame_ms": FRAME_MS, "action_hold_ms_reference": ACTION_HOLD_MS, "stages": STAGES,
           "example_schedule": demo, "budget": budget_check(demo["total_ms"]),
           "real_latency": summarize([], "policy_step"),
           "status": "IMPLEMENTED_NOT_MEASURED"}
    print(json.dumps(res, indent=1))
    if out:
        json.dump(res, open(out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
