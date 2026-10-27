#!/usr/bin/env python3
"""R45 GPU benchmark entry point: bounded, interruptible, and impossible to mistake for training.

Hard rules encoded here:
  * it refuses to run without an explicitly approved GPU-hour budget (--approve-hours > 0),
  * it stops at the approved wall-clock cap and writes a partial report instead of continuing,
  * it writes its own PID, so a hung run can be identified without inspecting unrelated processes,
  * it never imports or calls the trainer.

Each configuration is measured with the same sample set, warm-up excluded from the steady-state
statistics, so throughput numbers are comparable.
"""
import argparse
import json
import os
import signal
import sys
import time

STOP = {"requested": False, "reason": None}


class BudgetGuard:
    """Wall-clock budget with an explicit, testable expiry."""

    def __init__(self, max_seconds):
        self.max_seconds = float(max_seconds)
        self.t0 = time.monotonic()
        self.reason = None

    def elapsed(self):
        return time.monotonic() - self.t0

    def remaining(self):
        return max(0.0, self.max_seconds - self.elapsed())

    def expired(self):
        if STOP["requested"] or self.elapsed() >= self.max_seconds:
            self.reason = STOP["reason"] or "budget_exhausted"
            return True
        return False


def _sig(signum, _frame):
    STOP["requested"] = True
    STOP["reason"] = "signal %d" % signum


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--approve-hours", type=float, default=0.0)
    ap.add_argument("--configs", default="[]", help="JSON list of {name, micro_batch, attn}")
    ap.add_argument("--steps-per-config", type=int, default=20)
    args = ap.parse_args()

    if not (args.approve_hours > 0):
        print("BLOCKED: no approved GPU budget (--approve-hours must be > 0). "
              "R45 requires explicit research-lead approval before any GPU benchmark.",
              file=sys.stderr)
        return 2
    if args.approve_hours > 2.0:
        print("BLOCKED: requested budget %.2f h exceeds the 2 h benchmark ceiling" %
              args.approve_hours, file=sys.stderr)
        return 2
    out = os.path.abspath(args.out)
    if os.path.exists(out):
        print("BLOCKED: output directory exists: " + out, file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    with open(os.path.join(out, "bench.pid"), "w") as fh:
        fh.write("%d\n" % os.getpid())

    guard = BudgetGuard(args.approve_hours * 3600.0)
    configs = json.loads(args.configs)
    report = {"schema": "r45-gpu-bench/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                 time.gmtime()),
              "pid": os.getpid(), "approved_hours": args.approve_hours,
              "cap_seconds": guard.max_seconds, "configs": [], "status": "RUNNING"}
    exit_code = 1
    try:
        for cfg in configs:
            if guard.expired():
                report["stopped_early"] = guard.reason
                break
            row = {"config": cfg, "steps": [], "status": "NOT_RUN"}
            report["configs"].append(row)
            for step in range(args.steps_per_config):
                if guard.expired():
                    row["status"] = "TRUNCATED"
                    break
                t0 = time.perf_counter()
                time.sleep(0.001)                     # placeholder for one real training step
                row["steps"].append(time.perf_counter() - t0)
            if row["steps"]:
                warm = row["steps"][: max(1, len(row["steps"]) // 5)]
                steady = row["steps"][len(warm):] or row["steps"]
                row["warmup_s"] = sum(warm) / len(warm)
                row["steady_s"] = sum(steady) / len(steady)
                row["samples_per_s"] = (cfg.get("micro_batch", 1) / row["steady_s"]
                                        if row["steady_s"] else None)
                row["status"] = "OK"
        report["status"] = "PASS"
        exit_code = 0
    except Exception as exc:                              # pragma: no cover
        report["status"] = "ERROR"
        report["error"] = repr(exc)
        exit_code = 1
    finally:
        report["elapsed_s"] = guard.elapsed()
        with open(os.path.join(out, "bench_report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
    print("STATUS=%s elapsed=%.1fs out=%s" % (report["status"], report["elapsed_s"], out))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
