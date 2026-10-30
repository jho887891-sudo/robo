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
    """Fail closed until a REAL OpenVLA forward/backward/optimizer benchmark exists.

    This entrypoint must never report samples/sec or PASS from a placeholder timer.
    The --approve-hours option only records a requested future budget; it does NOT
    grant research-lead authorization or enable CUDA work in this version.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--approve-hours", type=float, default=0.0)
    ap.add_argument("--configs", default="[]")
    ap.add_argument("--steps-per-config", type=int, default=20)
    args = ap.parse_args()

    import math
    if not math.isfinite(args.approve_hours) or args.approve_hours <= 0:
        print("BLOCKED_AUTHORIZATION: explicit research-lead GPU approval is required.",
              file=sys.stderr)
        return 2
    if args.approve_hours > 0.5:
        print("BLOCKED_BUDGET: 30-minute single-GPU benchmark ceiling exceeded.",
              file=sys.stderr)
        return 2
    if args.steps_per_config < 1:
        print("BLOCKED_CONTRACT: steps-per-config must be positive.", file=sys.stderr)
        return 2
    try:
        configurations = json.loads(args.configs)
        if not isinstance(configurations, list) or not configurations:
            raise ValueError("nonempty configuration list required")
    except (ValueError, TypeError) as exc:
        print("BLOCKED_CONTRACT: %s" % exc, file=sys.stderr)
        return 2
    if os.path.exists(os.path.abspath(args.out)):
        print("BLOCKED_OUTPUT: output directory already exists.", file=sys.stderr)
        return 2
    print("BLOCKED_NOT_IMPLEMENTED: no real OpenVLA model forward/backward/optimizer "
          "measurement is wired. No GPU/CUDA access and no output created.",
          file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())
