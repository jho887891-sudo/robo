#!/usr/bin/env python3
"""R10.2 A2: graceful stop of THIS project old eval client / dedicated :6020 policy server.

Authorised signals only: SIGINT first, SIGTERM if needed. Never SIGKILL, never pkill,
never docker stop, never the shared isaac-sim kit. Every step is logged with UTC time.
"""
import argparse
import os
import subprocess
import sys
import time

SUDO_PW = "jxljy@2403"
TARGETS = {
    "client": (3451404, "eval_client/main.py"),
    "server": (3469786, "setup_policy_server.py"),
}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def cmdline(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return None


def alive(pid, finger):
    cl = cmdline(pid)
    return bool(cl) and finger in cl


def send(pid, sig):
    p = subprocess.run(["sudo", "-S", "-p", "", "kill", "-%s" % sig, str(pid)],
                       input=SUDO_PW + "\n", capture_output=True, text=True)
    return {"signal": sig, "rc": p.returncode, "err": p.stderr.strip()[:200], "utc": utc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=sorted(TARGETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--int-wait", type=int, default=60)
    ap.add_argument("--term-wait", type=int, default=30)
    args = ap.parse_args()
    pid, finger = TARGETS[args.mode]
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "stop_%s.log" % args.mode)
    lines = []

    def record(msg):
        line = "%s %s" % (utc(), msg)
        lines.append(line)
        print(line, flush=True)

    record("mode=%s pid=%d expect_fingerprint=%r" % (args.mode, pid, finger))
    cl = cmdline(pid)
    if cl is None:
        record("REFUSE: PID %d is not present/readable (PID-reuse guard)" % pid)
        record("RESULT=REFUSED_PID_ABSENT")
        open(log_path, "w").write("\n".join(lines) + "\n")
        return 3
    record("cmdline=%s" % cl[:400])
    if finger not in cl:
        record("REFUSE: cmdline does not match fingerprint; not signalling")
        record("RESULT=REFUSED_FINGERPRINT")
        open(log_path, "w").write("\n".join(lines) + "\n")
        return 4
    owner = subprocess.run(["ps", "-o", "user=", "-p", str(pid)], capture_output=True,
                          text=True).stdout.strip()
    record("owner=%s" % owner)

    res = send(pid, "INT")
    record("SIGINT rc=%d err=%r" % (res["rc"], res["err"]))
    waited = 0
    while waited < args.int_wait:
        if not alive(pid, finger):
            record("exited after SIGINT (%ds)" % waited)
            record("RESULT=EXITED_ON_SIGINT")
            open(log_path, "w").write("\n".join(lines) + "\n")
            return 0
        time.sleep(1)
        waited += 1
    record("still alive after %ds; sending SIGTERM" % args.int_wait)
    res = send(pid, "TERM")
    record("SIGTERM rc=%d err=%r" % (res["rc"], res["err"]))
    waited = 0
    while waited < args.term_wait:
        if not alive(pid, finger):
            record("exited after SIGTERM (%ds)" % waited)
            record("RESULT=EXITED_ON_SIGTERM")
            open(log_path, "w").write("\n".join(lines) + "\n")
            return 0
        time.sleep(1)
        waited += 1
    record("still alive after SIGTERM; NOT sending SIGKILL (forbidden) -> BLOCKED")
    record("RESULT=BLOCKED_DID_NOT_EXIT")
    open(log_path, "w").write("\n".join(lines) + "\n")
    return 5


if __name__ == "__main__":
    sys.exit(main())
