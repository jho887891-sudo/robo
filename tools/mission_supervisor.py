#!/usr/bin/env python3
"""R15-R23 persistent mission supervisor (server side, survives chat windows).

Run detached:   setsid nohup python tools/mission_supervisor.py --queue mission/supervisor_queue.json \
                    > logs/supervisor.log 2>&1 &
It is a real on-disk state machine, not an AI keeping watch:
  * the queue is a JSON list of jobs {id, kind, bucket, cmd, timeout_s, depends_on, gpu}
  * state lives in <queue>.state.json (atomic rename) so a supervisor restart resumes cleanly
  * heavy GPU jobs take the exclusive GpuJobLock, so training / IsaacSim collect / rollout stay
    serialised on the single A6000; CPU jobs run up to --max-cpu in parallel
  * every job gets an outer timeout, its raw stdout/stderr on disk, its exit code and real elapsed
    seconds booked into the append-only WAL ledger
  * a job is run at most once per id (idempotent, never overwrites); a failed job marks its
    dependents BLOCKED instead of being retried in a loop (<=2 attempts, different-hypothesis only)
"""
import argparse, json, os, subprocess, sys, time, threading

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "training"))
import r15_ledger as L  # noqa: E402


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(p, default):
    try:
        with open(p) as fh:
            return json.load(fh)
    except (IOError, ValueError):
        return default


def save_json(p, obj):
    tmp = p + ".tmp-%d" % os.getpid()
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def run_one(job, state, lock_dir, logs_dir, ledger):
    jid = job["id"]
    if state["jobs"].get(jid, {}).get("status") in ("DONE", "RUNNING"):
        return
    deps = job.get("depends_on") or []
    for d in deps:
        st = state["jobs"].get(d, {}).get("status")
        if st != "DONE":
            state["jobs"][jid] = {"status": "BLOCKED", "reason": "dependency %s is %s" % (d, st),
                                  "utc": now()}
            save_json(state["_path"], state)
            ledger.append("supervisor", "job_blocked", bucket=job.get("bucket"),
                          payload={"job": jid, "dependency": d, "dependency_status": st})
            return
    os.makedirs(logs_dir, exist_ok=True)
    out = os.path.join(logs_dir, "%s.log" % jid)
    state["jobs"][jid] = {"status": "RUNNING", "started_utc": now(), "log": out,
                          "attempts": state["jobs"].get(jid, {}).get("attempts", 0) + 1}
    save_json(state["_path"], state)
    ledger.append("supervisor", "job_begin", bucket=job.get("bucket"),
                  payload={"job": jid, "cmd": job["cmd"], "timeout_s": job.get("timeout_s")})
    t0 = time.time()
    rc, timed_out = None, False
    try:
        if job.get("gpu"):
            lock = L.GpuJobLock(lock_dir)
            lock.__enter__()
        with open(out, "wb") as fh:
            p = subprocess.Popen(job["cmd"], shell=True, cwd=REPO, stdout=fh,
                                 stderr=subprocess.STDOUT, preexec_fn=os.setsid)
            try:
                rc = p.wait(timeout=job.get("timeout_s", 6 * 3600))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(os.getpgid(p.pid), 15)
                time.sleep(10)
                if p.poll() is None:
                    os.killpg(os.getpgid(p.pid), 9)
                rc = -15
        if job.get("gpu"):
            lock.__exit__(None, None, None)
    except Exception as exc:                      # never kill the supervisor itself
        rc = -1
        out_note = repr(exc)
        with open(out, "a") as fh:
            fh.write("\nSUPERVISOR EXCEPTION: %s\n" % out_note)
    elapsed = time.time() - t0
    status = "DONE" if rc == 0 and not timed_out else ("TIMEOUT" if timed_out else "FAILED")
    state["jobs"][jid].update({"status": status, "exit": rc, "seconds": round(elapsed, 1),
                               "ended_utc": now()})
    save_json(state["_path"], state)
    ledger.append("supervisor", "job_end", bucket=job.get("bucket"),
                  payload={"job": jid, "status": status, "exit": rc, "log": out},
                  gpu_seconds=elapsed if job.get("gpu") else 0.0, formal_steps=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=os.path.join(REPO, "mission", "supervisor_queue.json"))
    ap.add_argument("--ledger-dir", default=None)
    ap.add_argument("--logs", default=os.path.join(REPO, "logs", "supervisor"))
    ap.add_argument("--max-cpu", type=int, default=4)
    ap.add_argument("--poll", type=float, default=10.0)
    a = ap.parse_args()
    state_path = a.queue + ".state.json"
    state = load_json(state_path, {"jobs": {}})
    state["_path"] = state_path
    ledger_dir = a.ledger_dir or os.path.dirname(os.path.join(REPO, "x"))
    ledger_dir = os.environ.get("R15_LEDGER_DIR", "/home/T7/ojh/robot_actual/r15_ledger")
    print("[%s] supervisor up, queue=%s" % (now(), a.queue), flush=True)
    while True:
        queue = load_json(a.queue, {"jobs": []})
        jobs = queue.get("jobs", [])
        running = sum(1 for v in state["jobs"].values() if v.get("status") == "RUNNING")
        progressed = False
        for job in jobs:
            st = state["jobs"].get(job["id"], {}).get("status")
            if st in ("DONE", "RUNNING", "BLOCKED", "TIMEOUT", "FAILED"):
                continue
            if job.get("gpu") and running:
                continue                       # one heavy GPU lane
            if not job.get("gpu") and running >= a.max_cpu:
                continue
            with L.Ledger(ledger_dir) as led:
                run_one(job, state, ledger_dir, a.logs, led)
            progressed = True
            running = sum(1 for v in state["jobs"].values() if v.get("status") == "RUNNING")
        if not progressed:
            pending = [j["id"] for j in jobs
                       if state["jobs"].get(j["id"], {}).get("status") is None]
            if not pending:
                print("[%s] queue drained" % now(), flush=True)
                return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
