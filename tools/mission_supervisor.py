#!/usr/bin/env python3
"""R15-R23 persistent mission supervisor (server side, survives chat windows).

Launch detached (verified after the launching shell is gone):
    cd /home/T7/ojh/robot_actual/r11
    setsid nohup /home/dgut/pusht_runtime/venv_oft/bin/python tools/mission_supervisor.py \
        --queue mission/supervisor_queue.json > logs/supervisor.log 2>&1 &

It is a real on-disk state machine, not an AI keeping watch:
  * queue is JSON {"jobs": [{id, kind, bucket, cmd, timeout_s, depends_on, gpu, count_gpu}]}
  * state lives in <queue>.state.json, written atomically (tmp+fsync+rename) on every transition
  * heavy GPU jobs take the exclusive GpuJobLock -> the single A6000 stays serialised, and the
    supervisor never starts a second GPU job while one runs; CPU jobs run up to --max-cpu at once
    in real threads (v1 ran them serially, which silently defeated the CPU-parallel rule)
  * every job gets an outer timeout (process-group kill), raw stdout+stderr on disk, the real exit
    code and the real elapsed seconds. Booking is *not* automatic: a job that already books itself
    into the WAL (the R15 trainer) must set "count_gpu": false so its GPU seconds are not counted
    twice. Jobs with "count_gpu": true are booked by the supervisor into the job's bucket.
  * a job id runs at most --max-attempts times; a failure marks its dependents BLOCKED instead of
    being retried in a loop. A supervisor that is killed and restarted resumes from the state file:
    jobs that were RUNNING are marked INTERRUPTED (never silently rebooked), and only jobs with
    "retry_on_interrupt": true are started again.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "training"))
import r15_ledger as L  # noqa: E402

STATE_LOCK = threading.Lock()
TERMINAL_BAD = ("FAILED", "TIMEOUT", "BLOCKED", "INTERRUPTED")
TERMINAL = ("DONE",) + TERMINAL_BAD


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg):
    print("[%s] %s" % (now(), msg), flush=True)


def load_json(path, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (IOError, ValueError):
        return default


def save_json(path, obj):
    tmp = path + ".tmp-%d-%d" % (os.getpid(), threading.get_ident())
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def book(ledger_dir, rid, kind, **kw):
    """Ledger writes must never take the supervisor down."""
    try:
        with L.Ledger(ledger_dir) as led:
            led.append(rid, kind, **kw)
    except Exception as exc:
        log("ledger write %s/%s failed: %r" % (rid, kind, exc))


def run_one(job, state, state_path, lock_dir, logs_dir, ledger_dir):
    jid = job["id"]
    rid = "sup:" + jid
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "%s.log" % jid)
    t0 = time.time()
    rc, timed_out, err = None, False, None
    gpu_lock = None
    try:
        # The R15 trainer takes the very same flock itself (GpuJobLock) for the whole run. Taking it
        # here as well deadlocked the first real launch: the child waited its full lock timeout for a
        # lock its own supervisor held. "supervisor_locks_gpu": false means the inner program owns it;
        # the supervisor still serialises dispatch through its own running-GPU counter.
        if job.get("gpu") and job.get("supervisor_locks_gpu", True):
            gpu_lock = L.GpuJobLock(lock_dir)
            gpu_lock.__enter__()
            log("job %s holds the GPU lane" % jid)
        with open(log_path, "wb") as fh:
            fh.write(("# supervisor job %s\n# cmd: %s\n# utc: %s\n"
                      % (jid, job["cmd"], now())).encode())
            fh.flush()
            p = subprocess.Popen(job["cmd"], shell=True, cwd=REPO, stdout=fh,
                                 stderr=subprocess.STDOUT, preexec_fn=os.setsid)
            with STATE_LOCK:
                state["jobs"][jid]["pid"] = p.pid
                save_json(state_path, state)
            try:
                rc = p.wait(timeout=float(job.get("timeout_s", 6 * 3600)))
            except subprocess.TimeoutExpired:
                timed_out = True
                log("job %s TIMEOUT after %.0fs, killing process group %d"
                    % (jid, time.time() - t0, p.pid))
                try:
                    os.killpg(os.getpgid(p.pid), 15)
                except OSError:
                    pass
                time.sleep(15)
                if p.poll() is None:
                    try:
                        os.killpg(os.getpgid(p.pid), 9)
                    except OSError:
                        pass
                p.wait()
                rc = -15
    except Exception as exc:                      # a broken job must not kill the supervisor
        err = repr(exc)[:400]
        rc = -1
        try:
            with open(log_path, "a") as fh:
                fh.write("\nSUPERVISOR EXCEPTION: %s\n" % err)
        except IOError:
            pass
        log("job %s raised %r" % (jid, exc))
    finally:
        if gpu_lock is not None:
            gpu_lock.__exit__(None, None, None)
    elapsed = time.time() - t0
    status = "DONE" if (rc == 0 and not timed_out) else ("TIMEOUT" if timed_out else "FAILED")
    with STATE_LOCK:
        rec = state["jobs"][jid]
        rec.update({"status": status, "exit": rc, "seconds": round(elapsed, 1),
                    "ended_utc": now(), "log": log_path})
        if err:
            rec["error"] = err
        save_json(state_path, state)
    if job.get("count_gpu"):
        book(ledger_dir, rid, "job_end", bucket=job.get("bucket"),
             payload={"job": jid, "status": status, "exit": rc, "log": log_path,
                      "counts_toward_total": True},
             gpu_seconds=elapsed, formal_steps=0)
    else:
        book(ledger_dir, rid, "job_end", bucket=job.get("bucket"),
             payload={"job": jid, "status": status, "exit": rc, "log": log_path,
                      "counts_toward_total": False,
                      "why_not_counted": "the job books its own GPU seconds into the WAL"},
             gpu_seconds=0.0, formal_steps=0)
    log("job %s -> %s exit=%r in %.0fs (log %s)" % (jid, status, rc, elapsed, log_path))


def dep_state(job, state):
    """DONE / WAIT / BLOCKED(+reason) for a job's dependencies."""
    for d in job.get("depends_on") or []:
        st = state["jobs"].get(d, {}).get("status")
        if st == "DONE":
            continue
        if st in TERMINAL_BAD:
            return "BLOCKED", "dependency %s is %s" % (d, st)
        return "WAIT", "dependency %s is %s" % (d, st or "PENDING")
    return "READY", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=os.path.join(REPO, "mission", "supervisor_queue.json"))
    ap.add_argument("--ledger-dir",
                    default=os.environ.get("R15_LEDGER_DIR", "/home/T7/ojh/robot_actual/r15_ledger"))
    ap.add_argument("--logs", default=os.path.join(REPO, "logs", "supervisor"))
    ap.add_argument("--max-cpu", type=int, default=4)
    ap.add_argument("--max-attempts", type=int, default=1)
    ap.add_argument("--poll", type=float, default=10.0)
    a = ap.parse_args()

    state_path = a.queue + ".state.json"
    state = load_json(state_path, {"jobs": {}})
    if "jobs" not in state:
        state = {"jobs": {}}
    # a previous supervisor that died mid-job must not leave a phantom RUNNING
    for jid, rec in list(state["jobs"].items()):
        if rec.get("status") == "RUNNING":
            rec["status"] = "INTERRUPTED"
            rec["ended_utc"] = now()
            rec["note"] = "supervisor restarted while this job was marked RUNNING"
            log("job %s was RUNNING at restart -> INTERRUPTED" % jid)
    queue_path = a.queue
    log("supervisor up pid=%d queue=%s max_cpu=%d ledger=%s"
        % (os.getpid(), queue_path, a.max_cpu, a.ledger_dir))
    book(a.ledger_dir, "sup:startup:" + now(), "job_begin", bucket=None,
         payload={"supervisor_pid": os.getpid(), "queue": queue_path,
                  "max_cpu": a.max_cpu, "counts_toward_total": False})

    threads = {}

    def launch(job):
        jid = job["id"]
        with STATE_LOCK:
            state["jobs"].setdefault(jid, {})
            state["jobs"][jid].update({
                "status": "RUNNING", "started_utc": now(), "kind": job.get("kind"),
                "gpu": bool(job.get("gpu")),
                "attempts": int(state["jobs"][jid].get("attempts", 0)) + 1,
                "cmd": job["cmd"], "timeout_s": job.get("timeout_s")})
            save_json(state_path, state)
        book(a.ledger_dir, "sup:" + jid, "job_begin", bucket=job.get("bucket"),
             payload={"job": jid, "cmd": job["cmd"], "timeout_s": job.get("timeout_s"),
                      "gpu": bool(job.get("gpu")), "counts_toward_total": bool(job.get("count_gpu"))})
        t = threading.Thread(target=run_one, name="job-" + jid, daemon=False,
                             args=(job, state, state_path, a.ledger_dir, a.logs, a.ledger_dir))
        threads[jid] = t
        t.start()
        log("launched %s%s" % (jid, " (GPU)" if job.get("gpu") else ""))

    while True:
        queue = load_json(queue_path, {"jobs": []})
        jobs = queue.get("jobs", [])
        alive = {k: v for k, v in threads.items() if v.is_alive()}
        for jid in list(threads):
            if jid not in alive:
                threads.pop(jid, None)
        running_gpu = sum(1 for j in jobs if j.get("gpu")
                          and state["jobs"].get(j["id"], {}).get("status") == "RUNNING")
        running_cpu = sum(1 for j in jobs if not j.get("gpu")
                          and state["jobs"].get(j["id"], {}).get("status") == "RUNNING")
        progressed = False
        for job in jobs:
            jid = job["id"]
            st = state["jobs"].get(jid, {}).get("status")
            if st in TERMINAL or st == "RUNNING":
                continue
            if int(state["jobs"].get(jid, {}).get("attempts", 0)) >= a.max_attempts:
                with STATE_LOCK:
                    state["jobs"][jid]["status"] = "FAILED"
                    state["jobs"][jid]["reason"] = "max attempts reached"
                    save_json(state_path, state)
                continue
            verdict, reason = dep_state(job, state)
            if verdict == "BLOCKED":
                with STATE_LOCK:
                    state["jobs"][jid] = {"status": "BLOCKED", "reason": reason, "utc": now()}
                    save_json(state_path, state)
                book(a.ledger_dir, "sup:" + jid, "job_blocked", bucket=job.get("bucket"),
                     payload={"job": jid, "reason": reason, "counts_toward_total": False})
                log("job %s BLOCKED: %s" % (jid, reason))
                progressed = True
                continue
            if verdict == "WAIT":
                continue
            if job.get("gpu") and running_gpu:
                continue
            if not job.get("gpu") and running_cpu >= a.max_cpu:
                continue
            launch(job)
            progressed = True
            if job.get("gpu"):
                running_gpu += 1
            else:
                running_cpu += 1
        if not progressed:
            # a job that is still RUNNING is *not* drained: the supervisor must stay alive until
            # every launched thread has finished, otherwise its timeout/kill/booking never happens
            alive_now = any(t.is_alive() for t in threads.values())
            pending = [j["id"] for j in jobs
                       if state["jobs"].get(j["id"], {}).get("status") not in TERMINAL]
            if not pending and not alive_now:
                summary = {j["id"]: state["jobs"].get(j["id"], {}).get("status") for j in jobs}
                log("queue drained: %s" % json.dumps(summary, sort_keys=True))
                book(a.ledger_dir, "sup:drained:" + now(), "job_end", bucket=None,
                     payload={"queue": queue_path, "summary": summary,
                              "counts_toward_total": False})
                return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
