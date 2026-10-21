#!/usr/bin/env python3
"""Push-T (and later 12-task) DEV closed-loop driver: run the OFFICIAL client, record the OFFICIAL verdict.

The mission forbids inventing a success signal, so this tool never scores anything itself: it launches
RoboDojo's own client (`scripts/robodojo.sh client`), then reads the `_result.json` that the official
eval client wrote and parses its `details[].success` flags. Nothing else counts as success.

Budget: the mission caps DEV rollouts at 20 episodes per task, consumed in stages 1 -> 5 -> 20. This
tool enforces the cumulative cap from its own append-only run ledger before it launches anything.

Usage (stage 1 = one episode, the smoke):
    python tools/r15_loop_dev.py --task push_T --policy-name OpenVLA_R42 --port 18765 \
        --episodes 1 --out runs/r15_loop_dev/push_T_stage1.json
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import r15_stats as S  # noqa: E402

DEFAULT_ROOT = "/home/T7/ojh/robot_actual/RoboDojo"
DEV_CAP_PER_TASK = 20
STAGE_CUM = {1: 1, 2: 5, 3: 20}


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_ledger(path):
    recs = []
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def append_ledger(path, rec):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def episodes_used(recs, task):
    """Cumulative DEV episodes actually OBSERVED.

    A run that crashed or never produced an official _result.json (verdict NO_RESULT) must not eat
    the budget - otherwise a broken launch silently spends one of the 20 allowed rollouts.
    """
    return sum(int(r.get("episodes_observed") or 0) for r in recs
               if r.get("task") == task and r.get("stage") in STAGE_CUM
               and r.get("verdict") not in (None, "NO_RESULT"))


def build_cmd(args):
    """The official client command.

    The Isaac Sim client runs INSIDE the rd_setup container (network mode host; the container mounts
    only RoboDojo -> /workspace/RoboDojo, and the r11 repo is NOT visible there). Running it on the
    host would need an isaacsim python this repo does not pin, so the container is the
    provenance-clean path: pass --docker rd_setup and the same arguments run there. eval_result lives
    inside the bind mount, so result parsing still happens on the host.
    """
    script_root = args.root_inner if args.docker else args.root
    inner = ["bash", os.path.join(script_root, "scripts", "robodojo.sh"), "client",
             "--dataset", args.dataset,
             "--task", args.task,
             "--env-cfg", args.env_cfg,
             "--policy-name", args.policy_name,
             "--policy-host", args.host,
             "--policy-port", str(args.port),
             "--eval-num", str(args.episodes),
             "--seed", str(args.seed),
             "--env-gpu", str(args.env_gpu),
             "--ckpt", args.ckpt_label,
             "--action-type", args.action_type]
    if args.docker:
        cmd = ["docker", "exec"]
        for kv in (args.docker_env or []):
            cmd += ["-e", kv]
        cmd += [args.docker, "bash", "-lc",
                "cd %s && %s" % (args.root_inner, " ".join(inner))]
        return cmd
    return inner


def newest_result(root, task, policy):
    """Newest _result.json the official client wrote for (task, policy)."""
    pats = [os.path.join(root, "eval_result", task, "**", "_result.json"),
            os.path.join(root, "eval_result", "**", task, "**", "_result.json")]
    found = []
    for p in pats:
        found.extend(glob.glob(p, recursive=True))
    found = sorted(set(found), key=lambda p: os.path.getmtime(p))
    return found[-1] if found else None


def parse_result(path):
    with open(path) as fh:
        d = json.load(fh)
    details = d.get("details") or d
    entries = []
    if isinstance(details, dict):
        items = details.items()
    else:
        items = enumerate(details or [])
    for key, val in items:
        if isinstance(val, dict) and "success" in val:
            entries.append({"layout": str(key), "success": bool(val.get("success")),
                            "score": val.get("score")})
        elif isinstance(val, bool):
            entries.append({"layout": str(key), "success": val, "score": None})
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--policy-name", default="OpenVLA_R42")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--episodes", type=int, required=True, help="episodes in THIS stage (EVAL_NUM)")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="host path of the RoboDojo checkout")
    ap.add_argument("--root-inner", default="/workspace/RoboDojo",
                    help="the same tree as seen inside --docker")
    ap.add_argument("--docker", default="",
                    help="run the official client inside this container (e.g. rd_setup); empty = host")
    ap.add_argument("--docker-env", action="append", default=[],
                    help="KEY=VALUE passed with docker exec -e (e.g. SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0)")
    ap.add_argument("--dataset", default="RoboDojo")
    ap.add_argument("--env-cfg", default="arx_x5")
    ap.add_argument("--env-gpu", default="0")
    ap.add_argument("--ckpt-label", default="external")
    ap.add_argument("--action-type", default="joint",
                    help="the env accepts joint or ee; our policy emits ABSOLUTE JOINT targets")
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ledger = a.ledger or os.path.join(os.path.dirname(os.path.abspath(a.out)),
                                      "loop_dev_ledger.jsonl")
    recs = load_ledger(ledger)
    used = episodes_used(recs, a.task)
    cap_note = None
    if used + a.episodes > DEV_CAP_PER_TASK:
        cap_note = ("REFUSED: %s already used %d of %d DEV episodes; asking for %d more"
                    % (a.task, used, DEV_CAP_PER_TASK, a.episodes))
    if a.stage >= 2 and a.episodes != STAGE_CUM[a.stage] - STAGE_CUM[a.stage - 1]:
        cap_note = cap_note or ("REFUSED: stage %d must add exactly %d episodes (1->5->20 cumulative)"
                                % (a.stage, STAGE_CUM[a.stage] - STAGE_CUM[a.stage - 1]))
    if cap_note:
        print(cap_note)
        return 3

    cmd = build_cmd(a)
    run_id = "r15loop-%s-s%d-%s" % (a.task, a.stage, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    env = dict(os.environ)
    env["ROBODOJO_RUN_ID"] = run_id
    rec = {"utc_start": utcnow(), "task": a.task, "policy_name": a.policy_name, "stage": a.stage,
           "episodes_requested": a.episodes, "episodes_used_before": used, "seed": a.seed,
           "robodojo_run_id": run_id, "cmd": " ".join(cmd), "action_type": a.action_type,
           "ckpt_label": a.ckpt_label, "root": a.root}
    if a.dry_run:
        rec["dry_run"] = True
        print(json.dumps(rec, indent=1))
        return 0

    t0 = time.time()
    rec["utc_end"] = None
    timed_out = False
    try:
        p = subprocess.run(cmd, cwd=a.root, env=env, capture_output=True, text=True,
                           timeout=a.timeout_s)
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as exc:
        # Keep the partial client output: a timeout is the most informative failure there is
        # (it means the policy is slower than the simulator's episode budget), and losing it
        # would leave a verdict with no evidence.
        timed_out = True
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else exc.stdout
        err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else exc.stderr
        rc = None
    rec["exit"] = rc
    rec["timed_out"] = timed_out
    rec["wall_clock_s"] = round(time.time() - t0, 1)
    rec["stdout_tail"] = (out or "")[-4000:]
    rec["stderr_tail"] = (err or "")[-4000:]
    rec["utc_end"] = utcnow()

    res = newest_result(a.root, a.task, a.policy_name)
    rec["result_json"] = res
    if res:
        try:
            entries = parse_result(res)
            n = len(entries)
            k = sum(1 for e in entries if e["success"])
            lo, hi = S.wilson_ci(k, n) if n else (float("nan"), float("nan"))
            rec["entries"] = entries
            rec["episodes_observed"] = n
            rec["successes"] = k
            rec["success_rate"] = (k / n) if n else None
            rec["wilson_95_ci"] = [lo, hi]
            if n != a.episodes:
                rec["warn"] = ("asked for %d episodes but the official result has %d (the client "
                               "clamps EVAL_NUM to the task's eval_nums, and unstable envs are "
                               "excluded from the denominator)" % (a.episodes, n))
        except Exception as exc:                     # noqa: BLE001 - record, never crash silently
            rec["parse_error"] = repr(exc)
    else:
        rec["parse_error"] = "no _result.json was produced"

    if rec.get("successes"):
        rec["verdict"] = "OFFICIAL_SUCCESS"
    elif rec.get("episodes_observed"):
        rec["verdict"] = "OFFICIAL_FAILURE"
    elif timed_out:
        rec["verdict"] = "TIMEOUT_NO_RESULT"
    else:
        rec["verdict"] = "NO_RESULT"
    append_ledger(ledger, rec)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: rec.get(k) for k in
                      ("task", "stage", "exit", "episodes_observed", "successes",
                       "success_rate", "wilson_95_ci", "verdict", "result_json")}, indent=1))
    return 0 if rec["verdict"].startswith("OFFICIAL") else 4


if __name__ == "__main__":
    sys.exit(main())