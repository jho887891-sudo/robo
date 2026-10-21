#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2: the ONE shared multi-task trainer for the 7B OpenVLA-R42 LoRA stack.

This is the single entrypoint that trains ONE LoRA r=32 all-linear adapter + ONE 42D projector on
TWO or more structurally identical ARX-X5 tasks (push_T / stack_bowls / sweep_blocks).  It is
additive: nothing here modifies r11_train_pushT_r42.py, r15_ledger.py or r15_resume.py - it reuses
them.  The single-task R15 path keeps working unchanged.

What is task-conditioned, and how
---------------------------------
  * PROMPT     : one real instruction string per task (the /instruction stored in every episode of
                 that task, see training/m2_multitask_data.TASK_PROMPTS), wrapped in the same
                 "What action should the robot take to <instr>?" turn the R15 path uses.
                 assert_prompts_distinct() refuses a task set whose prompts/token ids collapse.
  * NORMALIZER : per task, from the versioned multi-task payload of training/m2_task_normalizer.py
                 (q01/q99 = that task's velocity-residual quantiles drive the 256-bin codec).
                 payload_for(payload, task) is called per row; there is NO global normalizer and
                 no silent fallback.  normalizer_sha256 covers every task at once.
  * FEATURE    : 42D [S,V,Acc] from that task's own NormalizationContract (contract_for()).
  * TASK ID    : the frozen, never-renumbered M2 task id (push_T=0, stack_bowls=1,
                 sweep_blocks=2) - the lookup key for the normalizer and for the task token.

Balanced sampling (documented, reproducible, resumable)
-------------------------------------------------------
plan_task_mix() implements M2_MIX_POLICY = "proportional_to_train_frames_with_floor":
  1. raw weight of a task = its TRAIN frame count / total TRAIN frames;
  2. any task below --task-floor (default 0.30) is clamped UP to the floor and the remaining mass
     is redistributed over the other tasks proportionally to their frames (repeated until every
     weight is >= floor); weights are then normalised to sum to 1;
  3. the horizon is one epoch (sum of the per-task TRAIN frame counts) and the per-task slot counts
     are the largest-remainder apportionment of (horizon * weight) - exact integers summing to the
     horizon;
  4. the slot ORDER is a deterministic greedy fair interleave ("largest deficit first", ties by
     task id), so the running mix never drifts far from the target mix.
Both the schedule and the slot frame list are pure functions of (tasks, frame counts, floor, seed),
so the whole data order is reproducible; schedule_sha256 and the FrameSampler order_hash are
written into the evidence and into every checkpoint, and a resume whose frame list differs fails
closed (r15_resume.FrameSampler.from_state compares frames_sha256).

Hard caps
---------
CAPS = {1 task: 1000, 2 tasks: 2000, 3 tasks: 3000} cumulative multi-task optimizer steps per
stage (--max-tasks-stage chooses the row).  A target above the cap, a resume whose start is already
above it, or a task set larger than the stage refuses BEFORE any GPU work starts.  The R15 ledger
(training/r15_ledger.py, bucket TWO_TASK_PILOT) is the second, append-only accounting of the same
steps; every step is booked into the WAL as it happens.

Everything below the model boundary (task mix, caps, resume guards, teacher-forced CE, evidence
shape, dev frame selection) is pure numpy and CPU-testable with fake logits; the 7B model is only
touched inside run().
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("training", "diagnostics", "code/xpolicylab/policy/OpenVLA_Pure",
             "code/xpolicylab/policy/OpenVLA_R42"):
    _p = os.path.join(REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:                                                 # imported as a top-level module
    import m2_task_normalizer as TN                   # noqa: E402  (resolves m2_multitask_data)
except ImportError:                                  # imported as training.m2_train_shared
    from training import m2_task_normalizer as TN      # noqa: E402

M2 = TN.M2                    # the SAME m2_multitask_data module object the normalizer uses

RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
LEDGER_DIR = os.environ.get("R15_LEDGER_DIR", "/home/T7/ojh/robot_actual/r15_ledger")
SPLITS = os.environ.get("M2_SPLITS", os.path.join(REPO, "data", "m2_splits.json"))
NORMALIZER_JSON = os.environ.get("M2_NORMALIZER_JSON",
                                 "/home/dgut/r15_lanes/M2T/m2_task_normalizer.json")
EVIDENCE_DIR = os.environ.get("M2_EVIDENCE_DIR", "/home/dgut/r15_lanes/M2T")

SCHEMA_EVIDENCE = "m2-train-evidence-v1"
SCHEMA_MIX = "m2-task-mix-v1"
MIX_POLICY = "proportional_to_train_frames_with_floor"
STEPS_PER_TASK_STAGE = 1000
CAPS = {1: 1000, 2: 2000, 3: 3000}          # cumulative multi-task optimizer steps per stage
DEFAULT_TASK_FLOOR = 0.30
DEFAULT_DEV_FRAMES_PER_TASK = 8
N_ACTIONS = 14
N_SUPERVISED = N_ACTIONS + 1                # 14 action tokens + STOP(2)
DEFAULT_BUCKET = "TWO_TASK_PILOT"
INSTR_TEMPLATE = "What action should the robot take to %s?"


class M2TrainError(RuntimeError):
    """Every fail-closed M2 trainer violation (wrong task, wrong cap, wrong resume, ...)."""


# ------------------------------------------------------------------------------- utilities
def log(msg):
    print("[%s] %s" % (datetime.now(timezone.utc).strftime("%H:%M:%S"), msg), flush=True)


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp-%d" % os.getpid()
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


# --------------------------------------------------------------------- task set and hard caps
def parse_tasks(text):
    """'push_T,stack_bowls' -> validated task list in frozen task-id order (fail closed)."""
    if isinstance(text, (list, tuple)):
        items = list(text)
    else:
        items = [x.strip() for x in str(text).split(",")]
    tasks = []
    for raw in items:
        if not raw:
            continue
        if raw not in M2.TASK_IDS:
            raise M2TrainError("unknown task %r; the M2 stage knows %s" % (raw, list(M2.TASKS)))
        if raw in tasks:
            raise M2TrainError("task %r listed twice; the mix is per task, not per lane" % (raw,))
        tasks.append(raw)
    if not tasks:
        raise M2TrainError("--tasks is empty")
    return sorted(tasks, key=lambda t: M2.TASK_IDS[t])


def prompts_for(tasks, include_task_token=False):
    """Per-task prompt text (real instruction string, optionally preceded by the task token)."""
    out = {}
    for t in tasks:
        built = M2.build_prompt(t, include_token=include_task_token)
        out[t] = INSTR_TEMPLATE % built["prompt"]
    return out


def assert_prompts_distinct(tasks, prompts=None, token_ids=None):
    """The task conditioning must actually differ per task (string and, if given, token level)."""
    tasks = parse_tasks(tasks)
    prompts = prompts_for(tasks) if prompts is None else dict(prompts)
    missing = [t for t in tasks if t not in prompts]
    if missing:
        raise M2TrainError("no prompt for task(s) %s" % missing)
    vals = [prompts[t] for t in tasks]
    if any(not isinstance(v, str) or len(v.strip()) < 5 for v in vals):
        raise M2TrainError("every task prompt must be a non-empty instruction string: %r" % (vals,))
    if len(set(vals)) != len(vals):
        raise M2TrainError("two tasks share the same prompt, so the model cannot be conditioned: "
                           "%r" % (vals,))
    for t in tasks:
        if M2.task_prompt(t) not in prompts[t]:
            raise M2TrainError("%r: prompt does not contain the frozen instruction %r"
                               % (prompts[t], M2.task_prompt(t)))
    if token_ids is not None:
        ids = [tuple(int(x) for x in token_ids[t]) for t in tasks]
        if len(set(ids)) != len(ids):
            raise M2TrainError("two tasks tokenise to the same prompt ids")
    return {"tasks": tasks, "prompts": prompts,
            "prompt_sha256": {t: hashlib.sha256(prompts[t].encode()).hexdigest() for t in tasks}}


def cumulative_step_cap(max_tasks_stage):
    """Cumulative multi-task optimizer-step cap of a stage (1->1000, 2->2000, 3->3000)."""
    n = int(max_tasks_stage)
    if n not in CAPS:
        raise M2TrainError("--max-tasks-stage must be one of %s, got %r"
                           % (sorted(CAPS), max_tasks_stage))
    return int(CAPS[n])


def assert_stage_caps(tasks, max_tasks_stage, target_global_step, start_global_step=0):
    """Refuse BEFORE any GPU work: too many tasks for the stage, or too many cumulative steps."""
    tasks = parse_tasks(tasks)
    n = int(max_tasks_stage)
    cap = cumulative_step_cap(n)
    if len(tasks) > n:
        raise M2TrainError("%d tasks (%s) exceed --max-tasks-stage %d" % (len(tasks), tasks, n))
    target = int(target_global_step)
    start = int(start_global_step)
    if target <= 0:
        raise M2TrainError("--target-global-step must be a positive cumulative step count")
    if start < 0:
        raise M2TrainError("start_global_step must be >= 0, got %d" % start)
    if start > cap:
        raise M2TrainError("the checkpoint is already at step %d, above the stage cumulative cap "
                           "%d for %d task(s): refusing to continue" % (start, cap, n))
    if target > cap:
        raise M2TrainError("target global step %d exceeds the stage cumulative cap %d for %d "
                           "task(s) (2 tasks -> 2000 multi-task optimizer steps): refusing to "
                           "start" % (target, cap, n))
    if start >= target:
        return {"cap": cap, "max_tasks_stage": n, "already_at_target": True,
                "steps_requested": 0, "tasks": tasks}
    return {"cap": cap, "max_tasks_stage": n, "already_at_target": False,
            "steps_requested": target - start, "tasks": tasks}


# ------------------------------------------------------------------------------- task mix
def compute_task_weights(tasks, frames_per_task, floor=DEFAULT_TASK_FLOOR):
    """Proportional-to-frames weights with a hard floor per task (fixed-point, deterministic)."""
    tasks = parse_tasks(tasks)
    floor = float(floor)
    if not (0.0 < floor < 1.0):
        raise M2TrainError("--task-floor must be in (0,1), got %r" % (floor,))
    if floor * len(tasks) > 1.0 + 1e-12:
        raise M2TrainError("floor %.3f x %d tasks exceeds 1.0" % (floor, len(tasks)))
    counts = {}
    for t in tasks:
        n = int(frames_per_task.get(t, 0))
        if n <= 0:
            raise M2TrainError("task %s has no TRAIN frames" % t)
        counts[t] = n
    total = float(sum(counts.values()))
    raw = {t: counts[t] / total for t in tasks}
    weights = dict(raw)
    for _ in range(len(tasks) + 1):
        below = [t for t in tasks if weights[t] < floor - 1e-12]
        if not below:
            break
        for t in below:
            weights[t] = floor
        free = [t for t in tasks if weights[t] > floor + 1e-12]
        rest = 1.0 - floor * len(below)
        mass = sum(raw[t] for t in free)
        for t in free:
            weights[t] = rest * raw[t] / mass if mass > 0 else rest / max(1, len(free))
    weights = {t: weights[t] / sum(weights.values()) for t in tasks}
    if any(weights[t] < floor - 1e-9 for t in tasks):
        raise M2TrainError("floor not satisfiable for %s (weights %s)" % (tasks, weights))
    return raw, weights, counts


def apportion(horizon, weights, tasks):
    """Largest-remainder apportionment: exact integer counts summing to the horizon."""
    horizon = int(horizon)
    if horizon <= 0:
        raise M2TrainError("horizon must be > 0")
    exact = {t: horizon * float(weights[t]) for t in tasks}
    base = {t: int(np.floor(exact[t])) for t in tasks}
    remainder = horizon - sum(base.values())
    if remainder < 0:
        raise M2TrainError("apportionment is negative (%d)" % remainder)
    order = sorted(tasks, key=lambda t: (-(exact[t] - base[t]), M2.TASK_IDS[t]))
    for t in order[:remainder]:
        base[t] += 1
    if sum(base.values()) != horizon:
        raise M2TrainError("apportionment does not sum to the horizon")
    return base


def interleave_schedule(tasks, counts, horizon):
    """Deterministic fair interleave: at each slot take the task furthest behind its target."""
    tasks = list(tasks)
    horizon = int(horizon)
    if sum(int(counts[t]) for t in tasks) != horizon:
        raise M2TrainError("counts do not sum to the horizon")
    used = {t: 0 for t in tasks}
    schedule = []
    for k in range(horizon):
        best, best_deficit = None, None
        for t in tasks:
            target = (k + 1) * float(counts[t]) / horizon
            deficit = target - used[t]
            if best is None or deficit > best_deficit + 1e-12 or (
                    abs(deficit - best_deficit) <= 1e-12 and M2.TASK_IDS[t] < M2.TASK_IDS[best]):
                best, best_deficit = t, deficit
        used[best] += 1
        schedule.append(best)
    if any(used[t] != int(counts[t]) for t in tasks):
        raise M2TrainError("interleave lost the apportioned counts: %s vs %s" % (used, counts))
    return schedule


def plan_task_mix(tasks, frames_per_task, floor=DEFAULT_TASK_FLOOR, horizon=None, seed=0):
    """The documented, reproducible balanced multi-task sampling plan."""
    tasks = parse_tasks(tasks)
    raw, weights, counts = compute_task_weights(tasks, frames_per_task, floor=floor)
    h = int(horizon) if horizon else int(sum(counts.values()))
    slots = apportion(h, weights, tasks)
    schedule = interleave_schedule(tasks, slots, h)
    mix = {"schema": SCHEMA_MIX, "policy": MIX_POLICY,
           "policy_description": ("weight_t = max(floor, train_frames_t / total_train_frames), "
                                  "renormalised; counts = largest-remainder(horizon * weight_t); "
                                  "order = deterministic largest-deficit interleave"),
           "tasks": tasks, "floor": float(floor), "horizon": h, "seed": int(seed),
           "frames_per_task": {t: int(frames_per_task[t]) for t in tasks},
           "raw_weights": {t: float(raw[t]) for t in tasks},
           "weights": {t: float(weights[t]) for t in tasks},
           "counts": {t: int(slots[t]) for t in tasks},
           "schedule": schedule,
           "schedule_sha256": sha256_json(schedule),
           "expected_steps_fraction": {t: float(slots[t]) / float(h) for t in tasks}}
    return mix


def assert_mix_reproducible(mix, frames_per_task, floor=None, seed=None):
    """Recompute the mix and refuse if the recorded one is not reproducible from the same inputs."""
    again = plan_task_mix(mix["tasks"], frames_per_task,
                          floor=mix["floor"] if floor is None else floor,
                          horizon=mix["horizon"], seed=mix["seed"] if seed is None else seed)
    if again["schedule_sha256"] != mix["schedule_sha256"]:
        raise M2TrainError("task mix is not reproducible: recorded %s, recomputed %s"
                           % (mix["schedule_sha256"], again["schedule_sha256"]))
    return True


def build_slot_frames(mix, frames_by_task):
    """The ordered (qualified_episode, t) list of one epoch, one slot per schedule entry.

    Each task walks its own TRAIN frames in a stable order and CYCLES if the floor forces more
    slots than the task has distinct frames (documented; only possible when the floor binds).
    """
    counters = {t: 0 for t in mix["tasks"]}
    out = []
    for t in mix["schedule"]:
        frames = frames_by_task.get(t) or []
        if not frames:
            raise M2TrainError("task %s has no TRAIN frames for the slot list" % t)
        i = counters[t]
        ep, tt = frames[i % len(frames)]
        counters[t] = i + 1
        out.append((M2.qualified(t, ep), int(tt)))
    for t in mix["tasks"]:
        if counters[t] != int(mix["counts"][t]):
            raise M2TrainError("%s: %d slots consumed, %d planned"
                               % (t, counters[t], mix["counts"][t]))
    if len(out) != int(mix["horizon"]):
        raise M2TrainError("slot list length %d != horizon %d" % (len(out), mix["horizon"]))
    return out


def slot_task(slot):
    qualified_ep, _t = slot
    task, _ep = M2.split_qualified(qualified_ep)
    return task


def make_sampler(slot_frames, seed):
    """Reuse the R15 resumable FrameSampler verbatim over the QUALIFIED slot list."""
    import r15_resume as RS
    return RS.FrameSampler([(q, int(t)) for q, t in slot_frames], int(seed))


def count_schedule_prefix(schedule, position):
    """Per-task slots consumed after that many draws (drives per-task logging and evidence)."""
    counts = {}
    for t in schedule[:int(position)]:
        counts[t] = counts.get(t, 0) + 1
    return counts


# ---------------------------------------------------------------------------------- frames
def task_train_frames(index, split, task):
    """Every history-safe TRAIN frame of one task as [(bare_episode, t)], stable order."""
    train, _dev = TN.split_episodes_for_task(split, task)
    frames = []
    for ep in train:
        T = int(index["frames"][ep])
        for t in range(M2.MIN_HISTORY_FRAMES, T - 1):
            frames.append((ep, int(t)))
    if not frames:
        raise M2TrainError("%s: no history-safe TRAIN frames" % task)
    return frames


def dev_frames_for_task(index, split, task, n_frames, seed):
    """A deterministic DEV frame list: spread over the task's DEV episodes, same for every eval."""
    _train, dev = TN.split_episodes_for_task(split, task)
    n = int(n_frames)
    if n <= 0:
        raise M2TrainError("--dev-frames-per-task must be > 0")
    if n > len(dev):
        raise M2TrainError("%s: only %d DEV episodes for %d DEV frames" % (task, len(dev), n))
    rs = np.random.RandomState(int(seed) + 7919 * M2.task_id(task))
    order = list(rs.permutation(len(dev)))
    picked = [dev[i] for i in order[:n]]
    out = []
    for ep in picked:
        T = int(index["frames"][ep])
        lo, hi = M2.MIN_HISTORY_FRAMES, T - 2
        if hi <= lo:
            raise M2TrainError("%s/%s is too short for a DEV frame (T=%d)" % (task, ep, T))
        t = int(lo + (hi - lo) // 2)          # exact middle: stable, no RNG drift
        out.append((ep, t))
    return out


def dev_frames_sha256(frames):
    return sha256_json([[str(ep), int(t)] for ep, t in frames])
# ----------------------------------------------------------------------- labels and CE math
def task_contract(payload, task):
    return TN.contract_for(payload, task)


def row_for(payload, task, state, action, t, episode=None):
    """42D feature + 14D residual with THAT TASK's normalizer and a task-tagged row."""
    contract = task_contract(payload, task)
    return M2.build_row(state, action, int(t), contract, task, episode=episode)


def labels_for(payload, task, prompt_ids, residual):
    """The R42 causal label block, encoded with THAT TASK's residual quantiles."""
    import r42_adapter as R
    ctx = TN.decode_context(payload, task)
    return R.build_labels(prompt_ids, np.asarray(residual, np.float32), ctx["q01"], ctx["q99"])


def _softmax_rows(x):
    x = np.asarray(x, np.float64)
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def token_ce_from_logits(logits, labels, n_prompt, n_supervised=N_SUPERVISED):
    """Teacher-forced CE of the n_supervised positions, in NATS, from raw logits (numpy only).

    HF causal shift: the label at position p is predicted from the logits at position p-1, so the
    15 supervised positions consume logits[n_prompt-1 ... n_prompt+13].
    """
    lg = np.asarray(logits, np.float32)
    if lg.ndim == 3:
        if lg.shape[0] != 1:
            raise M2TrainError("token_ce_from_logits expects one sequence, got %s" % (lg.shape,))
        lg = lg[0]
    lab = np.asarray(labels).reshape(-1)
    n_prompt = int(n_prompt)
    if lg.ndim != 2:
        raise M2TrainError("logits must be [L,V], got %s" % (lg.shape,))
    if len(lab) != lg.shape[0]:
        raise M2TrainError("labels length %d != logits length %d" % (len(lab), lg.shape[0]))
    if n_prompt < 1 or n_prompt + n_supervised > len(lab):
        raise M2TrainError("prompt length %d cannot carry %d supervised positions (L=%d)"
                           % (n_prompt, n_supervised, len(lab)))
    rows = np.arange(n_prompt - 1, n_prompt - 1 + n_supervised)
    tgts = lab[n_prompt:n_prompt + n_supervised]
    logits_rows = lg[rows]
    mx = logits_rows.max(axis=1, keepdims=True)
    lse = np.log(np.exp(logits_rows - mx).sum(axis=1)) + mx[:, 0]
    valid = tgts != -100
    picked = logits_rows[np.arange(len(tgts)), np.clip(tgts, 0, logits_rows.shape[1] - 1)]
    ce = (lse - picked)
    if not valid.any():
        raise M2TrainError("no supervised positions in the label block")
    if not np.isfinite(ce[valid]).all():
        raise M2TrainError("non-finite teacher-forced CE")
    return {"ce": float(ce[valid].mean()), "per_position": [float(x) for x in ce[valid]],
            "n_supervised": int(valid.sum())}


def action_window_probs(logits, n_prompt, n_actions=N_ACTIONS, n_bins=256):
    """Per-dim 256-bin distribution over the legal action window + its argmax bin."""
    import r42_adapter as R
    lg = np.asarray(logits, np.float32)
    if lg.ndim == 3:
        if lg.shape[0] != 1:
            raise M2TrainError("action_window_probs expects one sequence")
        lg = lg[0]
    n_prompt = int(n_prompt)
    rows = np.arange(n_prompt - 1, n_prompt - 1 + n_actions)
    if rows[-1] >= lg.shape[0]:
        raise M2TrainError("not enough logits for %d action positions" % n_actions)
    lo = R.TOKEN_BEGIN_IDX + 1
    hi = lo + n_bins
    win = lg[rows][:, lo:hi]
    if win.shape != (n_actions, n_bins):
        raise M2TrainError("action window shape %s != (%d,%d)"
                           % (win.shape, n_actions, n_bins))
    probs = _softmax_rows(win).astype(np.float32)
    return {"pred_bins": probs.argmax(axis=1).astype(np.int64), "pred_probs": probs}


def bins_from_residual(payload, task, residual):
    """The codec bins of a residual (the ground truth the CE is measured against)."""
    import r42_adapter as R
    ctx = TN.decode_context(payload, task)
    _idx, ids = R.token_ids_for_residual(np.asarray(residual, np.float32), ctx["q01"], ctx["q99"])
    return np.asarray(ids, np.int64) - R.TOKEN_BEGIN_IDX - 1


def ce_window_from_probs(probs, bins_true):
    """CE (nats) of the true bins under the RENORMALISED 256-way window distribution.

    This is the metric the P6 gate compares against the no-op / constant-velocity baselines, so
    both sides live in the same 256-bin space (the full-vocabulary CE incl. STOP is recorded as
    ce_full and is what the optimizer minimises).
    """
    p = np.asarray(probs, np.float64)
    b = np.asarray(bins_true, np.int64).reshape(-1)
    if p.ndim != 2 or p.shape[0] != b.shape[0]:
        raise M2TrainError("ce_window_from_probs shapes %s vs %s" % (p.shape, b.shape))
    if b.min() < 0 or b.max() >= p.shape[1]:
        raise M2TrainError("true bin outside the 256-bin window: %r" % (b.tolist(),))
    clicked = np.clip(p[np.arange(len(b)), b], 1e-12, 1.0)
    return float(-np.log(clicked).mean())


def evaluate_frame(predict, payload, frame_task, episode, t, prompt_tasks):
    """One DEV frame under EVERY task prompt (counterfactual), so task routing can be tested.

    predict(prompt_task, frame_task, episode, t) -> {logits, labels, n_prompt, residual}.
    The CE numbers are the CORRECT-prompt ones; the counterfactual distributions are what prove
    that the task conditioning actually moves the output (gate P6.routing).
    """
    prompt_tasks = list(prompt_tasks)
    out = predict(frame_task, frame_task, episode, int(t))
    n_prompt = int(out["n_prompt"])
    ce = token_ce_from_logits(out["logits"], out["labels"], n_prompt)
    win = action_window_probs(out["logits"], n_prompt)
    if "residual" not in out:
        raise M2TrainError("predict() must return the row residual so the true bins can be "
                           "recomputed")
    bins_true = bins_from_residual(payload, frame_task, out["residual"])
    x = {}
    for pt in prompt_tasks:
        if pt == frame_task:
            x[pt] = {"pred_probs": win["pred_probs"], "pred_bins": win["pred_bins"],
                     "ce_window": ce_window_from_probs(win["pred_probs"], bins_true)}
            continue
        o2 = predict(pt, frame_task, episode, int(t))
        w2 = action_window_probs(o2["logits"], o2["n_prompt"])
        x[pt] = {"pred_probs": w2["pred_probs"], "pred_bins": w2["pred_bins"],
                 "ce_window": ce_window_from_probs(w2["pred_probs"], bins_true)}
    return {"task": frame_task, "episode": episode, "t": int(t), "ce": float(ce["ce"]),
            "ce_window": ce_window_from_probs(win["pred_probs"], bins_true),
            "pred_bins": win["pred_bins"], "pred_probs": win["pred_probs"],
            "bins_true": bins_true, "counterfactual": x}


def evaluate_dev_ce(predict, payload, tasks, dev_frames, log_fn=None):
    """Per-task DEV teacher-forced CE + per-frame distributions (model boundary is the callable)."""
    tasks = parse_tasks(tasks)
    res = {}
    for task in tasks:
        frames = dev_frames[task]
        entries = [evaluate_frame(predict, payload, task, ep, t, tasks) for ep, t in frames]
        ces = [e["ce_window"] for e in entries]
        full = [e["ce"] for e in entries]
        if not ces or not np.isfinite(ces).all() or not np.isfinite(full).all():
            raise M2TrainError("%s: DEV CE is empty or non-finite" % task)
        res[task] = {"frames": entries, "ce_mean": float(np.mean(ces)),
                     "ce_full_mean": float(np.mean(full)), "ce_frames": ces,
                     "frames_sha256": dev_frames_sha256(frames)}
        if log_fn:
            log_fn("DEV %s ce_window=%.4f ce_full=%.4f nats over %d frames"
                   % (task, res[task]["ce_mean"], res[task]["ce_full_mean"], len(ces)))
    return res


def eval_archive_from(entries_by_task):
    """The npz payload: per frame task the frames/CE/true bins, plus the counterfactual prompts."""
    out = {}
    for task, block in entries_by_task.items():
        frames = block["frames"]
        out[task + "/episodes"] = np.array([e["episode"] for e in frames], dtype=object)
        out[task + "/t"] = np.array([int(e["t"]) for e in frames], dtype=np.int64)
        out[task + "/ce"] = np.array([float(e["ce"]) for e in frames], dtype=np.float64)
        out[task + "/ce_window"] = np.array([float(e["ce_window"]) for e in frames],
                                            dtype=np.float64)
        out[task + "/pred_bins"] = np.stack([e["pred_bins"] for e in frames]).astype(np.int64)
        out[task + "/pred_probs"] = np.stack([e["pred_probs"] for e in frames]).astype(np.float32)
        out[task + "/bins_true"] = np.stack([e["bins_true"] for e in frames]).astype(np.int64)
        prompts = sorted(frames[0]["counterfactual"]) if frames else []
        out[task + "/prompt_tasks"] = np.array(prompts, dtype=object)
        for pt in prompts:
            out["x|%s|%s" % (task, pt)] = np.stack(
                [e["counterfactual"][pt]["pred_probs"] for e in frames]).astype(np.float32)
    return out


def write_eval_archive(path, entries_by_task, data_contract=None):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    arrays = eval_archive_from(entries_by_task)
    if data_contract is not None:
        arrays["data_contract"] = np.array(json.dumps(data_contract, sort_keys=True))
    np.savez_compressed(path, **arrays)
    return sha256_file(path)


def load_eval_archive(path):
    """Read + validate the DEV eval archive written by m2_train_shared.write_eval_archive."""
    if not os.path.isfile(path):
        raise M2TrainError("DEV eval archive not found: %s" % path)
    z = np.load(path, allow_pickle=True)
    tasks = sorted({k.split("/")[0] for k in z.files if "/" in k}, key=lambda t: M2.TASK_IDS[t])
    if not tasks:
        raise M2TrainError("DEV eval archive has no per-task blocks: %s" % path)
    out = {"tasks": tasks, "blocks": {}}
    for task in tasks:
        for k in ("episodes", "t", "ce", "ce_window", "pred_bins", "pred_probs", "bins_true"):
            if task + "/" + k not in z.files:
                raise M2TrainError("DEV eval archive is missing %s/%s" % (task, k))
        eps = [str(x) for x in z[task + "/episodes"]]
        ts = [int(x) for x in z[task + "/t"]]
        ce = np.asarray(z[task + "/ce"], np.float64)
        cew = np.asarray(z[task + "/ce_window"], np.float64)
        bins = np.asarray(z[task + "/pred_bins"], np.int64)
        probs = np.asarray(z[task + "/pred_probs"], np.float32)
        true_bins = np.asarray(z[task + "/bins_true"], np.int64)
        if not (len(eps) == len(ts) == len(ce) == len(cew) == bins.shape[0]
                == probs.shape[0] == true_bins.shape[0]):
            raise M2TrainError("%s: DEV archive block lengths disagree" % task)
        if bins.shape[1:] != (N_ACTIONS,) or probs.shape[1:] != (N_ACTIONS, 256)                 or true_bins.shape[1:] != (N_ACTIONS,):
            raise M2TrainError("%s: DEV archive shapes %s/%s/%s"
                               % (task, bins.shape, probs.shape, true_bins.shape))
        finite = np.isfinite(ce).all() and np.isfinite(cew).all() and np.isfinite(probs).all()
        if not finite:
            raise M2TrainError("%s: DEV archive has non-finite values" % task)
        prompt_tasks = ([str(x) for x in z[task + "/prompt_tasks"]]
                        if task + "/prompt_tasks" in z.files else [task])
        x = {}
        for pt in prompt_tasks:
            key = "x|%s|%s" % (task, pt)
            if key not in z.files:
                raise M2TrainError("DEV eval archive is missing the counterfactual block %s" % key)
            xp = np.asarray(z[key], np.float32)
            if xp.shape != probs.shape:
                raise M2TrainError("%s: counterfactual shape %s != %s"
                                   % (key, xp.shape, probs.shape))
            if not np.isfinite(xp).all():
                raise M2TrainError("%s: counterfactual has non-finite values" % key)
            cew_x = np.asarray([ce_window_from_probs(xp[i], true_bins[i])
                                for i in range(xp.shape[0])], np.float64)
            x[pt] = {"pred_probs": xp, "pred_bins": xp.argmax(axis=2).astype(np.int64),
                     "ce_window": cew_x, "ce_window_mean": float(cew_x.mean())}
        out["blocks"][task] = {"episodes": eps, "t": ts, "ce": ce, "ce_window": cew,
                               "pred_bins": bins, "pred_probs": probs, "bins_true": true_bins,
                               "counterfactual": x, "prompt_tasks": prompt_tasks,
                               "ce_mean": float(cew.mean()),
                               "frames_sha256": dev_frames_sha256(list(zip(eps, ts)))}
    if "data_contract" in z.files:
        out["data_contract"] = str(z["data_contract"])
    return out


# ------------------------------------------------------------------------------- evidence
def evidence_skeleton(args, tasks, payload, prompts, mix, dev_frames, caps, data_contract,
                      start_global_step=0, prior_ce_curves=None):
    return {
        "schema": SCHEMA_EVIDENCE,
        "run_id": None, "utc_start": utcnow(), "utc_end": None,
        "out": os.path.abspath(args.out),
        "tasks": list(tasks),
        "task_ids": {t: M2.TASK_IDS[t] for t in tasks},
        "max_tasks_stage": int(args.max_tasks_stage),
        "target_global_step": int(args.target_global_step),
        "start_global_step": int(start_global_step),
        "caps": caps,
        "task_mix": mix,
        "prompts": prompts["prompts"], "prompt_sha256": prompts["prompt_sha256"],
        "include_task_token": bool(getattr(args, "include_task_token", False)),
        "normalizer_sha256": TN.normalizer_sha256(payload),
        "per_task_normalizer_sha256": {t: TN.sha256_json(payload["tasks"][t]) for t in tasks},
        "normalizer_json": os.path.abspath(args.normalizer),
        "splits_path": os.path.abspath(args.splits),
        "split_sha256": sha256_file(args.splits),
        "dev_frames": {t: {"frames": [[ep, int(tt)] for ep, tt in dev_frames[t]],
                           "frames_sha256": dev_frames_sha256(dev_frames[t])} for t in tasks},
        "ce_curves": {t: list((prior_ce_curves or {}).get(t, [])) for t in tasks},
        "per_task_loss": {t: [] for t in tasks},
        "per_task_loss_summary": {t: {} for t in tasks},
        "steps_per_task": {t: 0 for t in tasks},
        "data_contract": data_contract,
        "train_only": True, "dev_used_for_gradients": False,
        "resume_from": getattr(args, "resume_from", None) or None,
        "eval_archive": None, "eval_archive_sha256": None,
        "ledger": {"bucket": args.bucket, "run_id": None, "steps_booked": 0},
        "log": [],
    }


def refresh_loss_summary(ev, window=10):
    for t, rows in ev["per_task_loss"].items():
        vals = [r["loss"] for r in rows]
        if not vals:
            ev["per_task_loss_summary"][t] = {"n": 0}
            continue
        ev["per_task_loss_summary"][t] = {
            "n": len(vals), "first": vals[0], "last": vals[-1],
            "first_window_mean": float(np.mean(vals[:window])),
            "last_window_mean": float(np.mean(vals[-window:])),
            "min": float(np.min(vals)), "max": float(np.max(vals))}
    return ev


# ------------------------------------------------------------------------- resume guards
def assert_resume_compatible(trainer_state, tasks, normalizer_sha, mix, max_tasks_stage=None):
    """Fail closed unless the checkpoint was trained on EXACTLY this task set / mix / normalizer."""
    tasks = parse_tasks(tasks)
    if not isinstance(trainer_state, dict):
        raise M2TrainError("trainer_state must be a dict")
    ck_tasks = trainer_state.get("tasks")
    if ck_tasks is None:
        raise M2TrainError("the checkpoint records no task set; it is not a multi-task checkpoint")
    ck_tasks = parse_tasks(list(ck_tasks))
    if list(ck_tasks) != list(tasks):
        raise M2TrainError("the checkpoint was trained on tasks %s but this run asks for %s; a "
                           "resume across a different task set is refused" % (ck_tasks, tasks))
    if trainer_state.get("normalizer_sha256") is None:
        raise M2TrainError("the checkpoint records no multi-task normalizer sha")
    if str(trainer_state["normalizer_sha256"]) != str(normalizer_sha):
        raise M2TrainError("the checkpoint multi-task normalizer sha %s != this run %s"
                           % (trainer_state["normalizer_sha256"], normalizer_sha))
    ck_mix = trainer_state.get("task_mix") or {}
    if not ck_mix:
        raise M2TrainError("the checkpoint records no task mix")
    if ck_mix.get("schedule_sha256") != mix.get("schedule_sha256"):
        raise M2TrainError("the checkpoint task-mix schedule %s != this run %s (the data "
                           "order changed; refusing to resume)"
                           % (ck_mix.get("schedule_sha256"), mix.get("schedule_sha256")))
    if max_tasks_stage is not None:
        ck_stage = int(trainer_state.get("max_tasks_stage", 0) or 0)
        if ck_stage and ck_stage > int(max_tasks_stage):
            raise M2TrainError("the checkpoint belongs to a %d-task stage, this run allows %d"
                               % (ck_stage, max_tasks_stage))
    return {"tasks": tasks, "normalizer_sha256": str(normalizer_sha),
            "schedule_sha256": ck_mix.get("schedule_sha256")}


# ------------------------------------------------------------------------------- artefacts
def load_m2_artifacts(tasks, splits_path=None, normalizer_path=None):
    """Load + cross-validate the splits artefact and the multi-task normalizer payload."""
    tasks = parse_tasks(tasks)
    splits_path = splits_path or SPLITS
    normalizer_path = normalizer_path or NORMALIZER_JSON
    if not os.path.isfile(splits_path):
        raise M2TrainError("splits artefact missing: %s" % splits_path)
    with open(splits_path) as fh:
        splits_doc = json.load(fh)
    split_map = splits_doc.get("splits") or {}
    missing = [t for t in tasks if t not in split_map]
    if missing:
        raise M2TrainError("splits artefact %s has no split for %s" % (splits_path, missing))
    dev_lists = {t: [k.split("/", 1)[-1] for k in split_map[t]["dev"]] for t in tasks}
    payload = TN.load_payload(normalizer_path, tasks=tasks, dev_lists=dev_lists)
    for t in tasks:      # the normalizer's fit list must equal the split's TRAIN list, exactly
        want = sorted(k.split("/", 1)[-1] for k in split_map[t]["train"])
        got = sorted(payload["tasks"][t]["fit_episodes"])
        if want != got:
            raise M2TrainError("%s: normalizer was fitted on %d episodes but the split has %d "
                               "TRAIN episodes (first difference: %s vs %s)"
                               % (t, len(got), len(want),
                                  sorted(set(want) - set(got))[:3],
                                  sorted(set(got) - set(want))[:3]))
    return splits_doc, split_map, payload, os.path.abspath(splits_path), \
        os.path.abspath(normalizer_path)


def data_contract_for(tasks, payload, mix, splits_path, cams):
    return {
        "schema": SCHEMA_EVIDENCE,
        "tasks": list(tasks),
        "task_ids": {t: M2.TASK_IDS[t] for t in tasks},
        "normalizer_sha256": TN.normalizer_sha256(payload),
        "per_task_normalizer_sha256": {t: TN.sha256_json(payload["tasks"][t]) for t in tasks},
        "task_mix_schedule_sha256": mix["schedule_sha256"],
        "task_mix_policy": mix["policy"], "task_floor": mix["floor"],
        "split_sha256": sha256_file(splits_path),
        "camera_order": list(M2.CAMERA_ORDER[:int(cams)]),
        "gripper_dims": list(M2.GRIPPER_DIMS), "base_dim": M2.BASE_DIM, "n42": M2.N42,
        "alignment_version": M2.ALIGNMENT_VERSION,
        "target_representation": M2.TARGET_REPRESENTATION,
        "action_tokens": N_SUPERVISED,
        "frame_pool": "t in [2, T-2] per task",
    }
# ------------------------------------------------------------------------- model boundary
def _make_prompt(proc, task, include_task_token=False):
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    import r42_adapter as R
    text = INSTR_TEMPLATE % M2.build_prompt(task, include_token=include_task_token)["prompt"]
    b = PurePromptBuilder("openvla")
    b.add_turn("human", text)
    ids = proc.tokenizer(b.get_prompt(), return_tensors="np").input_ids
    return R.normalize_prompt_ids(ids.reshape(-1).tolist())


def _episode_images(proc, episode, t, cams, hdf5_dir):
    import h5py
    import cv2
    import torch
    from PIL import Image
    import r42_adapter as R
    tiles = []
    with h5py.File(os.path.join(hdf5_dir, episode), "r") as h:
        for cam in R.CAMERA_ORDER[:int(cams)]:
            buf = h["vision/" + cam + "/colors"][int(t)]
            img = cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
            tiles.append(proc.image_processor.apply_transform(
                Image.fromarray(np.ascontiguousarray(img))))
    return torch.cat(tiles, dim=0).unsqueeze(0).to(torch.bfloat16).to("cuda")


def _read_state_action(hdf5_dir, episode):
    import h5py
    import r42_train_contract as TC
    with h5py.File(os.path.join(hdf5_dir, episode), "r") as h:
        return TC.read14(h, "state"), TC.read14(h, "action")


# ------------------------------------------------------------------------------- the run
def run(args):
    """The GPU entry.  Booked into the R15 WAL step by step; full-state resumable."""
    import torch
    import r15_ledger as L15
    import r15_resume as RS
    import r42_adapter as R
    import r11_train_pushT_r42 as R11          # reuse the proven single-task model builder

    tasks = parse_tasks(args.tasks)
    prompts = prompts_for(tasks, include_task_token=args.include_task_token)
    assert_prompts_distinct(tasks, prompts)
    _doc, split_map, payload, splits_path, norm_path = load_m2_artifacts(
        tasks, args.splits, args.normalizer)
    indexes = {t: M2.load_task_index(t, args.hdf5_root) for t in tasks}
    frames_by_task = {t: task_train_frames(indexes[t], split_map[t], t) for t in tasks}
    mix = plan_task_mix(tasks, {t: len(frames_by_task[t]) for t in tasks},
                        floor=args.task_floor, seed=args.seed)
    assert_mix_reproducible(mix, {t: len(frames_by_task[t]) for t in tasks})
    slot_frames = build_slot_frames(mix, frames_by_task)
    dev_frames = {t: dev_frames_for_task(indexes[t], split_map[t], t,
                                         args.dev_frames_per_task, args.seed) for t in tasks}
    normalizer_sha = TN.normalizer_sha256(payload)
    dcontract = data_contract_for(tasks, payload, mix, splits_path, args.cams)

    start_step = 0
    prior_curves, prior_steps = {}, {}
    if args.resume_from:
        _man, ts = RS.verify_and_load_manifest(args.resume_from)
        start_step = int(ts["global_step"])
        assert_resume_compatible(ts, tasks, normalizer_sha, mix, args.max_tasks_stage)
        prior_curves = ts.get("ce_curves") or {}
        prior_steps = ts.get("steps_per_task") or {}
    caps = assert_stage_caps(tasks, args.max_tasks_stage, args.target_global_step, start_step)
    if caps["already_at_target"]:
        log("checkpoint is already at %d >= target %d; nothing to do"
            % (start_step, args.target_global_step))
        return {"verdict": "ALREADY_AT_TARGET", "global_step_reached": start_step}

    out = args.out
    if os.path.isdir(out) and os.listdir(out):
        raise M2TrainError("output directory must be NEW and empty: %s" % out)
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    run_id = args.run_id or ("m2-%dtask-%d-%s"
                            % (len(tasks), args.target_global_step,
                               datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    need = int(args.target_global_step) - start_step
    gpu_est = float(need) * float(args.est_seconds_per_step) + float(args.time_margin)
    ev = evidence_skeleton(args, tasks, payload, prompts, mix, dev_frames, caps, dcontract,
                           start_global_step=start_step, prior_ce_curves=prior_curves)
    ev["run_id"] = run_id
    ev["ledger"]["run_id"] = run_id
    ev["steps_per_task"] = {t: int(prior_steps.get(t, 0)) for t in tasks}
    _write_json(os.path.join(out, "stage_evidence.json"), ev)

    ledger = L15.Ledger(args.ledger_dir)
    with ledger:
        guard = ledger.assert_can_start(args.bucket, gpu_est, requested_steps=need, run_id=run_id)
        if int(guard["left_steps"]) < need:
            raise M2TrainError("the ledger has only %d formal steps left but this stage needs %d"
                               % (int(guard["left_steps"]), need))
        ledger.append(run_id, "job_begin", bucket=args.bucket, payload={
            "tasks": tasks, "target_global_step": int(args.target_global_step),
            "start_global_step": start_step, "steps_requested": need,
            "max_tasks_stage": int(args.max_tasks_stage), "stage_cap": int(caps["cap"]),
            "normalizer_sha256": normalizer_sha,
            "task_mix": {"policy": mix["policy"], "floor": mix["floor"],
                         "weights": mix["weights"], "counts": mix["counts"],
                         "schedule_sha256": mix["schedule_sha256"], "horizon": mix["horizon"]},
            "resume_from": args.resume_from or None, "gpu_estimate_s": round(gpu_est, 1),
            "guard": guard})
    log("M2 %s: %d -> %d (%d steps), tasks=%s, bucket=%s, cap=%d, estimate=%.0fs"
        % (run_id, start_step, args.target_global_step, need, tasks, args.bucket, caps["cap"],
           gpu_est))

    # ---- environment guard FIRST (prismatic picks ACTION_DIM from sys.argv), then the model
    ev["argv"] = R11.force_arx_platform()
    RS.seed_everything(args.seed)
    proc, vla, proj, modeling_sha = R11.build_model(args.cams)
    prompt_ids = {t: _make_prompt(proc, t, args.include_task_token) for t in tasks}
    assert_prompts_distinct(tasks, prompts, token_ids=prompt_ids)
    ev["runtime_contract"] = R.assert_runtime_contract()
    ev["modeling_sha256"] = modeling_sha
    ev["prompt_token_ids_sha256"] = {t: sha256_json(prompt_ids[t]) for t in tasks}
    base_id = RS.attest_base_model(R11.MODEL)
    ev["base_model"] = base_id

    params = [p for p in vla.parameters() if p.requires_grad] + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(x) for x in args.milestones.split(",") if x.strip()], gamma=0.1)
    sampler = make_sampler(slot_frames, args.seed)
    ev["sampler_order_hash"] = sampler.order_hash()
    if args.save_every <= 0:
        args.save_every = 100
        log("NOTE: --save-every <= 0 defaulted to 100 (a crash must stay resumable)")
    if args.resume_from:
        state = RS.load_full_state(args.resume_from, model=vla, proj=proj, opt=opt, sched=sched,
                                   frames=[(q, int(t)) for q, t in slot_frames],
                                   model_kind="peft", expected_contract=dcontract,
                                   expected_base=base_id, expected_global_step=start_step)
        sampler = state["sampler"]
        log("resumed from %s at global_step=%d (next frame %s)"
            % (args.resume_from, start_step, sampler.next()))
        sampler.position -= 1
    ev["data_contract"] = dcontract
    _write_json(os.path.join(out, "stage_evidence.json"), ev)

    def predict(prompt_task, frame_task, episode, t):
        state_, action_ = _read_state_action(indexes[frame_task]["hdf5_dir"], episode)
        row = row_for(payload, frame_task, state_, action_, t, episode=episode)
        lab = labels_for(payload, frame_task, prompt_ids[prompt_task], row["residual"])
        pv = _episode_images(proc, episode, t, args.cams, indexes[frame_task]["hdf5_dir"])
        pr = torch.from_numpy(row["features"]).reshape(1, 42).to("cuda")
        inp = torch.tensor([lab["input_ids"]], dtype=torch.long, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                    proprio=pr, proprio_projector=proj, labels=inp.clone())
        return {"logits": o.logits[0].float().cpu().numpy(),
                "labels": np.asarray(lab["labels"], np.int64), "n_prompt": int(lab["n_prompt"]),
                "residual": np.asarray(row["residual"], np.float32)}

    def record_dev_ce(step):
        blocks = evaluate_dev_ce(predict, payload, tasks, dev_frames, log_fn=log)
        for t in tasks:
            ev["ce_curves"][t].append({"global_step": int(step),
                                       "ce": float(blocks[t]["ce_mean"]),
                                       "ce_full": float(blocks[t]["ce_full_mean"]),
                                       "frames_sha256": blocks[t]["frames_sha256"]})
        ev["eval_archive"] = os.path.abspath(os.path.join(out, "dev_eval_latest.npz"))
        ev["eval_archive_sha256"] = write_eval_archive(ev["eval_archive"], blocks, dcontract)
        ev["dev_ce_latest"] = {t: float(blocks[t]["ce_mean"]) for t in tasks}
        _write_json(os.path.join(out, "stage_evidence.json"), ev)
        return blocks

    t0 = time.time()
    vla.train()
    proj.train()
    if not args.resume_from:
        record_dev_ce(0)                     # each task's own STEP-0 DEV CE (forgetting anchor)
    global_step = start_step
    step_times = []
    while global_step < args.target_global_step:
        elapsed = time.time() - t0
        if elapsed > gpu_est - args.time_margin:
            log("stopping at global_step=%d: time guard (%.0fs used of %.0fs)"
                % (global_step, elapsed, gpu_est))
            break
        qualified_ep, t = sampler.next()
        task, episode = M2.split_qualified(qualified_ep)
        if task not in tasks:
            raise M2TrainError("sampler produced task %r outside the stage task set %s"
                               % (task, tasks))
        st = time.time()
        state_, action_ = _read_state_action(indexes[task]["hdf5_dir"], episode)
        contract = task_contract(payload, task)
        row = M2.build_row(state_, action_, int(t), contract, task, episode=episode)
        # R15 discipline: prove per step that no row >= t+1 leaked into the training row
        M2.assert_row_no_future_leakage(state_, action_, int(t), contract, task)
        ev["leak_checks"] = int(ev.get("leak_checks", 0)) + 1
        pv = _episode_images(proc, episode, t, args.cams, indexes[task]["hdf5_dir"])
        pr = torch.from_numpy(row["features"]).reshape(1, 42).to("cuda")
        lab = labels_for(payload, task, prompt_ids[task], row["residual"])
        inp = torch.tensor([lab["input_ids"]], dtype=torch.long, device="cuda")
        tgt = torch.tensor([lab["labels"]], dtype=torch.long, device="cuda")
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                    labels=tgt, proprio=pr, proprio_projector=proj)
            loss = o.loss
        if not torch.isfinite(loss):
            with open(os.path.join(out, "FAILURES.md"), "w") as fh:
                fh.write("# M2 shared trainer stopped\n\nnon-finite loss at global_step %d "
                         "(%s/%s t=%d)\n" % (global_step + 1, task, episode, t))
            with ledger:
                ledger.append(run_id, "abort", bucket=args.bucket,
                              payload={"global_step": global_step, "task": task,
                                       "reason": "non_finite_loss"},
                              gpu_seconds=time.time() - t0, formal_steps=0)
            raise M2TrainError("non-finite loss at global_step %d (%s)"
                               % (global_step + 1, task))
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        sched.step()
        global_step += 1
        step_times.append(time.time() - st)
        ev["steps_per_task"][task] = int(ev["steps_per_task"].get(task, 0)) + 1
        ev["per_task_loss"][task].append({"global_step": global_step,
                                          "loss": round(float(loss), 6),
                                          "episode": episode, "t": int(t), "utc": utcnow()})
        with ledger:
            ledger.append(run_id, "step_commit", bucket=args.bucket,
                          payload={"global_step": global_step, "task": task,
                                   "loss": round(float(loss), 6), "grad_norm": round(gn, 4),
                                   "lr": sched.get_last_lr()[0], "episode": episode,
                                   "t": int(t)},
                          formal_steps=1, gpu_seconds=max(0.0, time.time() - st))
        if global_step % args.log_every == 0 or global_step == args.target_global_step:
            rows = ev["per_task_loss"]
            flat = {t: (rows[t][-1]["loss"] if rows[t] else None) for t in tasks}
            ev["log"].append({"global_step": global_step, "task": task, "loss": float(loss),
                              "grad_norm": gn, "lr": sched.get_last_lr()[0],
                              "per_task_last_loss": flat,
                              "steps_per_task": dict(ev["steps_per_task"]), "utc": utcnow()})
            log("step=%d task=%s loss=%.4f gnorm=%.2f lr=%.2e per-task=%s"
                % (global_step, task, float(loss), gn, sched.get_last_lr()[0],
                   {k: (None if v is None else round(v, 4)) for k, v in flat.items()}))
            refresh_loss_summary(ev)
            _write_json(os.path.join(out, "stage_evidence.json"), ev)
        if global_step % args.save_every == 0 or global_step == args.target_global_step:
            record_dev_ce(global_step)       # forgetting control at every save point
            ck = RS.save_full_state(
                os.path.join(out, "checkpoints", "step%d" % global_step),
                model=vla, proj=proj, opt=opt, sched=sched, sampler=sampler,
                global_step=global_step, run_id=run_id, data_contract=dcontract,
                model_kind="peft",
                extra={"tasks": tasks, "task_ids": {t: M2.TASK_IDS[t] for t in tasks},
                       "max_tasks_stage": int(args.max_tasks_stage),
                       "stage_cap": int(caps["cap"]),
                       "task_mix": {"policy": mix["policy"], "floor": mix["floor"],
                                    "weights": mix["weights"], "counts": mix["counts"],
                                    "horizon": mix["horizon"], "seed": mix["seed"],
                                    "schedule_sha256": mix["schedule_sha256"]},
                       "normalizer_sha256": normalizer_sha,
                       "per_task_normalizer_sha256": {
                           t: TN.sha256_json(payload["tasks"][t]) for t in tasks},
                       "ce_curves": ev["ce_curves"],
                       "steps_per_task": ev["steps_per_task"],
                       "base_model": base_id, "modeling_sha256": modeling_sha,
                       "target": int(args.target_global_step)})
            ev.setdefault("checkpoints", []).append(
                {"global_step": global_step,
                 "lora_sha256": ck["manifest"]["files"].get(
                     "lora_adapter/adapter_model.safetensors"),
                 "projector_sha256": ck["manifest"]["files"]["proprio_projector_42.pt"],
                 "dir": ck["dir"]})
            log("saved full-state checkpoint step=%d (tasks=%s, ce=%s)"
                % (global_step, tasks, ev.get("dev_ce_latest")))
            _write_json(os.path.join(out, "stage_evidence.json"), ev)

    ev["global_step_reached"] = global_step
    ev["steps_executed"] = global_step - start_step
    ev["utc_end"] = utcnow()
    ev["wall_clock_s"] = round(time.time() - t0, 1)
    ev["seconds_per_step"] = round(float(np.mean(step_times)), 3) if step_times else None
    ev["peak_mem_gb"] = round(float(torch.cuda.max_memory_allocated()) / 1e9, 3)
    ev["caps"]["respected"] = bool(global_step <= caps["cap"])
    ev["verdict"] = "REACHED_TARGET" if global_step == args.target_global_step else "PARTIAL"
    ev["final_ce"] = {t: (ev["ce_curves"][t][-1]["ce"] if ev["ce_curves"][t] else None)
                      for t in tasks}
    refresh_loss_summary(ev)
    with ledger:
        ledger.append(run_id, "job_end", bucket=args.bucket,
                      payload={"global_step": global_step, "verdict": ev["verdict"],
                               "tasks": tasks, "steps_booked_here": global_step - start_step,
                               "wall_clock_s": ev["wall_clock_s"],
                               "totals_after": ledger.totals()},
                      gpu_seconds=0.0, formal_steps=0)
        tot = ledger.totals()
        ev["ledger"]["steps_booked"] = int(tot["buckets"][args.bucket]["formal_steps"]) \
            if args.bucket in tot["buckets"] else None
        ev["ledger"]["totals"] = {"gpu_seconds_used": tot["gpu_seconds_used"],
                                  "formal_steps_used": tot["formal_steps_used"]}
    _write_json(os.path.join(out, "stage_evidence.json"), ev)
    log("M2 %s reached step %d in %.1fs (%.3fs/step, peak %.2f GB) final CE %s"
        % (ev["verdict"], global_step, ev["wall_clock_s"], ev["seconds_per_step"] or -1,
           ev["peak_mem_gb"], ev["final_ce"]))
    return ev


# ------------------------------------------------------------------------------------ main
def build_parser():
    ap = argparse.ArgumentParser(description="M2: ONE shared multi-task 7B OpenVLA-R42 LoRA "
                                             "trainer (push_T / stack_bowls / sweep_blocks)")
    ap.add_argument("--tasks", required=True, help="comma separated, e.g. push_T,stack_bowls")
    ap.add_argument("--target-global-step", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume-from", default="")
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--max-tasks-stage", type=int, default=2)
    ap.add_argument("--est-seconds-per-step", type=float, default=3.5)
    ap.add_argument("--time-margin", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cams", type=int, default=3)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--task-floor", type=float, default=DEFAULT_TASK_FLOOR)
    ap.add_argument("--dev-frames-per-task", type=int, default=DEFAULT_DEV_FRAMES_PER_TASK)
    ap.add_argument("--include-task-token", action="store_true",
                    help="prepend the frozen <task:name#id> token to the instruction")
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--normalizer", default=NORMALIZER_JSON)
    ap.add_argument("--hdf5-root", default=None)
    ap.add_argument("--ledger-dir", default=None)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--milestones", default="1000,2500")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the mix/caps/plan evidence on CPU and exit (no model, no GPU)")
    return ap


def run_plan_only(args):
    """CPU-only dry run: validates every fail-closed guard and prints the plan (no model load)."""
    tasks = parse_tasks(args.tasks)
    prompts = prompts_for(tasks, include_task_token=args.include_task_token)
    assert_prompts_distinct(tasks, prompts)
    _doc, split_map, payload, splits_path, _norm = load_m2_artifacts(
        tasks, args.splits, args.normalizer)
    indexes = {t: M2.load_task_index(t, args.hdf5_root) for t in tasks}
    frames_by_task = {t: task_train_frames(indexes[t], split_map[t], t) for t in tasks}
    mix = plan_task_mix(tasks, {t: len(frames_by_task[t]) for t in tasks},
                        floor=args.task_floor, seed=args.seed)
    assert_mix_reproducible(mix, {t: len(frames_by_task[t]) for t in tasks})
    slot_frames = build_slot_frames(mix, frames_by_task)
    dev_frames = {t: dev_frames_for_task(indexes[t], split_map[t], t,
                                         args.dev_frames_per_task, args.seed) for t in tasks}
    caps = assert_stage_caps(tasks, args.max_tasks_stage, args.target_global_step, 0)
    dcontract = data_contract_for(tasks, payload, mix, splits_path, args.cams)
    plan = {"schema": "m2-plan-v1", "tasks": tasks,
            "task_ids": {t: M2.TASK_IDS[t] for t in tasks}, "prompts": prompts, "caps": caps,
            "normalizer_sha256": TN.normalizer_sha256(payload),
            "per_task_normalizer_sha256": {t: TN.sha256_json(payload["tasks"][t]) for t in tasks},
            "task_mix": {k: v for k, v in mix.items() if k != "schedule"},
            "slot_frames": len(slot_frames),
            "dev_frames": {t: dev_frames[t] for t in tasks},
            "data_contract": dcontract}
    print(json.dumps(plan, indent=1, sort_keys=True))
    return plan


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.plan_only:
        run_plan_only(args)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
