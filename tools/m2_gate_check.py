#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Machine verdict for gate P6_M2_TWO: is the shared 2-task R42 LoRA actually multi-task?

The verdict is computed from two artefacts the trainer writes - stage_evidence.json (the CE
curves, the task mix and the step accounting) and dev_eval_latest.npz (per-DEV-frame predicted
256-bin distributions) - plus a FRESH recomputation of the two baselines from the HDF5 and the
task-aware normalizer payload.  Nothing is read from a table or copied from an older run.

Action-window alignment (issue #26, audit PR #25): the model prepends 769 multimodal vision
tokens, so the 14 legal action rows live at pad + n_prompt - 1 where pad = logits_len - len(labels).
Both the teacher-forced CE and the window probabilities now apply that per-prompt offset; an eval
archive without the 'vision-prefix-aware-v2' marker is refused (LEGACY_INVALID_FOR_P6).

Pre-registered conditions (all must hold for PASS):
  P6.nll_smoothed_vs_noop
                       per-task eps-smoothed NLL of the true 256-bin window is BELOW the no-op
                       baseline (A_hat = S, i.e. R_hat = -V) on the SAME DEV frames, same eps both
                       sides (PRIMARY);
  P6.nll_smoothed_vs_constant_velocity
                       ... and below the constant-velocity baseline (A_hat = S+V, R_hat = 0);
  P6.ce_vs_noop_legacy_hybrid_score / P6.ce_vs_constant_velocity_legacy_hybrid_score
                       the old hybrid score, reported for continuity, never decisive;
  P6.routing_not_degenerate
                       the per-task predicted 256-bin distributions must be separable: minimum
                       pairwise Jensen-Shannon divergence above --routing-min-js, and a
                       leave-one-out nearest-centroid task classifier above
                       --routing-min-accuracy (default 0.5 + 0.5/n_tasks), i.e. the model has not
                       collapsed onto one task's output distribution;
  P6.no_catastrophic_forgetting
                       for EVERY task, every save-point DEV CE <= that task's own step-0 CE +
                       --forget-tol nats;
  P6.cumulative_step_cap
                       cumulative multi-task optimizer steps <= CAPS[max_tasks_stage]
                       (2 tasks -> 2000) and the booked/claimed step counts agree;
  P6.dev_disjoint      the DEV frames used for every number above are DEV episodes of the same
                       task in data/m2_splits.json (never a TRAIN episode).

Baseline metric definition (documented because the repo only defines POINT baselines)
-------------------------------------------------------------------------------------
The R15/R16 baselines are point predictors (identity A=S, constant velocity A=S+V) scored with
an L1 metric.  A CE gate needs a distribution, so each baseline is scored as the same proper
score of the same target: a hit on the exact quantisation bin costs 0 nats, a miss costs the
same as an uninformed guess, ln(256) = 5.545 nats, i.e. CE_base = (1 - hit_rate) * ln(256).
hit_rate and the L1 of the same baseline are reported next to it so the number can be
re-derived independently.  The model side is the CE of the SAME true bins under the
renormalised 256-way window distribution (training/m2_train_shared.ce_window_from_probs).

Exit codes: 0 PASS, 3 FAIL (ran, conditions not met), 4 CANNOT_EVALUATE (fail closed: inputs
missing/inconsistent - nothing is scored and no PASS can be printed).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("training", "diagnostics", "code/xpolicylab/policy/OpenVLA_Pure",
             "code/xpolicylab/policy/OpenVLA_R42"):
    _p = os.path.join(REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:                                                 # imported as top-level modules
    import m2_task_normalizer as TN                   # noqa: E402  (resolves m2_multitask_data)
    import m2_train_shared as MT                      # noqa: E402
except ImportError:                                  # imported as tools.m2_gate_check
    from training import m2_task_normalizer as TN      # noqa: E402
    from training import m2_train_shared as MT         # noqa: E402

M2 = TN.M2                    # the SAME m2_multitask_data module object every M2 module uses
assert MT.M2 is M2, "m2_train_shared resolved a different m2_multitask_data module"

GATE = "P6_M2_TWO"
EXIT_PASS, EXIT_FAIL, EXIT_CANNOT = 0, 3, 4
SPLITS = os.environ.get("M2_SPLITS", os.path.join(REPO, "data", "m2_splits.json"))
LN_BINS = float(np.log(256.0))
# PRE-REGISTERED smoothing for the 256-bin action window (issue #26 item 4).
# p = (1 - eps) * onehot + eps / 256 on BOTH sides, so a hit and a miss are priced by the same
# distribution instead of the legacy "0 nats on a hit, ln(256) on a miss" hybrid (which is eps->0 for
# hits and eps->1 for misses and is therefore not a likelihood at all).
EPS_SMOOTH = 1e-3


def smoothed_nll_from_probs(probs, bins_true, eps=EPS_SMOOTH):
    """Coherent negative log-likelihood of the true bins under the model's smoothed window.

    probs is [n_actions, 256] already renormalised over the legal window; smoothing is applied here,
    once, with the pre-registered eps.
    """
    p = np.asarray(probs, np.float64)
    b = np.asarray(bins_true, np.int64)
    if p.ndim == 3:                       # [n_frames, n_actions, 256] as the archive stores it
        if b.ndim != 2 or b.shape != p.shape[:2]:
            raise GateCannotEvaluate("smoothed_nll frames %s vs bins %s" % (p.shape, b.shape))
        per = [smoothed_nll_from_probs(p[i], b[i], eps) for i in range(p.shape[0])]
        return float(np.mean(per))
    b1 = b.reshape(-1)
    if p.ndim != 2 or p.shape[0] != b1.shape[0]:
        raise GateCannotEvaluate("smoothed_nll shapes %s vs %s" % (p.shape, b.shape))
    ps = (1.0 - float(eps)) * p + float(eps) / 256.0
    picked = ps[np.arange(len(b1)), np.clip(b1, 0, 255)]
    return float(-np.log(picked).mean())


def smoothed_nll_from_bins(pred_bins, bins_true, eps=EPS_SMOOTH):
    """Same likelihood for a POINT predictor: p = 1 at the predicted bin, smoothed with eps."""
    pred = np.asarray(pred_bins, np.int64).reshape(-1)
    b = np.asarray(bins_true, np.int64).reshape(-1)
    if pred.shape != b.shape:
        raise GateCannotEvaluate("smoothed_nll_from_bins shapes %s vs %s" % (pred.shape, b.shape))
    hit = pred == b
    p = np.where(hit, (1.0 - float(eps)) + float(eps) / 256.0, float(eps) / 256.0)
    return float(-np.log(p).mean())


class GateCannotEvaluate(Exception):
    """The gate refuses to score: the artefacts are missing, inconsistent or from another run."""


def check(cid, requirement, measured, passed):
    return {"id": cid, "requirement": requirement, "measured": measured, "passed": bool(passed)}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------- baselines
def baseline_bins(payload, task, state, action, t):
    """The codec bins of the truth and of the two point baselines, for ONE frame."""
    import r42_adapter as R
    ctx = TN.decode_context(payload, task)
    s = np.asarray(state, np.float32)
    a = np.asarray(action, np.float32)
    t = int(t)
    if t < M2.MIN_HISTORY_FRAMES or t > len(s) - 2:
        raise GateCannotEvaluate("frame %d is outside the history-safe pool (T=%d)" % (t, len(s)))
    v = s[t] - s[t - 1]
    r_true = (a[t] - s[t]) - v
    r_noop = -v                       # A_hat = S        -> R_hat = -V
    r_cv = np.zeros_like(v)           # A_hat = S + V    -> R_hat = 0
    out = {}
    for name, res in (("true", r_true), ("noop", r_noop), ("cv", r_cv)):
        _idx, ids = R.token_ids_for_residual(res, ctx["q01"], ctx["q99"])
        out[name] = np.asarray(ids, np.int64) - R.TOKEN_BEGIN_IDX - 1
    # Physical L1 of each baseline's 14D command against the true action.
    # The no-op baseline COMMANDS THE CURRENT STATE (A_hat = S[t]), so its error is |S[t] - A[t]|.
    # The previous expression, |(S[t] - S[t]) - A[t]| = |A[t]|, is the error of commanding ZERO JOINT
    # POSITIONS - a different (and much worse) predictor - which silently understated every reported
    # no-op margin (audit PR#27).
    out["l1_noop"] = float(np.abs(s[t] - a[t]).mean())
    out["l1_cv"] = float(np.abs((s[t] + v) - a[t]).mean())
    out["l1_zero_position_LEGACY"] = float(np.abs(np.zeros_like(a[t]) - a[t]).mean())
    return out


def recompute_baselines(payload, tasks, blocks, hdf5_root):
    """Fresh, model-free recomputation of both baselines on exactly the archive's DEV frames."""
    import r42_train_contract as TC
    import h5py
    out = {}
    for task in tasks:
        block = blocks[task]
        dirs = M2.task_dir(task, hdf5_root)
        per_frame = []
        cache = {}
        for ep, t in zip(block["episodes"], block["t"]):
            if ep not in cache:
                p = os.path.join(dirs, ep)
                if not os.path.isfile(p):
                    raise GateCannotEvaluate("DEV episode missing: %s" % p)
                with h5py.File(p, "r") as h:
                    cache[ep] = (TC.read14(h, "state"), TC.read14(h, "action"))
            state, action = cache[ep]
            b = baseline_bins(payload, task, state, action, t)
            per_frame.append(b)
        true = np.stack([b["true"] for b in per_frame])
        noop = np.stack([b["noop"] for b in per_frame])
        cv = np.stack([b["cv"] for b in per_frame])
        hit_noop = float((noop == true).mean())
        hit_cv = float((cv == true).mean())
        out[task] = {
            "n_frames": int(true.shape[0]),
            "true_bins_sha256": MT.sha256_json(true.tolist()),
            "hit_rate_noop": hit_noop, "hit_rate_cv": hit_cv,
            # legacy hybrid score, kept for continuity only (NOT a likelihood)
            "ce_noop": (1.0 - hit_noop) * LN_BINS,
            "ce_cv": (1.0 - hit_cv) * LN_BINS,
            # coherent smoothed NLL on the same frames, same eps as the model side
            "nll_smoothed_noop": smoothed_nll_from_bins(noop, true),
            "nll_smoothed_cv": smoothed_nll_from_bins(cv, true),
            "l1_noop": float(np.mean([b["l1_noop"] for b in per_frame])),
            "l1_cv": float(np.mean([b["l1_cv"] for b in per_frame])),
        }
    return out


# ------------------------------------------------------------------------------ routing
def _js(p, q, eps=1e-12):
    p = np.asarray(p, np.float64) + eps
    q = np.asarray(q, np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))          # noqa: E731
    return max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))


def histograms(probs):
    """One 256-bin histogram per frame: the mean predicted window distribution over dims."""
    p = np.asarray(probs, np.float64)
    if p.ndim != 3:
        raise GateCannotEvaluate("expected [N,14,256] probabilities, got %s" % (p.shape,))
    h = p.mean(axis=1)
    return h / h.sum(axis=1, keepdims=True)


def routing_metrics(blocks, tasks, min_win_rate=0.9):
    """Does the TASK PROMPT actually route the output, or did the model collapse onto one task?

    The per-task marginal predictions are standardised by each task's own residual quantiles, so
    two tasks can have similar-looking marginals even for a perfectly conditioned model; comparing
    marginals is therefore NOT used as a pass/fail test (it is reported for information only).
    The archive instead carries COUNTERFACTUAL predictions - every DEV frame is re-evaluated under
    every OTHER task's prompt - so routing is measured directly:

      (a) on every scored frame the model must be BETTER (lower CE) with its own task's prompt
          than with any other task's prompt;
      (b) the mean CE it pays for the wrong prompt must exceed --routing-min-ce-penalty;
      (c) it must not emit one constant bin vector for every frame.

    A model that ignores the prompt gives (a) a win rate of exactly 0 (ties never count as a win)
    and (b) a penalty of exactly 0, so prompt collapse cannot pass by accident.
    """
    pairs = {}
    for ft in tasks:
        own = np.asarray(blocks[ft]["ce_window"], np.float64)
        for pt in blocks[ft]["prompt_tasks"]:
            if pt == ft:
                continue
            other = np.asarray(blocks[ft]["counterfactual"][pt]["ce_window"], np.float64)
            if other.shape != own.shape:
                raise GateCannotEvaluate("%s under prompt %s has %d frames, expected %d"
                                         % (ft, pt, other.shape[0], own.shape[0]))
            pairs["%s<-prompt:%s" % (ft, pt)] = {
                "own_prompt_win_rate": float((own < other).mean()),
                "mean_ce_penalty_nats": float(np.mean(other - own)),
                "n_frames": int(own.shape[0])}
    if not pairs:
        raise GateCannotEvaluate("no counterfactual prompt pairs to score (need >= 2 tasks)")
    min_win = min(v["own_prompt_win_rate"] for v in pairs.values())
    min_penalty = min(v["mean_ce_penalty_nats"] for v in pairs.values())
    # informational: marginal separation of the CORRECT-prompt distributions
    marg = {t: histograms(blocks[t]["pred_probs"]).mean(axis=0) for t in tasks}
    marg = {t: marg[t] / marg[t].sum() for t in tasks}
    js = {}
    for i, a in enumerate(tasks):
        for b in tasks[i + 1:]:
            js["%s|%s" % (a, b)] = _js(marg[a], marg[b])
    varied = {t: int(max(1, np.unique(np.asarray(blocks[t]["pred_bins"]), axis=0).shape[0]))
              for t in tasks}
    identical = {}
    for ft in tasks:
        ownp = np.asarray(blocks[ft]["pred_probs"], np.float64)
        for pt in blocks[ft]["prompt_tasks"]:
            if pt == ft:
                continue
            o = np.asarray(blocks[ft]["counterfactual"][pt]["pred_probs"], np.float64)
            identical["%s<-prompt:%s" % (ft, pt)] = float(
                np.mean(np.abs(o - ownp).reshape(o.shape[0], -1).max(axis=1) == 0.0))
    return {"per_pair": pairs,
            "min_own_prompt_win_rate": min_win,
            "min_wrong_prompt_ce_penalty": min_penalty,
            "min_win_rate_required": float(min_win_rate),
            "prompt_identical_fraction": identical,
            "marginal_pairwise_js_informational": js,
            "distinct_pred_bins_per_task": varied,
            "any_frame_variation": bool(any(v > 1 for v in varied.values()))}

# --------------------------------------------------------------------------- step budget
def cap_check(ev, ledger_dir=None):
    tasks = ev.get("tasks") or []
    stage = int(ev.get("max_tasks_stage", 0) or 0)
    if stage not in MT.CAPS:
        raise GateCannotEvaluate("evidence max_tasks_stage %r is not a known stage" % (stage,))
    cap = int(MT.CAPS[stage])
    reached = int(ev.get("global_step_reached", -1))
    start = int(ev.get("start_global_step", 0))
    target = int(ev.get("target_global_step", -1))
    executed = int(ev.get("steps_executed", reached - start))
    booked = (ev.get("ledger") or {}).get("steps_booked")
    checks = []
    checks.append(check("P6.cap.task_count",
                        "task count <= --max-tasks-stage",
                        {"n_tasks": len(tasks), "max_tasks_stage": stage},
                        len(tasks) <= stage))
    checks.append(check("P6.cap.cumulative_steps",
                        "cumulative multi-task optimizer steps <= the stage cap",
                        {"cap": cap, "target": target, "reached": reached},
                        reached <= cap and target <= cap))
    checks.append(check("P6.cap.stage_completed",
                        "the stage actually reached its declared cumulative target",
                        {"target": target, "reached": reached},
                        reached == target))
    checks.append(check("P6.cap.booking_matches_steps",
                        "steps booked/claimed agree with the executed steps",
                        {"executed": executed, "reached": reached, "start": start,
                         "ledger_steps_booked": booked},
                        executed == reached - start
                        and (booked is None or int(booked) >= executed)))
    if ledger_dir:
        wal = os.path.join(ledger_dir, "events.jsonl")
        if os.path.isfile(wal):
            run_id = ev.get("run_id")
            seen = {}
            with open(wal) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except ValueError:
                        raise GateCannotEvaluate("corrupt ledger WAL line: %r" % line[:80])
                    if e.get("run_id") != run_id:
                        continue
                    if e.get("kind") == "step_commit":
                        gs = int((e.get("payload") or {}).get("global_step", -1))
                        if gs in seen:
                            raise GateCannotEvaluate("duplicate step_commit %d for run %s"
                                                     % (gs, run_id))
                        seen[gs] = e
            checks.append(check("P6.cap.wal_steps",
                                "the R15 WAL books exactly one step per executed step",
                                {"run_id": run_id, "wal_steps": len(seen),
                                 "executed": executed},
                                len(seen) == executed))
        else:
            checks.append(check("P6.cap.wal_steps",
                                "the R15 WAL is present for this run (informational)",
                                {"wal": wal, "status": "absent"}, True))
    return checks, cap


# ------------------------------------------------------------------------- forgetting
def forgetting_check(ev, tol):
    curves = ev.get("ce_curves") or {}
    tasks = ev.get("tasks") or []
    checks, table = [], {}
    for t in tasks:
        pts = curves.get(t) or []
        if len(pts) < 2:
            raise GateCannotEvaluate("%s: CE curve has %d points; the step-0 anchor and at least "
                                     "one save point are required" % (t, len(pts)))
        anchor = float(pts[0]["ce"])
        worst = max(float(p["ce"]) for p in pts[1:])
        worst_step = int([p["global_step"] for p in pts
                          if float(p["ce"]) == worst][0])
        best = min(float(p["ce"]) for p in pts)
        table[t] = {"step0_ce": anchor, "worst_ce": worst, "worst_step": worst_step,
                    "final_ce": float(pts[-1]["ce"]), "best_ce": best,
                    "delta_worst_minus_step0": worst - anchor, "tolerance": float(tol),
                    "n_save_points": len(pts)}
        checks.append(check("P6.no_catastrophic_forgetting.%s" % t,
                            "every save-point DEV CE <= that task's step-0 CE + tolerance",
                            table[t], (worst - anchor) <= float(tol)))
    return checks, table


# ------------------------------------------------------------------------------- main
def _write(path, report):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=False)


def score(evidence_path, eval_path, normalizer_path, splits_path=None, hdf5_root=None,
          forget_tol=0.10, routing_min_js=0.02, routing_min_win_rate=0.9,
          routing_min_ce_penalty=0.05, ledger_dir=None, dev_check=True):
    """Compute the P6_M2_TWO verdict.  Returns (report, exit_code)."""
    report = {"gate": GATE, "evidence": os.path.abspath(evidence_path),
              "eval_archive": os.path.abspath(eval_path),
              "normalizer": os.path.abspath(normalizer_path),
              "checks": [], "per_task": {}, "verdict": "CANNOT_EVALUATE", "exit": EXIT_CANNOT,
              "action_window_alignment": "vision-prefix-aware-v2",
              "ce_definitions": {
                  "primary": "eps-smoothed NLL of the true 256-bin action window: "
                             "p = (1-eps) * p_model + eps/256 with eps = %g pre-registered; the "
                             "point baselines are scored as a point likelihood with the SAME eps so "
                             "a hit and a miss are priced coherently" % EPS_SMOOTH,
                  "legacy": "(1 - hit_rate) * ln(256): 0 nats on a hit, ln(256) on a miss. Kept for "
                            "continuity with the previous report only - it is a heuristic bin-error "
                            "score, not a likelihood, and is NOT used to decide P6."},
              "metric_definitions": {
                  "model_ce": "mean over the 14 action dims of -log p_window(true_bin), "
                              "256-way window renormalised (nats)",
                  "model_nll_smoothed": "eps-smoothed NLL of the true bins (the primary metric)",
                  "baseline_nll_smoothed": "the same eps-smoothed NLL for the point baseline's bins",
                  "baseline_ce": "LEGACY (1 - hit_rate) * ln(256) hybrid score, reported only",
                  "forgetting": "worst save-point CE minus that task's step-0 CE",
                  "routing": "pairwise Jensen-Shannon divergence of the per-task mean predicted "
                             "256-bin distributions + leave-one-out nearest-centroid accuracy"}}

    def cannot(msg):
        report["error"] = msg
        report["checks"].append(check("inputs.consistent",
                                      "every input artefact is present, complete and from the "
                                      "same run", {"error": msg}, False))
        report["verdict"] = "CANNOT_EVALUATE"
        report["exit"] = EXIT_CANNOT
        return report, EXIT_CANNOT

    if not os.path.isfile(evidence_path):
        return cannot("stage evidence missing: %s" % evidence_path)
    try:
        with open(evidence_path) as fh:
            ev = json.load(fh)
    except ValueError as exc:
        return cannot("stage evidence is not JSON: %s" % exc)
    if ev.get("schema") != MT.SCHEMA_EVIDENCE:
        return cannot("evidence schema %r != %r" % (ev.get("schema"), MT.SCHEMA_EVIDENCE))
    tasks = ev.get("tasks")
    if not tasks:
        return cannot("evidence records no task set")
    try:
        tasks = MT.parse_tasks(list(tasks))
    except MT.M2TrainError as exc:
        return cannot(str(exc))
    report["tasks"] = tasks
    report["evidence_sha256"] = sha256_file(evidence_path)
    report["run_id"] = ev.get("run_id")

    try:
        payload = TN.load_payload(normalizer_path, tasks=tasks)
    except (TN.M2NormalizerError, OSError) as exc:
        return cannot("multi-task normalizer rejected: %s" % exc)

    # ---- the eval archive must be the same run's, on the same frames, with sane distributions
    try:
        arch = MT.load_eval_archive(eval_path)
    except (MT.M2TrainError, OSError) as exc:
        return cannot(str(exc))
    if sorted(arch["tasks"]) != sorted(tasks):
        return cannot("eval archive covers tasks %s but the evidence covers %s"
                      % (arch["tasks"], tasks))
    for t in tasks:
        want = ((ev.get("dev_frames") or {}).get(t) or {}).get("frames_sha256")
        got = arch["blocks"][t]["frames_sha256"]
        if want is None:
            return cannot("evidence records no DEV frame list for %s" % t)
        if str(want) != str(got):
            return cannot("%s: the eval archive frames %s are not the evidence DEV frames %s"
                          % (t, got, want))
        probs = arch["blocks"][t]["pred_probs"]
        sums = probs.sum(axis=2)
        if not np.allclose(sums, 1.0, atol=1e-3):
            return cannot("%s: predicted window distributions do not sum to 1" % t)
        if not np.array_equal(probs.argmax(axis=2), arch["blocks"][t]["pred_bins"]):
            return cannot("%s: pred_bins is not the argmax of pred_probs" % t)

    # ---- DEV/TRAIN disjointness of the frames that every number below is computed on
    if dev_check:
        if not os.path.isfile(splits_path):
            return cannot("splits artefact missing: %s" % splits_path)
        with open(splits_path) as fh:
            split_map = (json.load(fh).get("splits") or {})
        bad = []
        for t in tasks:
            if t not in split_map:
                return cannot("splits artefact has no split for %s" % t)
            dev = {k.split("/", 1)[-1] for k in split_map[t]["dev"]}
            train = {k.split("/", 1)[-1] for k in split_map[t]["train"]}
            for ep in arch["blocks"][t]["episodes"]:
                if ep in train:
                    bad.append("%s/%s is a TRAIN episode" % (t, ep))
                elif ep not in dev:
                    bad.append("%s/%s is not a DEV episode" % (t, ep))
        report["checks"].append(check("P6.dev_disjoint",
                                      "every scored frame is a DEV episode of the same task",
                                      {"violations": bad[:10], "n_violations": len(bad)},
                                      not bad))
        if bad:
            report["verdict"] = "FAIL"
            report["exit"] = EXIT_FAIL
            _write_placeholder(report)
            return report, EXIT_FAIL
    else:
        report["checks"].append(check("P6.dev_disjoint",
                                      "DEV/TRAIN check explicitly disabled", {"disabled": True},
                                      True))

    # ---- baselines, recomputed here on the same frames
    try:
        base = recompute_baselines(payload, tasks, arch["blocks"], hdf5_root)
    except GateCannotEvaluate as exc:
        return cannot(str(exc))
    except (OSError, KeyError) as exc:
        return cannot("baseline recomputation failed: %r" % (exc,))
    for t in tasks:
        got = arch["blocks"][t]["bins_true"]
        if MT.sha256_json(got.tolist()) != base[t]["true_bins_sha256"]:
            return cannot("%s: the archive's true bins disagree with the bins recomputed from the "
                          "HDF5 + normalizer payload (the archive was made with another payload "
                          "or other frames)" % t)
        model_ce = float(arch["blocks"][t]["ce_window"].mean())
        model_nll = smoothed_nll_from_probs(arch["blocks"][t]["pred_probs"], got)
        base[t]["model_ce"] = model_ce
        base[t]["model_nll_smoothed"] = model_nll
        base[t]["margin_vs_noop"] = base[t]["ce_noop"] - model_ce
        base[t]["margin_vs_cv"] = base[t]["ce_cv"] - model_ce
        base[t]["margin_nll_vs_noop"] = base[t]["nll_smoothed_noop"] - model_nll
        base[t]["margin_nll_vs_cv"] = base[t]["nll_smoothed_cv"] - model_nll
        report["per_task"][t] = base[t]
        # PRIMARY, coherent comparison (same eps on both sides, issue #26 item 4)
        report["checks"].append(check(
            "P6.nll_smoothed_vs_noop.%s" % t,
            "coherent eps-smoothed NLL of the true bins is below the no-op baseline (A_hat=S) on "
            "the same frames, same eps=%g on both sides" % EPS_SMOOTH,
            {"model_nll": model_nll, "baseline_nll": base[t]["nll_smoothed_noop"],
             "hit_rate": base[t]["hit_rate_noop"], "l1": base[t]["l1_noop"],
             "margin_nats": base[t]["margin_nll_vs_noop"], "eps": EPS_SMOOTH},
            model_nll < base[t]["nll_smoothed_noop"]))
        report["checks"].append(check(
            "P6.nll_smoothed_vs_constant_velocity.%s" % t,
            "coherent eps-smoothed NLL of the true bins is below the constant-velocity baseline "
            "(A_hat=S+V) on the same frames, same eps=%g on both sides" % EPS_SMOOTH,
            {"model_nll": model_nll, "baseline_nll": base[t]["nll_smoothed_cv"],
             "hit_rate": base[t]["hit_rate_cv"], "l1": base[t]["l1_cv"],
             "margin_nats": base[t]["margin_nll_vs_cv"], "eps": EPS_SMOOTH},
            model_nll < base[t]["nll_smoothed_cv"]))
        report["checks"].append(check(
            "P6.ce_vs_noop_legacy_hybrid_score.%s" % t,
            "LEGACY hybrid score (0 nats on a hit, ln256 on a miss: not a likelihood) below the "
            "no-op baseline - reported for continuity, not used to decide P6",
            {"model_ce": model_ce, "baseline_ce": base[t]["ce_noop"],
             "hit_rate": base[t]["hit_rate_noop"], "l1": base[t]["l1_noop"],
             "margin_nats": base[t]["margin_vs_noop"]},
            model_ce < base[t]["ce_noop"]))
        report["checks"].append(check(
            "P6.ce_vs_constant_velocity_legacy_hybrid_score.%s" % t,
            "LEGACY hybrid score (0 nats on a hit, ln256 on a miss) below the constant-velocity "
            "baseline - reported for continuity, not used to decide P6",
            {"model_ce": model_ce, "baseline_ce": base[t]["ce_cv"],
             "hit_rate": base[t]["hit_rate_cv"], "l1": base[t]["l1_cv"],
             "margin_nats": base[t]["margin_vs_cv"]},
            model_ce < base[t]["ce_cv"]))

    # ---- routing / collapse
    routing = routing_metrics(arch["blocks"], tasks, min_win_rate=routing_min_win_rate)
    report["routing"] = routing
    if len(tasks) < 2:
        report["checks"].append(check("P6.routing_not_degenerate",
                                      "at least two tasks are needed to prove task routing",
                                      {"tasks": tasks}, False))
    else:
        report["checks"].append(check(
            "P6.routing_not_degenerate.own_prompt_wins",
            "on every scored frame the model is better (lower CE) with its OWN task prompt than "
            "with any other task prompt (ties never count as a win)",
            {"min_own_prompt_win_rate": routing["min_own_prompt_win_rate"],
             "required": routing["min_win_rate_required"], "per_pair": routing["per_pair"]},
            routing["min_own_prompt_win_rate"] >= routing["min_win_rate_required"]))
        report["checks"].append(check(
            "P6.routing_not_degenerate.wrong_prompt_penalty",
            "evaluating a frame under another task's prompt costs CE (the model really uses the "
            "task tag instead of one collapsed output distribution)",
            {"min_ce_penalty_nats": routing["min_wrong_prompt_ce_penalty"],
             "required": float(routing_min_ce_penalty),
             "prompt_identical_fraction": routing["prompt_identical_fraction"]},
            routing["min_wrong_prompt_ce_penalty"] >= float(routing_min_ce_penalty)))
        report["checks"].append(check(
            "P6.routing_not_degenerate.marginals_informational",
            "per-task marginal predicted distributions (reported, never required: the targets "
            "are standardised per task, so equal marginals are possible)",
            {"pairwise_js": routing["marginal_pairwise_js_informational"],
             "required_if_used": float(routing_min_js)}, True))
        report["checks"].append(check(
            "P6.routing_not_degenerate.not_constant",
            "the model does not emit one constant bin vector for every frame of every task",
            {"distinct_pred_bins_per_task": routing["distinct_pred_bins_per_task"]},
            routing["any_frame_variation"]))

    # ---- forgetting
    try:
        fchecks, ftable = forgetting_check(ev, forget_tol)
    except GateCannotEvaluate as exc:
        return cannot(str(exc))
    report["forgetting"] = ftable
    report["checks"].extend(fchecks)

    # ---- cap
    try:
        cchecks, cap = cap_check(ev, ledger_dir=ledger_dir)
    except GateCannotEvaluate as exc:
        return cannot(str(exc))
    report["cap"] = cap
    report["checks"].extend(cchecks)

    failed = [c for c in report["checks"] if not c["passed"]]
    report["verdict"] = "PASS" if not failed else "FAIL"
    report["exit"] = EXIT_PASS if not failed else EXIT_FAIL
    report["n_checks"] = len(report["checks"])
    report["n_failed"] = len(failed)
    return report, report["exit"]


def _write_placeholder(report):
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="P6_M2_TWO machine gate verdict")
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--eval", required=True, dest="eval_path")
    ap.add_argument("--normalizer", required=True)
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--hdf5-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--forget-tol", type=float, default=0.10)
    ap.add_argument("--routing-min-js", type=float, default=0.02)
    ap.add_argument("--routing-min-win-rate", type=float, default=0.9)
    ap.add_argument("--routing-min-ce-penalty", type=float, default=0.05)
    ap.add_argument("--ledger-dir", default=None)
    ap.add_argument("--no-dev-check", action="store_true")
    a = ap.parse_args(argv)
    report, code = score(a.evidence, a.eval_path, a.normalizer, splits_path=a.splits,
                         hdf5_root=a.hdf5_root, forget_tol=a.forget_tol,
                         routing_min_js=a.routing_min_js,
                         routing_min_win_rate=a.routing_min_win_rate,
                         routing_min_ce_penalty=a.routing_min_ce_penalty,
                         ledger_dir=a.ledger_dir, dev_check=not a.no_dev_check)
    _write(a.out, report)
    print(json.dumps({c["id"]: c["passed"] for c in report["checks"]}, indent=1))
    print("VERDICT %s (%s) -> %s" % (report["verdict"], GATE, a.out))
    return code


if __name__ == "__main__":
    sys.exit(main())
