#!/usr/bin/env python3
"""R16: wire the RoboDojo policy-client loop to the 42D [S,V,Acc] R checkpoint (CPU only).

Scope of this module, stated honestly: it is the *adapter + timing* layer between a RoboDojo
episode loop and r42_deploy_server.R42DeployServer.  It contains no model, no CUDA and no
simulator call, it never fabricates a rollout and it never reports a success rate.  A closed-loop
Push-T number can only come from R15-qualified weights on a GPU box running the container client;
this file is what makes that run *auditable* the moment such a checkpoint exists.

Wiring (all of it reuses the existing 42D contract, nothing is re-implemented here):

  RoboDojo client (sim side, container)                    this module              policy server
  ----------------------------------------                ------------             -------------
  TASK_ENV.get_obs()                                      .step(observation)
    -> observation {images, state, env_idx, timestamp}    - privileged guard   ->  refused
                                                          - camera resolution  ->  frozen order
                                                          - 14D state check
    model_client.call(get_action)                         LatencyRecorder      ->  R42DeployServer
      -> 14D joint command per control step                                     .act(images, state)
                                                                                 - 14 legal tokens
                                                                                 - S+V+R safety
  TASK_ENV.take_action(action), hold for control_dt      latch previous command

  .reset(env_idx)  -> server.reset(env_idx) + drop the held command of that env only
  .replay(log)     -> server.replay(log), bit-identical, zero weight update

Contract enforced here (fail closed, every rejection is a test in
tests/test_r42_robodojo_adapter.py):
  1. three RGB cameras resolved to the frozen order (cam_head, cam_left_wrist, cam_right_wrist);
     the client-native aliases cam_high / *_camera / *_wrist seen in OpenVLA_Pure/model.py are
     mapped explicitly, an unknown or doubly-aliased camera is refused, and a bare positional
     list is refused unless the caller declares the frozen order;
  2. the proprio state must be 14D and finite;
  3. privileged ground-truth fields (target_t, privileged_*, gt_*, oracle_*, ...) are refused
     BEFORE the policy is called - they may only be read with .diagnostics(observation), which
     never reaches .step()/the server;
  4. every step must return exactly 14 action tokens and a 14D command;
  5. the rate rule (checkpoint safety_limits.max_rate, or an explicit override) is applied at the
     client control period, so the emitted command is rate-limited against the last commanded
     value of the same env and structurally stays inside the physical box;
  6. truth for the numbers is wall-clock latency per stage, reported as p50/p95/p99 against the
     25 Hz (40 ms) control period - not a simulated or copied figure.
"""
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PURE = os.path.abspath(os.path.join(_HERE, "..", "OpenVLA_Pure"))
for _p in (_HERE, _PURE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r42_adapter as R  # noqa: E402

CONTROL_DT = 0.04                 # 25 Hz Push-T control period
CONTROL_HZ = 25.0
STATE_DIM = 14
N_ACTION_TOKENS = R.N_ACTIONS     # 14
PHASES = ("preprocess", "forward_total", "decode", "total")
PERCENTILE_METHOD = "linear interpolation, rank = (n - 1) * p / 100 (numpy default)"

# The frozen camera order of the R42 contract, plus the aliases the RoboDojo client actually uses
# (XPolicyLab/policy/OpenVLA_Pure/model.py: "cam_high", "cam_left_wrist", "cam_right_wrist" in the
# native path and the _IMAGE_CANDIDATES fallback).  Resolution is by name, never by dict order.
CAMERA_ORDER = tuple(R.CAMERA_ORDER)
CAMERA_ALIASES = {
    "cam_head": ("cam_head", "cam_high", "head_camera", "top_camera"),
    "cam_left_wrist": ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left"),
    "cam_right_wrist": ("cam_right_wrist", "right_camera", "right_wrist", "wrist_right"),
}

# A field with any of these names/prefixes is simulator ground truth or an oracle: it may describe
# the episode, it may never condition the policy (Push-T success is scored on these same objects).
PRIVILEGED_PREFIXES = ("target_", "privileged", "priv_", "gt_", "oracle_", "expert_", "demo_",
                       "ground_truth", "groundtruth")
PRIVILEGED_EXACT = ("target", "targets", "target_t", "groundtruth", "gt", "oracle")
PRIVILEGED_SCAN_DEPTH = 4


class R42RoboDojoError(R.R42Error):
    """Any wiring-contract violation on the RoboDojo -> 42D boundary."""


# ------------------------------------------------------------------------ small helpers
def _vec14(x, name):
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    if v.shape != (STATE_DIM,):
        raise R42RoboDojoError("%s must be a %dD vector, got shape %s"
                              % (name, STATE_DIM, v.shape))
    if not np.isfinite(v).all():
        raise R42RoboDojoError("%s has non-finite values" % name)
    return v


def percentile(values, q):
    """Percentile of a sample with linear interpolation, rank = (n - 1) * q / 100.

    Spelled out instead of calling numpy so the reported p50/p95/p99 are exactly the documented
    definition (and are unit-tested against a known sequence).
    """
    a = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if a.size == 0:
        raise R42RoboDojoError("percentile of an empty sample is undefined")
    q = float(q)
    if not (0.0 <= q <= 100.0):
        raise R42RoboDojoError("percentile must be inside [0, 100], got %r" % (q,))
    if a.size == 1:
        return float(a[0])
    pos = (a.size - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, a.size - 1)
    frac = pos - lo
    return float(a[lo] + (a[hi] - a[lo]) * frac)


class LatencyRecorder:
    """Per-stage wall-clock latency, reported against the control period.

    budget_ms defaults to control_dt * 1000 (40 ms at 25 Hz).  A sample violates the period when
    its total is *strictly greater* than the budget: 40.0 ms exactly still fits one period.
    """

    def __init__(self, control_dt=CONTROL_DT, budget_ms=None):
        dt = float(control_dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise R42RoboDojoError("control_dt must be finite and > 0, got %r" % (control_dt,))
        self.control_dt = dt
        self.budget_ms = float(budget_ms) if budget_ms is not None else dt * 1e3
        if not math.isfinite(self.budget_ms) or self.budget_ms <= 0.0:
            raise R42RoboDojoError("budget_ms must be finite and > 0")
        self.records = []

    def __len__(self):
        return len(self.records)

    @property
    def n(self):
        return len(self.records)

    def reset(self):
        self.records = []
        return self

    def record(self, latency_ms):
        """Store one step.  Missing phases, negatives and non-finite values are refused."""
        if not isinstance(latency_ms, dict):
            raise R42RoboDojoError("latency must be a dict with phases %s" % (PHASES,))
        missing = [p for p in PHASES if p not in latency_ms]
        if missing:
            raise R42RoboDojoError("latency record is missing phase(s) %s" % missing)
        rec = {}
        for p in PHASES:
            try:
                val = float(latency_ms[p])
            except (TypeError, ValueError):
                raise R42RoboDojoError("latency phase %r is not a number: %r"
                                       % (p, latency_ms[p]))
            if not math.isfinite(val):
                raise R42RoboDojoError("latency phase %r is not finite" % p)
            if val < 0.0:
                raise R42RoboDojoError("latency phase %r is negative (%r)" % (p, val))
            rec[p] = val
        rec["violates"] = bool(rec["total"] > self.budget_ms)
        self.records.append(rec)
        return dict(rec)

    def violations(self):
        return int(sum(1 for r in self.records if r["violates"]))

    def summary(self):
        n = len(self.records)
        phases = {}
        for p in PHASES:
            vals = [r[p] for r in self.records]
            if not vals:
                phases[p] = {"n": 0, "mean": 0.0, "min": 0.0, "p50": 0.0, "p95": 0.0,
                             "p99": 0.0, "max": 0.0}
                continue
            phases[p] = {"n": n, "mean": float(np.mean(vals)), "min": float(min(vals)),
                         "p50": percentile(vals, 50), "p95": percentile(vals, 95),
                         "p99": percentile(vals, 99), "max": float(max(vals))}
        viol = self.violations()
        worst = None
        if n:
            worst = max(PHASES, key=lambda p: phases[p]["p99"])
        return {"control_dt": self.control_dt,
                "control_hz": 1.0 / self.control_dt,
                "budget_ms": self.budget_ms,
                "n_samples": n,
                "violations": viol,
                "violation_rate": (float(viol) / float(n)) if n else 0.0,
                "percentile_method": PERCENTILE_METHOD,
                "worst_phase": worst,
                "phases": phases}

    def to_json(self, path):
        payload = {"summary": self.summary(), "samples": [dict(r) for r in self.records]}
        d = os.path.dirname(os.path.abspath(path))
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.write(chr(10))
        return path


# --------------------------------------------------------------------- observation checks
def _canonical_camera(name):
    key = str(name).strip().lower()
    for canon, aliases in CAMERA_ALIASES.items():
        if key in aliases:
            return canon
    return None


def privileged_paths(obj, prefix="", depth=0):
    """Dotted paths of every privileged ground-truth field found in the observation."""
    found = []
    if depth > PRIVILEGED_SCAN_DEPTH:
        return found
    if isinstance(obj, dict):
        for key, val in obj.items():
            path = "%s.%s" % (prefix, key) if prefix else str(key)
            low = str(key).strip().lower()
            if low in PRIVILEGED_EXACT or any(low.startswith(p) for p in PRIVILEGED_PREFIXES):
                found.append(path)
            found.extend(privileged_paths(val, path, depth + 1))
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            found.extend(privileged_paths(val, "%s[%d]" % (prefix, i), depth + 1))
    return found


def assert_no_privileged(observation):
    bad = privileged_paths(observation)
    if bad:
        raise R42RoboDojoError(
            "privileged ground-truth field(s) %s may never be fed to the 42D policy (they are "
            "diagnostics only, use .diagnostics(observation)); the observation is refused" % bad)
    return True


def resolve_images(observation):
    """Observation images -> validated RGB dict in the frozen camera order."""
    if not isinstance(observation, dict):
        raise R42RoboDojoError("observation must be a dict, got %r" % (type(observation),))
    if "images" not in observation:
        raise R42RoboDojoError("observation must carry an images entry")
    raw = observation["images"]
    names = observation.get("camera_names")
    channel_order = observation.get("channel_order")

    if isinstance(raw, dict):
        raw = dict(raw)
        if "channel_order" in raw:
            channel_order = raw.pop("channel_order")
        if names is not None:
            declared = [_canonical_camera(n) for n in names]
            if declared != list(CAMERA_ORDER):
                raise R42RoboDojoError("declared camera_names %s do not resolve to the frozen "
                                       "order %s" % (list(names), list(CAMERA_ORDER)))
        present = {}
        for key, val in raw.items():
            canon = _canonical_camera(key)
            if canon is None:
                raise R42RoboDojoError(
                    "unexpected camera key %r; the frozen order is %s (aliases %s)"
                    % (key, list(CAMERA_ORDER), CAMERA_ALIASES))
            if canon in present:
                raise R42RoboDojoError(
                    "ambiguous cameras: two keys resolve to %r, refusing to guess which image "
                    "is which" % canon)
            present[canon] = val
        missing = [c for c in CAMERA_ORDER if c not in present]
        if missing:
            raise R42RoboDojoError("missing camera(s) %s" % missing)
        ordered = {c: present[c] for c in CAMERA_ORDER}
    elif isinstance(raw, (list, tuple)):
        if names is None:
            raise R42RoboDojoError(
                "positional images require a declared camera_names == %s; a bare list cannot "
                "prove that the cameras were not swapped" % (list(CAMERA_ORDER),))
        if len(raw) != len(CAMERA_ORDER):
            raise R42RoboDojoError("expected %d positional images, got %d"
                                   % (len(CAMERA_ORDER), len(raw)))
        declared = [_canonical_camera(n) for n in names]
        if declared != list(CAMERA_ORDER):
            raise R42RoboDojoError("declared camera_names %s are not the frozen order %s"
                                   % (list(names), list(CAMERA_ORDER)))
        ordered = {c: img for c, img in zip(CAMERA_ORDER, raw)}
    else:
        raise R42RoboDojoError("images must be a dict keyed by camera name or a declared list, "
                               "got %r" % (type(raw),))

    if channel_order is not None:
        ordered["channel_order"] = channel_order
    try:
        imgs = R.validate_images(ordered)
    except R.R42Error as exc:
        raise R42RoboDojoError("camera contract violated: %s" % exc)
    if tuple(imgs.keys()) != CAMERA_ORDER:
        raise R42RoboDojoError("resolved camera order %s != frozen %s"
                               % (tuple(imgs.keys()), CAMERA_ORDER))
    return imgs


# ------------------------------------------------------------------------------- adapter
class R42RoboDojoAdapter:
    """RoboDojo observation -> 42D policy server -> rate-limited 14D command, with timing.

    server      : r42_deploy_server.R42DeployServer (or anything exposing .metadata, .act,
                  .reset, .replay with the same signatures); a stub backend makes it CPU-testable.
    control_dt  : the client control period in seconds (25 Hz -> 0.04).  The action returned by
                  one step() is held for exactly this long by the sim before the next observation.
    max_rate    : optional 14D rad/s override of the checkpoint rate rule.  None (default) uses
                  metadata safety_limits.max_rate when present, exactly as decode_with_metadata
                  would with prev_cmd/dt supplied; if the checkpoint declares no max_rate, no
                  rate rule is applied and only the physical box (applied by the server) binds.
    recorder    : LatencyRecorder; created with control_dt when omitted.
    """

    def __init__(self, server, control_dt=CONTROL_DT, max_rate=None, recorder=None):
        if not callable(getattr(server, "act", None)):
            raise R42RoboDojoError(
                "server must expose act(images, state, ...) (R42DeployServer); got %r"
                % (type(server),))
        metadata = getattr(server, "metadata", None)
        if not isinstance(metadata, dict):
            raise R42RoboDojoError("server must expose the validated 21-field R42 metadata dict")
        R.validate_metadata(metadata)
        dt = float(control_dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise R42RoboDojoError("control_dt must be finite and > 0, got %r" % (control_dt,))
        self.server = server
        self.metadata = metadata
        self.control_dt = dt
        self.max_rate = None if max_rate is None else _vec14(max_rate, "max_rate")
        if self.max_rate is not None and (self.max_rate < 0).any():
            raise R42RoboDojoError("max_rate must be non-negative")
        self.recorder = recorder if recorder is not None else LatencyRecorder(control_dt=dt)
        self._prev_cmd = {}
        self.n_steps = 0

    # ------------------------------------------------------------------ introspection
    def rate_rule(self):
        """(rate vector or None, source string) actually applied at the control boundary."""
        if self.max_rate is not None:
            return self.max_rate, "adapter_override"
        limits = self.metadata.get("safety_limits") or {}
        if limits.get("max_rate") is not None:
            return _vec14(limits["max_rate"], "safety_limits.max_rate"), "checkpoint"
        return None, "none"

    def contract(self):
        """The wiring as observed by the adapter, for evidence files and the R16 report."""
        rate, src = self.rate_rule()
        return {"camera_order": list(CAMERA_ORDER),
                "camera_aliases": {k: list(v) for k, v in CAMERA_ALIASES.items()},
                "state_dim": STATE_DIM,
                "proprio_dim": int(self.metadata["proprio_dim"]),
                "n_action_tokens": N_ACTION_TOKENS,
                "action_token_window": list(R.CODEC_WINDOW),
                "control_dt": self.control_dt,
                "control_hz": 1.0 / self.control_dt,
                "budget_ms": self.control_dt * 1e3,
                "rate_rule_source": src,
                "max_rate": None if rate is None else [float(x) for x in rate],
                "privileged_policy": "refused_before_policy",
                "task_id": self.metadata["task_id"],
                "alignment_version": self.metadata["alignment_version"],
                "normalizer_sha256": self.metadata["normalizer_sha256"],
                "deviation": "CPU wiring only; no closed-loop success rate is claimed",
                "device": getattr(self.server, "device", "unknown")}

    # ---------------------------------------------------------------------- lifecycle
    def reset(self, env_idx=0):
        """Clear the per-env 42D history and the held command of that env only."""
        env = int(env_idx)
        self.server.reset(env)
        self._prev_cmd.pop(env, None)
        return True

    def replay(self, log):
        """Re-execute a recorded step; delegated to the server, which never updates a weight."""
        return self.server.replay(log)

    def diagnostics(self, observation):
        """Read privileged fields for logging only.  Returned data never reaches the policy."""
        out = {}
        _collect_privileged(observation, out)
        return out

    # ------------------------------------------------------------------------- one step
    def step(self, observation, env_idx=None, record=False):
        """One control step: observation -> 14 legal tokens -> rate-limited 14D command."""
        t_wall = time.perf_counter()
        if not isinstance(observation, dict):
            raise R42RoboDojoError("observation must be a dict, got %r" % (type(observation),))
        assert_no_privileged(observation)          # before anything is computed or sent
        images = resolve_images(observation)
        state = _vec14(observation.get("state"), "observation.state")
        env = int(env_idx if env_idx is not None else observation.get("env_idx", 0))
        timestamp = observation.get("timestamp")
        want_log = bool(record or observation.get("record", False))

        out = self.server.act(images, state, env_idx=env, timestamp=timestamp, record=want_log)
        n_tok = int(out.get("n_action_tokens", -1))
        if n_tok != N_ACTION_TOKENS:
            raise R42RoboDojoError("policy produced %d action token(s); the 42D contract is %d"
                                   % (n_tok, N_ACTION_TOKENS))
        action = _vec14(out["action"], "server action")

        rate, _src = self.rate_rule()
        rate_info = {"enabled": rate is not None, "rate_limited": 0, "max_step": None}
        if rate is not None:
            action, rate_info = self._apply_rate(action, state, env, rate)
        self._prev_cmd[env] = action.copy()

        lat = dict(out.get("latency_ms") or {})
        lat["wall_ms"] = (time.perf_counter() - t_wall) * 1e3
        rec = self.recorder.record(lat)
        rec["budget_ms"] = self.recorder.budget_ms
        rec["wall_ms"] = lat["wall_ms"]
        self.n_steps += 1

        res = {"action14": action,
               "n_action_tokens": n_tok,
               "latency": rec,
               "env_idx": env,
               "rate": rate_info,
               "bins": np.asarray(out.get("bins", []), np.int64),
               "ids": np.asarray(out.get("ids", []), np.int64),
               "info": out.get("info", {})}
        if want_log and "log" in out:
            res["log"] = out["log"]
        return res

    def _apply_rate(self, action, state, env, rate):
        """Rate-limit at the control period, seeded (once) from the observed state.

        Safety: prev and the target are both inside the physical box (the server clipped the
        target; prev is checked here), so prev + clip(target - prev, -step, +step) stays inside
        the box by convexity - the same argument as r42_adapter.decode_command.  The result is
        re-checked against the box and refused if the constraints are inconsistent.
        """
        limits = self.metadata.get("safety_limits") or {}
        lo = _vec14(limits.get("amin"), "safety_limits.amin")
        hi = _vec14(limits.get("amax"), "safety_limits.amax")
        tol = 1e-4
        prev = self._prev_cmd.get(env)
        seeded = prev is None
        if seeded:
            prev = state.copy()
        if not np.all((prev >= lo - tol) & (prev <= hi + tol)):
            raise R42RoboDojoError(
                "previous command of env %s is outside the joint box; the rate rule is "
                "undefined, refusing to invent a command" % env)
        prev = np.clip(prev, lo, hi).astype(np.float32)
        max_step = (rate * self.control_dt).astype(np.float32)
        delta = action - prev
        limited = int((np.abs(delta) > max_step + 1e-6).sum())
        cmd = (prev + np.clip(delta, -max_step, max_step)).astype(np.float32)
        if np.any(cmd < lo - tol) or np.any(cmd > hi + tol):
            raise R42RoboDojoError("action_unsafe: rate-limited command left the joint box")
        cmd = np.clip(cmd, lo, hi).astype(np.float32)
        return cmd, {"enabled": True, "rate_limited": limited,
                     "max_step": [float(x) for x in max_step],
                     "seeded_from_state": bool(seeded)}


def _collect_privileged(obj, out, prefix="", depth=0):
    if depth > PRIVILEGED_SCAN_DEPTH:
        return out
    if isinstance(obj, dict):
        for key, val in obj.items():
            path = "%s.%s" % (prefix, key) if prefix else str(key)
            low = str(key).strip().lower()
            if low in PRIVILEGED_EXACT or any(low.startswith(p) for p in PRIVILEGED_PREFIXES):
                out[path] = val
            else:
                _collect_privileged(val, out, path, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            _collect_privileged(val, out, "%s[%d]" % (prefix, i), depth + 1)
    return out


# ------------------------------------------------------------------------------ self check
def _selftest():
    rec = LatencyRecorder(control_dt=CONTROL_DT)
    for total in (10.0, 20.0, 30.0, 40.0):
        rec.record({"preprocess": 0.5, "forward_total": total - 1.0, "decode": 0.5,
                    "total": total})
    s = rec.summary()
    assert s["budget_ms"] == 40.0 and s["n_samples"] == 4
    assert abs(s["phases"]["total"]["p95"] - 38.5) < 1e-9
    try:
        assert_no_privileged({"images": {}, "state": [], "target_t": [0, 0, 0]})
    except R42RoboDojoError:
        pass
    else:
        raise AssertionError("privileged target_t was not refused")
    return s


if __name__ == "__main__":
    print(json.dumps(_selftest(), indent=1, sort_keys=True))

