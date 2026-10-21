#!/usr/bin/env python3
"""R16 (G6): client-side RTT and the server-side latency decomposition over the real wire.

The server stamps latency_ms = (time.perf_counter() - start) * 1e3 around update_obs()+get_action()
only (RoboDojo/XPolicyLab/client_server/ws/model_server.py:250-295): it measures inference, not the
round trip.  The matrix Frame.sent_at is written by whoever constructs the frame and is never
compared anywhere in the stack, so nothing in the shipped code measures the client-visible
latency of one control step.

This module adds exactly that, without changing the transport:

  * server phases   : the four phases our model returns (preprocess / forward_total / decode /
                      total), asserted present and finite;
  * server span     : the same-clock span our model stamps around one infer call
                      (stamp.obs_iso -> stamp.reply_iso, plus stamp.server_span_ms from
                      perf_counter), which is skew-free across hosts;
  * client RTT      : wall clock from "about to send" to "reply decoded", measured on the env
                      side with perf_counter;
  * net_ms          : client_rtt_ms - server_span_ms, i.e. everything that is not the policy
                      (serialisation + socket + client bookkeeping);
  * clock_offset_ms : server stamp minus client send time; NON-ZERO means the two hosts'
                      clocks disagree, which is why net_ms, not the offset, is the number used.

Runs on CPU; importing this module does not import websockets, torch or the RoboDojo tree.
"""
import datetime
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PURE = os.path.abspath(os.path.join(_HERE, "..", "OpenVLA_Pure"))
for _p in (_HERE, _PURE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r42_adapter as R      # noqa: E402
import r42_env_io as E       # noqa: E402

PHASES = ("preprocess", "forward_total", "decode", "total")
ACTION_KEYS = tuple(E.JOINT_ORDER)
STAMP_KEYS = ("obs_iso", "reply_iso", "server_span_ms")


class R42WireError(R.R42Error):
    """Anything malformed coming back from the policy server."""


def _iso_to_epoch(value):
    """ISO-8601 (with or without Z) -> epoch seconds."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.datetime.fromisoformat(text).timestamp()


def parse_infer_payload(payload):
    """Validate an INFER_RESULT payload and split it into (action dict, 14D vector, phases)."""
    if not isinstance(payload, dict):
        raise R42WireError("infer payload must be a dict, got %r" % (type(payload),))
    if "actions" not in payload:
        raise R42WireError("infer payload has no 'actions' entry (keys %s)"
                           % sorted(payload))
    action_dict = payload["actions"]
    E.assert_action_dict(action_dict)
    cmd = E.action_dict_to_cmd14(action_dict)

    latency = payload.get("latency_ms")
    if not isinstance(latency, dict):
        raise R42WireError("the server must report the four latency phases as a dict, got %r"
                           % (latency,))
    phases = {}
    for phase in PHASES:
        if phase not in latency:
            raise R42WireError("server latency is missing phase %r (has %s)"
                               % (phase, sorted(latency)))
        try:
            value = float(latency[phase])
        except (TypeError, ValueError):
            raise R42WireError("latency phase %r is not a number: %r" % (phase, latency[phase]))
        if not np.isfinite(value) or value < 0.0:
            raise R42WireError("latency phase %r must be finite and >= 0, got %r"
                               % (phase, value))
        phases[phase] = value
    n_tokens = payload.get("n_action_tokens", R.N_ACTIONS)
    if int(n_tokens) != R.N_ACTIONS:
        raise R42WireError("server produced %r action tokens; the 42D contract is %d"
                           % (n_tokens, R.N_ACTIONS))
    codec = payload.get("codec_mode")
    if codec is not None and str(codec) != "fixed":
        raise R42WireError("server codec mode %r is not the trained fixed codec" % (codec,))
    return {"action_dict": action_dict, "action14": cmd, "phases": phases,
            "n_action_tokens": int(n_tokens),
            "codec_mode": None if codec is None else str(codec),
            "ids": [int(x) for x in payload.get("ids", [])],
            "stamp": payload.get("stamp")}


def decompose(client_send_epoch, client_recv_epoch, payload):
    """One infer round trip -> the full latency decomposition (all values in ms and >= 0-ish)."""
    parsed = parse_infer_payload(payload)
    sample = {"client_rtt_ms": (client_recv_epoch - client_send_epoch) * 1e3,
              "phases": parsed["phases"],
              "server_span_ms": None, "net_ms": None, "clock_offset_ms": None,
              "action14": parsed["action14"], "action_dict": parsed["action_dict"],
              "ids": parsed["ids"], "n_action_tokens": parsed["n_action_tokens"]}
    stamp = parsed["stamp"]
    if isinstance(stamp, dict):
        missing = [k for k in STAMP_KEYS if k not in stamp]
        if not missing:
            span = float(stamp["server_span_ms"])
            if not np.isfinite(span) or span < 0.0:
                raise R42WireError("server span stamp is not a non-negative number: %r" % span)
            sample["server_span_ms"] = span
            sample["net_ms"] = sample["client_rtt_ms"] - span
            sample["clock_offset_ms"] = (
                (_iso_to_epoch(stamp["obs_iso"]) - client_send_epoch) * 1e3)
    for key in ("client_rtt_ms",):
        if not np.isfinite(sample[key]) or sample[key] < 0.0:
            raise R42WireError("%s must be finite and >= 0, got %r" % (key, sample[key]))
    return sample


class R42WsTimingClient(object):
    """Synchronous env-side client: real WS round trips, RTT + phase accounting per step.

    This is the timing-aware sibling of client_server.ws.model_client.WsModelClient (same
    transport, same frames).  WsModelClient.get_action() returns payload['actions'] only, so it
    cannot report latency; this client keeps the whole payload and the send/receive wall clock.
    """

    def __init__(self, url=None, evaluation_id="r42-eval", trial_id="r42-trial",
                 action_case_id=None, repeat_index=None, client=None,
                 ws_ping_interval_s=20.0, ws_ping_timeout_s=20.0, control_dt=E.CONTROL_DT):
        import asyncio
        from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
        from r42_robodojo_adapter import LatencyRecorder

        self.trial_id = trial_id
        self.action_case_id = action_case_id
        self.repeat_index = repeat_index
        self.step = 0
        self.samples = []
        self.recorder = LatencyRecorder(control_dt=control_dt)
        self._loop = asyncio.new_event_loop()
        self._client = client or PolicyEvalClient(PolicyEvalClientConfig(
            url=url, evaluation_id=evaluation_id,
            ws_ping_interval_s=ws_ping_interval_s,
            ws_ping_timeout_s=ws_ping_timeout_s))
        self._loop.run_until_complete(self._client.connect(handshake=True))

    # ----------------------------------------------------------------- protocol operations
    def reset(self, payload=None):
        frame = self._loop.run_until_complete(self._client.reset(
            trial_id=self.trial_id, action_case_id=self.action_case_id,
            repeat_index=self.repeat_index, payload=payload))
        self.step = 0
        return frame.payload.get("result")

    def infer(self, observation, step=None):
        """One control step over the wire; returns the latency decomposition + the action."""
        step = self.step if step is None else int(step)
        send = time.time()
        frame = self._loop.run_until_complete(self._client.infer(
            observation, trial_id=self.trial_id, action_case_id=self.action_case_id,
            step=step))
        recv = time.time()
        sample = decompose(send, recv, dict(frame.payload))
        sample["step"] = step
        sample["server_stamp"] = frame.sent_at
        recorded = dict(sample["phases"])
        recorded["total"] = sample["phases"]["total"]
        self.recorder.record(recorded)
        self.samples.append({"step": step, "client_rtt_ms": sample["client_rtt_ms"],
                             "server_span_ms": sample["server_span_ms"],
                             "net_ms": sample["net_ms"],
                             "phases": sample["phases"]})
        self.step += 1
        return sample

    # --------------------------------------------------------------------------- summary
    def summary(self):
        server = {"p50": None, "p95": None, "p99": None, "n": len(self.samples)}
        spans = [s["server_span_ms"] for s in self.samples if s["server_span_ms"] is not None]
        nets = [s["net_ms"] for s in self.samples if s["net_ms"] is not None]
        if spans:
            from r42_robodojo_adapter import percentile
            server = {"p50": percentile(spans, 50), "p95": percentile(spans, 95),
                      "p99": percentile(spans, 99), "n": len(spans)}
        net = {"p50": None, "p95": None, "p99": None, "n": len(nets)}
        if nets:
            from r42_robodojo_adapter import percentile
            net = {"p50": percentile(nets, 50), "p95": percentile(nets, 95),
                   "p99": percentile(nets, 99), "n": len(nets)}
        rtts = [s["client_rtt_ms"] for s in self.samples]
        rtt = {"p50": None, "p95": None, "p99": None, "n": len(rtts)}
        if rtts:
            from r42_robodojo_adapter import percentile
            rtt = {"p50": percentile(rtts, 50), "p95": percentile(rtts, 95),
                   "p99": percentile(rtts, 99), "n": len(rtts)}
        out = self.recorder.summary()
        out["client_rtt_ms"] = rtt
        out["server_span_ms"] = server
        out["net_ms"] = net
        out["client_steps"] = len(self.samples)
        return out

    def close(self):
        if self._loop.is_closed():
            return
        try:
            self._loop.run_until_complete(self._client.close())
        finally:
            self._loop.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
