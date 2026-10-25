#!/usr/bin/env python3
"""R16 (G1/G2/G3): OpenVLA_R42 as an XPolicyLab PolicyServer model.

class Model(ModelTemplate) is the exact server-side interface the XPolicyLab WebSocket policy
server requires (XPolicyLab/model_template.py:42-97; ws/model_server.py:245-297 calls
update_obs(observation) then get_action() and replies with whatever get_action returns):

    update_obs(obs)     RoboDojo observation dict -> packed inputs (3 RGB cameras in the frozen
                        r42_adapter.CAMERA_ORDER + flat 14D state); fail-closed in r42_env_io
    get_action()        -> {"actions": <4-key dict>, "latency_ms": {preprocess, forward_total,
                        decode, total}, ...}; the 4-key dict is exactly what
                        eval_env.take_action() consumes
    reset()             -> resets EVERY env (the WS server never passes env_idx; G9)
    prepare_case(...)   -> bookkeeping hook
    on_trial_end(...)   -> bookkeeping hook

Construction is CPU-only and lazy: __init__ validates config + metadata and touches NO torch
device, so the whole interface can be imported and unit tested on a box whose GPU is busy.  The
7B base model, the LoRA adapter and the 42D proprio projector are only loaded on the first
get_action() (or an explicit warmup()), through the existing R42DeployServer.

The deployed file layout is identical to the other XPolicyLab policies:

    XPolicyLab/policy/OpenVLA_R42/{__init__.py, model.py, deploy.py, deploy.yml,
                                   setup_eval_policy_server.sh, r42_serving pieces}
"""
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PURE = os.path.abspath(os.path.join(_HERE, "..", "OpenVLA_Pure"))
for _p in (_HERE, _PURE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r42_adapter as R          # noqa: E402
import r42_deploy_server as S    # noqa: E402  (torch is only imported inside .load())
import r42_env_io as E           # noqa: E402
import r42_runtime_audit as Audit  # noqa: E402 (stdlib-only, no model/GPU imports)

LATENCY_PHASES = ("preprocess", "forward_total", "decode", "total")
LORA_DIRNAME = "lora_adapter"
PROJECTOR_NAME = "proprio_projector_42.pt"
CODEC_MODE = "fixed"
INSTRUCTION = S.INSTRUCTION      # the single trained instruction, one source of truth
DEFAULT_CONTROL_HZ = E.EXPECTED_CONTROL_HZ


# ------------------------------------------------------------------ template resolution
def _resolve_template():
    """Use the real XPolicyLab template when it is importable, else a faithful stand-in.

    Two situations are legitimate:
      * inside the RoboDojo serving tree  -> XPolicyLab.model_template is the real class;
      * inside the r11 repo alone (CPU audit, unit tests) -> RoboDojo is not on sys.path and the
        r11 repo deliberately does not vendor it, so a minimal stand-in with the same six-method
        surface is used and TEMPLATE_SOURCE records that this happened.
    """
    try:
        from XPolicyLab.model_template import ModelTemplate as T
        return T, "XPolicyLab.model_template"
    except Exception:
        pass

    class ModelTemplate(object):
        """Minimal stand-in for XPolicyLab/model_template.py (interface-compatible)."""

        RED = "\033[31m"
        RESET = "\033[0m"

        def __init__(self):
            self.model = None

        def _error_msg(self, msg):
            return "%s[ERROR] %s%s" % (self.RED, msg, self.RESET)

        def update_obs(self, obs):
            raise NotImplementedError(self._error_msg("update_obs() must be implemented"))

        def update_obs_batch(self, obs_list):
            raise NotImplementedError(self._error_msg("update_obs_batch() must be implemented"))

        def get_action(self):
            raise NotImplementedError(self._error_msg("get_action() must be implemented"))

        def get_action_batch(self, env_idx_list=None):
            raise NotImplementedError(self._error_msg("get_action_batch() must be implemented"))

        def reset(self):
            pass

        def prepare_case(self, case_meta=None):
            pass

        def on_trial_end(self, result=None):
            pass

    return ModelTemplate, "builtin_standin"


ModelTemplate, TEMPLATE_SOURCE = _resolve_template()
_CODE = "OpenVLA_R42"
_STATE_DIM_ORDER = tuple(
    "%s[%d]" % (key, i)
    for key, dim in zip(E.JOINT_ORDER, E.JOINT_DIMS)
    for i in range(dim)
)


class R42ModelError(R.R42Error):
    """Any violation of the XPolicyLab <-> 42D server contract."""


def _iso(epoch_seconds):
    """Epoch seconds -> ISO-8601 UTC (the same format the wire Frame uses)."""
    return datetime.datetime.fromtimestamp(
        float(epoch_seconds), datetime.timezone.utc).isoformat()


def _check_phases(latency_ms):
    """G6: the four server-side phases must be present, finite and non-negative."""
    if not isinstance(latency_ms, dict):
        raise R42ModelError("server latency must be a dict with phases %s, got %r"
                            % (list(LATENCY_PHASES), type(latency_ms)))
    out = {}
    for phase in LATENCY_PHASES:
        if phase not in latency_ms:
            raise R42ModelError("server latency is missing phase %r (has %s)"
                                % (phase, sorted(latency_ms)))
        try:
            value = float(latency_ms[phase])
        except (TypeError, ValueError):
            raise R42ModelError("latency phase %r is not a number: %r"
                                % (phase, latency_ms[phase]))
        if not np.isfinite(value) or value < 0.0:
            raise R42ModelError("latency phase %r must be finite and >= 0, got %r"
                                % (phase, value))
        out[phase] = value
    if out["total"] + 1e-9 < max(out["preprocess"], out["forward_total"], out["decode"]):
        raise R42ModelError("latency total %.6f ms is smaller than a single phase %r"
                            % (out["total"], out))
    return out


class Model(ModelTemplate):
    """42D [S,V,Acc] OpenVLA-R42 Push-T policy behind the XPolicyLab model interface."""

    def __init__(self, model_cfg):
        super().__init__()
        if not isinstance(model_cfg, dict):
            raise R42ModelError("deploy_cfg must be a dict, got %r" % (type(model_cfg),))
        self.model_cfg = dict(model_cfg)
        self.policy_name = model_cfg.get("policy_name", _CODE)
        self.device = str(model_cfg.get("device", "cuda"))
        self.control_hz = float(model_cfg.get("control_hz", DEFAULT_CONTROL_HZ))
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise R42ModelError("control_hz must be finite and > 0")
        self.control_dt = 1.0 / self.control_hz
        expected_hw = model_cfg.get("expected_hw", E.EXPECTED_RGB_SHAPE)
        self.expected_hw = None if expected_hw is None else tuple(int(x) for x in expected_hw)
        self.checkpoint_dir = model_cfg.get("checkpoint_dir")
        self.metadata_path = model_cfg.get("metadata_path")
        self.base_model_path = model_cfg.get("base_model_path")
        self.lora_dir = model_cfg.get("adapter_dir") or model_cfg.get("lora_dir")
        self.projector_path = model_cfg.get("proprio_projector_path")
        if self.checkpoint_dir:
            # default to the R15 full-state layout; never fall back to "no adapter at all"
            if not self.lora_dir:
                self.lora_dir = os.path.join(self.checkpoint_dir, LORA_DIRNAME)
            if not self.projector_path:
                self.projector_path = os.path.join(self.checkpoint_dir, PROJECTOR_NAME)
        self.instruction = model_cfg.get("prompt") or model_cfg.get("instruction") or INSTRUCTION
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise R42ModelError("instruction must be a non-empty string")
        if self.instruction.strip() != INSTRUCTION and \
                not bool(model_cfg.get("allow_prompt_override", False)):
            raise R42ModelError(
                "prompt %r differs from the instruction this checkpoint was trained on (%r); "
                "a different prompt silently invalidates the policy - set allow_prompt_override "
                "to accept that risk explicitly" % (self.instruction.strip(), INSTRUCTION))

        self._server = None
        self._inputs = None
        self._t_obs_wall = None
        self._t_obs_perf = None
        self._warned = set()
        self._n_steps = 0
        self.n_resets = 0
        self.last_payload = None
        self._audit_trace = None
        self._audit_episode_index = 0
        self._audit_episode_step = 0
        self._audit_prev_response_action14 = None
        self._audit_prev_response_global_step_index = None
        self._audit_current_observation_record_index = None
        self._audit_current_observation_index = None
        self._audit_observation_index = 0
        self._audit_current_camera_inputs = None
        self._audit_backend_ready_logged = False
        self.metadata = self._resolve_metadata()

        # G12: the served codec mode is printed and asserted, never implied
        self.codec_mode = str(self.metadata["codec_mode"])
        if self.codec_mode != CODEC_MODE:
            raise R42ModelError("served codec mode %r != %r: the 42D checkpoint was trained with "
                                "the fixed 256-bin codec; refusing to serve a different decode"
                                % (self.codec_mode, CODEC_MODE))
        if list(self.metadata["action_token_window"]) != list(R.CODEC_WINDOW):
            raise R42ModelError("action_token_window %s != %s"
                                % (self.metadata["action_token_window"],
                                   list(R.CODEC_WINDOW)))
        self._configure_runtime_audit()
        print("[%s] template=%s codec_mode=%s (asserted) window=%s n_actions=%d "
              "camera_order=%s control_hz=%.3g metadata=%s device=%s"
              % (_CODE, TEMPLATE_SOURCE, self.codec_mode.upper(), list(R.CODEC_WINDOW),
                 R.N_ACTIONS, list(E.CAMERA_ORDER), self.control_hz,
                 self._metadata_source, self.device), flush=True)
        if self.device.startswith("cuda"):
            print("[%s] backend is lazy: no model is loaded until the first get_action()"
                  % _CODE, flush=True)

    # ------------------------------------------------------------------ optional runtime audit
    def _configure_runtime_audit(self):
        """Enable a durable per-request trace only when an explicit path is supplied.

        If the path is set, a run ID and writable, non-existing destination are mandatory. The
        writer is initialized before lazy model loading so an audit setup failure cannot launch an
        unlogged policy server.
        """
        if "R42_AUDIT_LOG_PATH" not in os.environ:
            return
        path = os.environ.get("R42_AUDIT_LOG_PATH")
        run_id = os.environ.get("R42_AUDIT_RUN_ID")
        manifest_path = None
        manifest_sha256 = None
        if self.checkpoint_dir:
            candidate = os.path.join(self.checkpoint_dir, "MANIFEST.json")
            if os.path.isfile(candidate):
                with open(candidate, "rb") as fh:
                    manifest_sha256 = hashlib.sha256(fh.read()).hexdigest()
                manifest_path = candidate
        normalizer_payload_sha256 = None
        if self.metadata.get("normalizer") is not None:
            canonical_normalizer = json.dumps(
                self.metadata["normalizer"], sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")
            normalizer_payload_sha256 = hashlib.sha256(canonical_normalizer).hexdigest()
        run_metadata = {
            "policy_name": self.policy_name,
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_manifest_path": manifest_path,
            "checkpoint_manifest_sha256": manifest_sha256,
            "embedded_normalizer_payload_sha256": normalizer_payload_sha256,
            "base_model_path": self.base_model_path,
            "base_model_sha256": self.metadata.get("base_model_sha256"),
            "lora_dir": self.lora_dir,
            "projector_path": self.projector_path,
            "metadata_source": self._metadata_source,
            "task_id": self.metadata.get("task_id"),
            "device": self.device,
            "control_hz": self.control_hz,
            "codec_mode": self.codec_mode,
            "normalizer_sha256": self.metadata.get("normalizer_sha256"),
            "lora_sha256": self.metadata.get("lora_sha256"),
            "projector_sha256": self.metadata.get("projector_sha256"),
            "state_dim_order": list(_STATE_DIM_ORDER),
            "joint_group_order": list(E.JOINT_ORDER),
            "camera_order": list(E.CAMERA_ORDER),
            "raw_images_persisted": False,
            "camera_state_wallclock_sync": "UNKNOWN",
            "python_version": sys.version.split()[0],
            "pid": os.getpid(),
            "durability": "flush_and_fsync_each_record",
        }
        self._audit_trace = Audit.RuntimeAuditWriter(path, run_id, run_metadata)
        print("[%s] runtime audit enabled: %s (run_id=%s)"
              % (_CODE, self._audit_trace.path, run_id), flush=True)

    def _audit_observation_received(self, inputs):
        """Persist the validated observation before inference can fail."""
        if self._audit_trace is None:
            return

        state = np.asarray(inputs["state"], np.float32).reshape(-1)
        camera_inputs = {}
        for camera in E.CAMERA_ORDER:
            arr = np.ascontiguousarray(inputs["images"][camera])
            camera_inputs[camera] = {
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "shape": [int(x) for x in arr.shape],
                "dtype": str(arr.dtype),
            }
        record = self._audit_trace.write({
            "event": "observation_received",
            "observation_index": int(self._audit_observation_index),
            "episode_index": int(self._audit_episode_index),
            "episode_step_index": int(self._audit_episode_step),
            "global_step_index": int(self._n_steps),
            "env_idx": int(inputs["env_idx"]),
            "source_timestamp": inputs.get("timestamp"),
            "frequency_hz": inputs.get("frequency"),
            "observation_wall_utc": _iso(self._t_obs_wall),
            "state_dim_order": list(_STATE_DIM_ORDER),
            "state14": state.tolist(),
            "camera_inputs": camera_inputs,
            "camera_state_wallclock_sync": "UNKNOWN",
            "raw_images_persisted": False,
        })
        self._audit_current_observation_record_index = int(record["record_index"])
        self._audit_current_observation_index = int(self._audit_observation_index)
        self._audit_observation_index += 1
        self._audit_current_camera_inputs = camera_inputs

    def _audit_action_error(self, inputs, stage, exc):
        """Record a failed inference/contract stage against the received observation."""
        if self._audit_trace is None:
            return
        self._audit_trace.write({
            "event": "action_error",
            "episode_index": int(self._audit_episode_index),
            "episode_step_index": int(self._audit_episode_step),
            "global_step_index": int(self._n_steps),
            "env_idx": int(inputs["env_idx"]),
            "observation_record_index": self._audit_current_observation_record_index,
            "observation_index": self._audit_current_observation_index,
            "stage": str(stage),
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:2000],
        })

    def _audit_action_step(self, inputs, out, payload, stamp, phases):
        """Persist the generated response associated with the preceding observation record."""
        if self._audit_trace is None:
            return

        state = np.asarray(inputs["state"], np.float32).reshape(-1)
        response_action = np.asarray(out["action"], np.float32).reshape(-1)
        previous = self._audit_prev_response_action14
        state_minus_previous = None
        if previous is not None:
            state_minus_previous = (state - previous).astype(np.float32).tolist()

        camera_inputs = self._audit_current_camera_inputs or {}
        self._audit_trace.write({
            "event": "action_step",
            "observation_record_index": self._audit_current_observation_record_index,
            "observation_index": self._audit_current_observation_index,
            "episode_index": int(self._audit_episode_index),
            "episode_step_index": int(self._audit_episode_step),
            "global_step_index": int(self._n_steps),
            "env_idx": int(inputs["env_idx"]),
            "source_timestamp": inputs.get("timestamp"),
            "frequency_hz": inputs.get("frequency"),
            "observation_wall_utc": stamp["obs_iso"],
            "response_wall_utc": stamp["reply_iso"],
            "server_span_ms": float(stamp["server_span_ms"]),
            "state_dim_order": list(_STATE_DIM_ORDER),
            "state14": state.tolist(),
            "velocity14": np.asarray(out.get("velocity", []), np.float32).reshape(-1).tolist(),
            "features42": np.asarray(out.get("features", []), np.float32).reshape(-1).tolist(),
            "camera_inputs": camera_inputs,
            "prediction": {
                "token_ids14": list(payload["ids"]),
                "bins14": list(payload["bins"]),
                "residual14": np.asarray(out.get("residual", []), np.float32).reshape(-1).tolist(),
            },
            "server_response": {
                "action14": list(payload["action14"]),
                "action_dict": payload["actions"],
                "decoder_info": out.get("info", {}),
                "semantics": "server-generated response; client acknowledgement/execution is not observed here",
            },
            "previous_server_response_action14": (
                None if previous is None else previous.astype(np.float32).tolist()),
            "previous_response_global_step_index": self._audit_prev_response_global_step_index,
            "observed_state_minus_previous_server_response14": state_minus_previous,
            "comparison_semantics": (
                "current observed state minus previous server-generated response; not proof of actuation"),
            "camera_state_wallclock_sync": "UNKNOWN",
            "latency_ms": dict(phases),
            "n_action_tokens": int(payload["n_action_tokens"]),
            "codec_mode": self.codec_mode,
        })
        self._audit_prev_response_action14 = response_action.copy()
        self._audit_prev_response_global_step_index = int(self._n_steps)
        self._audit_episode_step += 1

    # ------------------------------------------------------------------ metadata resolution
    def _resolve_metadata(self):
        injected = self.model_cfg.get("metadata")
        if injected is not None:
            R.validate_metadata(injected, expect_task=self.model_cfg.get("task_id"))
            self._metadata_source = "deploy_cfg.metadata"
            return injected
        path = self.metadata_path
        if path is None and self.checkpoint_dir:
            candidate = os.path.join(self.checkpoint_dir, "r42_metadata.json")
            if os.path.isfile(candidate):
                path = candidate
        if path is not None:
            if not os.path.isfile(path):
                raise R42ModelError("metadata_path does not exist: %s" % path)
            meta = R.load_metadata(path, expect_task=self.model_cfg.get("task_id"))
            self._metadata_source = path
            return meta
        if not self.checkpoint_dir:
            raise R42ModelError(
                "deploy_cfg must provide one of metadata / metadata_path / checkpoint_dir "
                "(an R15 stage checkpoint has no r42_metadata.json, so checkpoint_dir goes "
                "through r42_metadata_bridge)")
        import r42_metadata_bridge as B
        meta, report = B.build_stage_metadata(
            self.checkpoint_dir, task_id=self.model_cfg.get("task_id", "push_T"),
            seed=int(self.model_cfg.get("seed", B.DEFAULT_SEED)),
            max_rate=self.model_cfg.get("max_rate", B.DEFAULT_MAX_RATE),
            base_model_sha256=self.model_cfg.get("base_model_sha256"),
            base_model_config=self.model_cfg.get("base_model_config",
                                                 B.DEFAULT_BASE_MODEL_CONFIG))
        print("[%s] metadata bridged from %s (manifest=%s, steps=%d, lora=%s..)"
              % (_CODE, self.checkpoint_dir, report["manifest_source"],
                 report["train_steps"], report["lora_sha256"][:12]), flush=True)
        self.bridge_report = report
        self._metadata_source = "r42_metadata_bridge(%s)" % self.checkpoint_dir
        return meta

    # ------------------------------------------------------------------------- lazy backend
    def _ensure_server(self):
        if self._server is not None:
            return self._server
        backend = self.model_cfg.get("backend")
        server = S.R42DeployServer(
            self.metadata, backend=backend, device=self.device,
            tokenizer_prompt_ids=self.model_cfg.get("tokenizer_prompt_ids"))
        if backend is None:
            if not self.base_model_path:
                raise R42ModelError("base_model_path is required to load the 7B backend")
            # Fail closed BEFORE touching CUDA: serving the pristine 7B (or 14D projector) as if
            # it were the R42 policy would look like a working server and produce nonsense.
            if not (self.lora_dir and os.path.isdir(self.lora_dir)):
                raise R42ModelError(
                    "no LoRA adapter at %r: refusing to serve the base 7B as if it were the R42 "
                    "policy (set adapter_dir or point checkpoint_dir at a full-state checkpoint)"
                    % (self.lora_dir,))
            if not (self.projector_path and os.path.isfile(self.projector_path)):
                raise R42ModelError(
                    "no 42D proprio projector at %r: refusing to serve a 14D/random projector"
                    % (self.projector_path,))
            try:
                import torch
                from transformers import AutoProcessor
            except ImportError as exc:
                raise R42ModelError("the real backend needs torch + transformers: %s" % exc)
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise R42ModelError("device=%r but CUDA is not available; refusing to fall back "
                                    "to CPU for a 7B deploy" % self.device)
            processor = AutoProcessor.from_pretrained(self.base_model_path,
                                                      trust_remote_code=True)
            server.processor = processor
            server.build_prompt(processor)
            server.load(self.base_model_path, lora_dir=self.lora_dir,
                        projector_path=self.projector_path)
            server.assert_frozen()
        self._server = server
        try:
            prompt_len = len(server.prompt_ids())
        except R.R42Error:
            prompt_len = -1
        print("[%s] backend ready: %s prompt_len=%d device=%s"
              % (_CODE, type(server.backend).__name__, prompt_len, self.device), flush=True)
        return server

    def warmup(self):
        """Load the backend eagerly (the server path does this on the first step anyway)."""
        self._ensure_server()
        return True

    # ------------------------------------------------------------------------------ hooks
    def prepare_case(self, case_meta=None):
        self.case_meta = dict(case_meta or {})
        return {"prepared": True, "env_idx": self.case_meta.get("env_idx")}

    def on_trial_end(self, result=None):
        self.trial_result = dict(result or {})
        return {"n_steps": self._n_steps}

    # -------------------------------------------------------------------------------- obs
    def update_obs(self, obs):
        """Observation dict -> packed inputs.  Fails closed before any GPU work happens."""
        self._inputs = E.obs_dict_to_inputs(obs, expected_hw=self.expected_hw,
                                            expected_hz=self.control_hz)
        E.assert_no_action_leak(self._inputs)
        self._t_obs_wall = time.time()
        self._t_obs_perf = time.perf_counter()
        obs_instr = self._inputs.get("instruction")
        if obs_instr and obs_instr.strip() != self.instruction.strip():
            self._warn("observation instruction %r is ignored: this checkpoint runs the fixed "
                       "training instruction %r" % (obs_instr.strip(), self.instruction.strip()))
        self._audit_observation_received(self._inputs)
        return None

    def update_obs_batch(self, obs_list):
        obs_list = list(obs_list)
        if len(obs_list) != 1:
            raise R42ModelError("OpenVLA_R42 serves one environment per server "
                                "(got %d observations)" % len(obs_list))
        self.update_obs(obs_list[0])
        return None

    def _warn(self, msg):
        if msg in self._warned:
            return
        self._warned.add(msg)
        print("[%s][WARN] %s" % (_CODE, msg), flush=True)

    # --------------------------------------------------------------------------- get_action
    def get_action(self):
        """One control step -> 4-key absolute joint action dict + the four latency phases."""
        if self._inputs is None:
            raise R42ModelError("get_action() requires a preceding update_obs(observation)")
        inputs = self._inputs
        try:
            server = self._ensure_server()
        except Exception as exc:
            self._audit_action_error(inputs, "backend_load", exc)
            raise
        if self._audit_trace is not None and not self._audit_backend_ready_logged:
            self._audit_trace.write({
                "event": "backend_ready",
                "backend_type": type(server.backend).__name__ if server.backend is not None else None,
                "device": self.device,
                "metadata_source": self._metadata_source,
                "checkpoint_dir": self.checkpoint_dir,
            })
            self._audit_backend_ready_logged = True
        t0 = time.perf_counter()
        try:
            out = server.act(inputs["images"], inputs["state"], env_idx=inputs["env_idx"],
                             timestamp=inputs["timestamp"])
        except Exception as exc:
            self._audit_action_error(inputs, "server_act", exc)
            raise
        try:
            phases = _check_phases(out.get("latency_ms"))
            action = E.cmd14_to_action_dict(out["action"])
            stamp = {"obs_iso": _iso(self._t_obs_wall),
                     "reply_iso": _iso(time.time()),
                     "server_span_ms": (time.perf_counter() - self._t_obs_perf) * 1e3}
            payload = {"actions": action,
                       "latency_ms": phases,
                       "stamp": stamp,
                       "n_action_tokens": int(out.get("n_action_tokens", -1)),
                       "codec_mode": self.codec_mode,
                       "env_idx": int(inputs["env_idx"]),
                       "action14": [float(x) for x in np.asarray(out["action"], np.float32).reshape(-1)],
                       "ids": [int(x) for x in np.asarray(out.get("ids", []), np.int64).reshape(-1)],
                       "bins": [int(x) for x in np.asarray(out.get("bins", []), np.int64).reshape(-1)],
                       "model_total_ms": (time.perf_counter() - t0) * 1e3}
            if payload["n_action_tokens"] != R.N_ACTIONS or len(payload["ids"]) != R.N_ACTIONS:
                raise R42ModelError("server produced %d action token(s) / %d id(s); the 42D contract "
                                    "is %d" % (payload["n_action_tokens"], len(payload["ids"]),
                                               R.N_ACTIONS))
        except Exception as exc:
            self._audit_action_error(inputs, "response_contract", exc)
            raise
        # Log only after the existing action-token contract passes; an enabled trace failure is
        # fatal so the audited run cannot silently continue without evidence.
        self._audit_action_step(inputs, out, payload, stamp, phases)
        self._n_steps += 1
        self.last_payload = payload
        return payload

    def get_action_batch(self, env_idx_list=None):
        raise R42ModelError("OpenVLA_R42 supports single-env evaluation only "
                            "(no get_action_batch in the 42D deploy contract)")

    # ------------------------------------------------------------------------------- reset
    def reset(self):
        """G9: reset ALL envs.  The WS server calls reset() with no arguments and ignores
        env_idx, so a per-env reset would leak history across episodes."""
        if self._server is not None:
            self._server.reset_all()
        self._inputs = None
        self.n_resets += 1
        self._audit_episode_index += 1
        self._audit_episode_step = 0
        self._audit_prev_response_action14 = None
        self._audit_prev_response_global_step_index = None
        self._audit_current_observation_record_index = None
        self._audit_current_observation_index = None
        self._audit_observation_index = 0
        self._audit_current_camera_inputs = None
        if self._audit_trace is not None:
            self._audit_trace.write({
                "event": "episode_reset",
                "episode_index": int(self._audit_episode_index),
                "reset_count": int(self.n_resets),
                "global_step_count": int(self._n_steps),
            })
        return True

    # --------------------------------------------------------------------------- evidence
    def contract(self):
        """What this model believes it is serving, for logs and the R16 report."""
        return {"policy_name": self.policy_name,
                "template_source": TEMPLATE_SOURCE,
                "codec_mode": self.codec_mode,
                "action_token_window": list(R.CODEC_WINDOW),
                "n_action_tokens": R.N_ACTIONS,
                "metadata_source": self._metadata_source,
                "device": self.device,
                "control_hz": self.control_hz,
                "control_dt": self.control_dt,
                "env_io": E.contract(),
                "latency_phases": list(LATENCY_PHASES),
                "task_id": self.metadata["task_id"],
                "normalizer_sha256": self.metadata["normalizer_sha256"],
                "lora_sha256": self.metadata["lora_sha256"],
                "projector_sha256": self.metadata["projector_sha256"],
                "reset_scope": "all_envs",
                "n_steps": self._n_steps,
                "n_resets": self.n_resets}
