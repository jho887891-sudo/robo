#!/usr/bin/env python3
"""R25 M2's explicit-task RoboDojo ModelTemplate adapter.

This is a separate opt-in M2 entry point, NEVER an alias for the frozen
single-task OpenVLA_R42.model.Model.  It uses a verified task-bound
metadata bundle from m2_deploy_contract and the task-specific server.

Real 7B is guarded by allow_real_7b_smoke and source artifact hashes;
no optimizer, background deployment, data edits, or model downloads.
Official closed-loop evaluation still requires remote approval and rig.
"""
import json
import os
import time

import m2_deploy_contract as D
import m2_deploy_server as S
import m2_multitask_data as MD
import r42_adapter as R
import r42_metadata_bridge as B
import r42_env_io as E

from . import model as Legacy


class M2ModelError(Legacy.R42ModelError):
    pass


class Model(Legacy.Model):
    """Single selected M2 task, served through XPolicyLab's existing wire contract."""

    def __init__(self, model_cfg):
        if not isinstance(model_cfg, dict):
            raise M2ModelError("M2 model configuration must be a dict")
        task = model_cfg.get("task_id")
        if task not in ("push_T", "stack_bowls"):
            raise M2ModelError("explicit M2 task_id is required (push_T or stack_bowls)")
        if "metadata" not in model_cfg and not model_cfg.get("metadata_path"):
            raise M2ModelError("M2 requires explicit verified metadata or metadata_path; "
                               "R15 bridge fallback is forbidden")
        if model_cfg.get("metadata") is not None and model_cfg.get("metadata_path"):
            raise M2ModelError("ambiguous metadata sources: choose one")
        if (model_cfg.get("backend") is not None and
                str(model_cfg.get("device", "cuda")).startswith("cuda")):
            raise M2ModelError("injected GPU backend cannot bypass the M2 real-action safety gate")
        self._m2_task = task
        # Legacy __init__ checks the old Push-T prompt before it calls our own
        # metadata resolver. The override is restricted to the *exact* frozen
        # M2 task instruction and then verified by our strict update_obs().
        cfg = dict(model_cfg)
        cfg["prompt"] = MD.task_prompt(task)
        cfg["allow_prompt_override"] = True
        super().__init__(cfg)
        D.M2SessionContract(self.metadata, task)
        self.policy_name = model_cfg.get("policy_name", "OpenVLA_R42_M2")
        self.instruction = MD.task_prompt(task)

    def _resolve_metadata(self):
        meta = self.model_cfg.get("metadata")
        if meta is not None:
            self._metadata_source = "deploy_cfg.metadata(M2)"
        else:
            path = self.model_cfg.get("metadata_path")
            if not path or not os.path.isfile(path):
                raise M2ModelError("M2 metadata_path missing: %r" % path)
            try:
                with open(path, encoding="utf-8") as fh:
                    obj = json.load(fh)
            except (OSError, ValueError) as exc:
                raise M2ModelError("invalid metadata file: %s" % exc)
            meta = obj.get("metadata", obj)
            self._metadata_source = path
        try:
            D.M2SessionContract(meta, self._m2_task)
        except (ValueError, KeyError, TypeError) as exc:
            raise M2ModelError("M2 metadata/task contract refused: %s" % exc)
        return meta

    def prepare_case(self, case_meta=None):
        cm = dict(case_meta or {})
        declared = cm.get("task", cm.get("task_id"))
        if declared is not None and declared != self._m2_task:
            raise M2ModelError("case task mismatch: model serves %s, got %r"
                               % (self._m2_task, declared))
        return super().prepare_case(case_meta)

    def update_obs(self, obs):
        """Unlike R15, wrong observation instruction is a hard error, never WARN."""
        info = E.obs_dict_to_inputs(obs, expected_hw=self.expected_hw,
                                    expected_hz=self.control_hz)
        E.assert_no_action_leak(info)
        instr = info.get("instruction")
        if not isinstance(instr, str) or instr.strip() != self.instruction.strip():
            raise M2ModelError("observation instruction/task mismatch for %s: %r"
                               % (self._m2_task, instr))
        self._inputs = info
        self._t_obs_wall = time.time()
        self._t_obs_perf = time.perf_counter()
        return None

    def _ensure_server(self):
        """Lazy CPU stub or explicitly opted-in validated real-7B smoke."""
        if self._server is not None:
            return self._server
        backend = self.model_cfg.get("backend")
        if backend is None and not bool(self.model_cfg.get("allow_real_7b_smoke", False)):
            raise M2ModelError(
                "real M2 GPU backend requires explicit allow_real_7b_smoke=True, "
                "verified source artifacts and privileged DEV smoke gates")
        server = S.M2TaskDeployServer(
            self.metadata, task=self._m2_task, backend=backend, device=self.device,
            tokenizer_prompt_ids=self.model_cfg.get("tokenizer_prompt_ids"))
        if backend is None:
            if not self.base_model_path:
                raise M2ModelError("real M2 backend requires base_model_path")
            if not (self.lora_dir and os.path.isdir(self.lora_dir)):
                raise M2ModelError("real M2 LoRA adapter missing; refusing pristine 7B")
            if not (self.projector_path and os.path.isfile(self.projector_path)):
                raise M2ModelError("real M2 42D projector missing")
            expected = {
                "base_model_sha256": os.path.join(self.base_model_path, "config.json"),
                "lora_sha256": os.path.join(
                    self.lora_dir, "adapter_model.safetensors"),
                "projector_sha256": self.projector_path,
            }
            for name, source in expected.items():
                actual = B.sha256_file(source)
                if self.metadata[name] != actual:
                    raise M2ModelError(
                        "%s differs from sealed training metadata; refusing real GPU smoke" % name)
            try:
                import torch
                from transformers import AutoProcessor
            except ImportError as exc:
                raise M2ModelError("real 7B runtime missing torch/transformers: %s" % exc)
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise M2ModelError("CUDA unavailable; never silently run 7B on CPU")
            proc = AutoProcessor.from_pretrained(self.base_model_path, trust_remote_code=True)
            server.processor = proc
            server.build_prompt(proc)  # calls M2 training's _make_prompt, NOT R15
            server.load(self.base_model_path, lora_dir=self.lora_dir,
                        projector_path=self.projector_path)
            server.assert_frozen()
            # This smoke has not yet independently proved tokenizer IDs
            # equal the checkpoint's training tokens. Do not claim parity.
        self._server = server
        return server

    def get_action(self):
        # CPU stub output is for contract testing.  The real 7B backend may be
        # LOADED in a strictly opt-in smoke, but this XPolicyLab entry cannot
        # issue robot commands until a validated prev_cmd/dt rate limiter and
        # physical joint safety authority are wired into the servo interface.
        if self.model_cfg.get("backend") is None:
            raise M2ModelError(
                "real M2 robot action emission BLOCKED: certified per-step "
                "rate control (prev_cmd/dt) and live deployment Gate not verified")
        return super().get_action()

    def contract(self):
        ans = super().contract()
        ans["strict_task_binding"] = True
        ans["trained_prompt"] = self.metadata["trained_prompt"]
        ans["tokenizer_prompt_parity"] = self.metadata["tokenizer_prompt_parity"]
        ans["m2_official_closed_loop"] = "NOT_VERIFIED"
        ans["real_robot_action_emission"] = "BLOCKED_PENDING_RATE_AND_SAFETY_GATE"
        return ans
