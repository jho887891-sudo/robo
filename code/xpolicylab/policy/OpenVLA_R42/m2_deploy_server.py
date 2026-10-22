#!/usr/bin/env python3
"""R25 M2 CPU-testable task-frozen R42 serving adapter.

This adapter preserves the frozen R15 server unchanged.  It inherits its
vision preprocessing, 14-token greedy/KV logic and GPU load path, while
binding every M2 session to ONE named task, that task's 42D normalizer,
training prompt and 14D residual codec.  Dynamic task switching is forbidden.

CPU stub verification is NOT real-7B checkpoint parity or a robot deploy.
A real model still needs the checkpoint, signed robot safety contract,
trained tokenizer-prompt parity and an official DEV privileged-state smoke.
"""
import os
import time

import numpy as np

import r42_adapter as R
import r42_deploy_server as S
import m2_deploy_contract as D
import m2_train_shared as MT


class M2TaskDeployServer(S.R42DeployServer):
    """One task per model instance, preventing cross-task state or decoder leakage."""

    def __init__(self, metadata, task, backend=None, processor=None, device="cuda",
                 tokenizer_prompt_ids=None, use_kv_cache=False,
                 allow_cv_diagnostic=False):
        # Deliberately do not run the legacy parent constructor: it interprets
        # the WHOLE M2 normalizer object as one single-task normalization.
        self.session = D.M2SessionContract(metadata, task)
        if type(allow_cv_diagnostic) is not bool:
            raise D.M2DeployContractError("allow_cv_diagnostic must be a literal bool")
        self.allow_cv_diagnostic = allow_cv_diagnostic
        self.task = task
        self.metadata = metadata
        self.backend = backend
        self.processor = processor
        self.device = str(device)
        self.use_kv_cache = bool(use_kv_cache)
        self.n_action_tokens = R.N_ACTIONS
        self.n_supervised = R.N_ACTIONS + 1
        self.action_token_window = R.CODEC_WINDOW
        self.codec_mode = self._assert_codec_mode()
        self._norm = self.session.normalizer
        self.contract = self.session.history.contract
        self._history = self.session.history
        self._prompt_ids = (None if tokenizer_prompt_ids is None else
                            R.normalize_prompt_ids(tokenizer_prompt_ids))
        self._pv_shape_logged = None
        self._lat_samples, self._lat_fwd = None, None
        if self.backend is not None:
            self._assert_backend_contract()
        self._announce()

    def assert_instruction(self, observation_instruction):
        """No old R15 WARN-and-ignore behavior: bad instruction is a hard failure."""
        expected = MT.M2.task_prompt(self.task)
        if not isinstance(observation_instruction, str) or (
                observation_instruction.strip() != expected):
            raise D.M2DeployContractError(
                "observation instruction/task mismatch for %s: expected %r got %r"
                % (self.task, expected, observation_instruction))
        return True

    def build_prompt(self, processor):
        """Use the EXACT M2 training prompt builder, including PurePromptBuilder wrapper."""
        if processor is None:
            raise D.M2DeployContractError("processor is required for trained prompt tokens")
        computed_ids = R.normalize_prompt_ids(MT._make_prompt(
            processor, self.task, include_task_token=self.metadata["include_task_token"]))
        # These are the actual ids produced by training/m2_train_shared.py
        # from the same prompt formatter. Textual prompt equality alone is
        # insufficient, since a changed tokenizer/processor can alter ids.
        expected_hash = self.metadata.get("trained_prompt_token_ids_sha256")
        actual_hash = MT.sha256_json(computed_ids)
        if actual_hash != expected_hash:
            raise D.M2DeployContractError(
                "M2 prompt token SHA256 mismatch for %s: training %s vs serving %s. "
                "Refusing prefill, no model inference performed."
                % (self.task, expected_hash, actual_hash))
        self._prompt_ids = computed_ids
        return list(self._prompt_ids)

    def _decode_tokens(self, pixel_values, feat):
        """The legacy CV debug arm needs THIS task's q01/q99, not the M2 outer payload."""
        if os.environ.get("R42_POLICY", "").lower() == "cv":
            if self.allow_cv_diagnostic is not True:
                raise D.M2DeployContractError(
                    "CV diagnostic policy override requires explicit "
                    "allow_cv_diagnostic=True; refusing to mislabel CV as VLA")
            _idx, ids = R.token_ids_for_residual(
                np.zeros(R.N_ACTIONS, np.float32),
                self._norm["q01"], self._norm["q99"])
            ids = [int(x) for x in ids]
            return ids, [
                {"k": k, "bin": int(tid) - R.TOKEN_BEGIN_IDX - 1,
                 "id": tid, "policy": "cv_baseline"} for k, tid in enumerate(ids)]
        return super()._decode_tokens(pixel_values, feat)

    def act(self, images, state, env_idx=0, timestamp=None, prev_cmd=None, dt=None,
            record=False):
        """Same four latency phases as R42, with explicit per-task SAFE decode."""
        t_start = time.perf_counter()
        imgs = self.session.check_images(images)
        feature = self._features(env_idx, state, timestamp)
        t_pre = time.perf_counter()
        pixel_values = self.preprocess(imgs)
        lat = {"preprocess": (time.perf_counter() - t_pre) * 1000.0}
        t_forward = time.perf_counter()
        ids, step_logs = self._decode_tokens(pixel_values, feature)
        policy_mode = ("cv_diagnostic" if os.environ.get("R42_POLICY", "").lower() == "cv"
                       else "vla")
        lat["forward_total"] = (time.perf_counter() - t_forward) * 1000.0
        t_decode = time.perf_counter()
        v = self._history.velocity(int(env_idx))
        bins = np.asarray(ids, np.int64) - (R.TOKEN_BEGIN_IDX + 1)
        if bins.shape != (R.N_ACTIONS,) or (bins < 0).any() or (bins > 255).any():
            raise D.M2DeployContractError("model returned illegal 14-token action bins")
        cmd, resid, info = self.session.decode(
            bins, state, v, prev_cmd=prev_cmd, dt=dt)
        lat["decode"] = (time.perf_counter() - t_decode) * 1000.0
        lat["total"] = (time.perf_counter() - t_start) * 1000.0
        lat = self._check_latency(lat)
        if self._lat_samples is None:
            self._lat_samples, self._lat_fwd = [], []
        self._lat_samples.append(float(lat["total"]))
        self._lat_fwd.append(float(lat["forward_total"]))
        out = {
            "action": cmd, "residual": resid, "bins": bins,
            "ids": np.asarray(ids, np.int64),
            "features": feature, "velocity": v,
            "info": info, "n_action_tokens": len(ids),
            "prompt_len": len(self.prompt_ids()) if not (
                os.environ.get("R42_POLICY", "").lower() == "cv"
                and self._prompt_ids is None) else 0,
            "latency_ms": lat, "step_logs": step_logs,
            "served_task": self.task, "policy_mode": policy_mode,
            "normalizer_sha256": self.metadata["normalizer_sha256"],
        }
        if record:
            out["log"] = {
                "task": self.task, "prompt_ids": (
                    self.prompt_ids() if self._prompt_ids is not None else []),
                "features": feature.tolist(),
                "state": np.asarray(state, np.float32).tolist(),
                "policy_mode": policy_mode,
                "replay_scope": "cpu_stub_only",
                "velocity": v.tolist(), "env_idx": int(env_idx)}
        return out

    def replay(self, log):
        """Recompute the recorded frame through the SAME task normalizer and codec."""
        # The logged token/features/state triple does NOT preserve three
        # RGB camera tensors. A real multimodal 7B must not replay from
        # pixel_values=None and call the output a faithful video-conditioned
        # prediction. Only CPU-only contract stubs can use this path.
        if self.processor is not None:
            raise D.M2DeployContractError(
                "multimodal 7B replay requires recorded RGB pixels; "
                "CPU stub-only replay cannot reconstruct visual context")
        if log.get("task") != self.task:
            raise D.M2DeployContractError("replay task disagrees with frozen serving task")
        if "features" not in log or "prompt_ids" not in log:
            raise D.M2DeployContractError("replay log lacks task/prompt/features")
        if self._prompt_ids is None or list(log["prompt_ids"]) != self.prompt_ids():
            raise D.M2DeployContractError("replay prompt tokens differ from live task binding")
        feat = np.asarray(log["features"], np.float32)
        if feat.shape != (42,) or not np.isfinite(feat).all():
            raise D.M2DeployContractError("replay requires finite 42D features")
        prefix = list(log["prompt_ids"])
        ids = []
        for _k in range(R.N_ACTIONS):
            logits = self.backend.next_token_logits(prefix + ids, None, feat)
            ids.append(R.TOKEN_BEGIN_IDX + 1 + self._argmax_in_window(logits))
        state = np.asarray(log.get("state"), np.float32)
        velocity = np.asarray(log.get("velocity"), np.float32)
        bins = np.asarray(ids, np.int64) - (R.TOKEN_BEGIN_IDX + 1)
        action, residual, info = self.session.decode(bins, state, velocity)
        return {"action": action, "residual": residual,
                "ids": np.asarray(ids, np.int64), "bins": bins,
                "info": info, "n_action_tokens": len(ids), "served_task": self.task,
                "replay_scope": "cpu_stub_only"}
