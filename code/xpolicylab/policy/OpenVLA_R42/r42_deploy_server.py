#!/usr/bin/env python3
"""R10.2 D is a CPU adapter, not a deployment server.  This is the real one (R11 P0-4).

Responsibilities (all fail-closed):
  * load base 7B + LoRA adapter + the 42D proprio projector + the 21-field R42 metadata,
  * three raw RGB cameras in the frozen order (cam_head, cam_left_wrist, cam_right_wrist),
    no BGR swap and no hidden 256-px resize of our own (the processor does what it does),
  * per-env 42D [S,V,Acc] history with reset(env_idx)/batch reset and no cross-env leakage,
  * legal 14-token autoregressive decode with the argmax restricted to 31744..31999,
  * physical joint/rate safety through r42_adapter.decode_with_metadata,
  * replay of a recorded step with no weight update (everything runs under torch.no_grad()).

Only the model-touching part imports torch/prismatic, and only inside load()/act(), so the whole
contract can be exercised on CPU with an injected backend (see tests/test_r11_p0_deploy.py).
"""
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

import r42_adapter as R  # noqa: E402

INSTRUCTION = "Push the T-shaped block to align it precisely with the gray T-shaped pad."


class R42DeployError(ValueError):
    """Raised for any deployment-contract violation."""
    pass


class R42DeployServer:
    """Deterministic 42D/R policy server; `backend` can be injected for CPU contract tests."""

    def __init__(self, metadata, backend=None, processor=None, device="cuda",
                 tokenizer_prompt_ids=None, use_kv_cache=False):
        try:
            R.validate_metadata(metadata)
        except R.R42Error as exc:
            raise R.R42Error("refusing to build a deploy server on invalid metadata: %s" % exc)
        self.metadata = metadata
        self.backend = backend
        self.processor = processor
        self.device = device
        self.use_kv_cache = bool(use_kv_cache)
        self.n_action_tokens = R.N_ACTIONS
        self.n_supervised = R.N_ACTIONS + 1
        self.action_token_window = R.CODEC_WINDOW
        self._norm = self.metadata["normalizer"]
        from r8_normalization import NormalizationContract
        self.contract = NormalizationContract.from_metadata(self._norm)
        self._history = R.EpisodeHistory(self.contract)
        self._prompt_ids = None
        if tokenizer_prompt_ids is not None:
            self._prompt_ids = R.normalize_prompt_ids(tokenizer_prompt_ids)
        if self.backend is not None:
            self._assert_backend_contract()

    # ------------------------------------------------------------------ load / freeze
    def _assert_backend_contract(self):
        if not hasattr(self.backend, "next_token_logits"):
            raise R.R42Error("backend must implement next_token_logits(input_ids, pixel_values, "
                             "proprio) -> [vocab] logits for the last position")

    def load(self, base_model_path, lora_dir=None, projector_path=None):
        """Real 7B load.  Not exercised by the CPU tests; R13 must verify it on GPU."""
        import torch
        import r8_vla_adapter as A
        if self.processor is None:
            raise self._err("processor is required for the real model path")
        model = _load_prismatic(base_model_path, self.device)
        if lora_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_dir)
        proj = A.build_projector(int(model.llm_dim), device=self.device)
        if projector_path:
            state = torch.load(projector_path, map_location="cpu")
            R.assert_projector_width(state["fc1.weight"].shape, expected=42)
            A.load_projector_state(proj, state)
        model.eval()
        proj.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        for p in proj.parameters():
            p.requires_grad_(False)
        backend = _TorchBackend(model, proj, self.device)
        self.backend = backend
        self._assert_backend_contract()
        return self

    @staticmethod
    def _err(msg):
        return R.R42Error(msg)

    def assert_frozen(self):
        params = getattr(self.backend, "parameters", None)
        if params is not None:
            for p in params():
                if getattr(p, "requires_grad", False):
                    raise R.R42Error("deployment backend still has trainable parameters")
        return True

    # -------------------------------------------------------------------------- prompt
    def prompt_ids(self):
        if self._prompt_ids is None:
            raise R.R42Error("prompt ids not provided: pass tokenizer_prompt_ids or call "
                             "build_prompt(processor)")
        return list(self._prompt_ids)

    def build_prompt(self, processor):
        prompt = ("What action should the robot take to " + INSTRUCTION + "?")
        ids = processor.tokenizer(prompt, return_tensors="np").input_ids
        self._prompt_ids = R.normalize_prompt_ids(ids.reshape(-1).tolist())
        return self._prompt_ids

    # ------------------------------------------------------------------- preprocessing
    def preprocess(self, images):
        """RGB uint8 dict -> processor pixel tensor.  No BGR swap, no manual resize."""
        if self.processor is None:
            return None
        import torch
        from PIL import Image
        tiles = []
        for cam in R.CAMERA_ORDER:
            tiles.append(self.processor.image_processor.apply_transform(
                Image.fromarray(images[cam], mode="RGB")))
        return torch.cat(tiles, dim=0).unsqueeze(0).to(torch.bfloat16).to(self.device)

    # ------------------------------------------------------------------------- rollout
    def reset(self, env_idx=0):
        self._history.reset(int(env_idx))
        return True

    def reset_all(self):
        for k in list(getattr(self._history, "_hist", {}).keys()):
            self._history.reset(k)
        return True

    def history_features(self, env_idx=0):
        return self._history.features(int(env_idx))

    def _features(self, env_idx, state, timestamp):
        return self._history.observe(int(env_idx), state, timestamp=timestamp)

    def _argmax_in_window(self, logits):
        w = np.asarray(logits, np.float32).reshape(-1)
        lo = R.TOKEN_BEGIN_IDX + 1
        cand = w[lo: lo + 256]
        if cand.size != 256:
            raise R.R42Error("backend returned only %d logits; need the full 31744..31999 window"
                            % w.size)
        return int(np.argmax(cand))

    def act(self, images, state, env_idx=0, timestamp=None, prev_cmd=None, dt=None,
            record=False):
        """One control step: 14 autoregressive tokens -> safe 14D joint command."""
        t_start = time.perf_counter()
        imgs = R.validate_images(images)
        feat = self._features(env_idx, state, timestamp)
        t_pre = time.perf_counter()
        pixel_values = self.preprocess(imgs)
        lat = {"preprocess": (time.perf_counter() - t_pre) * 1e3}
        t_fwd = time.perf_counter()
        ids, step_logs = self._decode_tokens(pixel_values, feat)
        lat["forward_total"] = (time.perf_counter() - t_fwd) * 1e3
        t_dec = time.perf_counter()
        v = self._history.velocity(env_idx)
        action, residual, info = R.decode_with_metadata(
            self.metadata, np.asarray(ids) - (R.TOKEN_BEGIN_IDX + 1), state, v,
            prev_cmd=prev_cmd, dt=dt)
        lat["decode"] = (time.perf_counter() - t_dec) * 1e3
        lat["total"] = (time.perf_counter() - t_start) * 1e3
        out = {"action": action, "residual": residual, "bins": np.asarray(ids) - (R.TOKEN_BEGIN_IDX + 1),
               "ids": np.asarray(ids, np.int64), "features": feat, "velocity": v, "info": info,
               "n_action_tokens": int(len(ids)), "prompt_len": len(self.prompt_ids()),
               "latency_ms": lat}
        if record:
            out["log"] = {"prompt_ids": self.prompt_ids(), "features": feat.tolist(),
                          "state": np.asarray(state, np.float32).tolist(),
                          "velocity": v.tolist(), "env_idx": int(env_idx)}
        return out

    def _decode_tokens(self, pixel_values, feat):
        """Greedy legal 14-token decoding; the window constraint is enforced per token."""
        prompt = self.prompt_ids()
        ids, logs = [], []
        for k in range(R.N_ACTIONS):
            prefix = prompt + ids
            logits = self.backend.next_token_logits(prefix, pixel_values, feat)
            tok = self._argmax_in_window(logits)
            if not (0 <= tok <= 255):
                raise R.R42Error("decoded bin %d outside 0..255" % tok)
            ids.append(R.TOKEN_BEGIN_IDX + 1 + tok)
            logs.append({"k": k, "bin": tok, "id": ids[-1]})
        if len(ids) != R.N_ACTIONS:
            raise R.R42Error("expected %d action tokens, produced %d" % (R.N_ACTIONS, len(ids)))
        return ids, logs

    def replay(self, log):
        """Re-execute a recorded step from its own log; never updates any weight."""
        if "features" not in log or "prompt_ids" not in log:
            raise R.R42Error("replay log must carry prompt_ids and features")
        self._history.reset(log.get("env_idx", 0))
        feat = np.asarray(log["features"], np.float32)
        prompt = list(log["prompt_ids"])
        ids = []
        for _k in range(R.N_ACTIONS):
            logits = self.backend.next_token_logits(prompt + ids, None, feat)
            ids.append(R.TOKEN_BEGIN_IDX + 1 + self._argmax_in_window(logits))
        state = np.asarray(log.get("state", np.zeros(14)), np.float32)
        velocity = np.asarray(log.get("velocity", np.zeros(14)), np.float32)
        action, residual, info = R.decode_with_metadata(
            self.metadata, np.asarray(ids) - (R.TOKEN_BEGIN_IDX + 1), state, velocity)
        return {"action": action, "residual": residual, "ids": np.asarray(ids, np.int64),
                "bins": np.asarray(ids) - (R.TOKEN_BEGIN_IDX + 1), "info": info,
                "n_action_tokens": len(ids)}


class _TorchBackend:
    """Wraps the real 7B (+42D projector) behind the same `next_token_logits` signature."""

    def __init__(self, model, projector, device="cuda"):
        self.model, self.projector, self.device = model, projector, device
        self._params = list(model.parameters()) + list(projector.parameters())

    def parameters(self):
        return iter(self._params)

    def next_token_logits(self, input_ids, pixel_values=None, proprio=None):
        import torch
        ids = torch.tensor([list(input_ids)], dtype=torch.long, device=self.device)
        pr = None if proprio is None else torch.from_numpy(
            np.asarray(proprio, np.float32)).reshape(1, -1).to(self.device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16) if self.device.startswith("cuda") \
                    else _nullcontext():
                out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 pixel_values=pixel_values, proprio=pr,
                                 proprio_projector=self.projector)
        return out.logits[0, -1].float().cpu().numpy()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _load_prismatic(base_model_path, device="cuda"):
    """Load the pristine OpenVLA 7B with the R11 poison guard applied before any import."""
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    R.assert_runtime_contract()
    return AutoModelForVision2Seq.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to(device)
