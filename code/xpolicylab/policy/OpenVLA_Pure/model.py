"""Pure OpenVLA policy server (autoregressive discrete action tokens) for RoboDojo push_T.

Variant B: 3 cameras + proprio projector, LoRA r=32 all-linear on OpenVLA-7B.
Actions are decoded autoregressively as 256-bin discrete tokens (open-loop chunk = 1).
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

if "aloha" not in " ".join(sys.argv).lower():
    sys.argv.append("--aloha")

from peft import PeftModel
from transformers import AutoImageProcessor, AutoConfig, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.policy.OpenVLA_OFT.model import ensure_hwc_uint8, extract_image, extract_prompt
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

try:
    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register("openvla", PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
except TypeError:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register("openvla", PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)

BINS = 256
BIN_CENTERS = np.linspace(-1.0, 1.0, BINS).astype(np.float32)

# KV-cache decoding: one multimodal forward for the prefix, then one token per step.
# Set PURE_NO_KV=1 to fall back to the (slow) full-recompute loop for equivalence checks.
_USE_KV = os.environ.get("PURE_NO_KV", "0") != "1"
_ACTION_LOG = os.environ.get("PURE_ACTION_LOG", "")
_CALL_IDX = {"n": 0}


def _log_call(rec):
    if not _ACTION_LOG:
        return
    try:
        with open(_ACTION_LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


_IMAGE_CANDIDATES = (
    ["cam_high", "cam_head", "head_camera", "top_camera"],
    ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"],
    ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"],
)


def _encode_obs_pure(observation, action_type, robot_action_dim_info, default_prompt):
    """Like the OFT encoder but WITHOUT the 256x256 pre-resize: the pure policy was
    trained on raw 480x640 frames passed straight into processor.apply_transform."""
    if "images" in observation and "state" in observation:
        raw = [observation["images"][k] for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")]
        state = np.asarray(observation["state"], dtype=np.float32)
        prompt = extract_prompt(observation, default_prompt)
    else:
        raw = [extract_image(observation, cands) for cands in _IMAGE_CANDIDATES]
        if robot_action_dim_info is None:
            raise ValueError("env_cfg is required when encoding raw environment observations.")
        state = pack_robot_state(
            observation, action_type, robot_action_dim_info, source_type="obs"
        ).astype(np.float32)
        prompt = extract_prompt(observation, default_prompt)
    return {
        "images_raw": [ensure_hwc_uint8(img) for img in raw],
        "state": state,
        "instruction": prompt,
    }


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.device = torch.device("cuda")
        self.num_images = int(model_cfg.get("num_images_in_input", 3))

        meta = json.load(open(model_cfg["stats_path"]))
        self.q01 = np.asarray(meta["q01"], dtype=np.float32)
        self.q99 = np.asarray(meta["q99"], dtype=np.float32)
        nb = np.load(model_cfg["bounds_path"])
        self.pmin = np.asarray(nb["pmin"], dtype=np.float32)
        self.pmax = np.asarray(nb["pmax"], dtype=np.float32)

        print("[OpenVLA_Pure] loading base %s" % model_cfg["base_model_path"], flush=True)
        self.processor = AutoProcessor.from_pretrained(model_cfg["base_model_path"], trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            model_cfg["base_model_path"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        vla.vision_backbone.set_num_images_in_input(self.num_images)
        vla = PeftModel.from_pretrained(vla, model_cfg["adapter_dir"], is_trainable=False)
        vla = vla.merge_and_unload()
        self.vla = vla.to(self.device).eval()
        print("[OpenVLA_Pure] LoRA merged (cameras=%d)" % self.num_images, flush=True)

        self.proprio_projector = ProprioProjector(proprio_dim=14, llm_dim=self.vla.llm_dim)
        self.proprio_projector.load_state_dict(
            torch.load(model_cfg["proprio_projector_path"], map_location="cpu")
        )
        self.proprio_projector = self.proprio_projector.to(self.device).eval()

        self.instr = model_cfg.get("prompt") or "Push the T-shaped block to align it precisely with the gray T-shaped pad."
        self.task_name = model_cfg.get("task_name")
        self.action_type = model_cfg.get("action_type", "joint")
        env_cfg = model_cfg.get("env_cfg") or model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = get_robot_action_dim_info(env_cfg) if env_cfg is not None else None
        self.default_prompt = model_cfg.get("prompt", self.task_name)
        self.observation_window = None
        self._latest_env_idx_list = [0]
        self._pids = None

    # ------------------------------------------------------------------ obs
    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", i) for i, obs in enumerate(obs_list)]
        self.observation_window = [
            _encode_obs_pure(obs, self.action_type, self.robot_action_dim_info, self.default_prompt)
            for obs in obs_list
        ]

    # ------------------------------------------------------------ inference
    def _prompt_ids(self):
        if self._pids is None:
            builder = PurePromptBuilder("openvla")
            builder.add_turn("human", "What action should the robot take to " + self.instr + "?")
            ids = self.processor.tokenizer(builder.get_prompt(), return_tensors="pt").input_ids.to(self.device)
            if int(ids[0, -1].item()) != 29871:
                ids = torch.cat([ids, torch.tensor([[29871]], dtype=ids.dtype, device=self.device)], dim=1)
            self._pids = ids
        return self._pids

    @torch.no_grad()
    def infer(self, observation):
        images = [
            self.processor.image_processor.apply_transform(Image.fromarray(img))
            for img in observation["images_raw"]
        ]
        pixel_values = torch.cat(images, dim=0).unsqueeze(0).to(torch.bfloat16).to(self.device)

        state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)[:14]
        proprio_np = np.clip(
            2.0 * (state - self.pmin) / (self.pmax - self.pmin + 1e-8) - 1.0, -1.0, 1.0
        ).astype(np.float32)
        proprio = torch.from_numpy(proprio_np).unsqueeze(0).to(self.device)

        prompt_ids = self._prompt_ids()
        tokens = []

        def _pick(logits):
            cand = logits[ACTION_TOKEN_BEGIN_IDX + 1: ACTION_TOKEN_BEGIN_IDX + 1 + BINS]
            return int(torch.argmax(cand).item())

        if _USE_KV:
            # Prefix (images + prompt + proprio) once, then 13 cached single-token steps.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.vla(
                    input_ids=prompt_ids,
                    attention_mask=torch.ones_like(prompt_ids),
                    pixel_values=pixel_values,
                    labels=prompt_ids.clone(),
                    proprio=proprio,
                    proprio_projector=self.proprio_projector,
                    use_cache=True,
                )
            past = out.past_key_values
            logits = out.logits[0, -1]
            for step in range(ACTION_DIM):
                tok = _pick(logits)
                tokens.append(tok)
                if step + 1 == ACTION_DIM:
                    break
                nxt = torch.tensor(
                    [[ACTION_TOKEN_BEGIN_IDX + tok + 1]], dtype=prompt_ids.dtype, device=self.device
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self.vla(input_ids=nxt, past_key_values=past, labels=None, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1]
        else:
            for _ in range(ACTION_DIM):
                if tokens:
                    action_ids = torch.tensor(
                        [[ACTION_TOKEN_BEGIN_IDX + t + 1 for t in tokens]], dtype=prompt_ids.dtype, device=self.device
                    )
                    model_input = torch.cat([prompt_ids, action_ids], dim=1)
                else:
                    model_input = prompt_ids
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self.vla(
                        input_ids=model_input,
                        attention_mask=torch.ones_like(model_input),
                        pixel_values=pixel_values,
                        labels=model_input.clone(),
                        proprio=proprio,
                        proprio_projector=self.proprio_projector,
                    )
                logits = out.logits[0, -1]
                tokens.append(_pick(logits))

        centers = BIN_CENTERS[np.clip(np.asarray(tokens), 0, BINS - 1)]
        action = (centers + 1.0) / 2.0 * (self.q99 - self.q01 + 1e-8) + self.q01
        action = action.reshape(1, -1).astype(np.float32)
        _CALL_IDX["n"] += 1
        _log_call({
            "call": _CALL_IDX["n"],
            "wall": round(time.time(), 3),
            "state": [round(float(v), 5) for v in np.asarray(state).reshape(-1)[:14]],
            "proprio_norm": [round(float(v), 4) for v in np.asarray(proprio_np).reshape(-1)[:14]],
            "tokens": [int(t) for t in tokens],
            "action": [round(float(v), 5) for v in action[0]],
            "delta_action_minus_state": [round(float(a - b), 5) for a, b in zip(action[0], np.asarray(state).reshape(-1)[:14])],
        })
        return action

    # --------------------------------------------------------------- actions
    def get_action(self, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")
        chunk = self.infer(self.observation_window[0])
        if self.robot_action_dim_info is None:
            return [chunk[0]]
        return unpack_robot_state(chunk, self.action_type, self.robot_action_dim_info, source_type="obs")

    def get_action_batch(self, env_idx_list=None, **kwargs):
        indices = env_idx_list or self._latest_env_idx_list
        return [self.get_action(**kwargs) for _ in indices]

    def reset(self):
        self.observation_window = None
        return
