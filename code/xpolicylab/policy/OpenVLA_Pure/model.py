"""Pure OpenVLA policy server (autoregressive discrete action tokens) for RoboDojo push_T.

Variant B: 3 cameras + proprio projector, LoRA r=32 all-linear on OpenVLA-7B.
Actions are decoded autoregressively as 256-bin discrete tokens (open-loop chunk = 1).

------------------------------------------------------------------------------
第二轮审查的最小修复（仅诊断/正确性，未改算法、未重训、未改旧检查点）
  [R2-1] 动作离散化统一走 action_codec（原来是本文件里手写的 BIN_CENTERS 查表，
         与训练脚本各写一份，存在 1 个 token 的偏移与 PAD 边界缺陷）。
  [R2-3] 任务指令改为显式契约：固定使用 config 的训练指令，
         观测里的 instruction 若不同则打印一次警告并记入动作日志，
         非 push_T 任务启动时打印“不具备跨任务泛化证据”的声明。
------------------------------------------------------------------------------
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
from XPolicyLab.policy.OpenVLA_Pure.pure_autoreg_mode import enable_pure_autoreg
from XPolicyLab.policy.OpenVLA_Pure import action_target as AT
from XPolicyLab.policy.OpenVLA_Pure import proprio_history as PH
from XPolicyLab.policy.OpenVLA_Pure.action_codec import (
    LEGAL_TOKEN_MAX,
    LEGAL_TOKEN_MIN,
    MODE_LEGACY,
    MODE_FIXED,
    N_BINS,
    decode_indices,
    denormalize,
    token_ids,
    validate_token_ids,
)
from XPolicyLab.policy.OpenVLA_Pure.deploy_contract import (
    TRAINED_TASKS,
    prompt_text,
    resolve_instruction,
)
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

BINS = N_BINS
# [R2-1] 说明：现有 6000 步检查点的标签是用旧编码（MODE_LEGACY）烧进权重的，
# 因此解码必须沿用同一栅格 LEVELS[idx]；action_codec.decode_indices 正是这一栅格。
# 编码侧（训练标签）已修正为 MODE_FIXED，见 training/train_pushT_pure2.py。
TRAIN_CODEC_MODE = MODE_LEGACY
DEFAULT_INSTRUCTION = "Push the T-shaped block to align it precisely with the gray T-shaped pad."

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
        AT.validate_stats(self.q01, self.q99)
        self.action_target_mode = AT.resolve_checkpoint_target(
            meta, model_cfg.get("action_target_mode", AT.ABSOLUTE))
        self.codec_mode = meta.get("codec_mode", MODE_LEGACY)
        if self.codec_mode not in (MODE_FIXED, MODE_LEGACY):
            raise ValueError("unknown checkpoint codec_mode: %s" % self.codec_mode)
        if self.action_target_mode == AT.RESIDUAL and self.codec_mode != MODE_FIXED:
            raise ValueError("residual checkpoint requires corrected fixed action codec")
        # [R6] proprio interface: legacy checkpoints (no metadata) stay 14D exactly as before.
        self.proprio_history_mode = PH.resolve_mode(meta)
        self.proprio_dim = PH.proprio_dim(self.proprio_history_mode)
        if self.proprio_dim != int(meta.get("proprio_dim", 14)):
            raise ValueError("checkpoint proprio_dim %r disagrees with mode %r"
                             % (meta.get("proprio_dim"), self.proprio_history_mode))
        if self.proprio_history_mode != PH.NONE:
            print("[OpenVLA_Pure] proprio history mode=%s dim=%d (%s)" % (
                self.proprio_history_mode, self.proprio_dim, PH.MODE_DESCRIPTION[self.proprio_history_mode]),
                flush=True)
        expected_masking = meta.get("action_input_masking")
        if expected_masking not in (None, "keep"):
            raise ValueError("checkpoint has incompatible action_input_masking=%r" % expected_masking)
        AT.verify_checkpoint_hashes(meta, model_cfg["adapter_dir"],
                                    model_cfg["proprio_projector_path"], self.action_target_mode)
        print("[OpenVLA_Pure] target=%s codec=%s autoreg=keep" %
              (self.action_target_mode, self.codec_mode), flush=True)
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
        # Source identity must match the real dynamic module used at training.
        # This also prevents silently mixing the two independent R3 masking forks.
        AT.verify_runtime_modeling(meta, vla, self.action_target_mode)
        vla.vision_backbone.set_num_images_in_input(self.num_images)
        enable_pure_autoreg(vla)
        vla = PeftModel.from_pretrained(vla, model_cfg["adapter_dir"], is_trainable=False)
        vla = vla.merge_and_unload()
        enable_pure_autoreg(vla)
        if not getattr(vla, "_pure_autoreg_input_mode", False):
            raise RuntimeError("pure autoreg action input mode was not enabled")
        self.vla = vla.to(self.device).eval()
        print("[OpenVLA_Pure] LoRA merged (cameras=%d)" % self.num_images, flush=True)

        self.proprio_projector = ProprioProjector(proprio_dim=self.proprio_dim, llm_dim=self.vla.llm_dim)
        state_dict = torch.load(model_cfg["proprio_projector_path"], map_location="cpu")
        width = int(state_dict["fc1.weight"].shape[1])
        PH.verify_projector_width(meta, width, self.proprio_history_mode)
        self.proprio_projector.load_state_dict(state_dict)
        self.proprio_projector = self.proprio_projector.to(self.device).eval()

        self.task_name = model_cfg.get("task_name")
        self.action_type = model_cfg.get("action_type", "joint")
        env_cfg = model_cfg.get("env_cfg") or model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = get_robot_action_dim_info(env_cfg) if env_cfg is not None else None

        # [R2-3] 指令来源显式化：默认 config（= 训练指令），观测指令被忽略并告警。
        self.instruction_source = model_cfg.get("instruction_source", "config")
        _res = resolve_instruction(
            config_prompt=model_cfg.get("prompt"),
            task_name=self.task_name,
            obs_instruction=None,
            instruction_source=self.instruction_source,
        )
        self.instr = _res["instruction"]
        self.default_prompt = model_cfg.get("prompt", self.task_name)
        self._warned = set()
        for w in _res["warnings"]:
            self._warn(w)
        print("[OpenVLA_Pure] instruction pinned from %s (task=%s): %r"
              % (self.instruction_source, self.task_name, self.instr), flush=True)
        if self.task_name not in TRAINED_TASKS:
            print("[OpenVLA_Pure][WARN] checkpoint trained on %s only; task=%s results are "
                  "NOT cross-task generalisation evidence" % (list(TRAINED_TASKS), self.task_name),
                  flush=True)

        self.observation_window = None
        self.proprio_history = PH.HistoryBuffer(self.proprio_history_mode)
        self._latest_env_idx_list = [0]
        self._pids = None
        self.last_decode = None   # [R2] 最近一次推理的 token / 概率轨迹（诊断用）

    @staticmethod
    def _warn(msg):
        print("[OpenVLA_Pure][WARN] %s" % msg, flush=True)

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
            builder.add_turn("human", prompt_text(self.instr))
            ids = self.processor.tokenizer(builder.get_prompt(), return_tensors="pt").input_ids.to(self.device)
            if int(ids[0, -1].item()) != 29871:
                ids = torch.cat([ids, torch.tensor([[29871]], dtype=ids.dtype, device=self.device)], dim=1)
            self._pids = ids
        return self._pids

    def _note_instruction(self, obs_instruction):
        """[R2-3] 观测指令只做核对与告警，不参与推理。"""
        res = resolve_instruction(
            config_prompt=self.instr,
            task_name=self.task_name,
            obs_instruction=obs_instruction,
            instruction_source=self.instruction_source,
        )
        for w in res["warnings"]:
            self._warn(w)
        return res

    @torch.no_grad()
    def infer(self, observation, env_idx=0):
        images = [
            self.processor.image_processor.apply_transform(Image.fromarray(img))
            for img in observation["images_raw"]
        ]
        pixel_values = torch.cat(images, dim=0).unsqueeze(0).to(torch.bfloat16).to(self.device)

        state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)[:14]
        # [R6] 14D: proprio = normalized current state (unchanged).
        #      28D: [current, previous OBSERVED state] from the per-env buffer; the
        #      first observation after reset() uses the documented zero-velocity rule.
        proprio_np = self.proprio_history.observe(state, self.pmin, self.pmax, env_idx)
        proprio = torch.from_numpy(proprio_np).unsqueeze(0).to(self.device)

        instr_info = self._note_instruction(observation.get("instruction"))
        prompt_ids = self._prompt_ids()
        tokens = []
        probs = []   # [R2] 每步 256 个窗口 token 的 softmax 最大值与其下标

        win = ACTION_TOKEN_BEGIN_IDX + 1   # 解码窗口起点 = 合法动作 token 最小值 31744

        def _pick(logits):
            cand = logits[win: win + BINS]
            p = torch.softmax(cand.float(), dim=-1)
            top = int(torch.argmax(p).item())
            probs.append({"idx": top, "p": float(p[top].item()),
                          "top3": [[int(j), float(p[j].item())]
                                   for j in torch.topk(p, 3).indices.tolist()]})
            return top

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

        # [R2-1] 统一解码：tokens 是解码窗口下标（0..255），token id = 31744+idx。
        tok_ids = token_ids(np.asarray(tokens, dtype=np.int64))
        validate_token_ids(tok_ids)          # 越界 / PAD 会直接抛异常，不静默
        action_norm = decode_indices(np.asarray(tokens, dtype=np.int64))
        decoded_target = denormalize(action_norm, self.q01, self.q99)
        action = AT.joint_command(state, decoded_target, self.action_target_mode)
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        if not np.isfinite(action).all():
            raise ValueError("nonfinite predicted absolute joint command")

        self.last_decode = {
            "tokens": [int(t) for t in tokens],
            "token_ids": [int(t) for t in tok_ids],
            "probs": probs,
            "action_codec_mode": self.codec_mode,
            "action_target_mode": self.action_target_mode,
            "proprio_history_mode": self.proprio_history_mode,
            "proprio_dim": int(self.proprio_dim),
            "proprio_norm": [round(float(v), 5) for v in np.asarray(proprio_np).reshape(-1)],
            "decoded_target": [float(v) for v in decoded_target],
            "instruction_used": instr_info["instruction"],
            "instruction_source": instr_info["source"],
        }

        _CALL_IDX["n"] += 1
        _log_call({
            "call": _CALL_IDX["n"],
            "wall": round(time.time(), 3),
            "state": [round(float(v), 5) for v in np.asarray(state).reshape(-1)[:14]],
            "proprio_norm": [round(float(v), 4) for v in np.asarray(proprio_np).reshape(-1)[:14]],
            "tokens": [int(t) for t in tokens],
            "token_ids": [int(t) for t in tok_ids],
            "action_target_mode": self.action_target_mode,
            "decoded_target": [round(float(v), 6) for v in decoded_target],
            "action": [round(float(v), 5) for v in action[0]],
            "delta_action_minus_state": [round(float(a - b), 5) for a, b in zip(action[0], np.asarray(state).reshape(-1)[:14])],
            "instruction_used": instr_info["instruction"],
            "instruction_source": instr_info["source"],
            "obs_instruction": instr_info["obs_instruction"],
            "obs_instruction_ignored": bool(instr_info["obs_ignored"]),
        })
        return action

    # --------------------------------------------------------------- actions
    def get_action(self, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")
        chunk = self.infer(self.observation_window[0],
                           env_idx=self._latest_env_idx_list[0] if self._latest_env_idx_list else 0)
        if self.robot_action_dim_info is None:
            return [chunk[0]]
        return unpack_robot_state(chunk, self.action_type, self.robot_action_dim_info, source_type="obs")

    def get_action_batch(self, env_idx_list=None, **kwargs):
        indices = env_idx_list or self._latest_env_idx_list
        return [self.get_action(**kwargs) for _ in indices]

    def reset(self):
        self.observation_window = None
        if getattr(self, "proprio_history", None) is not None:
            self.proprio_history.reset()
        return
