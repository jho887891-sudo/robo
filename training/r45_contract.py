"""Fail-closed R45 training contract for the approved 42D state history."""
from dataclasses import dataclass, field
import numpy as np

class ContractError(ValueError):
    pass

@dataclass(frozen=True)
class R45Config:
    target_contract: str = "joint_delta_v1"
    action_dim: int = 14
    proprio_dim: int = 42
    history_layout: str = "[S_t,V_t,Acc_t]"
    timestamp_alignment: str = "state[t-2],state[t-1],state[t]"
    normalization: str = "explicit approved TRAIN-only transform"
    cameras: tuple = field(default=("cam_head", "cam_left_wrist", "cam_right_wrist"))
    lora_r: int = 32
    lora_alpha: int = 16
    lora_targets: str = "q_proj/v_proj"
    epochs: int = 5
    train_samples: int = 108780
    dev_samples: int = 28268

    def __post_init__(self):
        if self.target_contract != "joint_delta_v1":
            raise ContractError("only joint_delta_v1 is allowed")
        if self.action_dim != 14:
            raise ContractError("action_dim must remain 14")
        if self.proprio_dim != 42:
            raise ContractError("proprio_dim must remain 42; refusing silent fallback")
        if self.history_layout != "[S_t,V_t,Acc_t]":
            raise ContractError("42D layout must be [S_t,V_t,Acc_t]")
        if self.timestamp_alignment != "state[t-2],state[t-1],state[t]":
            raise ContractError("timestamp alignment is not approved")
        if self.normalization != "explicit approved TRAIN-only transform":
            raise ContractError("normalization provenance is not approved")
        if tuple(self.cameras) != ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            raise ContractError("camera order must be the approved three-camera order")
        if self.lora_r != 32 or self.lora_alpha != 16:
            raise ContractError("LoRA must use r=32 and alpha=16")
        if self.lora_targets != "q_proj/v_proj":
            raise ContractError("LoRA target set must be explicit")
        if self.epochs != 5:
            raise ContractError("formal training must use five epochs")
        if self.train_samples != 108780 or self.dev_samples != 28268:
            raise ContractError("dataset counts must match the fixed R45 baseline")

    def transform(self, state, velocity, acceleration):
        parts = [np.asarray(x, dtype=np.float32).reshape(-1)
                 for x in (state, velocity, acceleration)]
        if any(x.shape != (14,) or not np.isfinite(x).all() for x in parts):
            raise ContractError("state, velocity, acceleration must be finite 14D vectors")
        return np.concatenate(parts).astype(np.float32)

    def validate_feature(self, feature):
        x = np.asarray(feature, dtype=np.float32).reshape(-1)
        if x.shape != (42,) or not np.isfinite(x).all():
            raise ContractError("feature must be finite 42D")
        return x

