"""Pure autoregressive OpenVLA input policy.

The OFT/parallel forward() zeros action-token embeddings based on labels.
For *causal autoregressive* training the preceding action-token embeddings must
remain visible. Do not change OFT's original behavior globally.
"""
import types

import torch


def enable_pure_autoreg(model):
    """Install a zero action-input mask on an OpenVLA core model instance.

    Apply to the core model BEFORE wrapping it with PEFT; PEFT then delegates to
    the same instance. Reapply after merge_and_unload in inference for clarity.
    The caller must retain causal attention (standard language_model causal mask).
    """
    if not hasattr(model, "_process_action_masks"):
        raise TypeError("model has no _process_action_masks: incompatible OpenVLA implementation")

    def preserve_action_embeddings(self, labels):
        if labels is None:
            raise ValueError("labels required for multimodal autoregressive forward")
        return torch.zeros_like(labels, dtype=torch.bool)

    model._process_action_masks = types.MethodType(preserve_action_embeddings, model)
    model._pure_autoreg_input_mode = True
    return model
