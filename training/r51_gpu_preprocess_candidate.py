"""Opt-in R51 tensor resize candidate; requires measured parity before training use.

This is independent of the current PIL production path. CPU invocation is for
contract diagnostics only; it is not a GPU performance measurement.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def validate_processor(image_processor):
    if image_processor.image_resize_strategy != "resize-naive":
        raise ValueError("candidate requires resize-naive OpenVLA processor")
    if not image_processor.use_fused_vision_backbone:
        raise ValueError("candidate requires fused two-tower OpenVLA processor")
    if len(image_processor.tvf_resize_params) != 2:
        raise ValueError("candidate requires two vision towers")
    for resize, crop in zip(image_processor.tvf_resize_params,
                            image_processor.tvf_crop_params):
        if tuple(resize["size"]) != (224, 224) or tuple(crop["output_size"]) != (224, 224):
            raise ValueError("unexpected OpenVLA resize/crop size")
    if len(image_processor.tvf_normalize_params) != 2:
        raise ValueError("missing OpenVLA tower normalization")


def tensor_preprocess(rgb, image_processor, *, device):
    """Return [B,18,224,224] BF16 in head/left/right order.

    rgb: CPU uint8 [B,3,H,W,3]. Interpolation differs from PIL on some pixels;
    this candidate must not replace the production path until a parity Gate is
    defined and passed on real fixed TRAIN samples.
    """
    validate_processor(image_processor)
    if rgb.ndim != 5 or rgb.shape[1] != 3 or rgb.shape[-1] != 3 or rgb.dtype != torch.uint8:
        raise ValueError("expected RGB uint8 [B,3,H,W,3]")
    if device != "cpu" and not str(device).startswith("cuda"):
        raise ValueError("device must be cpu or cuda")
    batch, cams = rgb.shape[:2]
    x = rgb.to(device=device, non_blocking=str(device).startswith("cuda"))
    x = x.reshape(batch * cams, *rgb.shape[2:]).permute(0, 3, 1, 2).float()
    # PIL bicubic resize writes uint8 RGB; explicitly quantize before ToTensor.
    x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False,
                      antialias=True).round().clamp_(0, 255).div_(255)
    towers = []
    for cfg in image_processor.tvf_normalize_params:
        mean = x.new_tensor(cfg["mean"]).view(1, 3, 1, 1)
        std = x.new_tensor(cfg["std"]).view(1, 3, 1, 1)
        towers.append(((x - mean) / std).reshape(batch, cams, 3, 224, 224))
    return torch.stack(towers, dim=2).reshape(batch, 18, 224, 224).to(torch.bfloat16)


def compare_with_reference(rgb, image_processor):
    """CPU-only numerical audit; reports error, never grants parity by tolerance."""
    from PIL import Image
    candidate = tensor_preprocess(rgb, image_processor, device="cpu")
    reference = torch.stack([
        torch.cat([image_processor.apply_transform(Image.fromarray(rgb[b, c].numpy()))
                   for c in range(3)], dim=0).to(torch.bfloat16)
        for b in range(rgb.shape[0])])
    difference = (candidate.float() - reference.float()).abs()
    return {"exact": bool(torch.equal(candidate, reference)),
            "different_elements": int(torch.count_nonzero(difference).item()),
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
            "element_count": int(difference.numel())}


def prepare_pil_resized_rgb(rgb, image_processor):
    """Preserve production PIL resize pixels; defer tensor normalization."""
    import numpy as np
    from PIL import Image
    import torchvision.transforms.functional as TVF

    validate_processor(image_processor)
    first, second = image_processor.tvf_resize_params
    if first != second:
        raise ValueError("different tower resizes require separate buffers")
    if rgb.ndim != 5 or rgb.shape[1] != 3 or rgb.shape[-1] != 3 or rgb.dtype != torch.uint8:
        raise ValueError("expected RGB uint8 [B,3,H,W,3]")
    resized = []
    for batch in rgb:
        cameras = []
        for pixels in batch:
            image = Image.fromarray(pixels.numpy())
            image = TVF.resize(image, **first)
            image = TVF.center_crop(image, **image_processor.tvf_crop_params[0])
            cameras.append(torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()))
        resized.append(torch.stack(cameras))
    return torch.stack(resized)


def normalize_resized_rgb(resized, image_processor, *, device):
    """Batched ToTensor/Normalize/assembly on requested device."""
    validate_processor(image_processor)
    if resized.ndim != 5 or tuple(resized.shape[1:]) != (3, 224, 224, 3):
        raise ValueError("expected resized [B,3,224,224,3]")
    if resized.dtype != torch.uint8:
        raise ValueError("resized pixels must be uint8")
    x = resized.to(device=device, non_blocking=str(device).startswith("cuda"))
    x = x.permute(0, 1, 4, 2, 3).float().div_(255)
    towers = []
    for cfg in image_processor.tvf_normalize_params:
        mean = x.new_tensor(cfg["mean"]).view(1, 1, 3, 1, 1)
        std = x.new_tensor(cfg["std"]).view(1, 1, 3, 1, 1)
        towers.append((x - mean) / std)
    return torch.stack(towers, dim=2).reshape(-1, 18, 224, 224).to(torch.bfloat16)


def compare_hybrid_with_reference(rgb, image_processor):
    """CPU-only exact-parity audit for the hybrid candidate."""
    from PIL import Image
    resized = prepare_pil_resized_rgb(rgb, image_processor)
    candidate = normalize_resized_rgb(resized, image_processor, device="cpu")
    reference = torch.stack([
        torch.cat([image_processor.apply_transform(Image.fromarray(rgb[b, c].numpy()))
                   for c in range(3)], dim=0).to(torch.bfloat16)
        for b in range(rgb.shape[0])])
    difference = (candidate.float() - reference.float()).abs()
    return {"exact": bool(torch.equal(candidate, reference)),
            "different_elements": int(torch.count_nonzero(difference).item()),
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
            "element_count": int(difference.numel())}
