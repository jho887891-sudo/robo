"""Opt-in CPU tensor assembly candidates for R50 P02.

Both methods preserve the exact three-camera channel order produced by the
frozen OpenVLA processor. The default remains the existing R45 cat path.
"""
from __future__ import annotations


def assemble_camera_tiles(tiles, method="cat"):
    import torch

    if len(tiles) != 3:
        raise ValueError("exactly three camera tiles are required")
    height, width = tiles[0].shape[-2:]
    if any(tile.ndim != 3 or tile.shape[-2:] != (height, width)
           or tile.dtype != torch.float32 or tile.device.type != "cpu"
           for tile in tiles):
        raise ValueError("camera tiles must be CPU float32 CHW with equal spatial shape")
    if method == "cat":
        return torch.cat(tiles, dim=0).unsqueeze(0).to(torch.bfloat16)
    if method == "preallocate_bf16":
        channels = sum(tile.shape[0] for tile in tiles)
        out = torch.empty((1, channels, height, width), dtype=torch.bfloat16)
        offset = 0
        for tile in tiles:
            n = tile.shape[0]
            out[0, offset:offset + n].copy_(tile)
            offset += n
        return out
    raise ValueError(f"unknown assembly method: {method}")
