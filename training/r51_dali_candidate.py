"""Optional DALI/nvJPEG decode probe for R51 F.

This module is never imported by the formal trainer. The decoder returns RGB
on CPU for a strict comparison with the frozen OpenCV path; copying back is a
diagnostic cost and must be included in a later end-to-end benchmark.
"""
from __future__ import annotations

import importlib.util


class DALIUnavailable(RuntimeError):
    pass


def dali_available():
    try:
        return importlib.util.find_spec("nvidia.dali") is not None
    except ModuleNotFoundError:
        return False


def require_dali():
    if not dali_available():
        raise DALIUnavailable("BLOCKED_DALI_DEPENDENCY: nvidia.dali is absent")
    from nvidia.dali import fn, types
    from nvidia.dali.pipeline import Pipeline
    return fn, types, Pipeline


class ThreeCameraDaliDecoder:
    """Fixed-size HDF5 JPEG batch, camera order preserved by flat row order."""

    def __init__(self, batch_size: int, *, device_id: int = 0, num_threads: int = 2):
        if not 1 <= batch_size <= 16 or not 1 <= num_threads <= 8 or device_id < 0:
            raise ValueError("invalid bounded DALI decoder configuration")
        fn, types, Pipeline = require_dali()  # fail before CUDA when absent
        self.batch_size = batch_size
        self.pipeline = Pipeline(batch_size=3 * batch_size, num_threads=num_threads,
                                 device_id=device_id, prefetch_queue_depth=1)
        with self.pipeline:
            encoded = fn.external_source(name="r51_jpegs", device="cpu",
                                         dtype=types.UINT8)
            decoded = fn.decoders.image(encoded, device="mixed",
                                        output_type=types.RGB)
            self.pipeline.set_outputs(decoded)
        self.pipeline.build()

    def decode_for_parity(self, encoded):
        """Return [B,3,H,W,3] uint8 CPU, including GPU->CPU diagnostic transfer."""
        import numpy as np
        if len(encoded) != self.batch_size or any(len(row) != 3 for row in encoded):
            raise ValueError("expected batch of three ordered camera JPEGs")
        flat = [bytes(jpeg) for row in encoded for jpeg in row]
        if any(not item for item in flat):
            raise ValueError("empty JPEG")
        self.pipeline.feed_input("r51_jpegs", [np.frombuffer(item, dtype=np.uint8)
                                                for item in flat])
        decoded = self.pipeline.run()[0].as_cpu()
        pixels = [np.asarray(decoded.at(i)) for i in range(len(flat))]
        if any(p.dtype != np.uint8 or p.ndim != 3 or p.shape[-1] != 3
               for p in pixels):
            raise ValueError("DALI RGB output contract mismatch")
        if len({p.shape for p in pixels}) != 1:
            raise ValueError("DALI outputs have mixed image sizes")
        return np.stack(pixels).reshape(self.batch_size, 3, *pixels[0].shape)
