"""CPU-only R50 DataLoader experiment built on the existing R45 RuntimeReader.

The parent process stores only sample identities. Every spawned worker constructs
its own OpenVLA processor and HDF5 reader after process startup.
"""
from __future__ import annotations

import os
from pathlib import Path

from training.r45_data import CAMERAS, RuntimeReader, worker_init


def load_openvla_processor(base_model: str, oft_root: str):
    import sys
    source = str(Path(oft_root).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    return AutoProcessor.from_pretrained(base_model, trust_remote_code=True,
                                         local_files_only=True)


class R50WorkerDataset:
    """Map dataset whose reader and processor are worker-local and lazily made."""

    def __init__(self, selected, hdf5_root, base_model, oft_root, cache_bytes=0):
        self.selected = [tuple(item) for item in selected]
        self.hdf5_root = str(hdf5_root)
        self.base_model = str(base_model)
        self.oft_root = str(oft_root)
        self.cache_bytes = int(cache_bytes)
        if self.cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        self._pid = None
        self._reader = None
        self._processor = None

    def __len__(self):
        return len(self.selected)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pid"] = None
        state["_reader"] = None
        state["_processor"] = None
        return state

    def close(self):
        if getattr(self, "_reader", None) is not None:
            self._reader.close()
        self._reader = None
        self._processor = None
        self._pid = None

    def __del__(self):
        self.close()

    def _local(self):
        pid = os.getpid()
        if self._pid != pid:
            self.close()
            self._reader = RuntimeReader(cache_bytes=self.cache_bytes)
            self._processor = load_openvla_processor(self.base_model, self.oft_root)
            self._pid = pid
        return self._reader, self._processor

    def __getitem__(self, index):
        task, episode, frame = self.selected[index]
        path = Path(self.hdf5_root) / task / "arx_x5" / "data" / episode
        reader, processor = self._local()
        pixels, timing = reader.load_images(processor, path, int(frame), 3, CAMERAS,
                                            device="cpu")
        return {"index": int(index), "pixels": pixels.squeeze(0),
                "hdf5_s": timing["hdf5_read_s"],
                "jpeg_preprocess_s": timing["jpeg_preprocess_s"],
                "cache_hit": bool(timing["cache_hit"])}


def create_loader(dataset, workers: int, prefetch: int, pin_memory: bool):
    import multiprocessing as mp
    from torch.utils.data import DataLoader
    if workers < 1 or workers > 8 or prefetch < 1 or prefetch > 4:
        raise ValueError("workers must be 1..8, prefetch 1..4")
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                      prefetch_factor=prefetch, persistent_workers=True,
                      pin_memory=bool(pin_memory), worker_init_fn=worker_init,
                      multiprocessing_context=mp.get_context("spawn"))
