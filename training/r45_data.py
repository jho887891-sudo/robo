#!/usr/bin/env python3
"""R45 index-based data pipeline for the joint_delta_v1 full-data training.

Why: the historical loaders open an HDF5 file per training sample
(training/m2_multitask_data.py:163,384, training/r11_train_pushT_r42.py:140,208,570). At
108,780 samples/epoch that is 108,780 open/decode/close cycles per epoch. This module keeps one
handle per episode per worker process, decodes the three JPEG cameras lazily, and never holds more
decoded pixels than an explicit byte budget.

Contract: sample i is (task, episode, t) with t in [2, T-2]; it yields state[t] (14D), action[t]
(14D), the three RGB frames in CAMERA_ORDER, and the normalised 42D feature built from
state[t-2..t] only (no future information).
"""
import hashlib
import json
import os
import time
from collections import OrderedDict

import numpy as np

BASE_DIM = 14
N42 = 42
MIN_HISTORY = 2
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
JOINTS = ("left_arm_joint_states", "left_ee_joint_states",
          "right_arm_joint_states", "right_ee_joint_states")


def _sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_index(splits, hdf5_root, which="train", tasks=None):
    """[(task, path, t)] for every valid frame of every listed episode."""
    entries = []
    for task in sorted(splits):
        if tasks and task not in tasks:
            continue
        for rel in splits[task][which]:
            path = os.path.join(hdf5_root, task, "arx_x5", "data", os.path.basename(rel))
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            import h5py
            with h5py.File(path, "r") as h:
                T = int(h["state"][JOINTS[0]].shape[0])
                cams = [c for c in CAMERAS if ("vision/%s/colors" % c) in h]
            if len(cams) != 3:
                raise ValueError("%s has cameras %s, expected 3" % (path, cams))
            for t in range(MIN_HISTORY, T - 1):
                entries.append((task, path, int(t)))
    return entries


def index_manifest(entries, seed=0):
    """Stable description of the index: counts per task and a content hash."""
    per_task = {}
    for task, _p, _t in entries:
        per_task[task] = per_task.get(task, 0) + 1
    body = {"n": len(entries), "per_task": per_task, "seed": int(seed),
            "first": [list(entries[0][::2]) if entries else []]}
    return {"n_samples": len(entries), "per_task": per_task,
            "index_sha256": _sha256_json([[e[0], os.path.basename(e[1]), e[2]] for e in entries])}


class EpochOrder:
    """Deterministic, no-replacement ordering for (base_seed, epoch)."""

    def __init__(self, n, base_seed, epoch):
        self.n = int(n)
        self.epoch = int(epoch)
        self.order = np.random.default_rng([int(base_seed), int(epoch)]).permutation(self.n)

    def hash(self):
        return hashlib.sha256(self.order.astype(np.int64).tobytes()).hexdigest()


# --------------------------------------------------------------------------- per-worker handles
_HANDLES = {}


def _handle(path, max_open=24):
    """One open HDF5 handle per episode per process (LRU-capped, never shared across forks)."""
    h = _HANDLES.get(path)
    if h is not None:
        _HANDLES.move_to_end(path) if hasattr(_HANDLES, "move_to_end") else None
        return h
    import h5py
    h = h5py.File(path, "r")
    _HANDLES[path] = h
    if len(_HANDLES) > max_open:
        oldest = next(iter(_HANDLES))
        if oldest != path:
            _HANDLES.pop(oldest).close()
    return h


def close_handles():
    for h in list(_HANDLES.values()):
        try:
            h.close()
        except Exception:
            pass
    _HANDLES.clear()


def worker_init(_worker_id):
    """DataLoader worker entry point: a fork must not reuse the parent's handles."""
    close_handles()


# --------------------------------------------------------------------------------- dataset
class R45Dataset:
    """Map-style dataset over the frozen index with a bounded decoded-image cache."""

    def __init__(self, entries, cache_bytes=256 * 1024 * 1024, decode=True):
        self.entries = entries
        self.cache_bytes = int(cache_bytes)
        self.decode = bool(decode)
        self._cache = {}
        self._cache_bytes = 0
        self.stats = {"reads": 0, "cache_hits": 0, "bytes": 0}

    def __len__(self):
        return len(self.entries)

    # -- state / action ----------------------------------------------------------------
    def _arrays(self, path):
        h = _handle(path)
        st = np.concatenate([np.asarray(h["state"][k], np.float64) for k in JOINTS], axis=1)
        ac = np.concatenate([np.asarray(h["action"][k], np.float64) for k in JOINTS], axis=1)
        return st, ac

    def _images(self, path, t):
        key = (path, t)
        hit = self._cache.get(key)
        if hit is not None:
            self.stats["cache_hits"] += 1
            return hit
        import cv2
        h = _handle(path)
        out = {}
        for cam in CAMERAS:
            raw = np.asarray(h["vision/%s/colors" % cam][t])
            buf = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("JPEG decode failed for %s t=%d cam=%s" % (path, t, cam))
            img = img[:, :, ::-1]                      # BGR -> RGB
            if img.shape[2] != 3 or img.dtype != np.uint8:
                raise ValueError("bad frame %s" % (img.shape,))
            out[cam] = np.ascontiguousarray(img)
        nbytes = sum(v.nbytes for v in out.values())
        if nbytes <= self.cache_bytes:
            self._cache[key] = out
            self._cache_bytes += nbytes
            while self._cache_bytes > self.cache_bytes and self._cache:
                old = next(iter(self._cache))
                self._cache_bytes -= sum(v.nbytes for v in self._cache.pop(old).values())
        self.stats["reads"] += 1
        return out

    def __getitem__(self, i):
        task, path, t = self.entries[i]
        st, ac = self._arrays(path)
        if not (MIN_HISTORY <= t < len(st) - 1):
            raise IndexError("t=%d outside the frozen window for %s" % (t, path))
        state = st[t].astype(np.float32)
        action = st[t + 1].astype(np.float32)          # A_t == S_{t+1} (verified bit-exact)
        hist = st[t - MIN_HISTORY:t + 1].astype(np.float32)
        item = {
            "index": int(i), "task": task, "episode": os.path.basename(path), "t": int(t),
            "state": state, "action": action, "history": hist,
            "delta": (action - state).astype(np.float32),
        }
        if self.decode:
            item["images"] = self._images(path, t)
        return item


def feature42(history, contract):
    """42D feature from the observed history only: [state_t, state_t-state_{t-1}, acc]."""
    h = np.asarray(history, np.float32)
    if h.shape != (MIN_HISTORY + 1, BASE_DIM):
        raise ValueError("history must be (3,14), got %s" % (h.shape,))
    now, prev, prev2 = h[2], h[1], h[0]
    v = now - prev
    acc = (now - prev) - (prev - prev2)
    feat = contract.transform(now, v.astype(np.float32), acc.astype(np.float32))
    if feat.shape != (N42,) or not np.isfinite(feat).all():
        raise ValueError("42D feature invalid")
    return feat.astype(np.float32)


class RuntimeReader:
    """Formal-entry reader: worker/process-local HDF5 handles, bounded image cache, phase timing."""

    def __init__(self, cache_bytes=256 * 1024 * 1024, max_open_handles=24,
                 batch_assembly="cat", cache_kind="processed"):
        self.cache_bytes = int(cache_bytes)
        self.max_open_handles = int(max_open_handles)
        if self.cache_bytes < 0 or self.max_open_handles < 1:
            raise ValueError("cache_bytes must be >= 0 and max_open_handles must be >= 1")
        if batch_assembly not in ("cat", "preallocate_bf16"):
            raise ValueError("batch_assembly must be cat or preallocate_bf16")
        if cache_kind not in ("none", "processed", "jpeg", "rgb"):
            raise ValueError("invalid cache_kind")
        self.batch_assembly = batch_assembly
        self.cache_kind = cache_kind
        self.cache_evictions = 0
        self._handles = OrderedDict()
        self._images = OrderedDict()
        self._cache_bytes = 0
        self.totals = {"hdf5_read_s": 0.0, "jpeg_preprocess_s": 0.0,
                       "state_label_s": 0.0, "cpu_to_gpu_s": 0.0,
                       "samples": 0}

    def _handle(self, path):
        path = os.fspath(path)
        h = self._handles.get(path)
        if h is not None:
            self._handles.move_to_end(path)
            return h
        if len(self._handles) >= self.max_open_handles:
            _, oldest = self._handles.popitem(last=False)
            oldest.close()
        import h5py
        h = h5py.File(path, "r")
        self._handles[path] = h
        return h

    def load_state_cache(self, plan, contract):
        t0 = time.perf_counter()
        cache = {}
        for key, path in sorted(plan["episode_paths"].items()):
            h = self._handle(path)
            cache[tuple(key.split("/", 1))] = contract.read14(h, "state"), contract.read14(h, "action")
        self.totals["hdf5_read_s"] += time.perf_counter() - t0
        return cache

    def load_images(self, proc, path, t, cams, camera_order, device="cuda"):
        """Read the 3-camera sample and report non-overlapping host-side phase times.

        cpu_to_gpu_s measures host-observed transfer duration, not CUDA kernel execution.
        A real GPU throughput benchmark must use synchronization/events separately.
        """
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        if cams != 3:
            raise ValueError("R45 reader requires exactly 3 cameras")
        key = (os.fspath(path), int(t), tuple(camera_order[:cams]))
        cached = None if self.cache_kind == "none" else self._images.get(key)
        read_s = 0.0
        jpeg_s = 0.0
        cache_hit = cached is not None
        if cache_hit:
            self._images.move_to_end(key)
            cached = cached[0]
        if self.cache_kind == "processed" and cache_hit:
            tiles = cached
        else:
            tiles, encoded, rgb = [], [], []
            for cam_index, cam in enumerate(camera_order[:cams]):
                if self.cache_kind == "rgb" and cache_hit:
                    img = cached[cam_index]
                else:
                    if self.cache_kind == "jpeg" and cache_hit:
                        raw = cached[cam_index]
                    else:
                        read_start = time.perf_counter()
                        raw = bytes(self._handle(path)["vision/" + cam + "/colors"][int(t)])
                        read_s += time.perf_counter() - read_start
                    encoded.append(raw)
                    decode_start = time.perf_counter()
                    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise ValueError("JPEG decode failed for %s t=%d cam=%s" % (path, t, cam))
                    img = np.ascontiguousarray(bgr[..., ::-1])
                    jpeg_s += time.perf_counter() - decode_start
                rgb.append(img)
                transform_start = time.perf_counter()
                tiles.append(proc.image_processor.apply_transform(Image.fromarray(img)))
                jpeg_s += time.perf_counter() - transform_start
            if not cache_hit and self.cache_kind != "none":
                if self.cache_kind == "processed":
                    value = tiles
                    size = sum(int(x.numel() * x.element_size()) for x in value)
                elif self.cache_kind == "jpeg":
                    value = encoded
                    size = sum(len(x) for x in value)
                else:
                    value = rgb
                    size = sum(int(x.nbytes) for x in value)
                if size <= self.cache_bytes:
                    self._images[key] = (value, size)
                    self._cache_bytes += size
                    while self._cache_bytes > self.cache_bytes:
                        _, (_, old_size) = self._images.popitem(last=False)
                        self._cache_bytes -= old_size
                        self.cache_evictions += 1
        transfer_start = time.perf_counter()
        if self.batch_assembly == "cat":
            pixel_values = torch.cat(tiles, dim=0).unsqueeze(0).to(torch.bfloat16).to(device)
        else:
            from training.r50_batch_builder import assemble_camera_tiles
            pixel_values = assemble_camera_tiles(tiles, self.batch_assembly).to(device)
        transfer_s = time.perf_counter() - transfer_start
        self.totals["hdf5_read_s"] += read_s
        self.totals["jpeg_preprocess_s"] += jpeg_s
        is_cuda = str(device).startswith("cuda")
        self.totals["cpu_to_gpu_s"] += transfer_s if is_cuda else 0.0
        self.totals["samples"] += 1
        return pixel_values, {
            "hdf5_read_s": read_s,
            "jpeg_preprocess_s": jpeg_s,
            # For device='cpu' no GPU transfer occurred. Preserve the legacy
            # key but use a separate CPU tensor-preparation field.
            "cpu_to_gpu_s": transfer_s if is_cuda else 0.0,
            "cpu_tensor_preparation_s": 0.0 if is_cuda else transfer_s,
            "transfer_device": str(device),
            "cache_hit": cache_hit,
            "cache_kind": self.cache_kind,
            "cache_evictions": self.cache_evictions,
            "cache_bytes": self._cache_bytes,
            "timing_kind": "host_wall_no_cuda_event",
        }

    def close(self):
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()
        self._images.clear()
        self._cache_bytes = 0
