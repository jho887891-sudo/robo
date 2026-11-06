"""CPU regression for bounded R51 image caches on the formal reader."""
import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from training.r45_data import CAMERAS, RuntimeReader


class Processor:
    image_processor = None

    def __init__(self):
        self.image_processor = self

    def apply_transform(self, image):
        pixels = np.asarray(image, dtype=np.uint8).copy()
        tile = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
        return torch.cat((tile, tile), dim=0)


class CacheTypeTests(unittest.TestCase):
    def test_all_cache_types_match_uncached_and_obey_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "episode.hdf5"
            with h5py.File(path, "w") as h:
                for camera_idx, camera in enumerate(CAMERAS):
                    data = h.create_dataset(
                        f"vision/{camera}/colors", (2,),
                        dtype=h5py.vlen_dtype(np.dtype("uint8")))
                    for frame in range(2):
                        pixels = np.full((12, 12, 3), 30 * camera_idx + frame * 17,
                                         np.uint8)
                        ok, jpeg = cv2.imencode(".jpg", pixels)
                        self.assertTrue(ok)
                        data[frame] = jpeg
            processor = Processor()
            baseline = RuntimeReader(cache_bytes=0, cache_kind="none")
            try:
                expected = [baseline.load_images(processor, path, frame, 3, CAMERAS,
                                                 device="cpu")[0] for frame in range(2)]
            finally:
                baseline.close()
            for kind in ("none", "jpeg", "rgb", "processed"):
                with self.subTest(kind=kind):
                    reader = RuntimeReader(cache_bytes=16384, cache_kind=kind)
                    try:
                        hits = []
                        for frame in (0, 0, 1, 0):
                            actual, timing = reader.load_images(processor, path, frame,
                                                                3, CAMERAS, device="cpu")
                            self.assertTrue(torch.equal(actual, expected[frame]))
                            self.assertLessEqual(timing["cache_bytes"], 16384)
                            hits.append(timing["cache_hit"])
                        self.assertEqual(hits[1], kind != "none")
                        self.assertEqual(kind != "none", len(reader._images) > 0)
                        if kind == "processed":
                            self.assertGreater(reader.cache_evictions, 0)
                    finally:
                        reader.close()


if __name__ == "__main__":
    unittest.main()
