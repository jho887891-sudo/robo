import unittest
import tempfile
from pathlib import Path

import torch

from training.r50_batch_builder import assemble_camera_tiles


class BatchBuilderTests(unittest.TestCase):
    def test_runtime_reader_opt_in_matches_current_path(self):
        import cv2
        import h5py
        import numpy as np
        from training.r45_data import CAMERAS, RuntimeReader

        class Processor:
            image_processor = None

            def __init__(self):
                self.image_processor = self

            def apply_transform(self, image):
                array = np.asarray(image, dtype=np.uint8)[:8, :8].copy()
                tile = torch.from_numpy(array).permute(2, 0, 1).float() / 255
                return torch.cat((tile, tile), dim=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.hdf5"
            with h5py.File(path, "w") as h:
                for i, camera in enumerate(CAMERAS):
                    pixels = np.full((12, 12, 3), (i + 1) * 37, np.uint8)
                    ok, jpeg = cv2.imencode(".jpg", pixels)
                    self.assertTrue(ok)
                    data = h.create_dataset(f"vision/{camera}/colors", (3,),
                                            dtype=h5py.vlen_dtype(np.dtype("uint8")))
                    for frame in range(3):
                        data[frame] = jpeg
            baseline = RuntimeReader(cache_bytes=0)
            candidate = RuntimeReader(cache_bytes=0, batch_assembly="preallocate_bf16")
            try:
                old, _ = baseline.load_images(Processor(), path, 2, 3, CAMERAS, device="cpu")
                new, _ = candidate.load_images(Processor(), path, 2, 3, CAMERAS, device="cpu")
            finally:
                baseline.close()
                candidate.close()
            self.assertTrue(torch.equal(old, new))
            self.assertEqual(tuple(new.shape), (1, 18, 8, 8))

    def test_preallocated_matches_current_cat_bit_exactly(self):
        generator = torch.Generator().manual_seed(50)
        tiles = [torch.randn((6, 9, 11), generator=generator) for _ in range(3)]
        reference = assemble_camera_tiles(tiles, "cat")
        candidate = assemble_camera_tiles(tiles, "preallocate_bf16")
        self.assertEqual(tuple(candidate.shape), (1, 18, 9, 11))
        self.assertEqual(candidate.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(reference, candidate))

    def test_noncontiguous_tiles_preserve_values(self):
        tiles = [torch.arange(6 * 8 * 8, dtype=torch.float32).reshape(6, 8, 8)
                 .transpose(1, 2) for _ in range(3)]
        self.assertTrue(torch.equal(assemble_camera_tiles(tiles),
                                    assemble_camera_tiles(tiles, "preallocate_bf16")))

    def test_bad_shape_or_method_fails_closed(self):
        good = torch.zeros((6, 8, 8))
        with self.assertRaises(ValueError):
            assemble_camera_tiles([good] * 2)
        with self.assertRaises(ValueError):
            assemble_camera_tiles([good, good, good[:, :7, :]])
        with self.assertRaises(ValueError):
            assemble_camera_tiles([good] * 3, "unknown")


if __name__ == "__main__":
    unittest.main()
