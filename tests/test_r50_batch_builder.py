import unittest

import torch

from training.r50_batch_builder import assemble_camera_tiles


class BatchBuilderTests(unittest.TestCase):
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
