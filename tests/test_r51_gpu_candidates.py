"""CPU contract checks for opt-in C/F candidates; never opens CUDA."""
import unittest

import torch

from training.r51_gpu_preprocess_candidate import normalize_resized_rgb, tensor_preprocess
from training.r51_dali_candidate import (DALIUnavailable, ThreeCameraDaliDecoder,
                                         dali_available, require_dali)


class Processor:
    image_resize_strategy = "resize-naive"
    use_fused_vision_backbone = True
    tvf_resize_params = [{"size": (224, 224)}, {"size": (224, 224)}]
    tvf_crop_params = [{"output_size": (224, 224)},
                       {"output_size": (224, 224)}]
    tvf_normalize_params = [{"mean": [0, 0, 0], "std": [1, 1, 1]},
                            {"mean": [0, 0, 0], "std": [1, 1, 1]}]


class CandidateTests(unittest.TestCase):
    def test_hybrid_keeps_camera_and_tower_order(self):
        rgb = torch.zeros((1, 3, 224, 224, 3), dtype=torch.uint8)
        for camera, value in enumerate((51, 102, 153)):
            rgb[:, camera] = value
        result = normalize_resized_rgb(rgb, Processor(), device="cpu")
        self.assertEqual(tuple(result.shape), (1, 18, 224, 224))
        for camera, value in enumerate((51, 102, 153)):
            for tower in (0, 1):
                expected = torch.tensor(value / 255, dtype=torch.bfloat16)
                self.assertTrue(torch.all(result[0, camera * 6 + tower * 3:camera * 6 + tower * 3 + 3]
                                          == expected))
        self.assertFalse(torch.cuda.is_initialized())

    def test_shape_and_processor_drift_fail_closed(self):
        with self.assertRaises(ValueError):
            tensor_preprocess(torch.zeros(1, 3, 10, 10, 3), Processor(), device="cpu")
        proc = Processor()
        proc.image_resize_strategy = "letterbox"
        with self.assertRaises(ValueError):
            normalize_resized_rgb(torch.zeros(1, 3, 224, 224, 3,
                                             dtype=torch.uint8), proc, device="cpu")

    def test_dali_dependency_fails_before_pipeline_or_cuda(self):
        if dali_available():
            self.skipTest("DALI installed; missing dependency path not applicable")
        with self.assertRaisesRegex(DALIUnavailable, "BLOCKED_DALI_DEPENDENCY"):
            require_dali()
        with self.assertRaises(DALIUnavailable):
            ThreeCameraDaliDecoder(1)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
