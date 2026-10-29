from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from diagnostics.r46_openvla_smoke import _build_r44_argv  # noqa: E402
from r44_train import parse_args  # noqa: E402


class R46R44ArgumentBridgeTests(unittest.TestCase):
    def test_child_wall_limit_is_not_forwarded_to_r44_parser(self):
        argv = _build_r44_argv(
            ["--base-model", "/models/openvla"],
            manifest_path=Path("/data/manifest.json"),
            output_path=Path("/tmp/r46-smoke"),
            approved_gpu_uuid="GPU-test",
        )

        self.assertNotIn("--max-wall-seconds", argv)
        parsed = parse_args(argv)
        self.assertEqual(parsed.epochs, 5)
        self.assertEqual(parsed.base_model, "/models/openvla")
        self.assertEqual(parsed.dataset_manifest, str(Path("/data/manifest.json")))
        self.assertTrue(parsed.gpu_window_confirmed)
        self.assertEqual(parsed.expected_gpu_uuid, "GPU-test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
