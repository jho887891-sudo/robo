import signal
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from training.r44_safe_stop import (
    R44StopController,
    atomic_save_checkpoint,
    stop_and_checkpoint_if_requested,
    configure_torch_memory_fraction,
)


class SafeStopTests(unittest.TestCase):
    def test_memory_fraction_disabled_by_default(self):
        result = configure_torch_memory_fraction(None)
        self.assertFalse(result["enabled"])
        self.assertFalse(result["compute_isolation"])
        self.assertFalse(result["hard_gpu_isolation"])

    def test_memory_fraction_configures_only_allocator(self):
        gib = 1024**3
        calls = {}
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                current_device=lambda: 0,
                get_device_properties=lambda index: SimpleNamespace(total_memory=50 * gib),
                set_per_process_memory_fraction=lambda fraction, device: calls.update(
                    fraction=fraction, device=device
                ),
            )
        )
        with patch.dict("sys.modules", {"torch": fake_torch}):
            result = configure_torch_memory_fraction(22, device=0)
        self.assertAlmostEqual(calls["fraction"], 22 / 50)
        self.assertEqual(calls["device"], 0)
        self.assertEqual(result["scope"], "PyTorch CUDA caching allocator only")
        self.assertFalse(result["compute_isolation"])
        self.assertFalse(result["hard_gpu_isolation"])

    def test_memory_fraction_rejects_over_device_capacity(self):
        gib = 1024**3
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                current_device=lambda: 0,
                get_device_properties=lambda index: SimpleNamespace(total_memory=20 * gib),
                set_per_process_memory_fraction=lambda fraction, device: self.fail(
                    "must reject before configuring allocator"
                ),
            )
        )
        with patch.dict("sys.modules", {"torch": fake_torch}):
            with self.assertRaisesRegex(ValueError, "exceeds visible device memory"):
                configure_torch_memory_fraction(22, device=0)

    def test_stop_file_requests_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            stop_path = Path(tmp) / "STOP"
            controller = R44StopController(stop_path)
            self.assertFalse(controller.should_stop())
            stop_path.write_text("stop", encoding="utf-8")
            self.assertTrue(controller.should_stop())
            self.assertIn("stop_file", controller.reason)

    def test_signal_handler_only_marks_request_until_safe_poll(self):
        controller = R44StopController("/nonexistent/STOP")
        controller._signal_handler(signal.SIGTERM, None)
        self.assertTrue(controller.stop_requested)
        self.assertEqual(controller.reason, "signal:SIGTERM")
        self.assertTrue(controller.should_stop())

    def test_manual_request_is_idempotent(self):
        controller = R44StopController("/nonexistent/STOP")
        controller.request_stop("manual")
        controller.request_stop("second")
        self.assertTrue(controller.stop_requested)
        self.assertEqual(controller.reason, "manual")

    def test_low_gpu_memory_requires_configured_consecutive_samples(self):
        controller = R44StopController(
            "/nonexistent/STOP", min_free_gpu_gb=24, consecutive_low_memory_samples=2
        )
        gib = 1024**3
        self.assertFalse(controller.observe_gpu_free_memory(23 * gib))
        self.assertFalse(controller.observe_gpu_free_memory(25 * gib))
        self.assertFalse(controller.observe_gpu_free_memory(23 * gib))
        self.assertTrue(controller.observe_gpu_free_memory(22 * gib))
        self.assertIn("below_threshold", controller.reason)

    def test_atomic_save_replaces_only_after_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "last.pt"
            target.write_bytes(b"old-good-checkpoint")

            def failing_writer(stream):
                stream.write(b"partial-new")
                raise RuntimeError("serialization failed")

            with self.assertRaisesRegex(RuntimeError, "serialization failed"):
                atomic_save_checkpoint(target, failing_writer)
            self.assertEqual(target.read_bytes(), b"old-good-checkpoint")
            self.assertEqual(list(Path(tmp).glob(".last.pt.*.tmp")), [])

    def test_atomic_save_commits_new_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "last.pt"
            target.write_bytes(b"old")
            atomic_save_checkpoint(target, lambda stream: stream.write(b"new-good"))
            self.assertEqual(target.read_bytes(), b"new-good")
            self.assertEqual(list(Path(tmp).glob(".last.pt.*.tmp")), [])

    def test_stop_boundary_saves_once_and_returns_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = R44StopController(Path(tmp) / "STOP")
            controller.request_stop("SIGTERM")
            target = Path(tmp) / "last.pt"
            state = {"global_step": 12, "epoch": 1}
            stopped = stop_and_checkpoint_if_requested(
                controller,
                destination=target,
                state_factory=lambda: state,
                writer_factory=lambda obj: lambda stream: stream.write(repr(obj).encode()),
            )
            self.assertTrue(stopped)
            self.assertEqual(target.read_bytes(), repr(state).encode())

    def test_no_stop_does_not_write_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = R44StopController(Path(tmp) / "STOP")
            target = Path(tmp) / "last.pt"
            stopped = stop_and_checkpoint_if_requested(
                controller,
                destination=target,
                state_factory=lambda: self.fail("must not build state"),
                writer_factory=lambda obj: lambda stream: stream.write(b"x"),
            )
            self.assertFalse(stopped)
            self.assertFalse(target.exists())

    def test_cuda_probe_applies_memory_guard(self):
        gib = 1024**3
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                mem_get_info=lambda: (23 * gib, 50 * gib),
            )
        )
        controller = R44StopController(
            "/nonexistent/STOP", min_free_gpu_gb=24, consecutive_low_memory_samples=1
        )
        with patch.dict("sys.modules", {"torch": fake_torch}):
            self.assertTrue(controller.observe_cuda_free_memory())
        self.assertIn("below_threshold", controller.reason)

    def test_cuda_probe_failure_requests_safe_stop(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )
        controller = R44StopController("/nonexistent/STOP", min_free_gpu_gb=24)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            self.assertTrue(controller.observe_cuda_free_memory())
        self.assertEqual(controller.reason, "cuda_unavailable_for_resource_guard")

    def test_invalid_gpu_memory_sample_requests_stop(self):
        controller = R44StopController(
            "/nonexistent/STOP", min_free_gpu_gb=24
        )
        self.assertTrue(controller.observe_gpu_free_memory(-1))
        self.assertEqual(controller.reason, "gpu_memory_probe_invalid")


if __name__ == "__main__":
    unittest.main()
