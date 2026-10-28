#!/usr/bin/env python3
"""R45 CPU-only regression gate. No model, CUDA context, GPU or server HDF5 needed."""
import ast
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def tree(relative):
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def find_func(parsed, name):
    return next(x for x in ast.walk(parsed)
                if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == name)


class TestFormalTrainerWiring(unittest.TestCase):
    def test_dev_reader_explicitly_bound_at_callsite(self):
        parsed = tree("training/r44_train.py")
        ev = find_func(parsed, "evaluate_dev")
        self.assertIn("r45_reader", [x.arg for x in ev.args.kwonlyargs])
        main = find_func(parsed, "main")
        calls = [n for n in ast.walk(main)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "evaluate_dev"]
        self.assertEqual(len(calls), 1)
        self.assertIn("r45_reader", [k.arg for k in calls[0].keywords])

    def test_train_reader_result_and_full_sample_timer(self):
        parsed = tree("training/r44_train.py")
        main = find_func(parsed, "main")
        assignments = [n for n in ast.walk(main) if isinstance(n, ast.Assign)]
        sample = [a for a in assignments if any(isinstance(t, ast.Name)
                  and t.id == "sample_start" for t in a.targets)]
        data = [a for a in assignments if any(isinstance(t, ast.Tuple)
                and [z.id for z in t.elts if isinstance(z, ast.Name)]
                == ["pv", "data_timing"] for t in a.targets)]
        self.assertEqual(len(sample), 1)
        self.assertEqual(len(data), 1)
        self.assertLess(sample[0].lineno, data[0].lineno)
        self.assertIsInstance(data[0].value, ast.Call)
        self.assertEqual(data[0].value.func.attr, "load_images")

    def test_fake_cpu_trainer_only_and_safe_stop_cursor(self):
        parsed = tree("training/r45_train.py")
        func = find_func(parsed, "build_model")
        self.assertTrue(any(isinstance(x, ast.Raise) for x in ast.walk(func)))
        source = (ROOT / "training/r45_train.py").read_text(encoding="utf-8")
        self.assertIn('epoch, committed_cursor, global_step, order, entries,', source)
        self.assertIn('if j + len(batch) >= group_end:', source)
        self.assertIn('normalizer_sha256', source)


class TestBenchmarkRefusal(unittest.TestCase):
    def test_refuses_both_unauthorized_and_placeholder_mode(self):
        script = ROOT / "diagnostics/r45_gpu_bench.py"
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "bench")
            base = [sys.executable, str(script), "--out", out]
            blocked = subprocess.run(base, capture_output=True, text=True, timeout=15)
            self.assertEqual(blocked.returncode, 2)
            self.assertFalse(Path(out).exists())
            not_ready = subprocess.run(base + [
                "--approve-hours", "0.5",
                "--configs", '[{"micro_batch":1,"name":"R45"}]'
            ], capture_output=True, text=True, timeout=15)
            self.assertEqual(not_ready.returncode, 3)
            self.assertIn("BLOCKED_NOT_IMPLEMENTED", not_ready.stderr)
            self.assertNotIn("samples_per_s", not_ready.stdout)
            self.assertFalse(Path(out).exists())


class TestRuntimeReader(unittest.TestCase):
    def test_hdf5_handles_evicted_and_closed(self):
        from training.r45_data import RuntimeReader
        opened = []

        class FakeHandle:
            def __init__(self, name):
                self.name = name
                self.closed = False
            def close(self):
                self.closed = True

        def open_file(name, mode):
            self.assertEqual(mode, "r")
            h = FakeHandle(name)
            opened.append(h)
            return h

        with patch.dict(sys.modules, {"h5py": types.SimpleNamespace(File=open_file)}):
            reader = RuntimeReader(cache_bytes=0, max_open_handles=2)
            reader._handle("ep1")
            reader._handle("ep2")
            reader._handle("ep1")  # refresh LRU
            reader._handle("ep3")
            self.assertTrue(opened[1].closed)   # ep2 evicted
            self.assertFalse(opened[0].closed)
            self.assertEqual(len(reader._handles), 2)
            reader.close()
            self.assertTrue(all(x.closed for x in opened))

    def test_image_cache_hit_has_zero_read_and_decode_time(self):
        import numpy as np
        from training.r45_data import RuntimeReader
        counters = {"jpeg": 0, "opens": 0}

        class Tile:
            def numel(self):
                return 16
            def element_size(self):
                return 4
            def unsqueeze(self, _):
                return self
            def to(self, *_args, **_kw):
                return self

        class FakeFile:
            def __getitem__(self, key):
                self.key = key
                return self
            def __getitem__(self, key):
                if isinstance(key, str):
                    self.key = key
                    return self
                return b"JPEG"
            def close(self):
                pass

        def fake_open(*_args):
            counters["opens"] += 1
            return FakeFile()

        def fake_decode(*_args):
            counters["jpeg"] += 1
            return np.ones((2, 2, 3), dtype=np.uint8)

        fake_torch = types.SimpleNamespace(
            bfloat16="bfloat16",
            cat=lambda _tiles, dim=0: Tile(),
        )
        fake_cv2 = types.SimpleNamespace(IMREAD_COLOR=1, imdecode=fake_decode)
        fake_pil = types.SimpleNamespace(Image=types.SimpleNamespace(
            fromarray=lambda arr: arr,
        ))
        class Processor:
            image_processor = types.SimpleNamespace(apply_transform=lambda _: Tile())

        reader = RuntimeReader(cache_bytes=1024, max_open_handles=2)
        with patch.dict(sys.modules, {
            "torch": fake_torch, "cv2": fake_cv2, "PIL": fake_pil,
            "h5py": types.SimpleNamespace(File=fake_open),
        }):
            _, a = reader.load_images(Processor(), "ep1", 2, 3,
                                     ("cam_head", "cam_left_wrist", "cam_right_wrist"),
                                     device="cpu")
            _, b = reader.load_images(Processor(), "ep1", 2, 3,
                                     ("cam_head", "cam_left_wrist", "cam_right_wrist"),
                                     device="cpu")
        self.assertFalse(a["cache_hit"])
        self.assertTrue(b["cache_hit"])
        self.assertEqual(counters["jpeg"], 3)
        self.assertEqual(counters["opens"], 1)
        self.assertEqual(b["hdf5_read_s"], 0)
        self.assertEqual(b["jpeg_preprocess_s"], 0)
        self.assertEqual(b["cpu_to_gpu_s"], 0.0)\n        self.assertEqual(b["transfer_device"], "cpu")\n        self.assertGreaterEqual(b["cpu_tensor_preparation_s"], 0.0)
        reader.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
