#!/usr/bin/env python3
"""CPU-only checks for fair absolute/residual experiment comparison."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SRC = Path(__file__).resolve().parents[1] / "training" / "compare_target_runs.py"
spec = importlib.util.spec_from_file_location("compare_target_runs", SRC)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def write_run(root, mode, val_l1=0.018, source="train", val_samples=None):
    root.mkdir(parents=True)
    (root / "_snapshot").mkdir()
    config = {
        "mechanism": "keep", "action_target_mode": mode,
        "steps": 300, "lr": 0.0005,
        "samples": [["train1", 20], ["train2", 40]],
        "val_samples": val_samples if val_samples is not None else [["val1", 20]],
        "sample_source": source,
    }
    def record(name, l1):
        return {"split": name, "n": 2 if name == "train" else 1,
                "gen_L1": l1, "identity_L1": 0.008, "repeat_previous_L1": 0.009,
                "pred_delta": 0.01, "gt_delta": 0.008,
                "per_dim_gen": [l1] * 14,
                "per_dim_identity": [0.008] * 14}
    status = {"result": "DONE", "final_eval": record("train", 0.012),
              "final_val_eval": record("val", val_l1)}
    (root / "_snapshot" / "config.json").write_text(json.dumps(config))
    (root / "status.json").write_text(json.dumps(status))


class ComparisonTests(unittest.TestCase):
    def test_matched_comparison_uses_validation(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "absolute", Path(d) / "residual"
            write_run(a, "absolute", 0.02)
            write_run(b, "residual", 0.01)
            report = C.compare(a, b)
            self.assertTrue(report["val"]["residual_improves"])
            self.assertEqual(report["val"]["residual_L1"], 0.01)
            self.assertTrue(report["train"]["absolute_beats_identity"] is False)

    def test_wrong_mode_refused(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "absolute", Path(d) / "residual"
            write_run(a, "absolute")
            write_run(b, "absolute")
            with self.assertRaisesRegex(ValueError, "wrong experiment"):
                C.compare(a, b)

    def test_mismatched_validation_refused(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "absolute", Path(d) / "residual"
            write_run(a, "absolute")
            write_run(b, "residual", val_samples=[["val2", 30]])
            with self.assertRaisesRegex(ValueError, "unmatched A/B"):
                C.compare(a, b)

    def test_missing_done_refused(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "absolute", Path(d) / "residual"
            write_run(a, "absolute")
            write_run(b, "residual")
            status_path = b / "status.json"
            status = json.loads(status_path.read_text()); status["result"] = "RUNNING"
            status_path.write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError, "missing completed"):
                C.compare(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
