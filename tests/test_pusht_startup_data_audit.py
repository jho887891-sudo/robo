import ast
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.pusht_startup_data_audit import (
    ALIGN_ATOL,
    STILL_THRESHOLDS_RAD_PER_SAMPLE,
    aggregate_rows,
    analyze_arrays,
)


def make_aligned_episode():
    state = np.zeros((6, 14), dtype=np.float32)
    # A controlled scalar trajectory: second difference is exactly one at t=2,3,4.
    state[:, 0] = [0.0, 0.0, 0.0, 1.0, 3.0, 6.0]
    state[:, 7] = [0.0, 0.0, 0.0, 0.1, 0.3, 0.6]
    action = state.copy()
    action[:-1] = state[1:]
    return state, action


class TestPushTStartupDataAudit(unittest.TestCase):
    def test_run_audit_binds_artifact_loader_to_shared_module(self):
        # This is a wiring regression: run_audit must import the module that owns
        # load_m2_artifacts and call it through that same local alias.
        path = os.path.join(REPO, "tools", "pusht_startup_data_audit.py")
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        run_audit = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_audit"
        )
        imports = {
            (alias.name, alias.asname)
            for node in ast.walk(run_audit)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertIn(("m2_train_shared", "M2T"), imports)
        loader_calls = [
            node.func for node in ast.walk(run_audit)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load_m2_artifacts"
        ]
        self.assertTrue(loader_calls, "run_audit must load the M2 artefact contract")
        self.assertTrue(all(
            isinstance(func.value, ast.Name) and func.value.id == "M2T"
            for func in loader_calls
        ), "load_m2_artifacts must be called through the M2T alias")

    def test_alignment_and_second_difference_contract(self):
        state, action = make_aligned_episode()
        report = analyze_arrays(state, action, start_window=3)
        self.assertEqual(report["action_next_state_alignment"]["compared_rows"], 5)
        self.assertEqual(report["action_next_state_alignment"]["exactly_equal_rows"], 5)
        self.assertEqual(report["action_next_state_alignment"]["rows_over_atol_1e-6"], 0)
        self.assertEqual(report["valid_residual_t_range_inclusive"], [2, 4])
        self.assertEqual(report["valid_residual_rows"], 3)
        # Residual at t is (S[t+1]-S[t]) - (S[t]-S[t-1]) = second difference.
        self.assertTrue(np.allclose(report["all_valid_rows"]["residual_14d"]["mean_signed_14d"][0],
                                    1.0))
        self.assertEqual(report["all_valid_rows"]["residual_arm_rms"]["n"], 3)

    def test_startup_residual_window_source_indices_are_explicit(self):
        state = np.zeros((40, 14), dtype=np.float32)
        action = state.copy()
        action[:-1] = state[1:]
        report = analyze_arrays(state, action, start_window=25)

        startup = report["startup_window"]
        self.assertEqual(startup["first_residual_window_valid_rows"], 25)
        self.assertEqual(startup["first_residual_window_source_t_range_inclusive"], [2, 26])

        # Pooled startup groups use valid_t < start_window, so for a 25-transition
        # window the valid residual rows are t=2..24 (23 rows), not t=2..26.
        valid_t = report["_arrays"]["valid_t"]
        group_startup_t = valid_t[valid_t < 25]
        self.assertEqual(group_startup_t.tolist(), list(range(2, 25)))
        self.assertEqual(len(group_startup_t), 23)

    def test_alignment_mismatch_is_detected_without_rewriting_data(self):
        state, action = make_aligned_episode()
        original = action.copy()
        action[1, 4] += 0.02
        report = analyze_arrays(state, action)
        self.assertEqual(report["action_next_state_alignment"]["rows_over_atol_1e-6"], 1)
        self.assertGreater(report["action_next_state_alignment"]["max_abs_error"], ALIGN_ATOL)
        self.assertTrue(np.array_equal(action[[0, 2, 3, 4, 5]], original[[0, 2, 3, 4, 5]]))

    def test_threshold_sensitivity_is_monotone(self):
        state, action = make_aligned_episode()
        report = analyze_arrays(state, action, start_window=4)
        vals = [report["startup_window"]["still_threshold_sensitivity"]["%g" % t]
                ["all_transitions_fraction_below_or_equal"]
                for t in STILL_THRESHOLDS_RAD_PER_SAMPLE]
        self.assertEqual(vals, sorted(vals))

    def test_empty_motion_is_not_mislabelled_as_task_success(self):
        state = np.zeros((5, 14), dtype=np.float32)
        action = state.copy()
        action[:-1] = state[1:]
        report = analyze_arrays(state, action)
        self.assertEqual(report["startup_window"]["first_motion_transition_by_threshold"]["0.001"],
                         None)
        self.assertEqual(report["all_valid_rows"]["residual_arm_rms"]["q50"], 0.0)

    def test_group_aggregation_uses_only_selected_rows(self):
        state, action = make_aligned_episode()
        report = analyze_arrays(state, action)
        arrays = report["_arrays"]
        rows = [{
            "velocity": arrays["velocity"][:1],
            "action_delta": arrays["action_delta"][:1],
            "residual": arrays["residual"][:1],
            "arm_rms_velocity": arrays["arm_rms_velocity"][:1],
            "arm_rms_delta": arrays["arm_rms_delta"][:1],
            "arm_rms_residual": arrays["arm_rms_residual"][:1],
        }]
        agg = aggregate_rows(rows)
        self.assertEqual(agg["n_rows"], 1)
        self.assertEqual(agg["residual_14d"]["n"], 1)

    def test_rejects_invalid_shape_and_nonfinite_data(self):
        state, action = make_aligned_episode()
        with self.assertRaises(ValueError):
            analyze_arrays(state[:, :13], action[:, :13])
        state[0, 0] = np.nan
        with self.assertRaises(ValueError):
            analyze_arrays(state, action)


if __name__ == "__main__":
    unittest.main()
