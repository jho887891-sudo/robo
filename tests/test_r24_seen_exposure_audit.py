"""CPU, no external data needed: actual seen/unseen classification regression tests."""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "r24_seen_exposure_audit",
    os.path.join(ROOT, "tools", "r24_seen_exposure_audit.py"))
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def fixture():
    return {
        "tasks": ["push_T", "stack_bowls"],
        "frames": {
            "push_T": {
                "train": [{"episode": "ep0.hdf5", "t": 2},
                          {"episode": "ep0.hdf5", "t": 3}],
                "dev": [{"episode": "ep1.hdf5", "t": 2}],
            },
            "stack_bowls": {
                "train": [{"episode": "ep0.hdf5", "t": 2}],
                "dev": [{"episode": "ep1.hdf5", "t": 4}],
            },
        },
    }


def test_train_split_does_not_imply_gradient_exposure():
    m = fixture()
    seen = {A.frame_key("push_T", "ep0.hdf5", 2)}
    result = A.summarise_seen(seen, m)
    assert result["push_T"]["selected_train"] == 2
    assert result["push_T"]["selected_train_seen"] == 1
    assert result["push_T"]["selected_train_unseen"] == 1
    assert result["stack_bowls"]["selected_train_seen"] == 0


def test_qualified_names_and_tasks_are_not_confused():
    m = fixture()
    seen = {A.frame_key("stack_bowls", "stack_bowls/ep0.hdf5", 2)}
    result = A.summarise_seen(seen, m)
    assert result["stack_bowls"]["selected_train_seen"] == 1
    assert result["push_T"]["selected_train_seen"] == 0


def test_dev_gradient_contamination_is_detectable():
    m = fixture()
    seen = {A.frame_key("push_T", "ep1.hdf5", 2)}
    result = A.summarise_seen(seen, m)
    assert result["push_T"]["selected_dev_seen_BAD_IF_NONZERO"] == 1


def test_selected_duplicate_records_fail_closed():
    m = fixture()
    m["frames"]["push_T"]["train"].append({"episode": "ep0.hdf5", "t": 2})
    with pytest.raises(ValueError, match="duplicate"):
        A.summarise_seen(set(), m)


def test_manifest_digest_covers_frozen_flag():
    m = fixture()
    m["frozen_before_model_run"] = True
    a = A.manifest_digest(m)
    m["frozen_before_model_run"] = False
    assert A.manifest_digest(m) != a
