import copy
import importlib.util
import os

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "tools", "r25_episode_cluster_audit.py")
spec = importlib.util.spec_from_file_location("r25", SRC)
r25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r25)


def fixtures():
    f = {"tasks": ["push_T"], "frames": {"push_T": {"dev": [
        {"episode": "a", "stratum": "still"},
        {"episode": "a", "stratum": "slow"},
        {"episode": "b", "stratum": "still"}]}}}
    current = copy.deepcopy(f)
    current["frozen_before_model_run"] = True
    current["manifest_sha256"] = r25.digest(current, {"manifest_sha256"})
    legacy = copy.deepcopy(current)
    legacy["manifest_sha256"] = r25.digest(legacy, {"manifest_sha256", "frozen_before_model_run"})
    archive = {"frame_manifest_sha256": legacy["manifest_sha256"], "split": "dev",
               "tasks": {"push_T": {"frames": [
                   {"stratum": "still", "model": 2.0, "cv": 1.0},
                   {"stratum": "slow", "model": 2.0, "cv": 1.0},
                   {"stratum": "still", "model": 1.0, "cv": 2.0}]}}}
    return archive, current, legacy


def test_episode_macro_vs_frame_pooled():
    z = r25.analyze(*fixtures(), iterations=1000)["tasks"]["push_T"]
    assert z["n_frames"] == 3 and z["n_independent_episodes"] == 2
    assert z["pooled_frame_mean_model_minus_cv"] == pytest.approx(1/3)
    assert z["equal_episode_macro_model_minus_cv"] == pytest.approx(0)
    assert z["kish_effective_episode_count_under_frame_weights"] == pytest.approx(9/5)


def test_manifest_hash_fails_closed():
    archive, current, legacy = fixtures()
    current["frames"]["push_T"]["dev"][0]["episode"] = "new"
    with pytest.raises(ValueError, match="content hash"):
        r25.analyze(archive, current, legacy)


def test_legacy_hash_mismatch_fails_closed():
    archive, current, legacy = fixtures()
    legacy["manifest_sha256"] = "0"*64
    with pytest.raises(ValueError, match="legacy manifest"):
        r25.analyze(archive, current, legacy)


def test_report_frame_order_mismatch_fails_closed():
    archive, current, legacy = fixtures()
    archive["tasks"]["push_T"]["frames"][1]["stratum"] = "still"
    with pytest.raises(ValueError, match="order mismatch"):
        r25.analyze(archive, current, legacy)


def test_bootstrap_reproducibility():
    assert r25.bootstrap([1.0, -1.0, 2.0], 1000, seed=12) == r25.bootstrap([1.0, -1.0, 2.0], 1000, seed=12)
