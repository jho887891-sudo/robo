#!/usr/bin/env python3
"""Regression for the first M2 launch failure: evidence_skeleton looked up prompts["prompts"] on a
plain {task: instruction} mapping and died with KeyError at the first save point, after the 7B had
already been loaded."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _s in ("training", "code/xpolicylab/policy/OpenVLA_Pure",
           "code/xpolicylab/policy/OpenVLA_R42"):
    sys.path.insert(0, os.path.join(REPO, _s))
import m2_train_shared as T  # noqa: E402

NORM = os.path.join(REPO, "data", "m2_task_normalizer_2task.json")
SPLITS = os.path.join(REPO, "data", "m2_splits.json")
pytestmark = pytest.mark.skipif(not (os.path.isfile(NORM) and os.path.isfile(SPLITS)),
                                reason="M2 normalizer/splits artefacts not present")


def _args(tmp_path):
    """A permissive args stand-in: the skeleton reads a handful of fields, and any field it grows
    later should show up as None instead of exploding this regression."""
    class _A:
        out = str(tmp_path / "run")
        max_tasks_stage = 2
        target_global_step = 2000
        normalizer = NORM
        splits = SPLITS
        include_task_token = False
        bucket = "TWO_TASK_PILOT"

        def __getattr__(self, name):
            return None

    return _A()


def test_plain_prompt_mapping_is_accepted(tmp_path):
    tasks = T.parse_tasks("push_T,stack_bowls")
    payload = json.load(open(NORM))
    prompts = T.prompts_for(tasks)                     # the plain mapping run() passes
    mix = {"weights": {t: 0.5 for t in tasks}, "counts": {t: 1 for t in tasks}, "horizon": 2,
           "schedule_sha256": "0" * 64}
    dev = {t: [("ep_%s.hdf5" % t, 5)] for t in tasks}
    ev = T.evidence_skeleton(_args(tmp_path), tasks, payload, prompts, mix, dev,
                             {"cumulative_cap": 2000}, {"probe": 1})
    assert ev["prompts"] == prompts
    assert sorted(ev["prompt_sha256"]) == sorted(tasks)
    assert all(len(v) == 64 for v in ev["prompt_sha256"].values())
    # the hashes must actually match the strings, not be placeholders
    import hashlib
    for t in tasks:
        assert ev["prompt_sha256"][t] == hashlib.sha256(prompts[t].encode()).hexdigest()


def test_rich_distinctness_dict_is_also_accepted(tmp_path):
    tasks = T.parse_tasks("push_T,stack_bowls")
    payload = json.load(open(NORM))
    rich = T.assert_prompts_distinct(tasks)            # {"tasks","prompts","prompt_sha256"}
    ev = T.evidence_skeleton(_args(tmp_path), tasks, payload, rich,
                             {"weights": {}, "counts": {}, "horizon": 0},
                             {t: [] for t in tasks}, {}, {})
    assert ev["prompts"] == rich["prompts"]
    assert ev["prompt_sha256"] == rich["prompt_sha256"]


def test_the_two_task_prompts_are_really_different():
    tasks = T.parse_tasks("push_T,stack_bowls")
    prompts = T.prompts_for(tasks)
    assert prompts["push_T"] != prompts["stack_bowls"]
    assert "T-shaped" in prompts["push_T"]
    assert "bowls" in prompts["stack_bowls"].lower()
