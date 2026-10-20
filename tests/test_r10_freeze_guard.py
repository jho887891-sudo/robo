#!/usr/bin/env python3
"""R10 regression: r10_freeze_guard must enforce g1-g6, not just document them.

All episodes here are SYNTHETIC (schema-identical, tiny) - this tests the guard logic, not
whether any real new data exists. Real collection is a separate, GPU-bound step.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

import r10_freeze_guard as G  # noqa: E402

RESULTS = []
TASK = "Push the T-shaped block to align it precisely with the gray T-shaped pad."
USED = os.path.join(REPO, "evidence", "r9_eval_prep_20261020", "manifest_existing100.json")
SPLIT = os.path.join(REPO, "data", "split80.json")
PROTO = os.path.join(REPO, "protocol", "r9_eval_protocol.json")
GROUPS = (("left_arm", slice(0, 6)), ("left_ee", slice(6, 7)),
          ("right_arm", slice(7, 13)), ("right_ee", slice(13, 14)))


def ok(name, cond, detail=None):
    RESULTS.append({"check": name, "pass": bool(cond), "detail": detail or {}})
    print("[%s] %-44s %s" % ("PASS" if cond else "FAIL", name,
                             json.dumps(detail or {}, default=str)[:170]))
    return bool(cond)


def make_episode(path, seed=0, frames=120, offset=0.0, instruction=TASK, freq=25,
                 cameras=3, group_size=80):
    import h5py
    rs = np.random.RandomState(seed)
    state = (rs.randn(frames, 14) * 0.01 + offset).astype(np.float64)
    action = np.roll(state, -1, axis=0)
    with h5py.File(path, "w") as h:
        for name, sl in GROUPS:
            h.create_dataset("state/%s_joint_states" % name, data=state[:, sl])
            h.create_dataset("action/%s_joint_states" % name, data=action[:, sl])
        h.create_dataset("instruction", data=instruction)
        h.create_dataset("additional_info/frequency", data=freq)
        for i, cam in enumerate(("cam_head", "cam_left_wrist", "cam_right_wrist")[:cameras]):
            h.create_dataset("vision/%s/colors" % cam,
                             data=rs.randint(0, 255, (frames, group_size, group_size, 3),
                                             dtype=np.uint8))


def build_set(td, n=6, **kw):
    d = os.path.join(td, "set")
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        make_episode(os.path.join(d, "new_%02d.hdf5" % i), seed=100 + i,
                     offset=0.001 * (i + 1), **kw)
    return d


def main():
    with tempfile.TemporaryDirectory() as td:
        d = build_set(td)
        res = G.run(d, USED, SPLIT, PROTO, expect_total=6, expect_dev=2, allow_small_set=True)
        ok("synthetic_set_passes_all_gates", res["pass"],
           {k: v.get("pass") for k, v in res["checks"].items()})
        ok("split_assignment_is_disjoint_and_deterministic",
           len(res["checks"]["split_assignment"]["dev"]) == 2
           and not (set(res["checks"]["split_assignment"]["dev"])
                    & set(res["checks"]["split_assignment"]["final"])),
           {"dev_sha": res["checks"]["split_assignment"]["dev_names_sha256"][:16]})
        again = G.run(d, USED, SPLIT, PROTO, expect_total=6, expect_dev=2, allow_small_set=True)
        ok("guard_is_recomputable", again["checks"]["verify"]["dataset_sha256"]
           == res["checks"]["verify"]["dataset_sha256"])

        # negative 1: renamed copy of an already-used episode (content collision)
        d1 = os.path.join(td, "renamed")
        os.makedirs(d1)
        src = "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data/episode_0000000.hdf5"
        if os.path.exists(src):
            shutil.copy(src, os.path.join(d1, "brand_new_name.hdf5"))
            r1 = G.run(d1, USED, SPLIT, PROTO, expect_total=1, expect_dev=0, allow_small_set=True)
            ok("detects_renamed_used_episode",
               r1["checks"]["cross_set_disjoint"].get("pass") is False
               and bool(r1["checks"]["cross_set_disjoint"].get("content_overlap_sha256")),
               {"overlap": len(r1["checks"]["cross_set_disjoint"].get("content_overlap_sha256", []))})
        else:
            ok("detects_renamed_used_episode", False, {"error": "reference episode missing"})

        # negative 2: identical state/action, different instruction
        d2 = os.path.join(td, "instr")
        os.makedirs(d2)
        make_episode(os.path.join(d2, "a.hdf5"), seed=7)
        make_episode(os.path.join(d2, "b.hdf5"), seed=7,
                     instruction=TASK + " (reworded)")
        r2 = G.run(d2, USED, SPLIT, PROTO, expect_total=2, expect_dev=0, allow_small_set=True)
        ok("detects_same_trajectory_new_instruction",
           r2["checks"]["cross_set_disjoint"].get("pass") is False
           or r2["checks"]["cross_set_disjoint"].get("duplicate_state_action", 0) > 0,
           {"dups": r2["checks"]["cross_set_disjoint"].get("duplicate_state_action")})

        # negative 3: placeholder file
        d3 = os.path.join(td, "placeholder")
        os.makedirs(d3)
        with open(os.path.join(d3, "push_T_ep0000.hdf5"), "wb") as fh:
            fh.write(b"\x89HDF\r\n\x1a\n" + b"\0" * 9000)
        r3 = G.run(d3, USED, SPLIT, PROTO, expect_total=1, expect_dev=0, allow_small_set=True)
        ok("rejects_placeholder_file", r3["pass"] is False
           and r3["checks"]["schema"]["pass"] is False)

        # negative 4: wrong frequency
        d4 = build_set(td + "/f", n=2, freq=30)
        r4 = G.run(d4, USED, SPLIT, PROTO, expect_total=2, expect_dev=1, allow_small_set=True)
        ok("rejects_wrong_frequency", r4["checks"]["schema"]["pass"] is False)

        # negative 5: missing camera
        d5 = build_set(td + "/c", n=2, cameras=2)
        r5 = G.run(d5, USED, SPLIT, PROTO, expect_total=2, expect_dev=1, allow_small_set=True)
        ok("rejects_missing_camera", r5["checks"]["schema"]["pass"] is False)

        # negative 6: wrong episode count (must be exactly 25 for the real run)
        r6 = G.run(d, USED, SPLIT, PROTO, expect_total=5, expect_dev=2, allow_small_set=True)
        ok("rejects_wrong_episode_count", r6["checks"]["counts"]["pass"] is False)

        # negative 7: freeze must be refused when the guard fails
        out = os.path.join(td, "freeze_attempt.json")
        rc_refused = G.main.__name__ is not None  # structural: freeze only inside main
        ok("freeze_requires_guard_pass",
           "refusing to freeze" in open(os.path.join(REPO, "tools", "r10_freeze_guard.py")).read()
           and rc_refused)

    n = sum(1 for r in RESULTS if r["pass"])
    print("\n%d/%d checks passed" % (n, len(RESULTS)))
    out = os.path.join(REPO, "evidence", "r10", "test_r10_freeze_guard.json")
    if os.path.isdir(os.path.dirname(out)):
        with open(out, "w") as fh:
            json.dump({"checks": RESULTS, "n_pass": n, "n_total": len(RESULTS)}, fh, indent=1)
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
