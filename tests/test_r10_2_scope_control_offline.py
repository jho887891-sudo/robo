#!/usr/bin/env python3
"""R10.2 D4/D5 regression: 12-task scope registry + control timing + offline metrics."""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

import r10_2_task_scope as TS  # noqa: E402
import r10_2_control_timing as CT  # noqa: E402
import r10_2_offline_eval as OE  # noqa: E402

RESULTS = []


def ok(name, cond, detail=None):
    RESULTS.append({"check": name, "pass": bool(cond), "detail": detail or {}})
    print("[%s] %-46s %s" % ("PASS" if cond else "FAIL", name,
                             json.dumps(detail or {}, default=str)[:160]))
    return bool(cond)


def raises(fn, exc=Exception):
    try:
        fn()
        return False
    except exc:
        return True


def main():
    reg = TS.build("/home/T7/ojh/robot_actual/RoboDojo",
                   "/home/T7/ojh/robot_actual/data/data/hdf5")
    ok("scope_has_12_tasks", reg["n_scope_tasks"] == 12 and len(reg["tasks"]) == 12)
    names = [r["task_id"] for r in reg["tasks"]]
    ok("scope_uses_spec_task_ids", names == list(TS.TASKS12))
    ok("scope_records_local_presence",
       reg["n_present_locally"] >= 3 and "push_T" in reg["present"],
       {"present": reg["n_present_locally"], "absent": len(reg["absent"])})
    push = [r for r in reg["tasks"] if r["task_id"] == "push_T"][0]
    ok("pushT_has_local_data_and_module",
       push["local_episodes"] == 100 and push["data_status"] == "LOCAL_AVAILABLE"
       and push["in_local_inventory"],
       {"eps": push["local_episodes"], "module": push["module"]})
    ok("unknown_is_kept_not_invented",
       any(r["embodiment_adapter_needed"] == "UNKNOWN" for r in reg["tasks"]),
       {"unknown_fields": True})
    ok("route_raises_for_unregistered_task",
       raises(lambda: TS.route(reg, "not_a_task"), KeyError))
    ok("stats_only_from_train",
       TS.assert_stats_source("TRAIN") is True
       and raises(lambda: TS.assert_stats_source("DEV"), ValueError)
       and raises(lambda: TS.assert_stats_source("FINAL"), ValueError))

    stages = CT.stage_timestamps(1000.0, {"capture": 8.0, "queue": 2.0, "prefill": 25.0,
                                         "decode14": 30.0, "sim_step": 5.0, "hold": 40.0})
    ok("stage_timeline_ordered",
       stages["capture"]["start_ms"] == 1000.0 and stages["total_ms"] == 110.0
       and stages["hold"]["start_ms"] == 1070.0, {"total_ms": stages["total_ms"]})
    ok("missing_stage_rejected",
       raises(lambda: CT.stage_timestamps(0.0, {"capture": 1.0}), ValueError))
    ok("budget_overrun_reported",
       CT.budget_check(110.0)["fits_frame"] is False and CT.budget_check(38.0)["fits_frame"] is True,
       {"overrun_ms": CT.budget_check(110.0)["overrun_ms"]})
    ok("no_latency_claim_without_data",
       CT.summarize([])["status"] == "NOT_MEASURED")
    ok("percentiles_correct",
       abs(CT.percentile(list(range(101)), 0.50) - 50.0) < 1e-9
       and abs(CT.percentile(list(range(101)), 0.95) - 95.0) < 1e-9
       and abs(CT.percentile([5.0], 0.99) - 5.0) < 1e-9)

    gt = np.zeros((4, 14))
    gt[:, 0] = [0.001, -0.002, 0.003, -0.004]
    pred = gt.copy()
    pred[0, 0] = 0.002
    ok("l1_frame_mean_exact", abs(OE.l1_frame(pred, gt) - (0.001 / 4 / 14)) < 1e-12,
       {"l1": OE.l1_frame(pred, gt)})
    ok("episode_macro_exact", abs(OE.episode_macro([0.1, 0.3]) - 0.2) < 1e-12)
    ok("r2_perfect_is_one", abs(OE.correction_r2(gt, gt) - 1.0) < 1e-12)
    ok("r2_bad_is_negative", OE.correction_r2(-gt, gt) < 0.0)
    sa = OE.sign_agreement(np.array([[-0.01, 0.02]]), np.array([[-0.01, 0.02]]))
    ok("sign_agreement_full", sa["agreement"] == 1.0 and sa["n"] == 2, sa)
    ok("per_arm_groups_present",
       set(OE.per_arm_l1(pred, gt)) == {"left", "right", "left_gripper", "right_gripper"})
    ok("motion_strata_nonempty", len(OE.motion_strata(pred * 10, gt * 10)) >= 1)
    ok("clip_rate_exact",
       abs(OE.clip_rate(np.array([[2.0] * 14]), np.full(14, -1.0), np.full(14, 1.0)) - 1.0) < 1e-12)

    n = sum(1 for r in RESULTS if r["pass"])
    print("\n%d/%d checks passed" % (n, len(RESULTS)))
    out = os.path.join(REPO, "evidence", "r10_2", "test_r10_2_scope_control_offline.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"checks": RESULTS, "n_pass": n, "n_total": len(RESULTS)}, open(out, "w"), indent=1)
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
