#!/usr/bin/env python3
"""Regression for R9 Gate D: eval protocol, manifest tooling, survey and the BLOCKED state."""
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

import r8_eval_manifest as M  # noqa: E402

RESULTS = []
DATA = "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data"
PROTO = os.path.join(REPO, "protocol", "r9_eval_protocol.json")
R8PROTO = os.path.join(REPO, "protocol", "r8_eval_protocol.json")
SPLIT = os.path.join(REPO, "data", "split80.json")


def ok(name, cond, detail=None):
    RESULTS.append({"check": name, "pass": bool(cond), "detail": detail or {}})
    print("[%s] %-44s %s" % ("PASS" if cond else "FAIL", name,
                             json.dumps(detail or {}, default=str)[:180]))
    return bool(cond)


def main():
    proto = json.load(open(PROTO))
    sha = M.hashlib.sha256(open(R8PROTO, "rb").read()).hexdigest()
    ok("protocol_extends_r8_by_hash", proto["extends"]["r8_protocol_sha256"] == sha,
       {"declared": proto["extends"]["r8_protocol_sha256"][:16], "actual": sha[:16]})
    gates = proto["gates"]
    ok("protocol_has_six_gates",
       all(any(k.startswith("g%d_" % i) for k in gates) for i in range(1, 7))
       and len(gates) == 6,
       {"gates": sorted(gates)})
    ok("protocol_blocks_acquisition",
       proto["status"] == "BLOCKED_DATA_ACQUISITION_AUTHORIZATION"
       and all(o["executed"] is False for o in proto["acquisition_routes"]),
       {"status": proto["status"]})
    ok("protocol_min_final_episodes_is_15", proto["splits"]["final_episodes_min"] == 15,
       proto["splits"])
    ok("protocol_forbids_peeking",
       any("before" in f and "FINAL" in f for f in proto["forbidden"]),
       {"n_forbidden": len(proto["forbidden"])})

    one = sorted(f for f in os.listdir(DATA) if f.endswith(".hdf5"))[0]
    rec1 = M.episode_record(os.path.join(DATA, one))
    rec2 = M.episode_record(os.path.join(DATA, one))
    ok("episode_record_is_deterministic", rec1["content_sha256"] == rec2["content_sha256"] and rec1["length"] > 100,
       {"frames": rec1["length"], "sha": rec1["content_sha256"][:16]})

    with tempfile.TemporaryDirectory() as td:
        for i, name in enumerate(sorted(os.listdir(DATA))[:2]):
            os.symlink(os.path.join(DATA, name), os.path.join(td, "ep_%d.hdf5" % i))
            os.symlink(os.path.join(DATA, name), os.path.join(td, "copy_%d.hdf5" % i))
        man = M.build(td)
        ok("detects_duplicate_content",
           man["n_episodes"] == 4 and man["n_unique_content"] == 2 and man["n_duplicate_content"] == 2,
           {"n": man["n_episodes"], "unique": man["n_unique_content"]})
        dj = M.check_disjoint(man, SPLIT)
        ok("duplicates_fail_disjointness", dj["ok"] is False and dj["n_duplicate_content"] == 2,
           {"overlap": dj["n_overlap_with_used"], "dups": dj["n_duplicate_content"]})
        ok("verify_roundtrip_ok", M.verify(man, td)["ok"] is True)

    man100 = json.load(open(os.path.join(REPO, "evidence", "r9_eval_prep_20261020",
                                          "manifest_existing100.json")))
    ok("existing_100_are_all_used_up",
       man100["n_episodes"] == 100 and man100["n_unique_content"] == 100
       and man100["dataset_sha256"] == "6757696397863af8058b296a80f9c50ccfc1ed824c43e8d6f6f122cbd610040a",
       {"dataset_sha256": man100["dataset_sha256"][:16]})
    dj100 = M.check_disjoint(man100, SPLIT)
    ok("disjoint_reports_100_overlap",
       dj100["ok"] is False and dj100["n_overlap_with_used"] == 100,
       {"overlap": dj100["n_overlap_with_used"]})

    survey = json.load(open(os.path.join(REPO, "evidence", "r9_eval_prep_20261020",
                                         "r9_data_sources.json")))
    verdicts = {c["id"]: c.get("verdict") for c in survey["candidates"]}
    ok("survey_verdicts",
       verdicts.get("local_push_T_100") == "USABLE_PUSH_T"
       and verdicts.get("robodojo_demo_arx_x5") == "DIFFERENT_TASK"
       and verdicts.get("openvla_offline_bundle_stage63") == "PLACEHOLDER_FIXTURES",
       verdicts)
    ok("survey_budget_positive_and_not_executed",
       survey["bytes_per_episode_estimate"] > 0
       and all(o["executed"] is False for o in survey["acquisition_options"])
       and survey["blocker"] == "BLOCKED_DATA_ACQUISITION_AUTHORIZATION",
       {"gb_per_episode": survey["gb_per_episode_estimate"]})

    n = sum(1 for r in RESULTS if r["pass"])
    print("\n%d/%d checks passed" % (n, len(RESULTS)))
    out = os.path.join(REPO, "evidence", "r9_eval_prep_20261020", "test_r9_eval_prep.json")
    if os.path.isdir(os.path.dirname(out)):
        with open(out, "w") as fh:
            json.dump({"checks": RESULTS, "n_pass": n, "n_total": len(RESULTS)}, fh, indent=1)
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
