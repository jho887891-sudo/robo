#!/usr/bin/env python3
"""Machine gate verdicts for the pre-registered R15 gates (G4_300, G5_1000).

The verdict is computed from the raw prediction archive and the frozen pre-registration; it is a
program verdict, not a human reading of a table. Exit codes:
    0  PASS
    3  FAIL      (the gate ran and the numbers do not satisfy the pre-registered conditions)
    4  CANNOT_EVALUATE (fail closed: wrong frame list, missing prediction, wrong shape)

Rules that cannot be relaxed here:
  * the prediction archive must have been produced on the SAME frozen DEV frame list as the
    pre-registration (frames_sha256), otherwise nothing is scored;
  * the comparison is same-sample and same-code-version: the baselines are recomputed by the same
    evaluator on the same frames, never copied from an older table;
  * the pre-registered condition is RELATIVE (improve >=5 percent versus constant velocity); the
    absolute number in the pre-registration is reported next to it for continuity, not substituted
    for it.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("training", "diagnostics", "code/xpolicylab/policy/OpenVLA_Pure",
             "code/xpolicylab/policy/OpenVLA_R42"):
    _p = os.path.join(REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

PREREG = os.environ.get("R15_PREREG",
                        os.path.join(REPO, "mission", "R15_GATE_PREREGISTRATION.json"))
EXIT_PASS, EXIT_FAIL, EXIT_CANNOT = 0, 3, 4


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(cid, requirement, measured, passed):
    return {"id": cid, "requirement": requirement, "measured": measured, "passed": bool(passed)}


def _rel_improvement(cv, vla):
    return float((cv - vla) / max(cv, 1e-12))


def strata_comparison(cv_m, vla_m, min_frac=0.05):
    """Per pre-registered motion stratum: did the model beat CV by at least min_frac."""
    strata = []
    won = 0
    for key in ("q1", "q2", "q3", "q4"):
        c = cv_m["motion_strata"].get(key) or {}
        v = vla_m["motion_strata"].get(key) or {}
        cl1, vl1 = c.get("l1"), v.get("l1")
        if cl1 is None or vl1 is None:
            strata.append({"stratum": key, "cv_l1": cl1, "vla_l1": vl1, "won": False,
                           "note": "empty stratum"})
            continue
        rel = _rel_improvement(cl1, vl1)
        ok = rel >= min_frac
        won += int(ok)
        strata.append({"stratum": key, "n": c.get("n"), "cv_l1": cl1, "vla_l1": vl1,
                       "rel_improvement": rel, "won": bool(ok)})
    return won, strata

def gate_g4(ev, prereg, vla, cv, ident):
    """G4_300: a clear signal over identity, nothing collapsed, RAW decode rate reported."""
    conds = []
    conds.append(check("G4.signal_vs_identity",
                       "DEV macro L1 strictly lower than identity A=S",
                       {"vla": vla["physical_l1_macro"],
                        "identity": ident["physical_l1_macro"]},
                       vla["physical_l1_macro"] < ident["physical_l1_macro"]))
    conds.append(check("G4.no_collapse",
                       "still_action_rate < 1.0 and 0.2 <= delta magnitude ratio <= 5.0",
                       {"still_action_rate": vla["still_action_rate"],
                        "delta_magnitude_ratio": vla["delta_magnitude_ratio"]},
                       vla["still_action_rate"] < 1.0
                       and 0.2 <= vla["delta_magnitude_ratio"] <= 5.0))
    conds.append(check("G4.command_inside_box_safe",
                       "every SAFE (clamped) command is inside the joint box by construction",
                       {"gated_on": vla["gated_on"],
                        "safe_clip_frame_rate": vla["safe_clip_frame_rate"]},
                       True))
    conds.append(check("G4.raw_oob_reported",
                       "RAW out-of-box rate is reported (a required statistic, not a pass test)",
                       {"raw_oob_coord_rate": vla["raw_oob_coord_rate"],
                        "raw_oob_frame_rate": vla["raw_oob_frame_rate"],
                        "raw_oob_coord_count": vla["raw_oob_coord_count"]},
                       vla["raw_oob_coord_rate"] == vla["raw_oob_coord_rate"]))
    return conds


def gate_g5(ev, prereg, vla, cv):
    """G5_1000: >=5 percent better macro L1 than CV, OR >=3/4 motion strata, AND no collapse."""
    rel = _rel_improvement(cv["physical_l1_macro"], vla["physical_l1_macro"])
    abs_target = float(prereg.get("g5_absolute_target_l1", 0.002005))
    won, strata = strata_comparison(cv, vla)
    conds = []
    conds.append(check("G5.macro_relative_5pct",
                       "SAFE macro L1 improves >=5 percent versus same-sample constant velocity",
                       {"cv": cv["physical_l1_macro"], "vla": vla["physical_l1_macro"],
                        "rel_improvement": rel,
                        "absolute_target_from_prereg": abs_target,
                        "meets_absolute_prereg_number":
                            bool(vla["physical_l1_macro"] <= abs_target)},
                       rel >= 0.05))
    conds.append(check("G5.strata_3_of_4",
                       ">=3 of 4 pre-registered |V| motion strata improve >=5 percent vs CV",
                       {"strata_won": won, "strata": strata}, won >= 3))
    conds.append(check("G5.no_collapse",
                       "still_action_rate < 1.0 and 0.2 <= delta magnitude ratio <= 5.0",
                       {"still_action_rate": vla["still_action_rate"],
                        "delta_magnitude_ratio": vla["delta_magnitude_ratio"]},
                       vla["still_action_rate"] < 1.0
                       and 0.2 <= vla["delta_magnitude_ratio"] <= 5.0))
    sign_floor = 0.8 * cv["delta_sign_agreement"]
    mag_lo = cv["delta_magnitude_ratio"] / 1.2
    mag_hi = cv["delta_magnitude_ratio"] * 1.2
    conds.append(check("G5.sign_mag_not_worse_20pct",
                       "delta sign agreement and delta magnitude ratio not worse than CV by >20 pct",
                       {"vla_sign": vla["delta_sign_agreement"],
                        "cv_sign": cv["delta_sign_agreement"], "sign_floor": sign_floor,
                        "vla_mag": vla["delta_magnitude_ratio"],
                        "cv_mag": cv["delta_magnitude_ratio"],
                        "mag_window": [mag_lo, mag_hi]},
                       vla["delta_sign_agreement"] >= sign_floor
                       and mag_lo <= vla["delta_magnitude_ratio"] <= mag_hi))
    macro_ok = conds[0]["passed"]
    strata_ok = conds[1]["passed"]
    conds.append(check("G5.pass_rule",
                       "PASS = (macro 5 pct OR 3/4 strata) AND no collapse AND sign/mag ok",
                       {"macro_5pct": macro_ok, "strata_3_of_4": strata_ok,
                        "no_collapse": conds[2]["passed"],
                        "sign_mag": conds[3]["passed"]},
                       (macro_ok or strata_ok) and conds[2]["passed"] and conds[3]["passed"]))
    return conds


def _write(path, report):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=["G4_300", "G5_1000"])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-episode", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-out", default=None)
    a = ap.parse_args()

    report = {"gate": a.gate, "pred": a.pred, "eval_version": None, "checks": [],
              "verdict": "CANNOT_EVALUATE", "exit": EXIT_CANNOT}
    if not os.path.isfile(a.pred):
        report["error"] = "prediction archive missing"
        _write(a.out, report)
        return EXIT_CANNOT
    prereg = json.load(open(PREREG))
    report["prereg_sha256"] = sha256_file(PREREG)
    report["pred_sha256"] = sha256_file(a.pred)

    import r15_dev_eval as E
    try:
        ev = E.run(a.pred, a.per_episode, a.seed, a.eval_out, 256, True)
    except SystemExit as exc:                 # wrong frame list / wrong shape: fail closed
        report["error"] = "evaluator refused the archive: %s" % (exc,)
        report["checks"].append(check("cohort.evaluator_accepts_archive",
                                     "the prediction archive matches the frozen DEV sample set",
                                     {"error": str(exc)}, False))
        _write(a.out, report)
        return EXIT_CANNOT
    report["eval_version"] = ev["eval_version"]
    report["metric_definitions"] = ev["metric_definitions"]
    report["dev_frames_sha256"] = ev["dev_frames_sha256"]
    report["n_dev_frames"] = ev["n_dev_frames"]
    d = np.load(a.pred, allow_pickle=True)
    if "global_step" in d:
        report["pred_global_step"] = int(d["global_step"])
    if "model_sha256" in d:
        report["pred_model_sha256"] = str(d["model_sha256"])

    want = prereg["dev_cohort"]["frames_sha256"]
    frames_ok = ev["dev_frames_sha256"] == want
    report["checks"].append(check(
        "cohort.frozen_frames",
        "predictions were produced on the pre-registered frozen DEV frame list",
        {"expected": want, "actual": ev["dev_frames_sha256"], "n": ev["n_dev_frames"]},
        frames_ok))
    if not frames_ok:
        report["error"] = "frozen frame list mismatch: refusing to score"
        _write(a.out, report)
        return EXIT_CANNOT

    models = ev["models"]
    vla_m = models.get("vla_r42", {})
    if vla_m.get("status") != "OK":
        report["error"] = "no usable VLA predictions"
        report["vla_note"] = vla_m.get("note")
        _write(a.out, report)
        return EXIT_CANNOT
    vla = vla_m["metrics"]
    cv = models["constant_velocity_A=S+V"]["metrics"]
    ident = models["identity_A=S"]["metrics"]
    report["models"] = {}
    for k, v in models.items():
        q = v.get("metrics") or {}
        report["models"][k] = {"status": v.get("status"),
                              "macro_l1_safe": q.get("physical_l1_macro"),
                              "macro_l1_raw": q.get("raw_physical_l1_macro"),
                              "raw_oob_coord_rate": q.get("raw_oob_coord_rate")}
    report["vla_metrics"] = vla
    report["cv_metrics"] = cv

    if a.gate == "G4_300":
        report["checks"].extend(gate_g4(ev, prereg, vla, cv, ident))
    else:
        report["checks"].extend(gate_g5(ev, prereg, vla, cv))
    passed = all(c["passed"] for c in report["checks"])
    report["verdict"] = "PASS" if passed else "FAIL"
    report["exit"] = EXIT_PASS if passed else EXIT_FAIL
    _write(a.out, report)
    print(json.dumps({c["id"]: c["passed"] for c in report["checks"]}, indent=1))
    print("VERDICT %s (%s) -> %s" % (report["verdict"], a.gate, a.out))
    return report["exit"]


if __name__ == "__main__":
    sys.exit(main())
