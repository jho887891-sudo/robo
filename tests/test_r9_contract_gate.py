#!/usr/bin/env python
"""Regression for tools/r9_contract_gate.py (R9 Gate A independent verification).

Two purposes:
  1 the committed 64-TRAIN frozen contract must verify PASS on real HDF5;
  2 each check must actually DETECT a corrupted contract, so a green gate cannot be
    produced by a silently broken checker.
"""
import importlib.util
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "tools", "r9_contract_gate.py")
CONTRACT = os.path.join(REPO, "evidence", "r9_gate_20261020", "frozen_contract_64train.json")
RESULTS = []


def load_gate():
    spec = importlib.util.spec_from_file_location("r9_contract_gate", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ok(name, cond, detail=None):
    RESULTS.append({"check": name, "pass": bool(cond), "detail": detail})
    print("[%s] %-52s %s" % ("PASS" if cond else "FAIL", name,
                             json.dumps(detail) if detail is not None else ""))
    return bool(cond)


def tampered(meta, **kw):
    m = json.loads(json.dumps(meta))
    for k, v in kw.items():
        m[k] = v
    return m


def main():
    G = load_gate()
    meta = json.load(open(CONTRACT))
    train, val = G.load_split(os.path.join(REPO, "data", "split80.json"))
    hdf5 = G.DEFAULT["hdf5_dir"]
    bounds = G.DEFAULT["bounds"]
    contracts = os.path.join(REPO, G.DEFAULT["contracts"])
    dev = G.partition(train)[1]

    # ---- 1. the committed artefact must verify
    ok("split_identity", G.check_split(meta, train, val)["pass"])
    q = G.check_quantiles(meta, hdf5, train)
    ok("quantiles_from_64_train", q["pass"], {k: q[k] for k in
                                              ("max_abs_diff_q01", "max_abs_diff_q99",
                                               "n_rows_64", "material_change_vs_80_episode_fit_max_rel")})
    ok("dtype_stability", G.check_dtype_stability(meta, hdf5, train)["pass"])
    sc = G.check_scales(meta, contracts, bounds, hdf5, train)
    ok("scales_match_A", sc["pass"], {"v": sc["max_abs_diff_v_scale"], "a": sc["max_abs_diff_a_scale"]})
    ok("zero_centred", G.check_zero_motion(meta)["pass"])
    cd = G.check_codec(meta, hdf5, val)
    ok("codec_roundtrip", cd["pass"], {k: cd[k] for k in
                                       ("max_abs_error_in_range", "one_bin_step_max",
                                        "distinguishable_motion_colliding_with_zero_bin",
                                        "out_of_range_fraction")})
    lk = G.check_no_future_leak(hdf5, val, meta)
    ok("no_future_leak", lk["pass"])
    ok("provenance", G.check_provenance(meta, contracts)["pass"])

    # ---- 2. corrupted contracts MUST be caught
    leaky = tampered(meta)
    leaky["frozen_from"] = dict(leaky["frozen_from"])
    leaky["frozen_from"]["train_episode_names"] = list(
        leaky["frozen_from"]["train_episode_names"]) + [dev[0]]
    ok("detects_dev_episode_in_fit", not G.check_split(leaky, train, val)["pass"],
       {"injected": dev[0]})

    bad80 = tampered(meta)
    bad80["frozen_from"] = dict(bad80["frozen_from"])
    bad80["frozen_from"]["n_train_episodes_for_quantiles"] = 80
    ok("detects_80_episode_provenance", not G.check_split(bad80, train, val)["pass"])

    wide = tampered(meta, q01=(np.asarray(meta["q01"], np.float64) * 1.15).tolist())
    ok("detects_wrong_quantiles", not G.check_quantiles(wide, hdf5, train)["pass"])

    narrow = tampered(meta, q99=[float(0.0)] * 14)
    narrow["q99"] = (np.asarray(narrow["q01"], np.float64) + 1e-9).tolist()
    cdn = G.check_codec(narrow, hdf5, val)
    ok("detects_degenerate_bins", not cdn["pass"],
       {"max_abs_error_in_range": cdn["max_abs_error_in_range"],
        "one_bin_step_max": cdn["one_bin_step_max"]})

    no_target = tampered(meta)
    no_target.pop("target_representation", None)
    ok("detects_missing_target_mode", not G.check_provenance(no_target, contracts)["pass"])

    wrong_grip = tampered(meta, gripper_policy={"gripper_dims": [6, 12], "model_dims": []})
    ok("detects_wrong_gripper_dims", not G.check_provenance(wrong_grip, contracts)["pass"])

    legacy = tampered(meta, normalization_mode="legacy_position")
    ok("detects_legacy_normalisation", not G.check_scales(legacy, contracts, bounds, hdf5, train)["pass"])

    n_pass = sum(1 for r in RESULTS if r["pass"])
    print("\n%d/%d checks passed" % (n_pass, len(RESULTS)))
    out = os.path.join(REPO, "evidence", "r9_gate_20261020", "test_r9_contract_gate.json")
    if os.path.isdir(os.path.dirname(out)):
        with open(out, "w") as fh:
            json.dump({"checks": RESULTS, "n_pass": n_pass, "n_total": len(RESULTS)}, fh, indent=1)
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
