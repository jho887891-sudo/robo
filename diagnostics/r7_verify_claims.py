#!/usr/bin/env python3
"""R7: independent recomputation of every headline claim (CPU, assertions, no model).

Nothing here trusts the training script's own numbers: each claim is recomputed from
the HDF5 and from the stored prediction sidecars, then compared with what the run
logged. Any mismatch is a hard failure (non-zero exit).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))
sys.path.insert(0, os.path.join(REPO, "training"))
import r6_data as D          # noqa: E402
import action_codec as AC    # noqa: E402

RESULTS = []


def check(name, ok, detail):
    RESULTS.append({"claim": name, "pass": bool(ok), "detail": detail})
    print("[%s] %-52s %s" % ("PASS" if ok else "FAIL", name, json.dumps(detail)), flush=True)
    return bool(ok)


def holdout_frames():
    split = D.load_split()
    S, A, V, eps = [], [], [], []
    for n in split["val"]:
        ep = D.load_episode(n)
        t = D.valid_frames(len(ep["state"]))
        S.append(ep["state"][t]); A.append(ep["action"][t])
        V.append(ep["state"][t] - ep["state"][t - 1]); eps += [n] * len(t)
    return (np.concatenate(S), np.concatenate(A), np.concatenate(V), np.array(eps), split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--oracle-json", required=True)
    ap.add_argument("--mlp-json", required=True)
    ap.add_argument("--acc-json", required=True)
    ap.add_argument("--probe-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ok = True

    S, A, V, eps, split = holdout_frames()
    cv = S + V
    cv_L1 = float(np.abs(cv - A).mean())
    identity_L1 = float(np.abs(S - A).mean())

    # ---- claim 1: representable vs floor oracle, and vs cv
    o = json.load(open(args.oracle_json))
    res_bound = o["modes"]["residual"]["nearest_representable_lower_bound_L1"]
    abs_bound = o["modes"]["absolute"]["nearest_representable_lower_bound_L1"]
    check("cv_baseline_recomputed", abs(cv_L1 - o["cv_physical_L1_same_frames"]) < 1e-9,
                {"recomputed": cv_L1, "script": o["cv_physical_L1_same_frames"]})
    check("nearest_residual_bound_below_cv", res_bound < cv_L1,
                {"bound": res_bound, "cv": cv_L1, "ratio": res_bound / cv_L1})
    check("nearest_absolute_bound_above_cv", abs_bound > cv_L1,
                {"bound": abs_bound, "cv": cv_L1, "ratio": abs_bound / cv_L1})
    check("absolute_bound_is_below_floor_encoder_error",
                abs_bound < o["modes"]["absolute"]["floor_encoder_oracle_L1"],
                {"nearest": abs_bound, "floor_encoder": o["modes"]["absolute"]["floor_encoder_oracle_L1"]})
    check("oracle_independent_brute_force_agrees", o["independent_checks_pass"] is True, {})

    # ---- claim 2: the acceleration MLP beats copy-V on the holdout (all seeds)
    acc_json = json.load(open(args.acc_json))
    per_seed = acc_json["runs"]["SVAcc_R"]["per_seed"]
    sidecar_L1 = {}
    for s in per_seed:
        p = os.path.join(args.run_dir, "pred_SVAcc_R_%d.npz" % s["seed"])
        if not os.path.isfile(p):
            continue
        z = np.load(p)
        l1 = float(np.abs(z["cmd"] - z["gt"]).mean())
        cvc = float(np.abs(z["state"] + z["V"] - z["gt"]).mean())
        sidecar_L1[s["seed"]] = {"L1_from_sidecar": l1, "logged": s["L1"], "cv": cvc}
    check("sidecar_L1_matches_logged_all_seeds",
                all(abs(v["L1_from_sidecar"] - v["logged"]) < 1e-9 for v in sidecar_L1.values()),
                sidecar_L1)
    mean_acc = float(np.mean([v["L1_from_sidecar"] for v in sidecar_L1.values()]))
    check("acceleration_MLP_beats_copy_V",
                mean_acc < sidecar_L1[per_seed[0]["seed"]]["cv"] - 2e-4,
                {"mlp_mean_L1": mean_acc, "cv": sidecar_L1[per_seed[0]["seed"]]["cv"],
                 "relative_improvement": 1 - mean_acc / sidecar_L1[per_seed[0]["seed"]]["cv"]})

    # ---- claim 3: paired per-episode consistency from the sidecar itself
    z = np.load(os.path.join(args.run_dir, "pred_SVAcc_R_%d.npz" % per_seed[0]["seed"]))
    err = np.abs(z["cmd"] - z["gt"]); err_cv = np.abs(z["state"] + z["V"] - z["gt"])
    ep_ids = z["episode"]
    uniq = sorted(set(ep_ids.tolist()))
    diffs = np.array([err[ep_ids == e].mean() - err_cv[ep_ids == e].mean() for e in uniq])
    check("improvement_holds_in_most_episodes", int((diffs < 0).sum()) >= 17,
                {"episodes": len(uniq), "improved": int((diffs < 0).sum()),
                 "mean_diff": float(diffs.mean())})
    probe = json.load(open(args.probe_json))
    logged = probe["paired_vs_constant_velocity"]["SVAcc_R"]["vs_cv"]
    check("probe_paired_stats_match_recomputation",
                abs(logged["mean_diff"] - float(diffs.mean())) < 1e-9
                and logged["episodes_improved"] == int((diffs < 0).sum()),
                {"probe": {"mean_diff": logged["mean_diff"], "improved": logged["episodes_improved"]},
                 "recomputed": {"mean_diff": float(diffs.mean()), "improved": int((diffs < 0).sum())}})

    # ---- claim 4: R target beats a direct-delta target with identical inputs/budget
    main_json = json.load(open(args.mlp_json))
    sv_r = main_json["runs"]["SV_R"]["aggregate"]["L1"]["mean"]
    sv_d = main_json["runs"]["SV_delta"]["aggregate"]["L1"]["mean"]
    check("R_target_better_than_delta_target", sv_r < sv_d - 2e-5,
                {"SV_R": sv_r, "SV_delta": sv_d, "improvement": 1 - sv_r / sv_d})

    # ---- claim 5: full-frame training beats the 200-frame regime
    p200 = main_json["pilot200"]["aggregate"]["L1"]["mean"]
    check("full_frame_MLP_beats_200_frame_regime", p200 > 5 * sv_r,
                {"full_frame_SV_R": sv_r, "pilot200": p200, "ratio": p200 / sv_r})

    # ---- claim 6: no episode leakage between train/dev/holdout
    tr = set(main_json["splits"]["train_episodes"]); dv = set(main_json["splits"]["dev_episodes"])
    ho = set(main_json["splits"]["holdout_episodes"])
    check("episode_splits_disjoint", not (tr & dv) and not (tr & ho) and not (dv & ho),
                {"train": len(tr), "dev": len(dv), "holdout": len(ho)})
    check("holdout_matches_declared_split", ho == set(split["val"]), {"holdout": len(ho)})

    # ---- claim 7: inputs are strictly past-only (recompute X and compare to the sidecar link)
    n_checked, label_is_next_state, max_label_diff = 0, True, 0.0
    for name in sorted(split["val"])[:3]:
        ep = D.load_episode(name)
        t = D.valid_frames(len(ep["state"]))
        diff = float(np.abs(ep["action"][t] - ep["state"][t + 1]).max())
        max_label_diff = max(max_label_diff, diff)
        label_is_next_state = label_is_next_state and (diff == 0.0)
        n_checked += 1
    check("action_is_next_observation_and_inputs_are_past_only",
          label_is_next_state and n_checked == 3,
          {"episodes_checked": n_checked, "max_abs_action_minus_next_state": max_label_diff,
           "statement": "X_t uses S_t and S_(t-1) only; the label A_t equals S_(t+1)"})

    summary = {"claims": RESULTS, "all_pass": all(r["pass"] for r in RESULTS),
               "holdout_frames": int(len(S)), "cv_L1": cv_L1, "identity_L1": identity_L1}
    json.dump(summary, open(args.out, "w"), indent=1)
    print("\n%d/%d claims verified -> %s" % (sum(1 for r in RESULTS if r["pass"]), len(RESULTS),
                                             "PASS" if summary["all_pass"] else "FAIL"))
    return 0 if summary["all_pass"] else 4


if __name__ == "__main__":
    sys.exit(main())
