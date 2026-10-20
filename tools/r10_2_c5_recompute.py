#!/usr/bin/env python3
"""R10.2 C5: independent CPU recomputation of the archived KV/full logits.

Reads only the .npz sidecar and the JSON, recomputes every reported number from the logits,
and fails if they disagree. This is the "independent recompute" half of the KV gate.
"""
import hashlib
import json
import os
import sys

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    run_dir = sys.argv[1]
    out_path = sys.argv[2]
    js = json.load(open(os.path.join(run_dir, "r8_kv_diagnose.json")))
    npz_path = js["logits_sidecar"]["path"]
    checks = {}

    checks["sidecar_sha256_matches_metadata"] = {
        "pass": sha256(npz_path) == js["logits_sidecar"]["sha256"],
        "file": sha256(npz_path), "json": js["logits_sidecar"]["sha256"]}

    z = np.load(npz_path)
    kv, fl = z["logits_kv"], z["logits_full"]
    checks["shape_dtype"] = {"pass": kv.shape == (14, 256) and fl.shape == (14, 256)
                             and kv.dtype == np.float32 and fl.dtype == np.float32,
                             "kv_shape": list(kv.shape), "kv_dtype": str(kv.dtype)}
    checks["finite"] = {"pass": bool(np.isfinite(kv).all() and np.isfinite(fl).all())}
    checks["columns_256"] = {"pass": int(z["column_index"].shape[0]) == 256,
                             "first": int(z["column_index"][0]), "last": int(z["column_index"][-1])}

    tok_kv = np.argmax(kv, axis=1)
    tok_full = np.argmax(fl, axis=1)
    json_kv = [s["token_kv"] for s in js["steps"]]
    json_full = [s["token_full"] for s in js["steps"]]
    checks["tokens_recomputed"] = {"pass": list(map(int, tok_kv)) == json_kv
                                   and list(map(int, tok_full)) == json_full,
                                   "kv": list(map(int, tok_kv)), "full": list(map(int, tok_full))}
    match = tok_kv == tok_full
    checks["match_count"] = {"pass": int(match.sum()) == int(js["n_match"]),
                             "recomputed": int(match.sum()), "reported": int(js["n_match"])}
    d = np.abs(kv - fl)
    mx = d.max(axis=1)
    mn = d.mean(axis=1)
    rep_mx = np.array([s["max_abs_dlogit"] for s in js["steps"]])
    rep_mn = np.array([s["mean_abs_dlogit"] for s in js["steps"]])
    checks["delta_matches"] = {"pass": bool(np.allclose(mx, rep_mx, atol=1e-6)
                                           and np.allclose(mn, rep_mn, atol=1e-6)),
                               "max_abs_diff_of_max": float(np.abs(mx - rep_mx).max())}
    srt = np.sort(fl, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    rep_margin = np.array([s["top1_minus_top2_full"] for s in js["steps"]])
    checks["margin_matches"] = {"pass": bool(np.allclose(margin, rep_margin, atol=1e-6)),
                                "max_abs_diff": float(np.abs(margin - rep_margin).max())}
    checks["first_mismatch"] = {"pass": (None if match.all() else int(np.argmin(match)))
                                == js["first_mismatch_step"],
                                "reported": js["first_mismatch_step"]}
    checks["kv_equivalent_flag"] = {"pass": bool(match.all()) == bool(js["KV_EQUIVALENT"])}
    token_range_ok = all(31744 <= int(js["steps"][k]["token_full"]) + 31744 <= 31999
                         for k in range(14))
    checks["token_ids_inside_window"] = {"pass": bool(token_range_ok),
                                         "min": int(min(tok_full)), "max": int(max(tok_full))}

    res = {"run_dir": os.path.abspath(run_dir), "npz_sha256": sha256(npz_path),
           "n_steps": int(kv.shape[0]), "recomputed_n_match": int(match.sum()),
           "recomputed_median_max_dlogit": float(np.median(mx)),
           "recomputed_median_margin": float(np.median(margin)),
           "checks": checks, "pass": all(c["pass"] for c in checks.values())}
    json.dump(res, open(out_path, "w"), indent=1)
    for k, v in checks.items():
        print("[%s] %-32s %s" % ("PASS" if v["pass"] else "FAIL", k,
                                 json.dumps({kk: vv for kk, vv in v.items() if kk != "pass"})[:150]))
    print("C5 independent recompute:", "PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
