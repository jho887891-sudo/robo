"""R6 Phase 1 / Task A: action<->state temporal alignment on the REAL HDF5 data.

Pure numpy: no torch, no GPU, no model. Reads only state/action (never feeds
S_{t+1} to any predictor) and reports E1/E2/E3 with per-arm, per-dimension,
per-split and motion-stratified breakdowns, plus an explicit anti-leak test.

  E1 = MAE(A_t, S_{t+1})          next-state alignment
  E2 = MAE(A_t, S_t)              hold-current-state error
  E3 = MAE(A_t, 2S_t - S_{t-1})   past-only constant-velocity extrapolation
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import r6_data as D  # noqa: E402

ERRS = ("e1_next_state", "e2_current_state", "e3_extrapolated", "e0_previous_state")
MOTION_BINS = (0.0, 1e-4, 1e-3, 5e-3, np.inf)
MOTION_LABELS = ("static_lt_1e-4", "small_1e-4_1e-3", "medium_1e-3_5e-3", "large_ge_5e-3")


def frame_errors(state, action):
    """Per-frame alignment errors for every legal t (t-1 and t+1 both exist)."""
    S, A = D.validate_arrays(state, action)
    t = D.valid_frames(len(S))
    prev, cur, nxt = S[t - 1], S[t], S[t + 1]
    return {
        "frames": t,
        "e1_next_state": np.abs(A[t] - nxt),
        "e2_current_state": np.abs(A[t] - cur),
        "e3_extrapolated": np.abs(A[t] - (2.0 * cur - prev)),
        "e0_previous_state": np.abs(A[t] - prev),
        "expert_delta": np.abs(A[t] - cur),
        "past_velocity": np.abs(cur - prev),
        "future_velocity": np.abs(nxt - cur),
    }


def shift_scan(state, action, shifts=(-3, -2, -1, 0, 1, 2, 3)):
    """MAE(A_t, S_{t+shift}) for a range of index shifts; the minimum proves the offset."""
    S, A = D.validate_arrays(state, action)
    T = len(S)
    out = {}
    for k in shifts:
        t = np.arange(max(0, -k), min(T, T - k), dtype=np.int64)
        if len(t) == 0:
            continue
        out[str(k)] = float(np.abs(A[t] - S[t + k]).mean())
    return out


def structural_checks(state, action, frequency=None):
    """dt regularity proxy, duplicate frames/states, tail validity."""
    S, A = D.validate_arrays(state, action)
    T = len(S)
    step = np.abs(np.diff(S, axis=0)).mean(axis=1)
    dup_consecutive = int(np.sum(np.all(np.diff(S, axis=0) == 0.0, axis=1)))
    rounded = np.round(S, 9)
    uniq = len({tuple(r) for r in rounded.tolist()})
    return {
        "frames": int(T),
        "frequency": frequency,
        "dt_seconds": (1.0 / frequency) if frequency else None,
        "identical_consecutive_state_frames": dup_consecutive,
        "unique_state_rows": int(uniq),
        "duplicate_state_rows": int(T - uniq),
        "frame_step_mean": float(step.mean()),
        "frame_step_median": float(np.median(step)),
        "frame_step_max": float(step.max()),
        "frame_step_p99": float(np.quantile(step, 0.99)),
        # A row of zeros in the step sequence would mean a stalled/repeated frame.
        "zero_step_frames": int(np.sum(step == 0.0)),
        "last_frame_finite": bool(np.isfinite(S[-1]).all() and np.isfinite(A[-1]).all()),
        "last_frame_abs_delta_mean": float(np.abs(A[-1] - S[-1]).mean()),
        "tail_frames_without_next_state": 1,   # t = T-1 has no S_{t+1}: E1 undefined there
    }


def leak_test(state, action, seed=0, every=5):
    """E3 must never read S_{t+1}: corrupting the next state of a frame must not move its E3.

    S_{t+1} is legal PAST data for later frames, so the check is per-frame: only the
    selected frames' own next state is scrambled, and only their E3 is compared.
    """
    S, A = D.validate_arrays(state, action)
    t = D.valid_frames(len(S))
    e3 = np.abs(A[t] - (2.0 * S[t] - S[t - 1]))
    sel = t[::every]
    rng = np.random.default_rng(seed)
    S2 = S.copy()
    fut = sel + 1
    S2[fut] = S[fut] + rng.normal(0.0, 1.0, size=S[fut].shape)   # garbage next states
    e3_after = np.abs(A[sel] - (2.0 * S2[sel] - S2[sel - 1]))
    # Also prove E1 really does depend on the same future rows (i.e. the test can fail).
    e1 = np.abs(A[sel] - S[fut])
    e1_after = np.abs(A[sel] - S2[fut])
    future_motion = S[t + 1] - S[t]
    expert_delta = A[t] - S[t]
    v, d = future_motion.reshape(-1), expert_delta.reshape(-1)
    corr = None
    if v.std() > 0 and d.std() > 0:
        corr = float(np.corrcoef(v, d)[0, 1])
    return {
        "frames_tested": int(len(sel)),
        "e3_unchanged_when_future_scrambled": bool(np.array_equal(e3[::every], e3_after)),
        "e3_max_abs_diff": float(np.abs(e3[::every] - e3_after).max()),
        "e1_changed_when_future_scrambled": bool((e1_after - e1).max() > 1e-6),
        "corr_future_motion_vs_expert_delta": corr,
        "corr_note": "future motion is NOT a legal input; this only proves the A_t=S_(t+1) alignment",
    }


def _pack(acc, item):
    for key in ERRS + ("expert_delta", "past_velocity", "future_velocity"):
        acc.setdefault(key, []).append(item[key])


def summarize(acc):
    """Aggregate per-frame error matrices into the Task A report blocks."""
    if not acc.get("e1_next_state"):
        raise ValueError("no frames accumulated")
    stack = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    out = {}
    for key in ERRS:
        e = stack[key]
        out[key] = {
            "mean": float(e.mean()),
            "median": float(np.median(e)),
            "p95": float(np.quantile(e, 0.95)),
            "max": float(e.max()),
            "per_arm": {"left": float(e[:, D.LEFT].mean()), "right": float(e[:, D.RIGHT].mean())},
            "per_dim": e.mean(axis=0).tolist(),
            "exact_zero_fraction": float(np.mean(np.all(e == 0.0, axis=1))),
            "max_abs_lt_1e-9_fraction": float(np.mean(e.max(axis=1) < 1e-9)),
            "max_abs_lt_1e-6_fraction": float(np.mean(e.max(axis=1) < 1e-6)),
            "max_abs_lt_1e-4_fraction": float(np.mean(e.max(axis=1) < 1e-4)),
        }
    motion = stack["expert_delta"].mean(axis=1)
    strata = {}
    for lo, hi, label in zip(MOTION_BINS[:-1], MOTION_BINS[1:], MOTION_LABELS):
        m = (motion >= lo) & (motion < hi)
        if not m.any():
            strata[label] = {"n": 0}
            continue
        strata[label] = {"n": int(m.sum()), "fraction": float(m.mean()),
                         **{k: float(stack[k][m].mean()) for k in ERRS}}
    return {"n_frames": int(len(stack["e1_next_state"])), "per_error": out,
            "motion_strata": strata,
            "mean_expert_delta_by_arm": {
                "left": float(stack["expert_delta"][:, D.LEFT].mean()),
                "right": float(stack["expert_delta"][:, D.RIGHT].mean())},
            "mean_past_velocity_by_arm": {
                "left": float(stack["past_velocity"][:, D.LEFT].mean()),
                "right": float(stack["past_velocity"][:, D.RIGHT].mean())}}


def run(split_names=("train", "val")):
    split = D.load_split()
    result = {"dim_names": D.DIM_NAMES, "hdf5_dir": D.HDF5_DIR, "split_path": D.SPLIT_PATH,
              "errors_definition": {"e1": "|A_t - S_(t+1)|", "e2": "|A_t - S_t|",
                                    "e3": "|A_t - (2S_t - S_(t-1))|", "e0": "|A_t - S_(t-1)|"},
              "splits": {}, "episodes": {}, "checks": {}}
    for name in split_names:
        acc = {}
        shifts = {}
        leak = {}
        per_ep = {}
        struct = {}
        for ep in D.load_episodes(split[name]):
            item = frame_errors(ep["state"], ep["action"])
            _pack(acc, item)
            per_ep[ep["episode"]] = {
                "n_frames": int(len(item["frames"])),
                "e1": float(item["e1_next_state"].mean()),
                "e2": float(item["e2_current_state"].mean()),
                "e3": float(item["e3_extrapolated"].mean()),
                "mean_expert_delta": float(item["expert_delta"].mean()),
            }
            for k, v in shift_scan(ep["state"], ep["action"]).items():
                shifts.setdefault(k, []).append(v)
            lk = leak_test(ep["state"], ep["action"])
            leak.setdefault("e3_unchanged", []).append(lk["e3_unchanged_when_future_scrambled"])
            leak.setdefault("e3_diff", []).append(lk["e3_max_abs_diff"])
            leak.setdefault("e1_changed", []).append(lk["e1_changed_when_future_scrambled"])
            leak.setdefault("corr", []).append(lk["corr_future_motion_vs_expert_delta"])
            st = structural_checks(ep["state"], ep["action"], ep["frequency"])
            for k, v in st.items():
                struct.setdefault(k, []).append(v)
        result["splits"][name] = summarize(acc)
        result["splits"][name]["episode_count"] = len(split[name])
        result["splits"][name]["shift_scan_mean_mae"] = {
            k: float(np.mean(v)) for k, v in sorted(shifts.items(), key=lambda kv: int(kv[0]))}
        result["splits"][name]["shift_scan_best_shift"] = min(
            result["splits"][name]["shift_scan_mean_mae"].items(), key=lambda kv: kv[1])[0]
        result["episodes"][name] = per_ep
        corrs = [c for c in leak["corr"] if c is not None]
        result["checks"][name] = {
            "leak_test_all_episodes_pass": bool(all(leak["e3_unchanged"])),
            "leak_test_max_abs_diff": float(max(leak["e3_diff"])),
            "leak_test_can_fail_e1_responds": bool(all(leak["e1_changed"])),
            "corr_future_motion_vs_expert_delta_min": float(min(corrs)) if corrs else None,
            "corr_future_motion_vs_expert_delta_max": float(max(corrs)) if corrs else None,
            "frequencies": sorted({s for s in struct["frequency"] if s is not None}),
            "frames_total": int(sum(struct["frames"])),
            "zero_step_frames_total": int(sum(struct["zero_step_frames"])),
            "identical_consecutive_state_frames_total": int(sum(struct["identical_consecutive_state_frames"])),
            "duplicate_state_rows_total": int(sum(struct["duplicate_state_rows"])),
            "all_last_frames_finite": bool(all(struct["last_frame_finite"])),
            "frame_step_max_over_episodes": float(max(struct["frame_step_max"])),
            "frame_step_median_mean": float(np.mean(struct["frame_step_median"])),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = run()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    for split_name, block in res["splits"].items():
        print("[%s] episodes=%d frames=%d  E1=%.6f E2=%.6f E3=%.6f  best_shift=%s" % (
            split_name, block["episode_count"], block["n_frames"],
            block["per_error"]["e1_next_state"]["mean"], block["per_error"]["e2_current_state"]["mean"],
            block["per_error"]["e3_extrapolated"]["mean"], block["shift_scan_best_shift"]))
        for label, st in block["motion_strata"].items():
            if st["n"]:
                print("   %-18s n=%6d (%.3f)  E1=%.6f E2=%.6f E3=%.6f" % (
                    label, st["n"], st["fraction"], st["e1_next_state"], st["e2_current_state"],
                    st["e3_extrapolated"]))
        print("   checks:", json.dumps(res["checks"][split_name]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
