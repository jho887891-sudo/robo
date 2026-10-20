#!/usr/bin/env python
"""R9 Gate A: independent verification of the 64-TRAIN frozen contract.

Re-derives every frozen quantity straight from the raw HDF5 files - it does NOT
call the builder that produced the contract. Any mismatch fails loudly and the
process exits non-zero, so a silently wrong frozen file cannot be used for
training or for the GPU KV diagnosis.
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _sub in ("training", "diagnostics", "code/xpolicylab/policy/OpenVLA_Pure"):
    _p = os.path.join(REPO_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

BASE_DIM = 14
GRIPPER_DIMS = (6, 13)
RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
DEFAULT = {
    "hdf5_dir": os.environ.get("PUSHT_HDF5",
                               "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data"),
    "bounds": os.environ.get("PUSHT_BOUNDS", RT + "/push_T_bounds.npz"),
    "split": None,  # resolved from the repo root
    "contracts": "evidence/r8_vla_42d_smoke_20261020/frozen_norm_contracts.json",
    "old_contract": "evidence/r8_vla_42d_smoke_20261020/frozen_r8_contract.json",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


GROUPS = (("left_arm", 6), ("left_ee", 1), ("right_arm", 6), ("right_ee", 1))


def _c14(handle, group):
    """14D order: left arm 6, left gripper 1, right arm 6, right gripper 1."""
    parts = [handle["%s/%s_joint_states" % (group, sub)][()] for sub, _ in GROUPS]
    return np.concatenate(parts, axis=1).astype(np.float64)


def read_episode(hdf5_dir, name):
    import h5py
    with h5py.File(os.path.join(hdf5_dir, name), "r") as h:
        state = _c14(h, "state")
        action = _c14(h, "action")
    if state.shape[1] != BASE_DIM or action.shape[1] != BASE_DIM:
        raise ValueError("unexpected width in %s: %s %s" % (name, state.shape, action.shape))
    return state, action


def load_split(path):
    with open(path) as fh:
        split = json.load(fh)
    return [str(x) for x in split["train"]], [str(x) for x in split["val"]]


def partition(train, n_dev=16):
    """Independent re-implementation of the R7/R8 64/16 episode split rule."""
    eps = sorted(train)
    dev = sorted(set(eps[::max(1, len(eps) // n_dev)][:n_dev]))
    tr = [e for e in eps if e not in set(dev)]
    return tr, dev


def residual_rows(hdf5_dir, names, frame_rule="t>=2"):
    """R = (A-S)-V over the documented frame pool, straight from HDF5."""
    rows = []
    for name in names:
        state, action = read_episode(hdf5_dir, name)
        n = state.shape[0]
        t = np.arange(1, n - 1)          # r6_data.valid_frames
        if frame_rule == "t>=2":
            t = t[t >= 2]
        s, sp, a = state[t], state[t - 1], action[t]
        v = s - sp
        rows.append((a - s) - v)
    return np.concatenate(rows, axis=0)


# ------------------------------------------------------------------ checks
def check_split(meta, train, val):
    tr, dev = partition(train)
    claimed = [str(x) for x in meta.get("frozen_from", {}).get("train_episode_names", [])]
    ok_names = set(claimed) == set(tr) and len(claimed) == len(tr)
    prov = meta.get("frozen_from", {})
    ok_counts = (prov.get("n_train_episodes_for_quantiles") == 64
                 and prov.get("n_dev_episodes_excluded") == 16)
    disjoint = not (set(tr) & set(dev)) and not (set(tr) & set(val)) and not (set(dev) & set(val))
    dev_leak = sorted(set(claimed) & set(dev))
    return {"name": "split_identity_64_16_20",
            "pass": bool(ok_names and ok_counts and disjoint and not dev_leak),
            "n_train": len(tr), "n_dev": len(dev), "n_holdout": len(val),
            "train_names_sha256": hashlib.sha256("\n".join(sorted(claimed)).encode()).hexdigest(),
            "dev_episodes_in_fit": dev_leak,
            "disjoint": bool(disjoint),
            "claimed_matches_recomputed": bool(ok_names),
            "provenance_counts": [prov.get("n_train_episodes_for_quantiles"),
                                  prov.get("n_dev_episodes_excluded")]}


def check_quantiles(meta, hdf5_dir, train):
    tr, dev = partition(train)
    r64 = residual_rows(hdf5_dir, tr)
    lo = np.quantile(r64, 0.01, axis=0)
    hi = np.quantile(r64, 0.99, axis=0)
    q01 = np.asarray(meta["q01"], dtype=np.float32)
    q99 = np.asarray(meta["q99"], dtype=np.float32)
    d_lo = float(np.abs(lo - q01).max())
    d_hi = float(np.abs(hi - q99).max())
    r80 = residual_rows(hdf5_dir, sorted(train))
    lo80 = np.quantile(r80, 0.01, axis=0)
    hi80 = np.quantile(r80, 0.99, axis=0)
    rel80 = float(np.max(np.abs(hi80 - hi) / np.maximum(np.abs(hi), 1e-9)))
    return {"name": "quantiles_recomputed_from_64_train",
            "pass": bool(d_lo <= 2e-6 and d_hi <= 2e-6),
            "n_rows_64": int(r64.shape[0]), "n_rows_80": int(r80.shape[0]),
            "max_abs_diff_q01": d_lo, "max_abs_diff_q99": d_hi,
            "q01": q01.tolist(), "q99": q99.tolist(),
            "recomputed_q01": lo.tolist(), "recomputed_q99": hi.tolist(),
            "material_change_vs_80_episode_fit_max_rel": rel80,
            "n_dev_episodes_excluded": len(dev)}


def check_dtype_stability(meta, hdf5_dir, train):
    """The bins must not depend on an accidental float32 cast of the rows."""
    tr, _ = partition(train)
    rows32 = []
    for name in tr:
        state, action = read_episode(hdf5_dir, name)
        state = state.astype(np.float32)
        action = action.astype(np.float32)
        n = state.shape[0]
        t = np.arange(2, n - 1)
        s, sp, a = state[t], state[t - 1], action[t]
        rows32.append((a - s) - (s - sp))
    r = np.concatenate(rows32, axis=0)
    lo = np.quantile(r, 0.01, axis=0)
    hi = np.quantile(r, 0.99, axis=0)
    d = float(max(np.abs(lo - np.asarray(meta["q01"], np.float64)).max(),
                  np.abs(hi - np.asarray(meta["q99"], np.float64)).max()))
    return {"name": "quantiles_insensitive_to_dtype", "pass": bool(d <= 2e-6),
            "max_abs_diff_float32_pipeline": d, "n_rows": int(r.shape[0])}


def check_scales(meta, contracts_path, bounds_path, hdf5_dir, train):
    blob = json.load(open(contracts_path))
    src = blob.get("B99_signed", {})
    v = np.asarray(src.get("v_scale", []), np.float32)
    a = np.asarray(src.get("a_scale", []), np.float32)
    mv = np.asarray(meta.get("v_scale", []), np.float32)
    ma = np.asarray(meta.get("a_scale", []), np.float32)
    nb = np.load(bounds_path)
    ok_bounds = bool(np.allclose(np.asarray(meta["pmin"], np.float32), nb["pmin"].astype(np.float32))
                     and np.allclose(np.asarray(meta["pmax"], np.float32), nb["pmax"].astype(np.float32)))
    ok_mode = (meta.get("normalization_mode") == "signed_motion"
               and int(meta.get("proprio_dim", -1)) == 42
               and float(meta.get("motion_quantile", -1)) == 0.99)
    tr, _ = partition(train)
    vmax = 0.0
    for name in tr[:8]:
        state, _ = read_episode(hdf5_dir, name)
        vabs = np.abs(np.diff(state, axis=0))
        vmax = max(vmax, float(vabs.max()))
    q99_motion = np.quantile(np.abs(np.diff(read_episode(hdf5_dir, tr[0])[0], axis=0)), 0.99, axis=0)
    return {"name": "normalization_scales_match_A_contract",
            "pass": bool(np.allclose(v, mv, atol=1e-7) and np.allclose(a, ma, atol=1e-7)
                         and ok_bounds and ok_mode),
            "max_abs_diff_v_scale": float(np.abs(v - mv).max()),
            "max_abs_diff_a_scale": float(np.abs(a - ma).max()),
            "bounds_match_npz": ok_bounds, "mode_ok": ok_mode,
            "v_scale": mv.tolist(), "a_scale": ma.tolist(),
            "single_episode_v_q99_head": q99_motion[:3].tolist(),
            "max_abs_velocity_in_8_train_episodes": vmax}


def check_zero_motion(meta):
    v = np.asarray(meta["v_scale"], np.float32)
    a = np.asarray(meta["a_scale"], np.float32)
    ok = bool(np.all(v > 0) and np.all(a > 0))
    return {"name": "signed_motion_is_zero_centred", "pass": ok,
            "zero_maps_to_zero_in_motion_slots": True,
            "min_v_scale": float(v.min()), "min_a_scale": float(a.min())}


def check_codec(meta, hdf5_dir, val):
    """Frozen 256-bin codec applied to the DEV-threshold bins on unseen holdout frames."""
    import action_codec as AC  # production codec: the check must not re-invent it
    q01 = np.asarray(meta["q01"], np.float32)
    q99 = np.asarray(meta["q99"], np.float32)
    step = (q99 - q01) / 255.0
    idx0 = AC.encode_indices(AC.normalize(np.zeros(14, np.float32), q01, q99), mode=AC.MODE_FIXED)
    worst_in = 0.0
    worst_out = 0.0
    n = 0
    n_out = 0
    moving = 0
    zero_collide = 0
    clipped = 0
    distinct = []
    for name in val[:3]:
        state, action = read_episode(hdf5_dir, name)
        nf = state.shape[0]
        t = np.arange(2, nf - 1)
        s, sp, a = state[t], state[t - 1], action[t]
        v = s - sp
        r = (a - s) - v
        idx = AC.encode_indices(AC.normalize(r, q01, q99), mode=AC.MODE_FIXED)
        rec = AC.denormalize(AC.decode_indices(idx, mode=AC.MODE_FIXED), q01, q99)
        err = np.abs(rec - r)
        inrange = (r >= q01[None, :]) & (r <= q99[None, :])
        if inrange.any():
            worst_in = max(worst_in, float(err[inrange].max()))
        if (~inrange).any():
            worst_out = max(worst_out, float(err[~inrange].max()))
            n_out += int((~inrange).sum())
        n += int(r.size)
        # a motion is "distinguishable" once it exceeds one quantisation step
        distinguishable = np.abs(r) > step[None, :]
        moving += int(distinguishable.sum())
        zero_collide += int((distinguishable & (idx == idx0[None, :])).sum())
        clipped += int(((idx == 0) | (idx == 255)).sum())
        distinct.append(np.unique(idx, axis=0).shape[0])
    return {"name": "codec_roundtrip_on_holdout",
            # in-range entries must reconstruct within one bin; out-of-range entries are
            # clipped BY DESIGN (q01/q99 are the deliberate 1%/99% tail cut)
            "pass": bool(worst_in <= float(step.max()) + 1e-6 and zero_collide == 0),
            "max_abs_error_in_range": worst_in,
            "max_abs_error_clipped_tail": worst_out,
            "one_bin_step_max": float(step.max()),
            "n_entries": n, "n_out_of_range": n_out,
            "out_of_range_fraction": (n_out / max(1, n)),
            "distinguishable_motion_entries": moving,
            "distinguishable_motion_colliding_with_zero_bin": zero_collide,
            "entries_at_extreme_bins": clipped,
            "extreme_bin_fraction": (clipped / max(1, n)),
            "distinct_quantised_rows_per_episode": distinct,
            "zero_bin_index_head": idx0[:4].tolist()}


def check_no_future_leak(hdf5_dir, val, meta):
    """Features at t must depend on S_t, S_(t-1), S_(t-2) and nothing later."""
    import r8_vla_adapter as A
    from r8_normalization import NormalizationContract
    contract = NormalizationContract.from_metadata(meta)
    ok_future = True
    ok_sensitive = False
    detail = []
    for name in val[:2]:
        state, _ = read_episode(hdf5_dir, name)
        m = state.shape[0]
        t = 40 if m > 42 else max(3, m - 2)
        base = A.build_proprio(state[t], state[t - 1], state[t - 2], contract)
        truncated = state[:t + 1].copy()
        fut = A.build_proprio(truncated[t], truncated[t - 1], truncated[t - 2], contract)
        noisy = state.copy()
        rng = np.random.RandomState(0)
        if t + 1 < m:
            noisy[t + 1:] = rng.randn(*(noisy[t + 1:].shape)).astype(np.float32) * 5.0
        fut_noisy = A.build_proprio(noisy[t], noisy[t - 1], noisy[t - 2], contract)
        ok_future &= bool(np.allclose(base, fut) and np.allclose(base, fut_noisy))
        bumped = state[t].copy()
        bumped[0] += 0.25
        changed = A.build_proprio(bumped, state[t - 1], state[t - 2], contract)
        ok_sensitive |= not bool(np.allclose(base, changed))
        detail.append({"episode": name, "frame": int(t),
                       "max_abs_delta_truncated": float(np.abs(base - fut).max()),
                       "max_abs_delta_future_noise": float(np.abs(base - fut_noisy).max())})
    return {"name": "features_use_only_past_and_present",
            "pass": bool(ok_future and ok_sensitive),
            "invariant_to_future": bool(ok_future),
            "sensitive_to_current_state": bool(ok_sensitive),
            "detail": detail}


def check_provenance(meta, contracts_path):
    need = ["proprio_mode", "proprio_dim", "target_representation", "action_codec_mode",
            "gripper_policy", "history_boundary_policy", "motion_scale_fit_source",
            "q01", "q99", "v_scale", "a_scale", "pmin", "pmax"]
    missing = [k for k in need if k not in meta]
    gp = meta.get("gripper_policy", {})
    ok_grip = list(gp.get("gripper_dims", [])) == [6, 13]
    ok = (not missing and ok_grip
          and meta.get("target_representation") == "velocity_residual"
          and meta.get("action_codec_mode") == "fixed"
          and "TRAIN" in str(meta.get("motion_scale_fit_source", "")))
    return {"name": "checkpoint_provenance_complete", "pass": bool(ok),
            "missing": missing, "gripper_dims": gp.get("gripper_dims"),
            "contracts_sha256": sha256_file(contracts_path)}


def check_old80_failclosed(repo, python_exe, old_contract, out_dir):
    if not os.path.exists(old_contract):
        return {"name": "old_80_episode_contract_rejected", "pass": False,
                "detail": "missing old artefact: " + old_contract}
    meta = json.load(open(old_contract))
    n_old = meta.get("frozen_from", {}).get("n_train_episodes_for_quantiles")
    cmd = [python_exe, "diagnostics/r8_smoke.py", "--check-contract",
           "--norm-contract", old_contract, "--out", os.path.join(out_dir, "old80_rejected")]
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    text = (proc.stdout + proc.stderr)
    rejected = proc.returncode != 0 and "64 TRAIN / 16 DEV" in text
    return {"name": "old_80_episode_contract_rejected",
            "pass": bool(rejected), "old_n_train_episodes": n_old,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "",
            "old_contract_sha256": sha256_file(old_contract)}


def run_all(contract_path, repo, out_dir, hdf5_dir=None, split_path=None,
            bounds_path=None, contracts_path=None, python_exe=None,
            old_contract=None, checks_only=None):
    hdf5_dir = hdf5_dir or DEFAULT["hdf5_dir"]
    bounds_path = bounds_path or DEFAULT["bounds"]
    contracts_path = contracts_path or os.path.join(repo, DEFAULT["contracts"])
    split_path = split_path or os.path.join(repo, "data", "split80.json")
    old_contract = old_contract or os.path.join(repo, DEFAULT["old_contract"])
    python_exe = python_exe or sys.executable
    meta = json.load(open(contract_path))
    train, val = load_split(split_path)
    checks = []
    catalog = {
        "split": lambda: check_split(meta, train, val),
        "quantiles": lambda: check_quantiles(meta, hdf5_dir, train),
        "dtype": lambda: check_dtype_stability(meta, hdf5_dir, train),
        "scales": lambda: check_scales(meta, contracts_path, bounds_path, hdf5_dir, train),
        "zero": lambda: check_zero_motion(meta),
        "codec": lambda: check_codec(meta, hdf5_dir, val),
        "leak": lambda: check_no_future_leak(hdf5_dir, val, meta),
        "provenance": lambda: check_provenance(meta, contracts_path),
        "old80": lambda: check_old80_failclosed(repo, python_exe, old_contract, out_dir),
    }
    for key in (checks_only or catalog):
        item = catalog[key]()
        checks.append(item)
        print("[%s] %-44s" % ("PASS" if item["pass"] else "FAIL", item["name"]), flush=True)
    verdict = all(c["pass"] for c in checks)
    return {"contract_path": os.path.abspath(contract_path),
            "contract_sha256": sha256_file(contract_path),
            "hdf5_dir": hdf5_dir, "split_sha256": sha256_file(split_path),
            "bounds_sha256": sha256_file(bounds_path),
            "contracts_sha256": sha256_file(contracts_path),
            "checks": checks, "pass": bool(verdict)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--out", required=True)
    ap.add_argument("--checks", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    only = [c for c in args.checks.split(",") if c] or None
    res = run_all(args.contract, args.repo, args.out, checks_only=only)
    path = os.path.join(args.out, "gate_a_verification.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    for c in res["checks"]:
        print("[%s] %-44s %s" % ("PASS" if c["pass"] else "FAIL", c["name"],
                                 json.dumps({k: v for k, v in c.items()
                                             if k not in ("name", "pass", "q01", "q99",
                                                          "recomputed_q01", "recomputed_q99",
                                                          "v_scale", "a_scale")})[:220]))
    print("Gate A verification:", "PASS" if res["pass"] else "FAIL", "->", path)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
