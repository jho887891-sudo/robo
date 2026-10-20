#!/usr/bin/env python3
"""R10 freeze guard: the thin layer that ACTUALLY enforces g1-g6 before any FINAL freeze.

The inherited tools/r8_eval_manifest.py::freeze does not check its own preconditions; relying
on "the protocol document says so" is not an implementation. This guard re-derives every
precondition from the real HDF5 files and refuses to freeze otherwise.

Checks (all must pass, each independently recomputable):
  1 schema      real HDF5: state/action 14D over the four joint groups, three RGB streams with
                real frame arrays, instruction, 25 Hz, >= 100 frames, finite values, aligned
  2 task        instruction contains the target task
  3 counts      exactly N episodes (default 25) with DEV/FINAL partition (5/20) by seeded hash
  4 internal    no duplicate name/content/state-action inside the candidate set
  5 cross-set   no overlap (name, content sha, state-action sha) with the frozen used 100
  6 identity    the used-reference manifest matches the frozen dataset hash in the protocol
  7 verify      rebuilding the manifest from disk reproduces the same hashes

Only if 1-7 pass does it write the frozen manifest (via r8_eval_manifest.freeze).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import r8_eval_manifest as M  # noqa: E402
import r9_final_disjoint as D  # noqa: E402

BASE_DIM = 14
JOINT_GROUPS = (("left_arm", 6), ("left_ee", 1), ("right_arm", 6), ("right_ee", 1))
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
MIN_BYTES_PER_FILE = 100_000


def _c14(h, group):
    parts = [h["%s/%s_joint_states" % (group, g)][()] for g, _ in JOINT_GROUPS]
    return np.concatenate(parts, axis=1).astype(np.float64)


def hdf5_schema(path, task="Push the T-shaped block"):
    """Real-file schema probe: fail closed on anything that is not a genuine episode."""
    import h5py
    problems = []
    info = {"file": os.path.basename(path), "bytes": os.path.getsize(path)}
    if info["bytes"] < MIN_BYTES_PER_FILE:
        problems.append("file too small (%d bytes): placeholder, not an episode" % info["bytes"])
    try:
        with h5py.File(path, "r") as h:
            keys = set(h.keys())
            if not {"state", "action"} <= keys:
                problems.append("missing state/action groups")
                return {"file": info["file"], "bytes": info["bytes"], "pass": False,
                        "problems": problems}
            state = _c14(h, "state")
            action = _c14(h, "action")
            info["frames"] = int(state.shape[0])
            if state.shape != action.shape or state.shape[1] != BASE_DIM:
                problems.append("state/action shape %s / %s is not (T,%d)"
                                % (state.shape, action.shape, BASE_DIM))
            if not (np.isfinite(state).all() and np.isfinite(action).all()):
                problems.append("non-finite state/action")
            if info["frames"] < 100:
                problems.append("only %d frames (< 100)" % info["frames"])
            ins = h["instruction"][()] if "instruction" in h else b""
            if isinstance(ins, bytes):
                ins = ins.decode("utf-8", "replace")
            elif hasattr(ins, "shape") and getattr(ins, "shape", ()) != ():
                first = ins.reshape(-1)[0] if ins.size else b""
                ins = first.decode("utf-8", "replace") if isinstance(first, bytes) else str(first)
            info["instruction"] = str(ins)
            if task.lower() not in info["instruction"].lower():
                problems.append("instruction is not the target task: %r" % info["instruction"][:60])
            freq = None
            if "additional_info/frequency" in h:
                freq = int(h["additional_info/frequency"][()])
            info["frequency"] = freq
            if freq != 25:
                problems.append("frequency %r != 25 Hz" % (freq,))
            cams = {}
            for cam in CAMERAS:
                key = "vision/%s/colors" % cam
                if key not in h:
                    problems.append("missing camera stream %s" % cam)
                    continue
                ds = h[key]
                shape = list(ds.shape)
                cams[cam] = {"shape": shape, "dtype": str(ds.dtype)}
                if len(shape) < 4 or shape[0] != info["frames"]:
                    problems.append("camera %s shape %s is not aligned with %d frames"
                                    % (cam, shape, info["frames"]))
                if shape[-1] not in (3, 4) or shape[1] < 64 or shape[2] < 64:
                    problems.append("camera %s has implausible geometry %s" % (cam, shape))
            info["cameras"] = cams
    except Exception as exc:  # noqa: BLE001
        problems.append("unreadable HDF5: %r" % (exc,))
    info["pass"] = not problems
    info["problems"] = problems
    return info


def assign_splits(episodes, n_dev, seed="r10-dev-v1"):
    """Seeded content-hash assignment: deterministic, protocol-documented."""
    if n_dev <= 0 or n_dev >= len(episodes):
        raise ValueError("n_dev must be in 1..N-1")
    ordered = sorted(episodes, key=lambda r: hashlib.sha256(
        (seed + r["content_sha256"]).encode()).hexdigest())
    dev = [r["file"] for r in ordered[:n_dev]]
    final = [r["file"] for r in ordered[n_dev:]]
    if set(dev) & set(final):
        raise AssertionError("dev/final overlap")
    return {"dev": dev, "final": final,
            "dev_names_sha256": hashlib.sha256("\n".join(sorted(dev)).encode()).hexdigest(),
            "final_names_sha256": hashlib.sha256("\n".join(sorted(final)).encode()).hexdigest(),
            "seed": seed}


def run(candidate_dir, used_manifest_path, split_path, protocol_path,
        expect_total=25, expect_dev=5, task_filter=None, allow_small_set=False):
    protocol = json.load(open(protocol_path))
    used = json.load(open(used_manifest_path))
    split = json.load(open(split_path))
    checks = {}
    files = sorted(f for f in os.listdir(candidate_dir) if f.endswith(".hdf5"))
    schema = [hdf5_schema(os.path.join(candidate_dir, f)) for f in files]
    checks["schema"] = {"pass": bool(schema) and all(s["pass"] for s in schema),
                        "n_files": len(files),
                        "failures": [s for s in schema if not s["pass"]][:5],
                        "frames_total": int(sum(s.get("frames", 0) for s in schema))}
    manifest = {"episodes": []}
    build_error = None
    if files:
        try:
            manifest = M.build(candidate_dir, task_filter)
        except Exception as exc:  # noqa: BLE001
            build_error = repr(exc)
    manifest["candidate_dir"] = os.path.abspath(candidate_dir)
    checks["build"] = {"pass": build_error is None, "error": build_error,
                       "n_episodes": len(manifest.get("episodes", []))}
    protocol_for_check = protocol
    if allow_small_set:
        # TEST-ONLY: relax the protocol episode minimum to the requested synthetic size.
        # The real run never sets this; the override is recorded below and in the output.
        protocol_for_check = json.loads(json.dumps(protocol))
        protocol_for_check["splits"]["dev_episodes"] = int(expect_dev)
        protocol_for_check["splits"]["final_episodes_min"] = max(1, int(expect_total) - int(expect_dev))
    try:
        disjoint = D.check(manifest, used, split, protocol_for_check)
        disjoint["small_set_override"] = bool(allow_small_set)
        disjoint["final_episodes_min_effective"] = int(
            protocol_for_check["splits"]["final_episodes_min"])
        disjoint["dev_episodes_effective"] = int(protocol_for_check["splits"]["dev_episodes"])
        checks["cross_set_disjoint"] = disjoint
    except Exception as exc:  # noqa: BLE001
        checks["cross_set_disjoint"] = {"pass": False, "error": repr(exc),
                                        "small_set_override": bool(allow_small_set)}
    checks["counts"] = {"pass": len(files) == int(expect_total),
                        "n_files": len(files), "expect_total": int(expect_total)}
    try:
        split_assign = assign_splits(manifest["episodes"], int(expect_dev)) if files else {}
        checks["split_assignment"] = {"pass": bool(split_assign)
                                      and len(split_assign["dev"]) == int(expect_dev),
                                      **split_assign}
    except Exception as exc:  # noqa: BLE001
        checks["split_assignment"] = {"pass": False, "error": repr(exc)}
    if build_error is None and files:
        try:
            verify = M.verify(manifest, candidate_dir)
        except Exception as exc:  # noqa: BLE001
            verify = {"ok": False, "problems": [repr(exc)]}
        checks["verify"] = {"pass": bool(verify.get("ok")),
                            "dataset_sha256": verify.get("dataset_sha256"),
                            "problems": verify.get("problems", [])}
    else:
        checks["verify"] = {"pass": False, "error": "manifest could not be built"}
    ok = all(c["pass"] for c in checks.values())
    return {"pass": bool(ok), "candidate_dir": os.path.abspath(candidate_dir),
            "checks": checks, "manifest": manifest, "protocol_sha256": hashlib.sha256(
                open(protocol_path, "rb").read()).hexdigest(),
            "used_manifest_sha256": hashlib.sha256(
                open(used_manifest_path, "rb").read()).hexdigest(),
            "note": "g1-g6 enforced here; a FAIL must leave the set unfrozen"}


def freeze_checked(res, protocol_path, git_sha, freeze_out):
    """Persist a 25=5+20 FINAL freeze with the split IDENTITIES inside its hashed manifest.

    Unlike M.freeze(), this rejects missing/failed gates, incomplete splits, an
    unpinned commit identity, and accidental overwrites of previous evidence.
    """
    import re
    if not res.get("pass") or not res.get("checks") or not all(
            c.get("pass") is True for c in res["checks"].values()):
        raise ValueError("refusing FINAL freeze: R10 gate not fully passed")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", git_sha or ""):
        raise ValueError("refusing FINAL freeze: require full 40-character Git commit SHA")
    if os.path.lexists(freeze_out):
        raise FileExistsError("refusing FINAL freeze: output already exists: " + freeze_out)
    episodes = res["manifest"]["episodes"]
    split = res["checks"]["split_assignment"]
    dev = split["dev"]
    final = split["final"]
    names = [rec["file"] for rec in episodes]
    if (len(names) != 25 or len(dev) != 5 or len(final) != 20
            or len(set(names)) != 25 or set(names) != set(dev) | set(final)
            or set(dev) & set(final)):
        raise ValueError("refusing FINAL freeze: expected exactly 25 episodes / 5 DEV / 20 FINAL")
    # The split and provenance are part of the manifest being SHA256-sealed.
    sealed = dict(res["manifest"])
    sealed["r10_split_assignment"] = split
    sealed["r10_provenance"] = {
        "protocol_sha256": res["protocol_sha256"],
        "used_manifest_sha256": res["used_manifest_sha256"],
        "guard_result_checks_sha256": hashlib.sha256(
            json.dumps(res["checks"], sort_keys=True).encode("utf-8")).hexdigest(),
    }
    return M.freeze(sealed, json.load(open(protocol_path)), git_sha, freeze_out, protocol_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-dir", required=True)
    ap.add_argument("--used-manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-total", type=int, default=25)
    ap.add_argument("--expect-dev", type=int, default=5)
    ap.add_argument("--task-filter", default=None)
    ap.add_argument("--freeze-out", default="")
    ap.add_argument("--git-sha", default="")
    ap.add_argument("--allow-small-set-for-tests", action="store_true",
                    help="TEST ONLY: relax the protocol episode minimum to --expect-total")
    args = ap.parse_args()
    # TEST-only overrides must never generate a frozen FINAL artefact.
    if args.freeze_out and (args.allow_small_set_for_tests
                            or args.expect_total != 25 or args.expect_dev != 5):
        print("refusing to freeze: require production 25/5 counts, no test override",
              file=sys.stderr)
        return 9
    if args.freeze_out and (not args.git_sha or os.path.abspath(args.freeze_out) == os.path.abspath(args.out)):
        print("refusing to freeze: full Git SHA required and freeze output must differ from gate evidence",
              file=sys.stderr)
        return 9
    res = run(args.candidate_dir, args.used_manifest, args.split, args.protocol,
              args.expect_total, args.expect_dev, args.task_filter,
              args.allow_small_set_for_tests)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    for name, c in res["checks"].items():
        print("[%s] %-22s %s" % ("PASS" if c.get("pass") else "FAIL", name,
                                 json.dumps({k: v for k, v in c.items() if k != "pass"},
                                            default=str)[:220]))
    if args.freeze_out:
        if not res["pass"]:
            print("refusing to freeze: guard failed", file=sys.stderr)
            return 9
        try:
            frozen = freeze_checked(res, args.protocol, args.git_sha, args.freeze_out)
        except (ValueError, FileExistsError) as exc:
            print(str(exc), file=sys.stderr)
            return 9
        print("FROZEN ->", args.freeze_out, "manifest_sha256", frozen["manifest_sha256"][:16])
    print("R10 freeze guard:", "PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 9


if __name__ == "__main__":
    sys.exit(main())
