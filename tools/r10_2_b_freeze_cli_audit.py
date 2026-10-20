#!/usr/bin/env python3
"""R10.2 B: CLI-level audit of the FINAL freeze path (PR #14 freeze_checked).

Every case is executed through the real command line, and each NEGATIVE case must produce a
non-zero exit code AND leave the target freeze file absent. The positive case must really
write a freeze file whose sealed manifest contains the 5 DEV + 20 FINAL identities.

All HDF5 here is SYNTHETIC: no real new episodes exist, so the outcome can only be
VERIFIED_CPU_TESTED, never "new FINAL ready".
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PY = sys.executable
GUARD = os.path.join(REPO, "tools", "r10_freeze_guard.py")
USED = os.path.join(REPO, "evidence", "r9_eval_prep_20261020", "manifest_existing100.json")
SPLIT = os.path.join(REPO, "data", "split80.json")
PROTO = os.path.join(REPO, "protocol", "r9_eval_protocol.json")
TASK = "Push the T-shaped block to align it precisely with the gray T-shaped pad."
GROUPS = (("left_arm", slice(0, 6)), ("left_ee", slice(6, 7)),
          ("right_arm", slice(7, 13)), ("right_ee", slice(13, 14)))
GIT_SHA = "a" * 40
RESULTS = []


def ok(name, cond, detail=None):
    RESULTS.append({"check": name, "pass": bool(cond), "detail": detail or {}})
    print("[%s] %-52s %s" % ("PASS" if cond else "FAIL", name,
                             json.dumps(detail or {}, default=str)[:170]))
    return bool(cond)


def make_episode(path, seed=0, frames=101, offset=0.0, instruction=TASK, freq=25,
                 cameras=3, cam_side=64, rgb_bad=False, truncate=None):
    import h5py
    rs = np.random.RandomState(seed)
    state = (rs.randn(frames, 14) * 0.01 + offset).astype(np.float64)
    action = np.roll(state, -1, axis=0)
    cams = ("cam_head", "cam_left_wrist", "cam_right_wrist")[:cameras]
    with h5py.File(path, "w") as h:
        for name, sl in GROUPS:
            h.create_dataset("state/%s_joint_states" % name, data=state[:, sl])
            h.create_dataset("action/%s_joint_states" % name, data=action[:, sl])
        h.create_dataset("instruction", data=instruction)
        h.create_dataset("additional_info/frequency", data=freq)
        for cam in cams:
            if rgb_bad:
                arr = np.zeros((frames, cam_side, cam_side), dtype=np.uint8)
            else:
                arr = rs.randint(0, 255, (frames, cam_side, cam_side, 3), dtype=np.uint8)
            h.create_dataset("vision/%s/colors" % cam, data=arr)
    if truncate:
        with open(path, "r+b") as fh:
            fh.truncate(truncate)


def build(td, n=25, name="set", **kw):
    d = os.path.join(td, name)
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        make_episode(os.path.join(d, "ep_%03d.hdf5" % i), seed=1000 + i,
                     offset=0.001 * (i + 1), **kw)
    return d


def run_guard(cand, out, freeze=None, extra=(), protocol=None):
    cmd = [PY, GUARD, "--candidate-dir", cand, "--used-manifest", USED,
           "--split", SPLIT, "--protocol", protocol or PROTO, "--out", out,
           "--expect-total", "25", "--expect-dev", "5"]
    if freeze:
        cmd += ["--freeze-out", freeze, "--git-sha", GIT_SHA]
    cmd += list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {"rc": p.returncode, "stdout": p.stdout[-1500:], "stderr": p.stderr[-1500:]}


def main():
    td = tempfile.mkdtemp(prefix="r10_2_b_")
    work = os.path.join(td, "ev")
    os.makedirs(work)
    try:
        # ------------------------------------------------ positive: real freeze write
        good = build(td, 25, "good")
        fz = os.path.join(work, "frozen_ok.json")
        r = run_guard(good, os.path.join(work, "guard_ok.json"), freeze=fz)
        wrote = os.path.exists(fz)
        frozen = json.load(open(fz)) if wrote else {}
        man = frozen.get("manifest", {})
        eps = man.get("episodes", [])
        split = man.get("r10_split_assignment", {})
        prov = man.get("r10_provenance", {})
        ok("positive_freeze_written", r["rc"] == 0 and wrote,
           {"rc": r["rc"], "bytes": os.path.getsize(fz) if wrote else 0})
        ok("frozen_seals_25_episodes", len(eps) == 25, {"n": len(eps)})
        ok("frozen_seals_5_dev_20_final",
           len(split.get("dev", [])) == 5 and len(split.get("final", [])) == 20
           and not (set(split.get("dev", [])) & set(split.get("final", []))),
           {"dev": len(split.get("dev", [])), "final": len(split.get("final", []))})
        ok("frozen_records_carry_both_hashes",
           all(len(e.get("content_sha256", "")) == 64
               and len(e.get("state_action_sha256", "")) == 64 for e in eps))
        ok("frozen_records_carry_seed_protocol_usedgit",
           bool(split.get("seed")) and len(prov.get("protocol_sha256", "")) == 64
           and len(prov.get("used_manifest_sha256", "")) == 64
           and frozen.get("git_sha") == GIT_SHA,
           {"seed": split.get("seed"), "protocol": prov.get("protocol_sha256", "")[:12],
            "git_sha": frozen.get("git_sha")})
        ok("frozen_manifest_sha256_recorded", len(frozen.get("manifest_sha256", "")) == 64)

        # ------------------------------------------------ negatives (CLI + no file)
        def negative(name, cand, extra=(), protocol=None, precreate=None):
            out_f = os.path.join(work, "neg_%s.json" % name)
            fz_f = os.path.join(work, "neg_freeze_%s.json" % name)
            if precreate:
                with open(fz_f, "w") as fh:
                    fh.write(precreate)
                before = hashlib.sha256(open(fz_f, "rb").read()).hexdigest()
            res = run_guard(cand, out_f, freeze=fz_f, extra=extra, protocol=protocol)
            if precreate:
                after = hashlib.sha256(open(fz_f, "rb").read()).hexdigest()
                return ok(name, res["rc"] != 0 and before == after,
                          {"rc": res["rc"], "preexisting_untouched": before == after})
            return ok(name, res["rc"] != 0 and not os.path.exists(fz_f),
                      {"rc": res["rc"], "freeze_file_absent": not os.path.exists(fz_f)})

        # 1 renamed used episode
        d = os.path.join(td, "renamed")
        os.makedirs(d)
        src = "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data/episode_0000000.hdf5"
        shutil.copy(src, os.path.join(d, "totally_new_name.hdf5"))
        negative("renamed_used_episode", d)

        # 2 instruction-only change on the same state/action
        d = os.path.join(td, "instr_only")
        os.makedirs(d)
        make_episode(os.path.join(d, "a.hdf5"), seed=7, offset=0.002)
        make_episode(os.path.join(d, "b.hdf5"), seed=7, offset=0.002,
                     instruction=TASK + " (paraphrased)")
        negative("instruction_only_change", d)

        # 3 placeholder 9KB
        d = os.path.join(td, "placeholder")
        os.makedirs(d)
        with open(os.path.join(d, "push_T_ep0000.hdf5"), "wb") as fh:
            fh.write(b"\x89HDF\r\n\x1a\n" + b"\0" * 9000)
        negative("placeholder_9kb", d)

        # 4 missing camera
        negative("missing_camera", build(td, 25, "nocam", cameras=2))

        # 5 RGB anomaly (2-D array where HxWx3 needed)
        negative("rgb_channel_anomaly", build(td, 25, "badrgb", rgb_bad=True))

        # 6 30 Hz
        negative("wrong_frequency_30hz", build(td, 25, "freq30", freq=30))

        # 7 fewer than 100 frames
        negative("too_few_frames", build(td, 25, "short", frames=60))

        # 8 wrong task instruction
        negative("wrong_task", build(td, 25, "wrongtask", instruction="Stack the three bowls."))

        # 9 wrong episode count
        d = build(td, 24, "count24")
        res = run_guard(d, os.path.join(work, "count24.json"),
                        freeze=os.path.join(work, "count24_freeze.json"))
        ok("wrong_episode_count_24", res["rc"] != 0
           and not os.path.exists(os.path.join(work, "count24_freeze.json")), {"rc": res["rc"]})

        # 10 duplicate content inside the candidate set
        d = build(td, 24, "dup")
        shutil.copy(os.path.join(d, "ep_000.hdf5"), os.path.join(d, "ep_024_copy.hdf5"))
        negative("internal_duplicate", d)

        # 11 protocol drift (one byte changed)
        drift = os.path.join(td, "protocol_drift.json")
        blob = json.load(open(PROTO))
        blob["metrics_primary"] = blob["metrics_primary"][:-1]
        json.dump(blob, open(drift, "w"), indent=1)
        negative("protocol_drift", build(td, 25, "drift"), protocol=drift)

        # 12 test-only override must never freeze
        negative("test_only_override", build(td, 6, "override"),
                 extra=["--allow-small-set-for-tests", "--expect-total", "6"])

        # 13 existing freeze path must not be overwritten
        negative("freeze_path_exists", build(td, 25, "exists"),
                 precreate="{\"pre_existing\": true}")

        # 14 candidate dir unbuildable (empty)
        empty = os.path.join(td, "empty")
        os.makedirs(empty)
        negative("unbuildable_manifest", empty)

        # 15 truncated HDF5 (corrupt mid-file)
        d = build(td, 25, "trunc")
        make_episode(os.path.join(d, "ep_000.hdf5"), seed=9999, truncate=50000)
        negative("corrupt_truncated_hdf5", d)

        n = sum(1 for r in RESULTS if r["pass"])
        summary = {"checks": RESULTS, "n_pass": n, "n_total": len(RESULTS),
                   "note": "synthetic HDF5 only; outcome is VERIFIED_CPU_TESTED, not new FINAL ready"}
        out = os.path.join(REPO, "evidence", "r10_2", "b_freeze_cli_audit.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(summary, open(out, "w"), indent=1)
        print("\n%d/%d checks passed -> %s" % (n, len(RESULTS), "PASS" if n == len(RESULTS) else "FAIL"))
        return 0 if n == len(RESULTS) else 1
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
