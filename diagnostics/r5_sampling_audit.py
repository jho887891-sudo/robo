#!/usr/bin/env python3
"""R5 阶段二：真实 HDF5 上的采样分布审计（只读，不训练）。

复用 training/r5_sampling.py 的**真实**实现：
  candidate_frames -> diversified_manifest(200) -> summarize_manifest
  spread_validation_frames(60)
并额外给出左右臂运动阈值下的分布，以及每个训练回合被选中的帧数。

用法：python diagnostics/r5_sampling_audit.py --out <json>
"""
import argparse
import json
import os
import sys

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "training"))

from r5_sampling import candidate_frames, diversified_manifest, summarize_manifest, spread_validation_frames  # noqa: E402

RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
HDF5 = os.environ.get("PUSHT_HDF5", "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
SPLIT = os.environ.get("PUSHT_SPLIT", RT + "/split80.json")


def c14(h, g):
    return np.concatenate([h[g + "/left_arm_joint_states"][:], h[g + "/left_ee_joint_states"][:],
                           h[g + "/right_arm_joint_states"][:], h[g + "/right_ee_joint_states"][:]],
                          axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--val-n", type=int, default=60)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--start", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args, _u = ap.parse_known_args()

    split = json.load(open(SPLIT))
    train, val = list(split["train"]), list(split["val"])
    rows = []
    lengths = {}
    for e in sorted(train):
        with h5py.File(os.path.join(HDF5, e), "r") as h:
            s, a = c14(h, "state"), c14(h, "action")
        lengths[e] = int(s.shape[0])
        rows += candidate_frames(e, s, a, stride=args.stride, start=args.start, tail=2)
    val_lengths = {}
    for e in sorted(val):
        with h5py.File(os.path.join(HDF5, e), "r") as h:
            val_lengths[e] = int(h["action/left_arm_joint_states"].shape[0])

    chosen = diversified_manifest(split, rows, args.n, seed=args.seed)
    summary = summarize_manifest(chosen)
    val_pairs = spread_validation_frames(split, val_lengths, args.val_n, seed=args.seed)

    res = {
        "git_commit": os.popen("git -C %s rev-parse HEAD" % REPO).read().strip(),
        "candidate_frames_total": len(rows),
        "candidate_episodes": len({r[0] for r in rows}),
        "stride": args.stride, "start": args.start, "seed": args.seed,
        "train": {
            "requested": args.n, "selected": len(chosen),
            "episodes_covered": len({r[0] for r in chosen}),
            "episodes_total": len(train),
            "distinct_frames": len({(r[0], r[1]) for r in chosen}),
            "activity": summary["activity_counts"],
            "activity_ratio": {k: round(v / len(chosen), 4) for k, v in summary["activity_counts"].items()},
            "threshold": summary["threshold"],
            "left_delta_mean": summary["left_delta_mean"],
            "right_delta_mean": summary["right_delta_mean"],
            "frames_per_episode_min": min(summary["samples_per_episode"].values()),
            "frames_per_episode_max": max(summary["samples_per_episode"].values()),
            "frames_per_episode_histogram": {},
        },
        "val": {
            "requested": args.val_n, "selected": len(val_pairs),
            "episodes_covered": len({r[0] for r in val_pairs}),
            "episodes_total": len(val),
            "distinct_frames": len({(r[0], r[1]) for r in val_pairs}),
            "frames_per_episode": {},
            "overlap_with_train_episodes": len({r[0] for r in val_pairs} & set(train)),
        },
    }
    hist = {}
    for e, c in summary["samples_per_episode"].items():
        hist[str(c)] = hist.get(str(c), 0) + 1
    res["train"]["frames_per_episode_histogram"] = dict(sorted(hist.items(), key=lambda kv: int(kv[0])))
    for e, t in val_pairs:
        res["val"]["frames_per_episode"][e] = res["val"]["frames_per_episode"].get(e, 0) + 1

    a = res["train"]["activity"]
    print("commit:", res["git_commit"])
    print("candidate frames: %d across %d/%d train episodes (stride=%d start=%d)" % (
        res["candidate_frames_total"], res["candidate_episodes"], len(train), args.stride, args.start))
    print("TRAIN selected: %d frames | episodes covered %d/%d | distinct frames %d" % (
        res["train"]["selected"], res["train"]["episodes_covered"], res["train"]["episodes_total"],
        res["train"]["distinct_frames"]))
    print("  motion split (threshold %.3f): left_only=%d (%.1f%%) right_only=%d (%.1f%%) both=%d (%.1f%%) neither=%d (%.1f%%)" % (
        summary["threshold"], a["left_only"], 100 * a["left_only"] / len(chosen),
        a["right_only"], 100 * a["right_only"] / len(chosen),
        a["both_active"], 100 * a["both_active"] / len(chosen),
        a["neither_active"], 100 * a["neither_active"] / len(chosen)))
    print("  mean |A-S| left=%.5f right=%.5f | frames/episode min=%d max=%d hist=%s" % (
        summary["left_delta_mean"], summary["right_delta_mean"],
        res["train"]["frames_per_episode_min"], res["train"]["frames_per_episode_max"],
        res["train"]["frames_per_episode_histogram"]))
    print("VAL selected: %d frames | episodes covered %d/%d | distinct frames %d | overlap with train episodes %d" % (
        res["val"]["selected"], res["val"]["episodes_covered"], res["val"]["episodes_total"],
        res["val"]["distinct_frames"], res["val"]["overlap_with_train_episodes"]))
    print("  val frames/episode: %s" % json.dumps({k: v for k, v in sorted(res["val"]["frames_per_episode"].items())}))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
        print("wrote", args.out)
    print("R5_SAMPLING_AUDIT_DONE")


if __name__ == "__main__":
    main()
