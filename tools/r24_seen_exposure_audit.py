#!/usr/bin/env python3
"""CPU-only R24 audit: distinguish TRAIN-split examples from truly gradient-exposed examples.

The R24 'train fit' manifest selects frames from TRAIN episodes, not from the
checkpoint's consumed sampler prefix. Its score therefore does not measure
memorisation or fitting of examples that actually received gradient updates.
This tool reconstructs that prefix and fails closed on any checkpoint/sampler
contract mismatch. No optimizer, GPU, or checkpoint write occurs.
"""
import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def frame_key(task, episode, tick):
    return (str(task), os.path.basename(str(episode)), int(tick))


def manifest_digest(manifest):
    body = json.dumps({k: v for k, v in manifest.items()
                       if k != "manifest_sha256"}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def summarise_seen(seen_keys, manifest):
    """Pure classifier; never infer 'seen' merely from train split membership."""
    seen = set(seen_keys)
    results = {}
    for task in manifest["tasks"]:
        block = manifest["frames"][task]
        train_keys = [frame_key(task, r["episode"], r["t"]) for r in block["train"]]
        dev_keys = [frame_key(task, r["episode"], r["t"]) for r in block["dev"]]
        if len(set(train_keys)) != len(train_keys):
            raise ValueError("duplicate selected TRAIN frame for " + task)
        if len(set(dev_keys)) != len(dev_keys):
            raise ValueError("duplicate selected DEV frame for " + task)
        if set(train_keys) & set(dev_keys):
            raise ValueError("TRAIN/DEV selected frames overlap for " + task)
        hit = sorted(set(train_keys) & seen)
        miss = sorted(set(train_keys) - seen)
        dev_hit = sorted(set(dev_keys) & seen)
        results[task] = {
            "selected_train": len(train_keys),
            "selected_train_seen": len(hit),
            "selected_train_unseen": len(miss),
            "seen_fraction_of_selected_train": len(hit) / len(train_keys) if train_keys else None,
            "seen_selected_train": [{"episode": ep, "t": t} for _, ep, t in hit],
            "unseen_selected_train": [{"episode": ep, "t": t} for _, ep, t in miss],
            "selected_dev": len(dev_keys),
            "selected_dev_seen_BAD_IF_NONZERO": len(dev_hit),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="M2 step2000 full-state checkpoint directory (read-only)")
    parser.add_argument("--manifest", default=os.path.join(REPO, "mission/r24/frame_manifest.json"))
    parser.add_argument("--splits", default=os.path.join(REPO, "data/m2_splits.json"))
    parser.add_argument("--normalizer", default=os.path.join(
        REPO, "data/m2_task_normalizer_2task.json"))
    parser.add_argument("--hdf5-root", default=None)
    parser.add_argument("--out", default=os.path.join(
        REPO, "mission/r24/seen_exposure_verified.json"))
    args = parser.parse_args()

    for name in ("training", "code/xpolicylab/policy/OpenVLA_R42",
                 "code/xpolicylab/policy/OpenVLA_Pure"):
        sys.path.insert(0, os.path.join(REPO, name))
    import m2_train_shared as M2
    import m2_multitask_data as MD
    import r15_resume as RS

    manifest = json.load(open(args.manifest))
    if manifest_digest(manifest) != manifest.get("manifest_sha256"):
        raise SystemExit("R24 manifest digest failed; no inference allowed")
    checkpoint_manifest, trainer = RS.verify_and_load_manifest(args.checkpoint)
    tasks = list(trainer.get("tasks") or [])
    if set(tasks) != set(manifest["tasks"]) or not tasks:
        raise SystemExit("checkpoint tasks differ from the R24 manifest")
    tmix = trainer.get("task_mix") or {}
    if not {"floor", "seed", "schedule_sha256"} <= set(tmix):
        raise SystemExit("checkpoint lacks frozen task mix; cannot reconstruct seen samples")

    _doc, splits, payload, _sp, _np = M2.load_m2_artifacts(
        tasks, args.splits, args.normalizer)
    indexes = {t: MD.load_task_index(t, args.hdf5_root) for t in tasks}
    frames = {t: M2.task_train_frames(indexes[t], splits[t], t) for t in tasks}
    mix = M2.plan_task_mix(tasks, {t: len(frames[t]) for t in tasks},
                            floor=float(tmix["floor"]), seed=int(tmix["seed"]))
    if mix["schedule_sha256"] != tmix["schedule_sha256"]:
        raise SystemExit("saved task mix hash differs; aborting seen/unseen audit")
    if mix["counts"] != tmix.get("counts") or mix["horizon"] != tmix.get("horizon"):
        raise SystemExit("task mix counts/horizon changed")
    slots = M2.build_slot_frames(mix, frames)
    saved_sampler = json.load(open(os.path.join(args.checkpoint, "sampler.json")))
    sampler = RS.FrameSampler.from_state(slots, saved_sampler)
    pos = sampler.position
    if pos != int(trainer["global_step"]) or not 0 <= pos <= len(slots):
        raise SystemExit("sampler position/global_step mismatch or wrapped epoch")

    seen_keys = set()
    for idx in sampler.perm[:pos]:
        qualified, tick = slots[idx]
        task, episode = MD.split_qualified(qualified)
        seen_keys.add(frame_key(task, episode, tick))

    comparison = summarise_seen(seen_keys, manifest)
    if any(x["selected_dev_seen_BAD_IF_NONZERO"] != 0 for x in comparison.values()):
        raise SystemExit("DEV contamination detected: checkpoint consumed selected DEV frames")

    result = {
        "schema": "r24-actual-exposure-v1",
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_global_step": int(trainer["global_step"]),
        "checkpoint_manifest_sha256": hashlib.sha256(
            open(os.path.join(args.checkpoint, "MANIFEST.json"), "rb").read()).hexdigest(),
        "frame_manifest_sha256": manifest["manifest_sha256"],
        "sampler_order_hash": sampler.order_hash(),
        "sampler_position": pos,
        "total_training_slots": len(slots),
        "unique_gradient_exposed_frames": len(seen_keys),
        "tasks": comparison,
        "interpretation": (
            "Only selected_train_seen examples have truly been exposed to gradients. "
            "The existing R24 fit archive mixes selected_train_seen and "
            "selected_train_unseen; do not use its TRAIN-vs-DEV means to rule out H1/H3."
        ),
        "no_optimizer_steps": True,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps({"sampler_position": pos,
                      "unique_gradient_exposed_frames": len(seen_keys),
                      "tasks": {t: {k: v for k, v in d.items() if not isinstance(v, list)}
                                for t, d in comparison.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
