#!/usr/bin/env python3
"""R25: auditable episode-cluster evaluation of R24 *offline SAFE* rows.

Consumes the preserved R24 stratum rows and the original and current frame
manifests. Does not conflate frame bootstrap and independent episodes; does
not claim deployment exactness, equivalence, nor a true H4 causal mechanism.
No torch, checkpoint, HDF5 or GPU required.
"""
import argparse
import hashlib
import json
import os
import random


def digest(data, omit):
    obj = {k: v for k, v in data.items() if k not in omit}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def mean(items):
    return sum(items) / len(items)


def bootstrap(values, iterations=20000, seed=0):
    if not values:
        raise ValueError("empty episode collection")
    rng = random.Random(seed)
    samples = sorted(mean([rng.choice(values) for _ in values])
                     for _ in range(iterations))
    return [samples[int(.025 * iterations)], samples[int(.975 * iterations)]]


def analyze(archive, authoritative, legacy, iterations=20000, seed=0):
    if digest(authoritative, {"manifest_sha256"}) != authoritative["manifest_sha256"]:
        raise ValueError("authoritative frame manifest content hash mismatch")
    if digest(legacy, {"manifest_sha256", "frozen_before_model_run"}) != legacy["manifest_sha256"]:
        raise ValueError("legacy manifest historical hash mismatch")
    if archive["frame_manifest_sha256"] != legacy["manifest_sha256"]:
        raise ValueError("stratum report is not sourced from supplied legacy manifest")
    if authoritative["frames"] != legacy["frames"]:
        raise ValueError("legacy and authoritative frame arrays DIFFER")
    if archive["split"] != "dev":
        raise ValueError("only DEV stratum audit supported")
    result = {"schema": "r25-episode-cluster-audit-v1",
              "source_archive_manifest_sha256": legacy["manifest_sha256"],
              "authoritative_manifest_sha256": authoritative["manifest_sha256"],
              "same_frame_arrays_verified": True,
              "resampling_unit": "episode",
              "metric": "paired(model-CV) offline SAFE mean-absolute-joint-error",
              "scope": "R24 matched offline approximate SAFE; M2 deployment metadata/rate limit not verified",
              "tasks": {}}
    for task in authoritative["tasks"]:
        records = authoritative["frames"][task]["dev"]
        reported = archive["tasks"][task]["frames"]
        if len(records) != len(reported):
            raise ValueError("length mismatch: " + task)
        by_ep = {}
        for rec, row in zip(records, reported):
            if rec["stratum"] != row["stratum"]:
                raise ValueError("stratum/order mismatch: " + task)
            pair = float(row["model"]) - float(row["cv"])
            by_ep.setdefault(rec["episode"], []).append(pair)
        ep_means = [mean(x) for x in by_ep.values()]
        counts = [len(x) for x in by_ep.values()]
        total = sum(counts)
        result["tasks"][task] = {
            "n_frames": total,
            "n_independent_episodes": len(by_ep),
            "frames_per_episode": {k: len(v) for k, v in by_ep.items()},
            "largest_episode_share": max(counts) / total,
            "kish_effective_episode_count_under_frame_weights": total**2 / sum(c*c for c in counts),
            "pooled_frame_mean_model_minus_cv": sum(sum(x) for x in by_ep.values()) / total,
            "equal_episode_macro_model_minus_cv": mean(ep_means),
            "episode_cluster_bootstrap95": bootstrap(ep_means, iterations, seed),
            "bootstrap_draws": iterations,
            "bootstrap_seed": seed,
            "interpretation": "Exploratory; few independent episodes; no equivalence claim",
        }
    return result


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--archive", default=os.path.join(repo, "mission/r24/stratum_audit.json"))
    p.add_argument("--manifest", default=os.path.join(repo, "mission/r24/frame_manifest.json"))
    p.add_argument("--legacy-manifest", default=os.path.join(repo, "mission/r24/frame_manifest_v1_hash_scope.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap", type=int, default=20000)
    args = p.parse_args()
    if args.bootstrap < 100:
        p.error("--bootstrap must be >= 100")
    with open(args.archive) as f:
        archive = json.load(f)
    with open(args.manifest) as f:
        authoritative = json.load(f)
    with open(args.legacy_manifest) as f:
        legacy = json.load(f)
    result = analyze(archive, authoritative, legacy, args.bootstrap)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
