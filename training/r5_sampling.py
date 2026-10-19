"""R5: sample a broader robot-motion distribution without touching validation data.

All scores must be computed only from TRAIN episode state/action arrays.
This module has no dependencies on PyTorch, HDF5 or the robot runtime.
"""
from collections import Counter
import random

import numpy as np

ARM_DIM = 7
ACTION_DIM = 14
PREFERENCES = ("left", "right", "both", "time")


def candidate_frames(episode, states, actions, stride=20, start=20, tail=2):
    """Return (episode, frame, left_abs_delta, right_abs_delta) candidates."""
    s, a = np.asarray(states, np.float32), np.asarray(actions, np.float32)
    if s.ndim != 2 or a.shape != s.shape or s.shape[1] != ACTION_DIM:
        raise ValueError("expected matching (frames,14) action/state arrays")
    if not np.isfinite(s).all() or not np.isfinite(a).all():
        raise ValueError("non-finite HDF5 state or action")
    if stride <= 0 or start < 0 or tail < 1:
        raise ValueError("invalid sampling window")
    delta = np.abs(a - s)
    return [(episode, t, float(delta[t, :ARM_DIM].mean()),
             float(delta[t, ARM_DIM:].mean()))
            for t in range(start, max(start, len(s) - tail), stride)]


def diversified_manifest(split, rows, n, seed=0):
    """Round-robin episodes, alternating priority for left/right/total motion.

    Covers as many TRAIN episodes as the budget permits and prefers frames with
    movement rather than repeatedly selecting near-identical static examples.
    Does not assume left/right motion exists: diagnostics expose missing coverage.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("positive integer n required")
    train = list(split["train"])
    val = set(split["val"])
    if len(train) != len(set(train)) or set(train) & val:
        raise ValueError("train/val split overlap or duplicate train episodes")
    if not train:
        raise ValueError("empty training episodes")
    pool = {e: [] for e in train}
    for row in rows:
        e, t, left, right = row
        if e not in pool or e in val:
            raise ValueError("candidate from non-training episode: " + str(e))
        if t < 0 or not np.isfinite([left, right]).all() or left < 0 or right < 0:
            raise ValueError("invalid candidate frame/motion")
        pool[e].append((e, int(t), float(left), float(right)))
    seen = set()
    for e, candidates in pool.items():
        for _, t, _, _ in candidates:
            if (e, t) in seen:
                raise ValueError("duplicate candidate episode/frame")
            seen.add((e, t))
    if n >= len(train) and any(not pool[e] for e in train):
        raise ValueError("cannot cover all training episodes: some have no candidate frames")
    if len(rows) < n:
        raise ValueError("not enough candidate frames for requested budget")
    rng = random.Random(seed)
    episode_order = sorted(train)
    rng.shuffle(episode_order)
    result = []
    used = set()
    round_idx = 0
    while len(result) < n:
        preference = PREFERENCES[round_idx % len(PREFERENCES)]
        progressed = False
        for episode in episode_order:
            remaining = [row for row in pool[episode] if (row[0], row[1]) not in used]
            if not remaining:
                continue
            if preference == "left":
                key = lambda r: (r[2], r[3], -r[1])
            elif preference == "right":
                key = lambda r: (r[3], r[2], -r[1])
            elif preference == "both":
                key = lambda r: (min(r[2], r[3]), r[2] + r[3], -r[1])
            else:
                # Coverage of the episode timeline, after high-motion candidates.
                key = lambda r: (min(abs(r[1] - u[1]) for u in pool[episode]
                                   if (u[0], u[1]) in used)
                                 if any((u[0], u[1]) in used for u in pool[episode])
                                 else r[1], r[2] + r[3], -r[1])
            chosen = max(remaining, key=key)
            used.add((chosen[0], chosen[1]))
            result.append(chosen)
            progressed = True
            if len(result) == n:
                break
        if not progressed:
            raise ValueError("candidate pool exhausted before requested sample count")
        round_idx += 1
    return result


def summarize_manifest(rows, threshold=0.005):
    if not rows:
        raise ValueError("empty sampling manifest")
    epis = Counter(e for e, _, _, _ in rows)
    mat = np.asarray([[l, r] for _, _, l, r in rows], np.float64)
    active = mat > threshold
    counts = {
        "left_only": int(np.sum(active[:, 0] & ~active[:, 1])),
        "right_only": int(np.sum(~active[:, 0] & active[:, 1])),
        "both_active": int(np.sum(active[:, 0] & active[:, 1])),
        "neither_active": int(np.sum(~active[:, 0] & ~active[:, 1])),
    }
    return {"sample_count": len(rows), "episode_count": len(epis),
            "samples_per_episode": dict(sorted(epis.items())),
            "threshold": threshold, "activity_counts": counts,
            "left_delta_mean": float(mat[:, 0].mean()),
            "right_delta_mean": float(mat[:, 1].mean())}


def spread_validation_frames(split, episode_lengths, n, seed=0):
    """Pick time-spread examples from *all* held-out episodes when n >= count.

    Low-discrepancy fractions 1/2, 1/4, 3/4, ... prevent taking only a
    few episodes from a skewed precomputed fixed-frame list.
    """
    train, val = set(split["train"]), list(split["val"])
    if train & set(val) or len(set(val)) != len(val):
        raise ValueError("train/val episode overlap or duplicate validation episodes")
    if not val or not isinstance(n, int) or n < len(val):
        raise ValueError("validation budget must cover every held-out episode")
    order = sorted(val)
    random.Random(seed).shuffle(order)
    count = {e: 0 for e in order}
    result = []
    while len(result) < n:
        made_progress = False
        for e in order:
            length = int(episode_lengths[e])
            usable = length - 2  # match training's terminal exclusion
            if usable <= 0:
                raise ValueError("validation episode too short: " + e)
            phase = count[e] + 1
            # van der Corput base 2: 1/2, 1/4, 3/4, 1/8, ...
            k, denom, numerator = phase, 1, 0
            while k:
                denominator_digit = k % 2
                denom *= 2
                numerator = numerator * 2 + denominator_digit
                k //= 2
            fraction = numerator / denom
            t = min(usable - 1, int(fraction * usable))
            used = {q[1] for q in result if q[0] == e}
            if t in used:
                nearby = [q for q in range(usable) if q not in used]
                if not nearby:
                    raise ValueError("insufficient unique validation frames: " + e)
                t = min(nearby, key=lambda q: (abs(q - t), q))
            result.append([e, t])
            count[e] += 1
            made_progress = True
            if len(result) == n:
                break
        if not made_progress:
            raise ValueError("could not fill validation frame manifest")
    return result
