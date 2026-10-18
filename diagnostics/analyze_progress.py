#!/usr/bin/env python3
"""Summarize push_T process-progress logs (block-to-target geometry per episode)."""
import glob
import json
import os
import sys


def summarize(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        return None
    eps = {}
    for r in rows:
        eps.setdefault(r.get("ep", 0), []).append(r)
    out = []
    for ep, rs in sorted(eps.items()):
        xy = [r["xy"] for r in rs if "xy" in r]
        ang = [r["ang_deg"] for r in rs if "ang_deg" in r]
        if not xy:
            continue
        i_min = xy.index(min(xy))
        n = len(xy)
        out.append({
            "ep": ep,
            "seed": rs[-1].get("seed"),
            "steps": rs[-1].get("step"),
            "xy_first": xy[0],
            "xy_min": min(xy),
            "xy_min_step": rs[i_min].get("step"),
            "xy_final": xy[-1],
            "gain": (xy[0] - min(xy)),
            "frac_lt_5cm": sum(1 for v in xy if v < 0.05) / float(n),
            "ang_first": ang[0] if ang else None,
            "ang_min": min(ang) if ang else None,
            "ang_final": ang[-1] if ang else None,
        })
    return out


def main():
    paths = sys.argv[1:]
    if not paths:
        paths = sorted(glob.glob("/home/T7/ojh/robot_actual/pushT_closed_loop/**/_progress/*.jsonl", recursive=True))
        paths += sorted(glob.glob("/tmp/progress_*.jsonl"))
    for p in paths:
        res = summarize(p)
        if not res:
            print("%s: no rows" % p)
            continue
        print("== %s (%d episodes)" % (p, len(res)))
        for r in res:
            print("   ep%-2s seed=%-5s steps=%-4s xy: %.3f -> %.4f (min @%s, gain %.3f, <5cm %.0f%%)  ang: %.1f -> %.1f (min %.1f)" % (
                r["ep"], r["seed"], r["steps"], r["xy_first"], r["xy_min"], r["xy_min_step"],
                r["gain"], 100.0 * r["frac_lt_5cm"],
                r["ang_first"] if r["ang_first"] is not None else float("nan"),
                r["ang_final"] if r["ang_final"] is not None else float("nan"),
                r["ang_min"] if r["ang_min"] is not None else float("nan"),
            ))


if __name__ == "__main__":
    main()
