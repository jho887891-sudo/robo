#!/usr/bin/env python3
"""从原始训练日志反推第二阶段的学习率调度，核对 README 的重建命令是否自证。

对应第一轮审查问题 7。日志是唯一的原始证据（启动脚本与 shell history 均未留存），
所以这里只做“日志能证明到什么程度”的判定，不做任何猜测：

  * 解析 step / lr 时间序列；
  * 找出 lr 首次下降的位置，给出“下降步数”的可证明区间 [上一记录步+1, 下降记录步]；
  * 与给定 milestone 声明比对：若声明值不在该区间内 -> CONTRADICTION；
  * 明确标注“无法唯一确定”，除非区间宽度为 1。

用法：
  python diagnostics/lr_schedule_forensics.py \
      --log logs/pure3c_train.log --stage 1 --claimed-milestone 1000
  python diagnostics/lr_schedule_forensics.py \
      --log logs/pure3c_cont.log  --stage 2 --claimed-milestone 1000
"""
import argparse
import os
import re
import sys

PAT = re.compile(r"step=(\d+)\s+loss=([0-9.]+)\s+lr=([0-9.eE+-]+)")


def parse(path):
    rows = []
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(3))))
    return rows


def analyse(rows):
    if not rows:
        return None
    drops = []
    for i in range(1, len(rows)):
        if rows[i][1] < rows[i - 1][1] - 1e-12:
            drops.append({"prev_step": rows[i - 1][0], "prev_lr": rows[i - 1][1],
                          "step": rows[i][0], "lr": rows[i][1],
                          "bracket": [rows[i - 1][0] + 1, rows[i][0]]})
    return {
        "points": len(rows),
        "first_step": rows[0][0],
        "last_step": rows[-1][0],
        "lr_set": sorted({r[1] for r in rows}),
        "drops": drops,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--stage", default="?")
    ap.add_argument("--claimed-milestone", type=int, default=None)
    args = ap.parse_args()

    rows = parse(args.log)
    info = analyse(rows)
    print("== stage %s : %s ==" % (args.stage, os.path.basename(args.log)))
    if not info:
        print("NO_STEP_LINES")
        return 0
    print("points=%d steps=%d..%d" % (info["points"], info["first_step"], info["last_step"]))
    print("lr values seen: %s" % ", ".join("%.2e" % v for v in info["lr_set"]))
    if not info["drops"]:
        print("no lr drop inside the logged window")
    for d in info["drops"]:
        print("lr drop: %.2e @step %d  ->  %.2e @step %d   (true drop step in [%d, %d])" % (
            d["prev_lr"], d["prev_step"], d["lr"], d["step"], d["bracket"][0], d["bracket"][1]))

    if args.claimed_milestone is not None:
        lo, hi = (info["drops"][0]["bracket"] if info["drops"] else (None, None))
        if lo is None:
            print("VERDICT: no lr drop inside logged window; cannot check milestone=%d" % args.claimed_milestone)
        elif lo <= args.claimed_milestone <= hi:
            print("VERDICT: COMPATIBLE with claimed milestone=%d (allowed interval [%d,%d]%s)" % (
                args.claimed_milestone, lo, hi,
                "; NOT uniquely determined" if hi > lo else "; uniquely determined"))
        else:
            print("VERDICT: CONTRADICTION - claimed milestone=%d outside the logged interval [%d,%d]" % (
                args.claimed_milestone, lo, hi))
    print("LR_FORENSICS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
