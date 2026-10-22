#!/usr/bin/env python3
"""R24: emit the training-evidence chart as dependency-free SVG.

matplotlib is not installed in the serving venv and adding a plotting dependency to a shared runtime is
not worth it, so the chart is written directly as SVG (renders in GitHub, diffs as text, no deps).

Four separate panels, because a single-step TRAIN loss and a teacher-forced DEV CE are different
estimators and must never share an axis:
  (a) train loss (raw per step)   (b) DEV ce_window   (c) learning rate   (d) physical SAFE L1
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "mission", "r24")
W, H = 780, 300
PAD = 52


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Panel(object):
    def __init__(self, x, y, title, xlabel, ylabel, logy=False):
        self.x, self.y, self.title = x, y, title
        self.xlabel, self.ylabel, self.logy = xlabel, ylabel, logy
        self.series = []

    def add(self, xs, ys, colour, label, dash=None, dots=True):
        self.series.append({"xs": list(xs), "ys": list(ys), "c": colour, "l": label,
                            "dash": dash, "dots": dots})

    def render(self):
        o = ['<g>']
        o.append('<rect x="%d" y="%d" width="%d" height="%d" fill="#fff" stroke="#ccc"/>'
                 % (self.x, self.y, W, H))
        o.append('<text x="%d" y="%d" font-size="12" font-weight="bold" fill="#222">%s</text>'
                 % (self.x + 8, self.y + 16, esc(self.title)))
        allx = [v for s in self.series for v in s["xs"]]
        ally = [v for s in self.series for v in s["ys"]]
        if not allx:
            o.append("</g>")
            return "".join(o)
        x0, x1 = min(allx), max(allx)
        if x1 == x0:
            x1 = x0 + 1
        y0, y1 = min(ally), max(ally)
        if self.logy:
            import math
            ally = [max(v, 1e-12) for v in ally]
            y0, y1 = math.log10(min(ally)), math.log10(max(ally))
        if y1 == y0:
            y1 = y0 + 1
        lo = self.y + 26
        hi = self.y + H - 34

        def px(v):
            return self.x + PAD + (v - x0) / (x1 - x0) * (W - PAD - 16)

        def py(v):
            import math
            vv = math.log10(max(v, 1e-12)) if self.logy else v
            return hi - (vv - y0) / (y1 - y0) * (hi - lo)

        for s in self.series:
            pts = " ".join("%.1f,%.1f" % (px(a), py(b)) for a, b in zip(s["xs"], s["ys"]))
            dash = ' stroke-dasharray="5,3"' if s["dash"] else ""
            o.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.4"%s/>'
                     % (pts, s["c"], dash))
            if s["dots"]:
                for a, b in zip(s["xs"], s["ys"]):
                    o.append('<circle cx="%.1f" cy="%.1f" r="1.9" fill="%s"/>' % (px(a), py(b), s["c"]))
        for i, s in enumerate(self.series):
            o.append('<text x="%d" y="%d" font-size="9" fill="%s">%s</text>'
                     % (self.x + PAD + 4, lo + 12 * i - 2, s["c"], esc(s["l"])))
        o.append('<text x="%d" y="%d" font-size="9" fill="#666">%s</text>'
                 % (self.x + PAD, self.y + H - 14, esc(self.xlabel)))
        o.append('<text x="%d" y="%d" font-size="9" fill="#666" transform="rotate(-90 %d %d)">%s</text>'
                 % (self.x + 12, self.y + H // 2, self.x + 12, self.y + H // 2, esc(self.ylabel)))
        o.append('<text x="%.1f" y="%d" font-size="8" fill="#888">%.4g</text>'
                 % (px(x0), hi + 11, x0))
        o.append('<text x="%.1f" y="%d" font-size="8" fill="#888">%.4g</text>'
                 % (px(x1) - 20, hi + 11, x1))
        o.append("</g>")
        return "".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "curves.svg"))
    a = ap.parse_args()

    p_train = Panel(0, 0, "(a) TRAIN loss, raw per step (batch=1) - NOT comparable to DEV CE",
                    "optimizer step", "train loss")
    for name, colour in (("r15_seg_300_500", "#1f77b4"), ("r15_seg_500_1000", "#17becf")):
        p = os.path.join(OUT, "r15_training_curve.json")
        if os.path.isfile(p):
            d = json.load(open(p)).get(name)
            if d:
                st = [s for s in d["steps"] if "loss" in s]
                p_train.add([s["step"] for s in st], [s["loss"] for s in st], colour, name)
    p = os.path.join(OUT, "m2_training_curve.json")
    if os.path.isfile(p):
        d = json.load(open(p)).get("m2_two_2000")
        if d:
            st = [s for s in d["steps"] if "loss" in s]
            p_train.add([s["step"] for s in st], [s["loss"] for s in st], "#ff7f0e", "m2_two_2000")

    p_dev = Panel(W + 12, 0, "(b) DEV teacher-forced ce_window (nats) - separate estimator",
                  "DEV eval index", "ce_window")
    if os.path.isfile(p):
        dev = json.load(open(p))["m2_two_2000"]["dev"]
        for task, colour in (("push_T", "#d62728"), ("stack_bowls", "#2ca02c")):
            pts = [(i, d["ce_window"]) for i, d in enumerate(dev) if d["task"] == task]
            if pts:
                p_dev.add([q[0] for q in pts], [q[1] for q in pts], colour, task)
    corr = os.path.join(REPO, "mission", "m2_ce_curve_corrected.json")
    if os.path.isfile(corr):
        cc = json.load(open(corr))
        for task, colour in (("push_T", "#d62728"), ("stack_bowls", "#2ca02c")):
            xs = [q["global_step"] for q in cc["points"] if task in q["ce_window"]]
            ys = [q["ce_window"][task] for q in cc["points"] if task in q["ce_window"]]
            p_dev.add(xs, ys, colour, task + " (corrected alignment)", dash=True)

    p_lr = Panel(0, H + 12, "(c) learning rate (log) - R15 decays at step 1000 = its final step",
                 "optimizer step", "lr (log10)")
    for name, colour in (("r15_seg_500_1000", "#1f77b4"), ("m2_two_2000", "#ff7f0e")):
        p = os.path.join(OUT, ("r15_training_curve.json" if name.startswith("r15")
                               else "m2_training_curve.json"))
        if os.path.isfile(p):
            d = json.load(open(p)).get(name)
            if d and d["lr_schedule"]:
                sch = d["lr_schedule"]
                xs = [s["first_step"] for s in sch] + [max(x["step"] for x in d["steps"])]
                ys = [s["lr"] for s in sch] + [sch[-1]["lr"]]
                p_lr.add(xs, ys, colour, name, dots=False)

    p_phys = Panel(W + 12, H + 12, "(d) physical SAFE 14D L1 per checkpoint - model vs CV",
                   "global step", "SAFE L1 (joint units)")
    pc = os.path.join(REPO, "mission", "m2_physical_curve.json")
    if os.path.isfile(pc):
        pts = json.load(open(pc))["points"]
        for task, dash in (("push_T", None), ("stack_bowls", "dash")):
            p_phys.add([q["global_step"] for q in pts],
                       [q[task]["model_safe_l1"] for q in pts], "#d62728", task + " model", dash=dash)
            p_phys.add([q["global_step"] for q in pts],
                       [q[task]["cv_l1"] for q in pts], "#1f77b4", task + " CV", dash=dash)

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">'
           % (2 * W + 12, 2 * H + 12, 2 * W + 12, 2 * H + 12),
           '<rect width="100%" height="100%" fill="#fafafa"/>']
    for panel in (p_train, p_dev, p_lr, p_phys):
        svg.append(panel.render())
    svg.append('<text x="8" y="%d" font-size="10" fill="#444">R24 - four estimators, four axes, '
               'deliberately never joined</text>' % (2 * H + 6))
    svg.append("</svg>")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("".join(svg))
    print("wrote", a.out, os.path.getsize(a.out), "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
