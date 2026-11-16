#!/usr/bin/env python3
"""Render the post's charts as static SVGs from data.json (standard library only), in the shared chart style
(chart_style.py: tokens, type scale, light/dark).
Usage: make_charts.py [DIR]   -- reads DIR/data.json, writes DIR/throughput.svg, jobs.svg, energy.svg.
Pending values (null) are drawn as dashed placeholders with a label, never as a guessed value."""
import json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart_style import W, TITLE, LABEL, SMALL, doc, text, text_w, hline, line, dot, key, nice_step

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(D, "data.json")))
HALO = " paint-order='stroke' stroke='var(--surf)' stroke-width='3' stroke-linejoin='round'"
PAD = 8   # same left edge as the timeline charts


def wrap(body):
    return f"<g transform='translate({PAD},0)'>{body}</g>"


def throughput():
    pts = data["throughput"]["points"]
    H, L, R, T, B = 380, 46, 24, 70, 50
    xmax = 512
    def X(n): return L + (math.log2(n) / math.log2(xmax)) * (W - 2 * PAD - L - R)
    vmax = max(p["tok_s"] for p in pts if p["tok_s"] is not None)
    ymax, step = vmax * 1.15, nice_step(vmax * 1.15, 3)
    def Y(v): return H - B - v / ymax * (H - T - B)
    b = [text(0, 20, "Decode throughput, all streams together", TITLE, "ink", weight="600"),
         text(0, 39, "output tokens per second, four RX 9070 XT", LABEL, "ink2")]
    k, _ = key([("batch configuration", "s1", "line"), ("chat configuration (1-8 streams)", "neutral", "line")], 0, 60)
    b.append(k)
    v = 0.0
    while v <= ymax:
        b.append(hline(L, W - 2 * PAD - R, Y(v), "axis" if v == 0 else "grid"))
        b.append(text(L - 7, Y(v) + 4, f"{v:,.0f}", SMALL, "muted", "end"))
        v += step
    for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        b.append(text(X(n), H - B + 18, str(n), SMALL, "muted", "middle"))
    b.append(text((L + W - 2 * PAD - R) / 2, H - 12, "concurrent streams (log scale)", SMALL, "ink2", "middle"))
    for kind, col in (("chat", "neutral"), ("batch", "s1")):
        s = [p for p in pts if p["kind"] == kind and p["tok_s"] is not None]
        if len(s) > 1:
            b.append(line([(X(p["streams"]), Y(p["tok_s"])) for p in s], col, 2))
        for p in s:
            b.append(dot(X(p["streams"]), Y(p["tok_s"]), col))
    for p in pts:   # selective direct labels: the batch points and the single-stream anchor
        if p["tok_s"] is None:
            x = X(p["streams"])
            b.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{Y(0):.1f}' y2='{Y(ymax*0.55):.1f}' stroke='var(--s1)' "
                     f"stroke-width='1.5' stroke-dasharray='4 4' stroke-opacity='0.6'/>")
            b.append(text(x, Y(ymax * 0.55) - 6, "pending", SMALL, "muted", "middle"))
        elif p["kind"] == "batch" or p["streams"] == 1:
            lab = f"{p['tok_s']:.0f}" + (" *" if p.get("pending") else "")
            x, y = X(p["streams"]), Y(p["tok_s"])
            if p["streams"] == 1:
                b.append(text(x + 2, y - 11, lab, LABEL, "ink", "start", "600", HALO))
            else:
                b.append(text(x - 9, y - 9, lab, LABEL, "ink", "end", "600", HALO))
    if any(p.get("pending") and p["tok_s"] is not None for p in pts):
        b.append(text(W - 2 * PAD - R, 60, "* earlier build, being re-measured", SMALL, "muted", "end"))
    desc = "; ".join(f"{p['streams']} streams: " + (f"{p['tok_s']} tok/s" if p["tok_s"] is not None else "pending")
                     + f" ({p['kind']})" for p in pts)
    return doc(W, H, wrap("".join(b)), "Decode throughput by number of concurrent streams", desc)


def bar(x0, x1, y, h, col, r=4):
    """Horizontal bar: square at the baseline, 4 px rounded data end."""
    r = min(r, (x1 - x0) / 2, h / 2)
    return (f"<path d='M{x0:.1f},{y:.1f} H{x1-r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y+r:.1f} V{y+h-r:.1f} "
            f"Q{x1:.1f},{y+h:.1f} {x1-r:.1f},{y+h:.1f} H{x0:.1f} Z' fill='var(--{col})'/>")


def hbars(rows, keys, title, subtitle, unit, fname_title, fmt="{:,.0f}"):
    L = 236                                   # label column
    R = 78                                    # room for the value at the bar tip
    bh = 18 if len(keys) > 1 else 20          # bar thickness (<= 24 px), 2 px surface gap between a row's bars
    rowh = len(keys) * (bh + 2) + 16
    T = 64 if len(keys) > 1 else 50
    H = T + rowh * len(rows) + 6
    vmax = max((r[k] for r in rows for k, _, _ in keys if r[k] is not None), default=1)
    x1max = W - 2 * PAD - R
    def X(v): return L + v / vmax * (x1max - L)
    b = [text(0, 20, title, TITLE, "ink", weight="600"), text(0, 39, subtitle, LABEL, "ink2")]
    if len(keys) > 1:
        k, _ = key([(lab, col, "box") for _, lab, col in keys], 0, 58)
        b.append(k.replace("fill-opacity='0.55'", ""))
    for i, r in enumerate(rows):
        y0 = T + i * rowh
        b.append(text(L - 12, y0 + (rowh - 16) / 2 + 4, r["job"], LABEL, "ink", "end"))
        for j, (k, lab, col) in enumerate(keys):
            y = y0 + j * (bh + 2)
            v = r[k]
            if v is None:
                w = (x1max - L) * 0.3
                b.append(f"<rect x='{L+0.75:.1f}' y='{y+0.75:.1f}' width='{w:.1f}' height='{bh-1.5:.1f}' rx='4' fill='none' "
                         f"stroke='var(--{col})' stroke-width='1.5' stroke-dasharray='4 3'/>")
                b.append(text(L + w + 8, y + bh / 2 + 4, "pending", SMALL, "muted"))
            else:
                b.append(bar(L, X(v), y, bh, col))
                b.append(text(X(v) + 6, y + bh / 2 + 4, fmt.format(v) + " " + unit, SMALL, "ink"))
    b.append(f"<line x1='{L}' x2='{L}' y1='{T-4}' y2='{H-10}' stroke='var(--axis)' stroke-width='1'/>")
    desc = "; ".join(r["job"] + ": " + ", ".join(f"{lab} " + (fmt.format(r[k]) + " " + unit if r[k] is not None else "pending")
                                                  for k, lab, _ in keys) for r in rows)
    return doc(W, H, wrap("".join(b)), fname_title, desc)


open(os.path.join(D, "throughput.svg"), "w").write(throughput())
open(os.path.join(D, "jobs.svg"), "w").write(hbars(
    data["jobs"]["rows"], [("plain_s", "plain engine", "neutral"), ("batch_s", "batch configuration", "s1")],
    "Whole job, wall time", "seconds, lower is better", "s", "Job wall time, plain engine vs batch configuration"))
erows = [{"job": r["job"], "kwh": (r["j_per_req"] / 3600 if r["j_per_req"] is not None else None)}
         for r in data["energy"]["rows"]]
open(os.path.join(D, "energy.svg"), "w").write(hbars(
    erows, [("kwh", "energy", "s1")],
    "Energy per 1,000 requests", "kWh, GPU boards + CPU package", "kWh", "Energy per 1,000 requests by job", fmt="{:.2f}"))
print("wrote throughput.svg jobs.svg energy.svg in", D)
