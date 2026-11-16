#!/usr/bin/env python3
"""Animated heat map: 40 layers x 384 experts, each cell's opacity its share of the layer's decode tokens (log
scale), first in stored order, then sliding into each layer's sorted order (busiest left), holding, and back.
Pure SVG + CSS (no script), so it plays in an <img>; prefers-reduced-motion shows the sorted state. Cells under
MIN_SHARE are left out (they would be nearly invisible). Standard library only.
Usage: heatmap_anim.py COUNTS.json OUT.svg [--vram-share 0.118]"""
import json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart_style import W, TITLE, LABEL, SMALL, doc, text

src, out = sys.argv[1], sys.argv[2]
vram = float(sys.argv[sys.argv.index("--vram-share") + 1]) if "--vram-share" in sys.argv else 0.118
c = json.load(open(src))["counts"]["decode"]
nl, ne = len(c), len(c[0])
PAD, MIN_SHARE = 8, 0.002
lo, hi = math.log10(MIN_SHARE), math.log10(0.05)
L0, T, R, B = 52, 74, 8, 40
gw = W - 2 * PAD - L0 - R
ch = 6
cw = gw / ne
H = T + nl * ch + B

cells = []
for l, row in enumerate(c):
    t = sum(row) or 1
    order = sorted(range(ne), key=lambda e: -row[e])
    rank = {e: i for i, e in enumerate(order)}
    for e, v in enumerate(row):
        s = v / t
        if s < MIN_SHARE:
            continue
        a = max(0.06, min(1.0, (math.log10(s) - lo) / (hi - lo)))
        dx = (rank[e] - e) * cw
        cells.append(f"<rect x='{L0 + e * cw:.1f}' y='{T + l * ch}' width='{cw:.2f}' height='{ch - 0.6:.1f}' "
                     f"fill-opacity='{a:.2f}' style='--dx:{dx:.1f}px'/>")

css = ("<style>.hm rect{fill:var(--s1);animation:sortx 9s ease-in-out infinite}"
       "@keyframes sortx{0%,18%{transform:translateX(0)}45%,82%{transform:translateX(var(--dx))}"
       "100%{transform:translateX(0)}}"
       ".vr{animation:vrfade 9s ease-in-out infinite}@keyframes vrfade{0%,30%{opacity:0}45%,82%{opacity:1}95%,100%{opacity:0}}"
       ".lab1{animation:l1 9s infinite}@keyframes l1{0%,30%{opacity:1}40%,88%{opacity:0}96%,100%{opacity:1}}"
       ".lab2{animation:l2 9s infinite}@keyframes l2{0%,30%{opacity:0}40%,88%{opacity:1}96%,100%{opacity:0}}"
       "@media (prefers-reduced-motion:reduce){.hm rect{animation:none;transform:translateX(var(--dx))}"
       ".vr,.lab2{animation:none;opacity:1}.lab1{animation:none;opacity:0}}</style>")
vx = L0 + vram * gw
b = [css,
     text(0, 20, "A few experts do most of the work, in every layer", TITLE, "ink", weight="600"),
     text(0, 39, "share of each layer's decode tokens per expert, 64 BIRD training questions", LABEL, "ink2"),
     f"<rect x='{L0}' y='{T}' width='{gw:.1f}' height='{nl * ch}' fill='none' stroke='var(--grid)'/>",
     f"<g class='hm' shape-rendering='crispEdges'>{''.join(cells)}</g>",
     f"<g class='vr'><line x1='{vx:.1f}' x2='{vx:.1f}' y1='{T - 4}' y2='{T + nl * ch + 4}' stroke='var(--ink)' "
     f"stroke-width='1.5' stroke-dasharray='4 3'/>"
     + text(vx + 6, T - 8, f"room on the cards (≈{vram:.0%})", SMALL, "ink", "start", "600") + "</g>",
     f"<g class='lab1'>{text(L0 + gw / 2, H - 10, 'experts in stored order', SMALL, 'ink2', 'middle')}</g>",
     f"<g class='lab2'>{text(L0 + gw / 2, H - 10, 'each layer sorted, busiest first', SMALL, 'ink2', 'middle')}</g>"]
for l in (0, 10, 20, 30, 39):
    b.append(text(L0 - 7, T + l * ch + ch - 1, str(l + 1), SMALL, "muted", "end"))
b.append(text(L0 - 36, T + nl * ch / 2, "layer", SMALL, "ink2", "middle",
              extra=f" transform='rotate(-90 {L0 - 36} {T + nl * ch / 2})'"))
desc = ("Animation: the 40 x 384 grid of expert use starts in stored order, scattered, then each layer's experts slide "
        "into sorted order, leaving a dark band of busy experts on the left of every layer, mostly inside the "
        f"{vram:.0%} the cards have room for.")
open(out, "w").write(doc(W, H, f"<g transform='translate({PAD},0)'>{''.join(b)}</g>", "Sorting experts by use", desc))
print(len(cells), "cells,", os.path.getsize(out) // 1024, "KB")
