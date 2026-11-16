#!/usr/bin/env python3
"""How concentrated expert use is: from a placement profile's per-layer routing counts, the share of expert uses
covered by the busiest x% of each layer's experts (layers averaged), for decode and prefill, with the share of
experts the cards keep resident marked. Standard library only, in the shared chart style.
Usage: experts.py PROFILE.json OUT.svg [--vram-share 0.118]"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart_style import W, TITLE, LABEL, SMALL, doc, text, hline, line, area, dot, key

prof, out = sys.argv[1], sys.argv[2]
vram = float(sys.argv[sys.argv.index("--vram-share") + 1]) if "--vram-share" in sys.argv else 0.118
d = json.load(open(prof))
PAD = 8
HALO = " paint-order='stroke' stroke='var(--surf)' stroke-width='3' stroke-linejoin='round'"


def curve(layers):
    """Mean over layers of the cumulative share of uses held by the top k experts, k = 0..n."""
    n = len(layers[0]); acc = [0.0] * (n + 1); used = 0
    for L in layers:
        s = sorted(L, reverse=True); t = sum(s)
        if not t:
            continue
        used += 1; c = 0
        for k, v in enumerate(s, 1):
            c += v; acc[k] += c / t
    return [a / used for a in acc], n


H, L, R, T, B = 360, 46, 24, 74, 50
x1 = W - 2 * PAD - R
def X(f): return L + f * (x1 - L)
def Y(v): return H - B - v * (H - T - B)

b = [text(0, 20, "A few experts do most of the work", TITLE, "ink", weight="600"),
     text(0, 39, "share of expert uses covered by each layer's busiest experts, BIRD profile", LABEL, "ink2")]
k, _ = key([("decode", "s1", "line"), ("prefill", "s2", "line")], 0, 60)
b.append(k)
# the part the cards keep resident
b.append(f"<rect x='{X(0):.1f}' y='{Y(1):.1f}' width='{X(vram) - X(0):.1f}' height='{Y(0) - Y(1):.1f}' "
         f"fill='var(--s1)' fill-opacity='0.07'/>")
b.append(text(X(vram) + 6, Y(0.08), f"\u2190 kept on the cards ({vram:.0%})", SMALL, "ink2", "start", extra=HALO))
b.append(text(X(vram) + 6, Y(0.08) + 14, "the rest: RAM cache, then SSD", SMALL, "muted", "start", extra=HALO))
for v in (0, 0.25, 0.5, 0.75, 1.0):
    b.append(hline(L, x1, Y(v), "axis" if v == 0 else "grid"))
    b.append(text(L - 7, Y(v) + 4, f"{v:.0%}", SMALL, "muted", "end"))
for f in (0, 0.25, 0.5, 0.75, 1.0):
    b.append(text(X(f), H - B + 18, f"{f:.0%}", SMALL, "muted", "middle"))
b.append(text((L + x1) / 2, H - 12, "share of each layer's 384 experts, busiest first", SMALL, "ink2", "middle"))
marks = []
for kind, col in (("prefill", "s2"), ("decode", "s1")):
    cv, n = curve(d["counts"][kind])
    b.append(line([(X(i / n), Y(c)) for i, c in enumerate(cv)], col, 2))
    i = round(vram * n); marks.append((kind, cv[i]))
    b.append(dot(X(i / n), Y(cv[i]), col))
    b.append(text(X(i / n) + 8, Y(cv[i]) - 8 if kind == "decode" else Y(cv[i]) + 16, f"{cv[i]:.0%}", LABEL, "ink",
                  "start", "600", HALO))
desc = (f"Busiest {vram:.0%} of each layer's experts (about what the cards keep resident): "
        + ", ".join(f"{k} {v:.0%} of uses" for k, v in marks)) + "."
open(out, "w").write(doc(W, H, f"<g transform='translate({PAD},0)'>{''.join(b)}</g>",
                         "Expert use concentration", desc))
print(desc)
