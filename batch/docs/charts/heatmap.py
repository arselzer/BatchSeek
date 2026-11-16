#!/usr/bin/env python3
"""Expert use as a heat map: 40 layers x 384 experts, each cell's opacity the expert's share of its layer's routed
tokens (log scale), from routing counts (counts[kind][layer][expert]). The cells are a PNG (alpha only, one hue)
embedded in an SVG with the shared chart style, so it works on light and dark backgrounds. Standard library only.
Usage: heatmap.py COUNTS.json OUT.svg [--kind decode]"""
import base64, json, math, os, struct, sys, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart_style import W, TITLE, LABEL, SMALL, doc, text

src, out = sys.argv[1], sys.argv[2]
kind = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else "decode"
c = json.load(open(src))["counts"][kind]
nl, ne = len(c), len(c[0])
SORTED = "--sorted" in sys.argv
PAD, RGB = 8, (0x2a, 0x78, 0xd6)   # the s1 blue (light); the alpha carries the value


def png(rows):
    """RGBA PNG from rows of alpha bytes (one hue)."""
    raw = b"".join(b"\x00" + b"".join(bytes((*RGB, a)) for a in r) for r in rows)
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", len(rows[0]), len(rows), 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


lo, hi = math.log10(2e-3), math.log10(0.05)   # share of a layer's routed tokens: 0.2% .. 5%
rows, top = [], []
for L in c:
    L = sorted(L, reverse=True) if SORTED else L
    t = sum(L) or 1
    rows.append([0 if v == 0 else max(0, min(255, round(255 * (math.log10(v / t) - lo) / (hi - lo)))) for v in L])
    top.append(max(L) / t)
img = base64.b64encode(png(rows)).decode()

L0, T, R, B = 52, 70, 8, 46
gw = W - 2 * PAD - L0 - R
ch = 6                                        # px per layer row
H = T + nl * ch + B
b = [text(0, 20, "A few experts do most of the work, in every layer" if SORTED else "Which experts each layer uses", TITLE, "ink", weight="600"),
     text(0, 39, f"share of each layer's {kind} tokens per expert, 64 BIRD training questions (log scale)", LABEL, "ink2"),
     f"<image x='{L0}' y='{T}' width='{gw}' height='{nl * ch}' preserveAspectRatio='none' "
     f"style='image-rendering:pixelated' href='data:image/png;base64,{img}'/>",
     f"<rect x='{L0}' y='{T}' width='{gw}' height='{nl * ch}' fill='none' stroke='var(--grid)'/>"]
for l in (0, 10, 20, 30, 39):
    b.append(text(L0 - 7, T + l * ch + ch - 1, str(l + 1), SMALL, "muted", "end"))
b.append(text(L0 - 36, T + nl * ch / 2, "layer", SMALL, "ink2", "middle",
              extra=f" transform='rotate(-90 {L0 - 36} {T + nl * ch / 2})'"))
for e in (() if SORTED else (0, 96, 192, 288, 383)):
    b.append(text(L0 + (e + 0.5) / ne * gw, T + nl * ch + 16, str(e), SMALL, "muted", "middle"))
b.append(text(L0 + gw / 2, H - 8, "each layer's experts, busiest first" if SORTED else "expert number (as stored in the model)",
              SMALL, "ink2", "middle"))
if SORTED:   # the share of experts the cards have room for
    vx = L0 + 0.118 * gw
    b.append(f"<line x1='{vx:.1f}' x2='{vx:.1f}' y1='{T - 4}' y2='{T + nl * ch + 4}' stroke='var(--ink)' stroke-width='1.5' stroke-dasharray='4 3'/>")
    b.append(text(vx + 6, T - 8, "room on the cards (≈12%)", SMALL, "ink", "start", "600"))
# legend: a small gradient of the same hue
lx, ly = W - 2 * PAD - 190, 52
b.append("<defs><linearGradient id='hm'><stop offset='0' stop-color='var(--s1)' stop-opacity='0.04'/>"
         "<stop offset='1' stop-color='var(--s1)' stop-opacity='1'/></linearGradient></defs>")
b.append(f"<rect x='{lx + 44}' y='{ly - 9}' width='90' height='10' fill='url(#hm)'/>")
b.append(text(lx + 40, ly, "0.2%", SMALL, "muted", "end"))
b.append(text(lx + 138, ly, "5%+", SMALL, "muted", "start"))
unused = sum(v == 0 for L in c for v in L) / (nl * ne)
desc = (f"{kind}: {nl} layers x {ne} experts; the busiest expert of a layer takes {min(top):.1%}-{max(top):.1%} of "
        f"its tokens; {unused:.0%} of expert slots unused in this sample.")
open(out, "w").write(doc(W, H, f"<g transform='translate({PAD},0)'>{''.join(b)}</g>", "Expert use heat map", desc))
print(desc)
