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
     text(0, 39, "share of expert uses covered by each layer's busiest experts, 64 BIRD training questions", LABEL, "ink2")]
k, _ = key([("decode", "s1", "line"), ("prefill", "s2", "line")], 0, 60)
b.append(k)
# the part the cards keep resident
b.append(f"<rect x='{X(0):.1f}' y='{Y(1):.1f}' width='{X(vram) - X(0):.1f}' height='{Y(0) - Y(1):.1f}' "
         f"fill='var(--s1)' fill-opacity='0.07'/>")
b.append(text(X(vram) + 6, Y(0.08), f"\u2190 room on the cards (≈{vram:.0%})", SMALL, "ink2", "start", extra=HALO))
b.append(text(X(vram) + 6, Y(0.08) + 14, "the rest: RAM cache, then SSD", SMALL, "muted", "start", extra=HALO))
for v in (0, 0.25, 0.5, 0.75, 1.0):
    b.append(hline(L, x1, Y(v), "axis" if v == 0 else "grid"))
    b.append(text(L - 7, Y(v) + 4, f"{v:.0%}", SMALL, "muted", "end"))
for f in (0, 0.25, 0.5, 0.75, 1.0):
    b.append(text(X(f), H - B + 18, f"{f:.0%}", SMALL, "muted", "middle"))
b.append(text((L + x1) / 2, H - 12, "share of each layer's 384 experts, busiest first", SMALL, "ink2", "middle"))
marks = []; series = {}
for kind, col in (("prefill", "s2"), ("decode", "s1")):
    cv, n = curve(d["counts"][kind]); series[kind] = [round(v, 4) for v in cv]
    b.append(line([(X(i / n), Y(c)) for i, c in enumerate(cv)], col, 2))
    i = round(vram * n); marks.append((kind, cv[i]))
    b.append(dot(X(i / n), Y(cv[i]), col))
    b.append(text(X(i / n) + 8, Y(cv[i]) - 8 if kind == "decode" else Y(cv[i]) + 16, f"{cv[i]:.0%}", LABEL, "ink",
                  "start", "600", HALO))
desc = (f"Busiest {vram:.0%} of each layer's experts (about what the cards keep resident): "
        + ", ".join(f"{k} {v:.0%} of uses" for k, v in marks)) + "."
# hover readout (inline use only; an <img> ignores the script): a crosshair and the two shares at the pointer
b.append(f"<g id='xh' style='display:none;pointer-events:none'><line id='xhl' y1='{Y(1):.1f}' y2='{Y(0):.1f}' "
         f"stroke='var(--ink2)' stroke-width='1' stroke-dasharray='3 3'/><circle id='xhd' r='4' fill='var(--s1)'/>"
         f"<circle id='xhp' r='4' fill='var(--s2)'/><rect id='xhb' rx='4' height='48' width='196' fill='var(--surf)' "
         f"stroke='var(--grid)'/><text id='xht1' font-size='{SMALL}' fill='var(--ink)' font-weight='600'></text>"
         f"<text id='xht2' font-size='{SMALL}' fill='var(--ink2)'></text><text id='xht3' font-size='{SMALL}' fill='var(--ink2)'></text></g>"
         f"<rect id='xhz' x='{L:.1f}' y='{Y(1):.1f}' width='{x1 - L:.1f}' height='{Y(0) - Y(1):.1f}' fill='transparent'/>")
js = """<script><![CDATA[(function(){
var D=%s,P=%s,L=%.1f,X1=%.1f,Y0=%.1f,Y1=%.1f,PAD=%d;
var s=document.currentScript.closest('svg'),q=function(i){return s.querySelector('#'+i)};
function Y(v){return Y0-v*(Y0-Y1)}
function at(a,f){var i=Math.round(f*(a.length-1));return a[i]}
function mv(e){var r=s.getBoundingClientRect(),vb=s.viewBox.baseVal,x=(e.clientX-r.left)*vb.width/r.width-PAD;
 var f=Math.max(0,Math.min(1,(x-L)/(X1-L)));var px=L+f*(X1-L),d=at(D,f),p=at(P,f);
 q('xh').style.display='';q('xhl').setAttribute('x1',px);q('xhl').setAttribute('x2',px);
 q('xhd').setAttribute('cx',px);q('xhd').setAttribute('cy',Y(d));q('xhp').setAttribute('cx',px);q('xhp').setAttribute('cy',Y(p));
 var bx=px+12>X1-200?px-208:px+12,by=Y1+70;q('xhb').setAttribute('x',bx);q('xhb').setAttribute('y',by);
 var t=[['xht1','busiest '+Math.round(f*100)+'%% of experts ('+Math.round(f*384)+' a layer)'],['xht2','decode: '+Math.round(d*100)+'%% of the work'],['xht3','prefill: '+Math.round(p*100)+'%% of the work']];
 t.forEach(function(r,i){var el=q(r[0]);el.textContent=r[1];el.setAttribute('x',bx+10);el.setAttribute('y',by+16+i*14)})}
var z=q('xhz');z.addEventListener('pointermove',mv);z.addEventListener('pointerdown',mv);
z.addEventListener('pointerleave',function(){q('xh').style.display='none'});})();]]></script>""" % (
    json.dumps(series["decode"]), json.dumps(series["prefill"]), L, x1, Y(0), Y(1), PAD)
b.append(js)
open(out, "w").write(doc(W, H, f"<g transform='translate({PAD},0)'>{''.join(b)}</g>",
                         "Expert use concentration", desc))
print(desc)
