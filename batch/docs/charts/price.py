#!/usr/bin/env python3
"""Electricity price slider: the cost of the 500-question BIRD job on this box (measured energy x price) against
the APIs' billed cost for the same job. Drag (or click) the slider; without script it shows EUR 0.30/kWh.
Usage: price.py OUT.svg"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart_style import W, TITLE, LABEL, SMALL, doc, text

out = sys.argv[1]
KWH = 0.0480                      # measured: 721 W x 240 s (GPU boards + CPU package), H54 default mode, BIRD 500
API = [("DeepSeek API", 0.031), ("Wafer", 0.021), ("DeepInfra", 0.015)]
P0, PMAX = 0.30, 0.50
PAD, L, R = 8, 186, 70
x1 = W - 2 * PAD - R
VMAX = 0.07
def X(v): return L + min(v, VMAX) / VMAX * (x1 - L)

bh, gap, T = 20, 8, 128
rows = [("This box (electricity)", KWH * P0)] + API
H = T + len(rows) * (bh + gap) + 40
b = [text(0, 20, "Cost of the 500-question job", TITLE, "ink", weight="600"),
     text(0, 39, "drag to set the electricity price", LABEL, "ink2")]
# slider
sx0, sx1, sy = L, x1, 78
def SX(p): return sx0 + p / PMAX * (sx1 - sx0)
b.append(f"<line x1='{sx0}' x2='{sx1}' y1='{sy}' y2='{sy}' stroke='var(--grid)' stroke-width='4' stroke-linecap='round'/>")
for p, lab in ((0.0, "€0"), (0.05, "solar surplus"), (0.30, "typical tariff"), (0.50, "€0.50")):
    b.append(f"<line x1='{SX(p):.1f}' x2='{SX(p):.1f}' y1='{sy + 6}' y2='{sy + 11}' stroke='var(--muted)'/>")
    b.append(text(SX(p), sy + 24, lab, SMALL, "muted", "middle"))
b.append(text(L - 12, sy + 4, "electricity", LABEL, "ink", "end"))
b.append(f"<circle class='pk' cx='{SX(P0):.1f}' cy='{sy}' r='9' fill='var(--s1)' stroke='var(--surf)' stroke-width='2' style='cursor:grab'/>")
b.append(text(SX(P0), sy - 16, f"€{P0:.2f}/kWh", SMALL, "ink", "middle", "600").replace("<text ", "<text class='pl' ", 1))
b.append(f"<rect class='pz' x='{sx0 - 12}' y='{sy - 16}' width='{sx1 - sx0 + 24}' height='32' fill='transparent' style='cursor:pointer'/>")
for i, (name, v) in enumerate(rows):
    y = T + i * (bh + gap)
    local = i == 0
    b.append(text(L - 12, y + bh / 2 + 5, name, LABEL, "ink", "end", "600" if local else "normal"))
    b.append(f"<rect{' class=\"lb\"' if local else ''} x='{L}' y='{y}' width='{max(X(v) - L, 1):.1f}' height='{bh}' rx='3' "
             f"fill='var(--{'s1' if local else 'neutral'})'/>")
    b.append(text(X(v) + 6, y + bh / 2 + 5, f"€{v:.3f}", LABEL, "ink", "start", "600" if local else "normal")
             .replace("<text ", "<text class='lv' " if local else "<text ", 1))
b.append(text(0, H - 12, "Electricity only: 0.048 kWh measured (GPUs and CPU), no hardware. API costs as billed on "
              "16 Nov 2026.", SMALL, "muted"))
b.append("<script><![CDATA[(function(){var KWH=%s,PMAX=%s,SX0=%s,SX1=%s,L=%s,X1=%s,VMAX=%s;" % (KWH, PMAX, sx0, sx1, L, x1, VMAX) + """
var s=document.currentScript.closest('svg'),q=function(c){return s.querySelector('.'+c)},drag=false;
function set(e){var r=s.getBoundingClientRect(),vb=s.viewBox.baseVal,x=(e.clientX-r.left)*vb.width/r.width-8;
 var p=Math.max(0,Math.min(PMAX,(x-SX0)/(SX1-SX0)*PMAX));p=Math.round(p*100)/100;var sx=SX0+p/PMAX*(SX1-SX0);
 q('pk').setAttribute('cx',sx);var pl=q('pl');pl.setAttribute('x',sx);pl.textContent='\\u20ac'+p.toFixed(2)+'/kWh';
 var v=KWH*p,bx=L+Math.min(v,VMAX)/VMAX*(X1-L);q('lb').setAttribute('width',Math.max(bx-L,1));
 var lv=q('lv');lv.setAttribute('x',bx+6);lv.textContent='\\u20ac'+v.toFixed(3)}
var z=q('pz'),k=q('pk');[z,k].forEach(function(el){el.addEventListener('pointerdown',function(e){drag=true;
 try{el.setPointerCapture(e.pointerId)}catch(_){};set(e)});el.addEventListener('pointermove',function(e){if(drag)set(e)});
 el.addEventListener('pointerup',function(){drag=false})});})();]]></script>""")
desc = (f"At €{P0:.2f}/kWh the job's electricity costs €{KWH * P0:.3f}; APIs: "
        + ", ".join(f"{n} €{v:.3f}" for n, v in API) + ". At €0.05/kWh (solar surplus) it is €{:.4f}.".format(KWH * 0.05))
open(out, "w").write(doc(W, H, f"<g transform='translate({PAD},0)'>{''.join(b)}</g>", "Electricity price slider", desc))
print(desc)
