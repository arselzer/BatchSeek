#!/usr/bin/env python3
"""Throughput by requests at once, one line per BIRD set, from h71_summary.py rows (standard library only, shared chart
style). y: output tokens per second while generating (output tokens / server generation time); a lighter dashed line per
set: the whole job (output tokens / first request to last reply). Every point has a hover title.
Usage: throughput2.py ROWS.jsonl OUT.svg"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart_style import W, TITLE, LABEL, SMALL, doc, text, hline, line, dot, key

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith('{')]
HALO = " paint-order='stroke' stroke='var(--surf)' stroke-width='3' stroke-linejoin='round'"
PAD = 8
SETS = [('bird', 'Original BIRD (SQL only)', 's2', {'bird8', 'bird64', 'bird256', 'bird500'}),
        ('birdhf', 'BIRD 2026 (reasoning, then SQL)', 's1', {'birdhf8', 'birdhf64', 'birdhf256', 'birdhf500'})]
series = []
for fam, label, col, us in SETS:
    pts = {}
    for r in rows:
        if r['u'] in us and (r['conc'] >= r['n'] or r['conc'] in (1, 8)) and not r.get('errors'):
            pts[min(r['conc'], r['n'])] = r
    series.append((label, col, [pts[k] for k in sorted(pts)]))
H, L, Rm, T, B = 398, 46, 24, 82, 50
xmax = 512
def X(n): return L + (n / xmax) * (W - 2 * PAD - L - Rm)
vmax = max(p['gen_tok_s'] for _, _, s in series for p in s)
ymax, step = vmax * 1.15, 50
def Y(v): return H - B - v / ymax * (H - T - B)
b = [text(0, 20, "Output tokens per second, all requests together", TITLE, "ink", weight="600"),
     text(0, 39, "Four RX 9070 XT, both BIRD versions. Hover or tap a point for its numbers.", LABEL, "ink2")]
v = 0.0
while v <= ymax:
    b.append(hline(L, W - 2 * PAD - Rm, Y(v), "axis" if v == 0 else "grid"))
    b.append(text(L - 7, Y(v) + 4, f"{v:,.0f}", SMALL, "muted", "end"))
    v += step
for n in (1, 64, 128, 256, 384, 500):
    b.append(text(X(n), H - B + 18, str(n), SMALL, "muted", "middle"))
b.append(text((L + W - 2 * PAD - Rm) / 2, H - 12, "questions at once", SMALL, "ink2", "middle"))
def n_of(p): return min(p['conc'], p['n'])
for label, col, s in series:
    if not s: continue
    b.append(line([(X(n_of(p)), Y(p['whole_tok_s'])) for p in s], col, 1.5, opacity=0.6, dash="5 4"))
    b.append(line([(X(n_of(p)), Y(p['gen_tok_s'])) for p in s], col, 2))
    for p in s:
        t1 = f"{label.split(' (')[0]}, {n_of(p)} at once"
        t2 = f"{p['gen_tok_s']:.0f} tok/s generating, {p['whole_tok_s']:.0f} over the whole job"
        t3 = f"{p['out_tokens']:,} tokens in {p['job_s']:.0f} s, {p.get('power_w')} W, {(f"{p['wh_per_1k']:.2f}" if p['wh_per_1k'] < 2 else f"{p['wh_per_1k']:.1f}")} Wh per 1k tokens"
        x, y = X(n_of(p)), Y(p['gen_tok_s'])
        b.append(f"<g class='pt' tabindex='0' data-x='{x:.1f}' data-y='{y:.1f}' data-t1=\"{t1}\" data-t2=\"{t2}\" data-t3=\"{t3}\" "
                 f"style='cursor:pointer;outline:none'><title>{t1}: {t2}; {t3}</title>{dot(x, y, col)}"
                 f"<circle cx='{x:.1f}' cy='{y:.1f}' r='14' fill='transparent'/></g>")
    last = s[-1]
    b.append(text(X(n_of(last)) - 9, Y(last['gen_tok_s']) - 9, f"{last['gen_tok_s']:.0f}", LABEL, "ink", "end", "600", HALO))
for i, (lab, col, sr) in enumerate([x for x in series if x[2]]):   # two columns, aligned with the style key below
    b.append(key([(lab, col, "line")], L + 8 + 190 * i, T - 26)[0])
b.append(key([("while generating", "ink2", "line")], L + 8, T - 8)[0])
b.append(key([("whole job, prompts included", "ink2", "dash")], L + 198, T - 8)[0])
# readout box, filled by the script below on hover, tap or keyboard focus
b.append("<g class='tip' visibility='hidden' pointer-events='none'><rect rx='6' fill='var(--surf)' stroke='var(--axis)'/>"
         + "".join(f"<text class='t{i}' font-size='{12 if i == 1 else 11}' fill='var(--{'ink' if i == 1 else 'ink2'})' "
                   f"font-weight='{600 if i == 1 else 400}'></text>" for i in (1, 2, 3)) + "</g>")
desc = "; ".join(f"{lab}: " + ", ".join(f"{n_of(p)} at once {p['gen_tok_s']:.0f} tok/s generating, {p['whole_tok_s']:.0f} whole job"
                                         for p in s) for lab, _, s in series)
script = """<script><![CDATA[(function(){var s=document.currentScript.closest('svg');if(!s)return;
var tip=s.querySelector('.tip'),r=tip.querySelector('rect'),t=[1,2,3].map(function(i){return tip.querySelector('.t'+i)});
var X1=%s;function show(g){var x=+g.dataset.x,y=+g.dataset.y;t[0].textContent=g.dataset.t1;t[1].textContent=g.dataset.t2;t[2].textContent=g.dataset.t3;
var w=Math.max.apply(null,t.map(function(e){return e.getComputedTextLength()}))+20,h=58,bx=x+14,by=y-h-10;
if(bx+w>X1)bx=x-14-w;if(by<4)by=y+14;r.setAttribute('x',bx);r.setAttribute('y',by);r.setAttribute('width',w);r.setAttribute('height',h);
t.forEach(function(e,i){e.setAttribute('x',bx+10);e.setAttribute('y',by+18+i*16)});tip.setAttribute('visibility','visible')}
function hide(){tip.setAttribute('visibility','hidden')}
s.querySelectorAll('.pt').forEach(function(g){g.addEventListener('mouseenter',function(){show(g)});g.addEventListener('mouseleave',hide);
g.addEventListener('focus',function(){show(g)});g.addEventListener('blur',hide);g.addEventListener('click',function(e){e.stopPropagation();show(g)})});
s.addEventListener('click',hide);})();]]></script>""" % (W - 2 * PAD - Rm)
open(sys.argv[2], 'w').write(doc(W, H, f"<g transform='translate({PAD},0)'>{''.join(b)}</g>" + script,
                                 "Output tokens per second by questions at once, two BIRD sets", desc))
print("wrote", sys.argv[2])
