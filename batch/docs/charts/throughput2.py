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
H, L, Rm, T, B = 380, 46, 24, 64, 50
xmax = 512
def X(n): return L + (n / xmax) * (W - 2 * PAD - L - Rm)
vmax = max(p['gen_tok_s'] for _, _, s in series for p in s)
ymax, step = vmax * 1.15, 50
def Y(v): return H - B - v / ymax * (H - T - B)
b = [text(0, 20, "Output tokens per second, all requests together", TITLE, "ink", weight="600"),
     text(0, 39, "solid: while generating; dashed: whole job, prompts included. Four RX 9070 XT.", LABEL, "ink2")]
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
        tip = (f"{label}, {n_of(p)} at once: {p['gen_tok_s']:.0f} tok/s while generating, {p['whole_tok_s']:.0f} over the "
               f"whole job ({p['out_tokens']:,} tokens in {p['job_s']:.0f} s), {p.get('power_w')} W")
        b.append(f"<g><title>{tip}</title>{dot(X(n_of(p)), Y(p['gen_tok_s']), col)}"
                 f"<circle cx='{X(n_of(p)):.1f}' cy='{Y(p['gen_tok_s']):.1f}' r='12' fill='transparent'/></g>")
    last = s[-1]
    b.append(text(X(n_of(last)) - 9, Y(last['gen_tok_s']) - 9, f"{last['gen_tok_s']:.0f}", LABEL, "ink", "end", "600", HALO))
b.append(key([(lab, col, "line") for lab, col, s in series if s], L + 8, T - 8)[0])
desc = "; ".join(f"{lab}: " + ", ".join(f"{n_of(p)} at once {p['gen_tok_s']:.0f} tok/s generating, {p['whole_tok_s']:.0f} whole job"
                                         for p in s) for lab, _, s in series)
open(sys.argv[2], 'w').write(doc(W, H, f"<g transform='translate({PAD},0)'>{''.join(b)}</g>",
                                 "Output tokens per second by questions at once, two BIRD sets", desc))
print("wrote", sys.argv[2])
