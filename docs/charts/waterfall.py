#!/usr/bin/env python3
"""Per-request waterfall of a batch run: one row per request (sorted by first token), split into waiting for its
prompt to be read, reading the prompt (the server's prompt time), prompt read but waiting for decoding to start,
and decoding. Decoding starts for everyone at DECODE_START (the first reconstructed decode steps; this server reads
every prompt before decoding). Hover shows one request's times (inline use). Standard library only.
Usage: waterfall.py RESULT.json OUT.svg --decode-start 390"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart_style import W, TITLE, LABEL, SMALL, doc, text, key, nice_step

res, out = sys.argv[1], sys.argv[2]
d0 = float(sys.argv[sys.argv.index("--decode-start") + 1])
R = [r for r in json.load(open(res))["requests"] if r.get("timings")]
rows = []
for r in R:
    t = r["timings"]; ttft = t["ttft_ms"] / 1000; pm = t["prompt_ms"] / 1000
    rows.append((ttft - pm, ttft, max(ttft, d0), r["wall_s"], t["predicted_n"], t["prompt_n"] + (t.get("cache_n") or 0)))
rows.sort(key=lambda x: (x[1], x[3]))
span = max(x[3] for x in rows)
PAD, L, Rr, T, B = 8, 46, 14, 76, 44
gw = W - 2 * PAD - L - Rr
rh = 0.62                      # px per request row (512 rows -> ~320 px)
H = T + len(rows) * rh + B
def X(s): return L + s / span * gw

segs = {"wait": [], "read": [], "idle": [], "dec": []}
for i, (a, b, c, e, n, p) in enumerate(rows):
    y = T + i * rh
    for k, s0, s1 in (("wait", 0, a), ("read", a, b), ("idle", b, c), ("dec", c, e)):
        if s1 - s0 > 0.05:
            segs[k].append(f"M{X(s0):.1f} {y:.2f}h{X(s1) - X(s0):.2f}v{rh:.2f}h{X(s0) - X(s1):.2f}z")
style = {"wait": ("s3", 0.30), "read": ("s2", 0.95), "idle": ("neutral", 0.45), "dec": ("s1", 0.85)}
b = [text(0, 20, "512 requests, one line each", TITLE, "ink", weight="600"),
     text(0, 39, "every prompt is read first, then all requests decode together", LABEL, "ink2")]
k, _ = key([("waiting for its prompt to be read", "s3", "box"), ("reading the prompt", "s2", "box"),
            ("prompt read, waiting for decoding", "neutral", "box"), ("decoding", "s1", "box")], 0, 58)
b.append(k)
for kname, ds in segs.items():
    col, op = style[kname]
    b.append(f"<path d='{''.join(ds)}' fill='var(--{col})' fill-opacity='{op}' shape-rendering='crispEdges'/>")
st = nice_step(span, 6); s = 0.0
while s <= span + 1e-9:
    b.append(text(X(s), T + len(rows) * rh + 16, f"{s:,.0f}", SMALL, "muted", "middle")); s += st
b.append(text(L + gw / 2, H - 6, "seconds since all requests were sent", SMALL, "ink2", "middle"))
b.append(text(L - 8, T + 8, "first", SMALL, "muted", "end"))
b.append(text(L - 8, T + len(rows) * rh, "last", SMALL, "muted", "end"))
b.append(text(L - 30, T + len(rows) * rh / 2, "requests, by first token", SMALL, "ink2", "middle",
              extra=f" transform='rotate(-90 {L - 30} {T + len(rows) * rh / 2})'"))
# hover: one request's numbers
data = [[round(a, 1), round(b_, 1), round(c, 1), round(e, 1), n, p] for a, b_, c, e, n, p in rows]
b.append(f"<g class='wh' style='display:none;pointer-events:none'><rect class='whr' height='2' fill='var(--ink)'/>"
         f"<rect class='whb' rx='4' width='238' height='76' fill='var(--surf)' stroke='var(--grid)'/>"
         + "".join(f"<text class='wht{i}' font-size='{SMALL}' fill='var(--{'ink' if i == 0 else 'ink2'})'"
                   + (" font-weight='600'" if i == 0 else "") + "></text>" for i in range(4)) + "</g>"
         f"<rect class='whz' x='{L}' y='{T}' width='{gw:.1f}' height='{len(rows) * rh:.1f}' fill='transparent'/>")
b.append("<script><![CDATA[(function(){var D=" + json.dumps(data) + f",L={L},GW={gw:.1f},T={T},RH={rh},SPAN={span:.1f};" + """
var s=document.currentScript.closest('svg'),q=function(c){return s.querySelector('.'+c)};
function mv(e){var r=s.getBoundingClientRect(),vb=s.viewBox.baseVal,x=(e.clientX-r.left)*vb.width/r.width-8,
 y=(e.clientY-r.top)*vb.height/r.height,i=Math.max(0,Math.min(D.length-1,Math.floor((y-T)/RH))),d=D[i];
 q('wh').style.display='';var hr=q('whr');hr.setAttribute('x',L);hr.setAttribute('width',GW);hr.setAttribute('y',T+i*RH-0.7);
 var bx=x+14>L+GW-244?x-250:x+14,by=Math.min(T+i*RH+8,T+D.length*RH-80),b=q('whb');b.setAttribute('x',bx);b.setAttribute('y',by);
 var t=['request '+(i+1)+' of '+D.length+' ('+d[5].toLocaleString('en')+'-token prompt)',
  'prompt read '+Math.round(d[0])+'-'+Math.round(d[1])+' s','decoding '+Math.round(d[2])+'-'+Math.round(d[3])+' s',
  d[4]+' tokens generated'];
 for(var k=0;k<4;k++){var el=q('wht'+k);el.textContent=t[k];el.setAttribute('x',bx+10);el.setAttribute('y',by+18+k*15)}}
var z=q('whz');z.addEventListener('pointermove',mv);z.addEventListener('pointerdown',mv);
z.addEventListener('pointerleave',function(){q('wh').style.display='none'});})();]]></script>""")
desc = (f"{len(rows)} requests sent at once. Prompts are read one after another from about 0 to "
        f"{max(x[1] for x in rows):.0f} s; decoding starts for all at about {d0:.0f} s and the last reply ends at {span:.0f} s.")
open(out, "w").write(doc(W, H, f"<g transform='translate({PAD},0)'>{''.join(b)}</g>", "Per-request waterfall", desc))
print(desc, os.path.getsize(out) // 1024, "KB")
