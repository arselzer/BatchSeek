#!/usr/bin/env python3
"""Run timeline: one serving run ("arm"), or two side by side, as stacked panels on one time axis (seconds since the
run's first request). Standard library only; writes a static SVG (no script) in the post's chart style
(chart_style.py: shared tokens, type scale, light/dark).

Usage:
  timeline.py --result R.json --window-log W.log --power P.tsv [--stats S.jsonl] --out OUT.svg
              [--title T] [--subtitle S] [--label "fixed 512"] [--ref 512] [--gpus 83:00,48:00,...]
              [--compare R2.json --compare-label "sustained 512"
               [--compare-window-log W2.log] [--compare-power P2.tsv] [--compare-stats S2.jsonl]]
  With --compare the two runs are drawn as two columns (small multiples) on the same time scale and the same y
  scales, so lengths and levels compare directly. The compare inputs default to the first run's files (same window).

Panels (each column):
  phase strip (with --steps): "reading prompts" / "decoding", from the reconstructed decode-step rate (a 5 s cell is
    decoding when steps/s > 0.05; gaps under 15 s inside decoding are merged).
  1 requests in flight, stacked and labelled in place (no legend): "waiting for prompt" (queued or prefilling: no
    first token yet), "prompt read, waiting to decode" (first token received, no decode steps running; with --steps)
    and "decoding" (first token received while decode steps run). Without --steps the lower band is "first token
    received".
  2 output tokens/s while decoding (with --steps): decode tokens/s reconstructed by decode_steps.py from per-request
    timings (run it with a numpy/scipy python: venv-vision/bin/python -I decode_steps.py RESULT.json --grid 5
    --out STEPS.json); faint 5 s cells, 25 s centred mean, a value-only label at the top of the curve,
    the server's average while decoding (output tokens / stats.step_ns_decode) as a dashed line over the decoding
    phase, and the --tps-ref line (default 300). The server's /stats decode_tps is not used: it reads 0 here.
  (--decode-batch) tokens per decode step = requests decoding, with time-weighted mean and full batch.
  3 KV cache memory in use, % of the pool (kv_util; only with --stats).
  4 power: GPU total (selected cards) and CPU package; a 10 s rolling mean over the faint 1 s trace.
Alignment: the request phase ends at the window log's "arm NAME: requests exit" line and starts max(wall_s) earlier.
"""
import argparse, bisect, datetime as dt, json, math, os, re, sys
from xml.sax.saxutils import escape

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart_style import (W, TITLE, HEAD, LABEL, SMALL, doc, text, text_w, hline, line, area, dot, key, nice_max,
                         nice_step)


def iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


# ---------------------------------------------------------------- data
def load_run(path):
    d = json.load(open(path))
    iv = []
    for r in d["requests"]:
        t = r.get("timings") or {}
        if not t or r.get("error"):
            continue
        s = float(r.get("sent_s") or 0.0)
        w = float(r["wall_s"])
        ttft = min(max(float(t.get("ttft_ms") or t.get("prompt_ms") or 0) / 1000, 0.0), w)
        iv.append((s, s + ttft, s + w, int(t.get("predicted_n") or 0)))
    t0 = min(a for a, *_ in iv)
    iv = [(a - t0, b - t0, c - t0, n) for a, b, c, n in iv]
    st = d.get("stats") or {}
    toks = sum(x[3] for x in iv)
    # each request's first token comes from prefill, so decode-step time produced toks - (requests with a token)
    dtoks = toks - sum(1 for x in iv if x[3] > 0)
    step_mean = dtoks / (st["step_ns_decode"] / 1e9) if st.get("step_ns_decode") else None
    return dict(iv=iv, span=max(c for _, _, c, _ in iv), tokens=toks, step_mean=step_mean, stats=st,
                n=len(iv), conc=d.get("concurrency"), refill=any(r.get("sent_s") is not None for r in d["requests"]))


def phase_bounds(window_log, arm, max_wall):
    end = None
    for line in open(window_log, errors="replace"):
        m = re.match(r"\[(\S+)\] arm (\S+): requests exit", line)
        if m and m.group(2) == arm:
            end = iso(m.group(1))
    if end is None:
        sys.exit(f"no 'arm {arm}: requests exit' line in {window_log}")
    return end - max_wall, end


def arm_name(result_path):
    b = os.path.basename(result_path)
    for suf in (".unmeasured.json", ".json"):
        if b.endswith(suf):
            return b[: -len(suf)]
    return b


def load_power(path, start, end, gpus):
    """Rows within [start, end] (unix); HH:MM:SS is placed on the date nearest the phase."""
    rows = [l.rstrip("\n").split("\t") for l in open(path)]
    hdr, rows = rows[0], rows[1:]
    gcols = [i for i, h in enumerate(hdr) if h.startswith("gpu_") and h.endswith("_W")]
    cpu = hdr.index("cpu_pkg_W") if "cpu_pkg_W" in hdr else None
    base = dt.datetime.fromtimestamp(start, dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    for r in rows:
        try:
            h, m, s = (int(x) for x in r[0].split(":"))
        except ValueError:
            continue
        t = base.timestamp() + h * 3600 + m * 60 + s
        if t < start - 43200: t += 86400        # crossed midnight after the phase's date
        if t > end + 43200: t -= 86400
        if start - 2 <= t <= end + 2:
            vals = {hdr[i][4:-2]: float(r[i] or 0) for i in gcols}
            out.append((t - start, vals, float(r[cpu]) if cpu is not None and r[cpu] else None))
    if not out:
        sys.exit(f"no power rows inside the request phase in {path}")
    cards = sorted(out[0][1])
    if gpus:
        sel = [c for c in cards if any(g in c for g in gpus)]
    else:   # cards that did work during the phase; an idle card (another job's or spare) is left out
        sel = [c for c in cards if max(v[c] for _, v, _ in out) >= 60]
    return [(t, sum(v[c] for c in sel), p) for t, v, p in out], sel, [c for c in cards if c not in sel]


def load_stats(path, start, end):
    out = []
    for l in open(path):
        try:
            o = json.loads(l)
        except ValueError:
            continue
        t = float(o["t"]) - start
        if -3 <= t <= end - start + 3:
            out.append((t, o))
    return out


def hover_layer(r, c, xspan, y0, y1, step=2.0):
    """Crosshair + readout over every panel (inline use only; an <img> ignores the script). Series are sampled every
    `step` seconds (nearest earlier sample) and embedded as JSON."""
    def sample(pts, idx=1):
        out, j = [], 0
        k = 0
        while k * step <= xspan:
            t = k * step
            while j + 1 < len(pts) and pts[j + 1][0] <= t:
                j += 1
            out.append(round(pts[j][idx]) if pts and pts[0][0] <= t else None)
            k += 1
        return out
    occ = r["occ"]
    ser = {"dec": sample([(t, d) for t, d, _, _ in occ]),
           "wait": sample([(t, w or 0) for t, _, _, w in occ]),
           "pre": sample([(t, p - (w or 0)) for t, _, p, w in occ]),
           "tps": sample(r["tps_s"]) if r.get("tps_s") else [],
           "kv": sample([(t, u) for t, u, _ in r["kv"]]) if r["kv"] else [],
           "gpu": sample(r["gpu_s"]), "cpu": sample(r["cpu_s"]) if r["cpu_s"] else []}
    g = (f"<g class='th' style='display:none;pointer-events:none'><line class='thl' y1='{y0:.1f}' y2='{y1:.1f}' "
         f"stroke='var(--ink2)' stroke-width='1' stroke-dasharray='3 3'/><rect class='thb' rx='4' width='206' "
         f"height='114' fill='var(--surf)' stroke='var(--grid)'/>"
         + "".join(f"<text class='tht{i}' font-size='{SMALL}' fill='var({'--ink' if i == 0 else '--ink2'})'"
                   f"{' font-weight="600"' if i == 0 else ''}></text>" for i in range(7)) + "</g>"
         f"<rect class='thz' x='{c.x0:.1f}' y='{y0:.1f}' width='{c.x1 - c.x0:.1f}' height='{y1 - y0:.1f}' fill='transparent'/>")
    js = ("<script><![CDATA[(function(){var S=" + json.dumps(ser) + f",ST={step},X0={c.x0:.1f},X1={c.x1:.1f},SPAN={xspan:.1f},"
          f"Y0={y0:.1f};" + """
var s=document.currentScript.closest('svg'),q=function(c){return s.querySelector('.'+c)};
function f(v){return v==null?'-':Math.round(v).toLocaleString('en')}
function mv(e){var r=s.getBoundingClientRect(),vb=s.viewBox.baseVal,x=(e.clientX-r.left)*vb.width/r.width-8;
 var u=Math.max(0,Math.min(1,(x-X0)/(X1-X0))),t=u*SPAN,i=Math.min(Math.round(t/ST),S.dec.length-1),px=X0+u*(X1-X0);
 q('th').style.display='';q('thl').setAttribute('x1',px);q('thl').setAttribute('x2',px);
 var bx=px+12>X1-210?px-218:px+12,by=Y0+40,b=q('thb');b.setAttribute('x',bx);b.setAttribute('y',by);
 var L=[Math.round(t)+' s',
  'waiting '+f(S.wait[i])+', processing prompt '+f(S.pre[i]),'generating '+f(S.dec[i]),
  S.tps.length?'output \u2248'+f(S.tps[i])+' tokens/s':'',S.kv.length?'KV cache '+f(S.kv[i]/1000)+'k tokens':'',
  'GPUs '+f(S.gpu[i])+' W',S.cpu.length?'CPU '+f(S.cpu[i])+' W':''].filter(function(z){return z});
 for(var k=0;k<7;k++){var el=q('tht'+k);el.textContent=L[k]||'';el.setAttribute('x',bx+10);el.setAttribute('y',by+17+k*14.5)}
 b.setAttribute('height',14+L.length*14.5)}
var z=q('thz');z.addEventListener('pointermove',mv);z.addEventListener('pointerdown',mv);
z.addEventListener('pointerleave',function(){q('th').style.display='none'});})();]]></script>""")
    return g + js


def occupancy(iv, span, samples, step=1.0):
    pts = []
    ss = [t for t, _ in samples]
    k = 0
    while k * step <= span:
        t = k * step
        pre = sum(1 for a, b, c, _ in iv if a <= t < b)
        dec = sum(1 for a, b, c, _ in iv if b <= t < c)
        wait = None
        if samples:
            j = min(max(bisect.bisect_left(ss, t), 0), len(ss) - 1)
            if j > 0 and abs(ss[j - 1] - t) < abs(ss[j] - t): j -= 1
            if abs(ss[j] - t) <= 5:
                wait = min(pre, int(samples[j][1].get("waiting") or 0))
        pts.append((t, dec, pre, wait))
        k += 1
    return pts



def rolling(pts, half=5.0):
    """Centred rolling mean over +-half seconds; pts sorted by t."""
    ts = [t for t, _ in pts]
    out, acc = [], [0.0]
    for _, v in pts:
        acc.append(acc[-1] + v)
    for t, _ in pts:
        i, j = bisect.bisect_left(ts, t - half), bisect.bisect_right(ts, t + half)
        out.append((t, (acc[j] - acc[i]) / (j - i)))
    return out


def prepare(result, window_log, power, stats, arm, gpus, steps=None):
    run = load_run(result)
    start, end = phase_bounds(window_log, arm or arm_name(result), run["span"])
    samples = load_stats(stats, start, end) if stats and os.path.exists(stats) else []
    pw, cards, idle = load_power(power, start, end, gpus)
    span = run["span"]
    occ = occupancy(run["iv"], span, samples)
    dec = [(t, d) for t, d, _, _ in occ]
    active = [d for _, d in dec if d > 0]
    gpu = [(min(max(t, 0), span), g) for t, g, _ in pw]
    cpu = [(min(max(t, 0), span), c) for t, _, c in pw if c is not None]
    # KV cache by memory (kv_util: the share of the pool's bytes in use), in %: each request also holds a fixed block of
    # state besides its tokens, so the token count alone understates how full the pool is
    kv = [(min(max(t, 0), span), 100.0 * o["kv_util"], 100.0) for t, o in samples if o.get("kv_util") is not None]
    peak = max(d for _, d in dec)
    return dict(run=run, start=start, end=end, span=span, occ=occ, dec=dec, samples=samples,
                have_wait=any(w is not None for *_, w in occ), peak=peak,
                peak_t=next(t for t, d in dec if d == peak), mean=sum(active) / len(active) if active else 0,
                gpu=gpu, cpu=cpu, gpu_s=rolling(gpu), cpu_s=rolling(cpu), cards=cards, idle=idle, kv=kv,
                pool=max((k or 0) for *_, k in kv) if kv else None,
                tok_per_step=(run["tokens"] / run["stats"]["steps_decode"]) if run["stats"].get("steps_decode") else None,
                **tps_series(steps, run))


def tps_series(steps_path, run, smooth_cells=3):
    """Decode tokens/s over time from decode_steps.py's output (--steps): per-cell values plus a centred rolling
    mean over smooth_cells cells. Empty when no steps file is given."""
    if not steps_path:
        return dict(tps=[], tps_s=[], tps_check=None, phases=[], dec_active=[])
    d = json.load(open(steps_path))
    cells = list(zip(d["t"], d["tok_s"], d["steps_per_s"]))
    ndec = dict(zip(d["t"], d["decoding"]))
    g = (cells[1][0] - cells[0][0]) if len(cells) > 1 else 5.0
    tps = [(t, v) for t, v, _ in cells]
    k = smooth_cells // 2
    tps_s = [(tps[i][0], sum(v for _, v in tps[max(0, i - k):i + k + 1]) / len(tps[max(0, i - k):i + k + 1]))
             for i in range(len(tps))]
    active = [v for _, v, sps in cells if sps > 0.05]   # cells in which decode steps run (> 1 step per 20 s)
    # phases: runs of decode-active / inactive cells; inactive gaps under 15 s inside decoding are merged into it
    segs = []
    for t, _, sps in cells:
        st = sps > 0.05
        a0, a1 = max(0.0, t - g / 2), min(run["span"], t + g / 2)
        if segs and segs[-1][2] == st:
            segs[-1][1] = a1
        else:
            segs.append([a0, a1, st])
    merged = []
    for sg in segs:
        if merged and not sg[2] and sg[1] - sg[0] < 15 and sg is not segs[-1]:
            merged[-1][1] = sg[1]
        elif merged and merged[-1][2] == sg[2]:
            merged[-1][1] = sg[1]
        else:
            merged.append(list(sg))
    pk = max(tps, key=lambda x: x[1])
    srv = run["stats"].get("steps_decode")
    rec = (d.get("check") or {}).get("decode_steps_total")
    return dict(tps=tps, tps_s=tps_s, tps_grid=g, tps_smooth=g * (2 * k + 1),
                tps_mean=sum(active) / len(active) if active else 0, phases=merged,
                peak_dec=ndec.get(pk[0]), dec_active=[(t, sps > 0.05) for t, _, sps in cells],
                tps_check=(abs(rec - srv) / srv * 100) if srv and rec else None)


HALO = " paint-order='stroke' stroke='var(--surf)' stroke-width='3' stroke-linejoin='round'"


def caption(items, x, y, width):
    """Flow key entries ((label, colour, kind)) and notes (plain str) left to right, wrapping at width.
    Returns (svg, lines used)."""
    out, cx, ln = [], 0.0, 0
    for it in items:
        if isinstance(it, str):
            w = text_w(it, SMALL)
            if cx > 0 and cx + w > width:
                ln, cx = ln + 1, 0.0
            out.append(text(x + cx, y + ln * 15, it, SMALL, "muted"))
        else:
            s, w = key([it], x + cx, y + ln * 15)
            if cx > 0 and cx + w > width:
                ln, cx = ln + 1, 0.0
                s, w = key([it], x, y + ln * 15)
            out.append(s)
        cx += w + 16
    return "".join(out), ln + 1


class Col:
    def __init__(self, x0, x1, xspan):
        self.x0, self.x1, self.xspan = x0, x1, xspan

    def X(self, t):
        return self.x0 + t / self.xspan * (self.x1 - self.x0)


def yaxis(c, top, h, ymax, step, fmt, ticks=True):
    b = []
    v = 0.0
    while v <= ymax + 1e-9:
        y = top + h - v / ymax * h
        b.append(hline(c.x0, c.x1, y, "axis" if v == 0 else "grid"))
        if ticks:
            b.append(text(c.x0 - 7, y + 4, fmt(v), SMALL, "muted", "end"))
        v += step
    return b


def band_labels(c, layers, Y):
    """Label each band in place, at its visual middle: x = the band's thickness-weighted centroid, y = the middle of
    the band there. If the text box does not fit there, try the thickest spot where it fits; else leave it to the key."""
    b, unlabelled = [], []
    for lab, col, up, lo in layers:
        xs = [c.X(t) for t, _ in up]
        th = [max(0.0, Y(l) - Y(u)) for (_, u), (_, l) in zip(up, lo)]
        half = (text_w(lab, SMALL) + 12) / 2

        def fit(xc):
            if xc - half < c.x0 or xc + half > c.x1:
                return None
            j0, j1 = bisect.bisect_left(xs, xc - half), bisect.bisect_right(xs, xc + half)
            if j1 <= j0:
                return None
            top_ = max(Y(v) for _, v in up[j0:j1])
            bot_ = min(Y(v) for _, v in lo[j0:j1])
            if bot_ - top_ < 17:
                return None
            k = min(bisect.bisect_left(xs, xc), len(xs) - 1)
            yc = (Y(up[k][1]) + Y(lo[k][1])) / 2
            return bot_ - top_, xc, min(max(yc, top_ + 9), bot_ - 9)

        tot = sum(th)
        best = None
        if tot > 0:
            xc = sum(x * w for x, w in zip(xs, th)) / tot
            xc = min(max(xc, c.x0 + half), c.x1 - half)
            best = fit(xc)
        if best is None:
            for xc in xs[::max(1, len(xs) // 200)]:
                f = fit(xc)
                if f and (best is None or f[0] > best[0]):
                    best = f
        if best:
            b.append(text(best[1], best[2] + 4, lab, SMALL, "ink", "middle", "600"))
        else:
            unlabelled.append((lab, col, "box"))
    return b, unlabelled


def render(a):
    runs = [(a.label, prepare(a.result, a.window_log, a.power, a.stats, a.arm, a.gpus, a.steps))]
    if a.compare:
        runs.append((a.compare_label, prepare(a.compare, a.compare_window_log or a.window_log,
                                              a.compare_power or a.power, a.compare_stats or a.stats,
                                              a.compare_arm, a.gpus, a.compare_steps)))
    two = len(runs) == 2
    xspan = max(r["span"] for _, r in runs)
    L, Rpad, GAPC = 46, 14, 34
    cw = (W - L - Rpad - (GAPC if two else 0)) / len(runs)
    cols = [Col(L + i * (cw + GAPC), L + i * (cw + GAPC) + cw, xspan) for i in range(len(runs))]
    PH = 104
    body = [text(0, 20, a.title or "Run timeline", TITLE, "ink", weight="600")]
    y = 20
    if a.subtitle:
        y += 19
        body.append(text(0, y, a.subtitle, LABEL, "ink2"))
    y += 30
    if two:
        for (lab, r), c in zip(runs, cols):
            body.append(text(c.x0, y, lab, 13, "ink", weight="600"))
            mode = "pool refill" if r["run"]["refill"] else "all sent at once"
            body.append(text(c.x0 + text_w(lab, 13) + 10, y, f"{r['run']['n']} requests, {mode}, {r['span']:.0f} s",
                             SMALL, "muted"))
        y += 26

    def panel(title, draw, captions):
        """captions: per-column caption item lists. draw(col, run, top) -> svg list."""
        nonlocal y
        body.append(text(0 if not two else 0, y, title, HEAD, "ink", weight="600"))
        y += 17
        lines = 1
        caps = []
        for (lab, r), c, items in zip(runs, cols, captions):
            s, n = caption(items, c.x0, y, c.x1 - c.x0)
            caps.append(s)
            lines = max(lines, n if items else 0)
        body.extend(caps)
        top = y + (lines * 15 if any(captions) else 0) + 6
        for (lab, r), c in zip(runs, cols):
            body.extend(draw(c, r, top))
        y = top + PH + 34
        return top

    y_top = y - 16
    # phase strip: reading prompts / decoding, from the reconstructed step rate (needs --steps)
    if any(r["phases"] for _, r in runs):
        for (_, r), c in zip(runs, cols):
            for a0, a1, st in r["phases"]:
                x0, x1 = c.X(a0), c.X(a1)
                col, op = ("s1", 0.30) if st else ("neutral", 0.30)
                body.append(f"<rect x='{x0:.1f}' y='{y-13:.1f}' width='{max(x1-x0-1.5, 1):.1f}' height='19' rx='3' "
                            f"fill='var(--{col})' fill-opacity='{op}'/>")
                name = "generating" if st else "processing prompts"
                if text_w(name, SMALL) + 10 < x1 - x0:
                    body.append(text((x0 + x1) / 2, y + 1, name, SMALL, "ink", "middle", "600"))
        y += 30

    # 1 requests in flight
    ymax1, st1 = nice_max(max(d + p for _, r in runs for _, d, p, _ in r["occ"]) * 1.08, 3)
    layers_by = []
    for _, r in runs:
        occ = r["occ"]
        dec = [(t, d) for t, d, _, _ in occ]
        tot = [(t, d + p) for t, d, p, _ in occ]
        zero = [(t, 0) for t, *_ in occ]
        if r["dec_active"]:   # split "after first token" by whether decode steps run (reconstructed step rate)
            ts = [t for t, _ in r["dec_active"]]
            g = r["tps_grid"]
            def act(t):
                i = min(max(bisect.bisect_left(ts, t - g / 2), 0), len(ts) - 1)
                return r["dec_active"][i][1]
            base = [("generating", "s1", [(t, v if act(t) else 0) for t, v in dec], zero),
                    ("prompt processed, waiting", "neutral", [(t, 0 if act(t) else v) for t, v in dec], zero)]
        else:
            base = [("first token received", "s1", dec, zero)]
        # queued and prefilling together: requests whose prompt is not read yet
        layers_by.append(base + [("waiting for prompt", "s3", tot, dec)])
    lab1 = []
    def draw1(c, r, top):
        Y = lambda v: top + PH - v / ymax1 * PH
        layers = layers_by[[rr for _, rr in runs].index(r)]
        b = yaxis(c, top, PH, ymax1, st1, lambda v: f"{v:,.0f}", c is cols[0])
        for lab, col, up, lo in layers:
            b.append(area([(c.X(t), Y(v)) for t, v in up], [(c.X(t), Y(v)) for t, v in lo], col, 0.30))
        for i, (lab, col, up, lo) in enumerate(layers):   # edges: tops of the bottom bands and of the outer band
            if i == len(layers) - 1 or all(v == 0 for _, v in lo):
                seg = []
                for t, v in up + [(None, 0)]:   # draw only where the band is present
                    if t is not None and v > 0:
                        seg.append((c.X(t), Y(v)))
                    elif seg:
                        b.append(line(seg, col, 1.5)); seg = []
        lb, _ = band_labels(c, layers, Y)
        return b + lb
    caps1 = []
    for (_, r), c, layers in zip(runs, cols, layers_by):
        # key lists only bands that cannot carry an in-place label
        _, unl = band_labels(c, layers, lambda v, top=0: PH - v / ymax1 * PH)
        caps1.append(unl)
    panel("Requests in flight", draw1, caps1)

    # 2b decode tokens/s reconstructed from per-request timings (decode_steps.py output via --steps)
    if any(r["tps"] for _, r in runs):
        meas = [float(x) for x in a.measured.split(",")] if a.measured else None   # t0,t1,tok/s from the server's KV counter
        ymaxt, stt = nice_max(max([v for _, r in runs for _, v in r["tps"]] + [a.tps_ref or 0] + ([meas[2]] if meas else [])) * 1.15, 3)
        def drawt(c, r, top):
            Y = lambda v: top + PH - min(v, ymaxt) / ymaxt * PH
            b = yaxis(c, top, PH, ymaxt, stt, lambda v: f"{v:,.0f}", c is cols[0])
            if not r["tps"]:
                return b + [text((c.x0 + c.x1) / 2, top + PH / 2, "no --steps file for this run", SMALL, "muted", "middle")]
            g = r["tps_grid"]
            stp = []
            for t, v in r["tps"]:   # 5 s cells as a step line
                stp += [(c.X(max(0, t - g / 2)), Y(v)), (c.X(min(xspan, t + g / 2)), Y(v))]
            b.append(area(stp, [(stp[0][0], Y(0)), (stp[-1][0], Y(0))], "s1", 0.08))
            b.append(line(stp, "s1", 1, opacity=0.35))
            b.append(line([(c.X(t), Y(v)) for t, v in r["tps_s"]], "s1", 2))
            pt, pv = max(r["tps"], key=lambda x: x[1])
            px, py = c.X(pt), Y(pv)
            plab = f"\u2248{pv:,.0f} tok/s"   # value only, beside the top of the curve
            pw = text_w(plab, SMALL)
            right = px + 8 + pw < c.x1
            pxl = px + 8 if right else px - 8
            boxes = []   # no peak label: the peak of a reconstruction depends on its grid (2-20 s cells: 345-538)
            if a.tps_ref:
                yr = Y(a.tps_ref)
                b.append(hline(c.x0, c.x1, yr, "muted", 1, "2 3"))
                rl = f"{a.tps_ref:g} tok/s"
                rw = text_w(rl, SMALL)
                rb = (c.x1 - 2 - rw, yr - 17, rw, 14)
                if any(not (rb[0] + rb[2] < o[0] or o[0] + o[2] < rb[0] or rb[1] + rb[3] < o[1] or o[1] + o[3] < rb[1])
                       for o in boxes):
                    rb = (c.x0 + 4, yr - 17, rw, 14)
                boxes += [rb, (c.x0, yr - 2, c.x1 - c.x0, 4)]   # the label and the dotted line itself
                b.append(text(rb[0], rb[1] + 11, rl, SMALL, "muted", extra=HALO))
            sm_ = r["run"]["step_mean"] or r["tps_mean"]   # server figure: output tokens / decode-step time
            dsegs = [sg for sg in r["phases"] if sg[2]]
            d0, d1 = (dsegs[0][0], dsegs[-1][1]) if dsegs else (0, r["span"])
            ym = Y(sm_)
            b.append(hline(c.X(d0), c.X(d1), ym, "ink2", 1.25, "6 4"))
            ml = f"{sm_:,.0f} tok/s average while generating"
            mw = text_w(ml, SMALL)
            for bx in [(c.X(d0) - 6 - mw, ym - 7, mw, 14), (c.X(d0) - 6 - mw, ym + 1, mw, 14), (c.x0 + 4, ym - 17, mw, 14), (c.x0 + 4, ym + 3, mw, 14),
                       (c.x1 - 4 - mw, ym + 3, mw, 14)]:
                if all(bx[0] + bx[2] < o[0] or o[0] + o[2] < bx[0] or bx[1] + bx[3] < o[1] or o[1] + o[3] < bx[1]
                       for o in boxes) and bx[1] + bx[3] < top + PH and bx[0] >= 0:
                    break
            b.append(text(bx[0], bx[1] + 11, ml, SMALL, "ink2", weight="600", extra=HALO))
            if meas and c is cols[0]:   # measured: all requests generating, none finished (KV-cache growth)
                mx0, mx1, my = c.X(meas[0]), c.X(meas[1]), Y(meas[2])
                b.append(f"<line x1='{mx0:.1f}' x2='{mx1:.1f}' y1='{my:.1f}' y2='{my:.1f}' stroke='var(--ink)' stroke-width='3' stroke-linecap='round'/>")
                ml2 = f"measured: {meas[2]:,.0f} tok/s"
                b.append(text(mx1 + 6, my + 4, ml2, SMALL, "ink", "start", "600", extra=HALO))
            return b
        capst = []
        for _, r in runs:
            if not r["tps"]:
                capst.append([]); continue
            it = [(f"{r['tps_smooth']:.0f} s mean", "s1", "line"), f"faint: {r['tps_grid']:.0f} s cells"]
            it.append("reconstructed from per-request timings;")
            if r["tps_check"] is not None:
                it.append(f"matches the server's step count within {r['tps_check']:.1f}%")
            capst.append(it)
        panel("Output tokens/s while generating", drawt, capst)

    if a.decode_batch:
        # 2 tokens per decode step
        ref = a.ref if a.ref is not None else runs[0][1]["run"]["conc"]
        ymax2, st2 = nice_max(max([d for _, r in runs for _, d in r["dec"]] + [ref or 0]) * 1.12, 3)
        def draw2(c, r, top):
            Y = lambda v: top + PH - v / ymax2 * PH
            b = yaxis(c, top, PH, ymax2, st2, lambda v: f"{v:,.0f}", c is cols[0])
            pts = []
            for i, (t, v) in enumerate(r["dec"]):
                if i: pts.append((c.X(t), Y(r["dec"][i - 1][1])))
                pts.append((c.X(t), Y(v)))
            b.append(area(pts, [(c.X(r["span"]), Y(0)), (c.X(0), Y(0))][::-1], "s1", 0.10))
            b.append(line(pts, "s1", 2))
            px, py = c.X(r["peak_t"]), Y(r["peak"])
            refw = text_w(f"full batch {ref:g}", SMALL)
            if ref:
                yr = Y(ref)
                b.append(hline(c.x0, c.x1, yr, "muted", 1, "2 3"))
                rx_left = px > c.x0 + refw + 50
                b.append(text(c.x0 + 4 if rx_left else c.x1 - 2, yr - 5, f"full batch {ref:g}", SMALL, "muted",
                              "start" if rx_left else "end", extra=HALO))
            ym = Y(r["mean"])
            b.append(hline(c.x0, c.X(r["span"]), ym, "ink2", 1.25, "6 4"))
            near_ref = ref and abs(ym - Y(ref)) < 18   # keep the mean label clear of the full-batch label
            b.append(text(c.x0 + 4, ym + 14 if near_ref else ym - 5, f"mean {r['mean']:.0f}", SMALL, "ink2", weight="600",
                          extra=HALO))
            for k in range(0, len(r["dec"]), 10):   # native tooltips per 10 s; no script
                ch = r["dec"][k:k + 10]
                t0_, t1_ = ch[0][0], min(ch[-1][0] + 1, r["span"])
                b.append(f"<rect x='{c.X(t0_):.1f}' y='{top}' width='{max(c.X(t1_)-c.X(t0_), 1):.1f}' height='{PH}' "
                         f"fill='transparent'><title>{t0_:.0f}-{t1_:.0f} s: {min(v for _, v in ch)}-"
                         f"{max(v for _, v in ch)} requests decoding</title></rect>")
            return b
        caps2 = []
        for _, r in runs:
            it = []
            if r["run"]["step_mean"]:
                it.append(f"decode steps ran at {r['run']['step_mean']:.0f} tok/s on average")
            if a.show_run_average:
                it.append(f"whole run {r['run']['tokens'] / r['span']:.0f} tok/s")
            it.append("mean: time-weighted while any request decodes")
            caps2.append(it)
        panel("Tokens per decode step (requests decoding)", draw2, caps2)

    # 3 KV cache
    if any(r["kv"] for _, r in runs):
        ymax3, st3 = 100, 25
        def draw3(c, r, top):
            Y = lambda v: top + PH - v / ymax3 * PH
            b = yaxis(c, top, PH, ymax3, st3, lambda v: f"{v:.0f}%", c is cols[0])
            if not r["kv"]:
                return b + [text((c.x0 + c.x1) / 2, top + PH / 2, "no /stats samples for this run", SMALL, "muted", "middle")]
            pts = [(c.X(t), Y(u)) for t, u, _ in r["kv"]]
            b.append(area(pts, [(pts[0][0], Y(0)), (pts[-1][0], Y(0))], "s1", 0.10))
            b.append(line(pts, "s1", 2))
            return b
        panel("KV cache memory (% of the pool)", draw3, [[] for _ in runs])

    # 4 power
    pmax = max(max(v for _, v in r["gpu"]) for _, r in runs)
    ymax4, st4 = nice_max(pmax * 1.04, 3)
    def draw4(c, r, top):
        Y = lambda v: top + PH - v / ymax4 * PH
        b = yaxis(c, top, PH, ymax4, st4, lambda v: f"{v:,.0f}", c is cols[0])
        for raw, sm, col in ((r["cpu"], r["cpu_s"], "s2"), (r["gpu"], r["gpu_s"], "s1")):
            if raw:
                b.append(line([(c.X(t), Y(v)) for t, v in raw], col, 1, opacity=0.22))
                b.append(line([(c.X(t), Y(v)) for t, v in sm], col, 2))
        # x axis for the last panel
        return b
    caps4 = []
    for _, r in runs:
        gm = sum(v for _, v in r["gpu"]) / len(r["gpu"])
        cm = sum(v for _, v in r["cpu"]) / max(1, len(r["cpu"]))
        caps4.append([(f"{len(r['cards'])} GPUs (avg {gm:,.0f} W)", "s1", "line"),
                      (f"CPU (avg {cm:,.0f} W)", "s2", "line"), "10 s mean over 1 s samples"])
    top4 = panel("Power (W)", draw4, caps4)
    bottom = top4 + PH
    for c in cols:
        step = nice_step(xspan, 6 if two else 8)
        t = 0.0
        while t <= xspan + 1e-9:
            body.append(text(c.X(t), bottom + 16, f"{t:,.0f}", SMALL, "muted", "middle"))
            t += step
        body.append(text((c.x0 + c.x1) / 2, bottom + 34, "seconds since the run's first request", SMALL, "ink2", "middle"))
    H = bottom + 44
    if not two:
        body.append(hover_layer(runs[0][1], cols[0], xspan, y_top, bottom))

    descs = []
    for lab, r in runs:
        rr = r["run"]
        gm = sum(v for _, v in r["gpu"]) / len(r["gpu"])
        cm = sum(v for _, v in r["cpu"]) / max(1, len(r["cpu"]))
        descs.append(
            (f"{lab}: " if two else "") +
            f"{rr['n']} requests ({'pool refill' if rr['refill'] else 'all sent at t=0'}), {rr['tokens']:,} output tokens, "
            f"request phase {r['span']:.0f} s ending {dt.datetime.fromtimestamp(r['end'], dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC. "
            f"Decode batch peak {r['peak']} at {r['peak_t']:.0f} s, time-weighted mean {r['mean']:.0f} while decoding"
            + (f", {r['tok_per_step']:.0f} tokens per decode step" if r["tok_per_step"] else "") +
            f". Decode-step mean {rr['step_mean'] or 0:.0f} tok/s; whole-run average {rr['tokens'] / r['span']:.0f} tok/s; "
            + ". "
            f"GPU power avg {gm:.0f} W over the {len(r['cards'])} GPUs; CPU avg {cm:.0f} W.")
    desc = " ".join(descs)
    # pad the viewBox left so the y tick labels and panel titles share one left edge
    svg = doc(W, H, f"<g transform='translate(8,0)'>{''.join(body)}</g>", a.title or "Run timeline", desc)
    open(a.out, "w").write(svg)
    print(f"{a.out}: {desc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", required=True)
    ap.add_argument("--window-log", required=True)
    ap.add_argument("--power", required=True)
    ap.add_argument("--stats")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title")
    ap.add_argument("--subtitle")
    ap.add_argument("--arm", help="arm name in the window log (default: result file name)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--ref", type=float, help="full-batch reference line on panel 2 (default: the run's concurrency; 0 hides it)")
    ap.add_argument("--show-run-average", action="store_true", help="also print the whole-run average tok/s in panel 2's caption")
    ap.add_argument("--measured", help="t0,t1,tok/s measured directly (KV-cache growth) drawn on the tokens/s panel")
    ap.add_argument("--steps", help="decode_steps.py output JSON for this run (enables the decode tokens/s panel)")
    ap.add_argument("--compare-steps", help="decode_steps.py output JSON for the compare run")
    ap.add_argument("--decode-batch", action="store_true", help="also draw the tokens-per-decode-step panel")
    ap.add_argument("--tps-ref", type=float, default=300, help="reference line on the output tokens/s panel (0 hides it)")
    ap.add_argument("--gpus", help="comma list of bus-id substrings to sum (default: cards that reach 60 W)")
    ap.add_argument("--compare", help="second result JSON, drawn as a second column")
    ap.add_argument("--compare-label", default="comparison")
    ap.add_argument("--compare-window-log")
    ap.add_argument("--compare-power")
    ap.add_argument("--compare-stats")
    ap.add_argument("--compare-arm")
    a = ap.parse_args()
    a.gpus = a.gpus.split(",") if a.gpus else None
    render(a)


if __name__ == "__main__":
    main()
