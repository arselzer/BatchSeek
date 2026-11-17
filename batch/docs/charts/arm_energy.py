#!/usr/bin/env python3
"""Per arm of a window: the client's request phase (the window log's "the requests' N ms" line, ending at that line's
time), output tokens, whole-job tok/s over that phase, and the energy of the four RX 9070 XT + CPU package integrated over time (power log (about every 1.2 s)) over exactly that phase. The power log's HH:MM:SS times roll over to the next day
when they go backwards (runs across midnight). Usage: arm_energy.py OUT_DIR POWER.tsv"""
import csv, datetime as dt, glob, json, os, re, sys
out, ptsv = sys.argv[1:3]
log = open(os.path.join(out, 'window.log')).read().splitlines()
# power samples with full timestamps
rows = list(csv.DictReader(open(ptsv), delimiter='\t'))
cols = [c for c in rows[0] if c.startswith('gpu_') and any(b in c for b in (':03:00', ':48:00', ':83:00', ':c6:00'))] + ['cpu_pkg_W']
start_day = None
for l in log:
    m = re.match(r'\[(\d{4}-\d\d-\d\d)T', l)
    if m: start_day = dt.date.fromisoformat(m.group(1)); break
samples, day, prev = [], start_day, None
for r in rows:
    try:
        t = dt.time.fromisoformat(r['utc']); w = sum(float(r[c]) for c in cols)
    except (ValueError, KeyError, TypeError):
        continue
    if prev is not None and t < prev: day = day + dt.timedelta(days=1)
    prev = t
    samples.append((dt.datetime.combine(day, t, dt.timezone.utc).timestamp(), w))
res = []
for f in sorted(glob.glob(os.path.join(out, '*.json')), key=os.path.getmtime):
    b = os.path.basename(f)
    if 'xstats' in b or '-U' not in b or b.endswith('.mem'): continue
    arm = re.sub(r'(\.unmeasured)?\.json$', '', b)
    line = next((l for l in log if f'arm {arm}: NVMe busy' in l and "the requests'" in l), None)
    if not line: print(json.dumps({'arm': arm, 'error': 'no request-phase line'})); continue
    t1 = dt.datetime.fromisoformat(line[1:21].replace('Z', '+00:00')).timestamp()
    ms = int(re.search(r"the requests' (\d+) ms", line).group(1)); t0 = t1 - ms / 1000
    R = json.load(open(f))['requests']
    tok = sum(r['timings'].get('predicted_n', 0) for r in R if r.get('timings'))
    inside = [w for t, w in samples if t0 <= t <= t1]
    # time-weighted: each sample's power over the time to the next one (the log ticks about every 1.2 s, not 1 s),
    # clipped to the request phase; equals the mean power times the phase length when the ticks are even
    pts = [(t, w) for t, w in samples if t0 - 5 <= t <= t1 + 5]
    j = 0.0
    for (ta, wa), (tb, wb) in zip(pts, pts[1:]):
        a0, b0 = max(ta, t0), min(tb, t1)
        if b0 > a0: j += (wa + wb) / 2 * (b0 - a0)
    wh = j / 3600.0
    u = re.search(r'-U([a-z0-9]+)-', arm + '-').group(1); c = int(re.search(r'-C(\d+)-', arm).group(1))
    res.append(dict(u=u, conc=c, n=len(R), tokens=tok, job_s=round(ms / 1000, 1), whole_tok_s=round(tok / (ms / 1000), 1),
                    samples=len(inside), mean_w=round(sum(inside) / len(inside)) if inside else None,
                    kwh=round(wh / 1000, 4), wh_per_1k=round(wh / tok * 1000, 2) if tok else None,
                    eur_030=round(wh / 1000 * 0.30, 4)))
    print(json.dumps(res[-1]))
