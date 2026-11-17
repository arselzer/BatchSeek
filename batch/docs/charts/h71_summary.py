#!/usr/bin/env python3
"""One row per arm of a multi-arm window (H71): set, concurrency, requests, output tokens, job time (first request to
last reply), whole-job and generation tok/s, peak tok/s (KV growth over >=10 s while >=90% of the slots generate and no
prompt is processed), mean power of the four RX 9070 XT + CPU package over the arm, Wh per 1,000 output tokens, kWh, cost
at EUR 0.30/kWh, and execution accuracy over all requests (a reference query that fails or times out counts as wrong,
as in BIRD's evaluator).
Usage: h71_summary.py OUT_DIR POWER.tsv STATS.jsonl [--no-score] > rows.json"""
import csv, glob, json, os, re, subprocess, sys, warnings, datetime as dt
warnings.filterwarnings('ignore')
out, ptsv, stats = sys.argv[1:4]; score = '--no-score' not in sys.argv
BD = '/home/as/ds41-claude-build/bird-data'
SC = '/home/as/ds41-claude-build/work/h1/bird/bird_ex.py'
SETS = {'birdhf': (f'{BD}/hf-2026/annotations-2026-sqlite.json', f'{BD}/minidev/MINIDEV/dev_databases', 'BIRD 2026'),
        'bird': (f'{BD}/minidev/MINIDEV/mini_dev_sqlite.json', f'{BD}/minidev/MINIDEV/dev_databases', 'BIRD original'),
        'birddev26': (f'{BD}/dev/dev_20240627/dev.json', f'{BD}/dev/dev_20240627/dev_databases', 'BIRD dev (2026 prompts)')}
log = open(os.path.join(out, 'window.log')).read()
st = [json.loads(l) for l in open(stats) if l.strip()]
busy = [s for s in st if s.get('running', 0) + s.get('waiting', 0) > 0]
segs = [[busy[0]]] if busy else []
for s in busy[1:]:
    (segs[-1].append(s) if s['t'] - segs[-1][-1]['t'] < 30 else segs.append([s]))
rows_p = list(csv.DictReader(open(ptsv), delimiter='\t'))
cols = [c for c in rows_p[0] if c.startswith('gpu_') and any(b in c for b in (':03:00', ':48:00', ':83:00', ':c6:00'))] + ['cpu_pkg_W']
def power(t0, t1):
    day = dt.datetime.utcfromtimestamp(t0).date(); w = []
    for r in rows_p:
        try:
            t = dt.datetime.combine(day, dt.time.fromisoformat(r['utc'])).replace(tzinfo=dt.timezone.utc).timestamp()
            if t0 <= t <= t1: w.append(sum(float(r[c]) for c in cols))
        except (ValueError, KeyError, TypeError):
            pass
    return sum(w) / len(w) if w else None
files = sorted([f for f in glob.glob(os.path.join(out, '*.json')) if 'xstats' not in f and '-U' in f and not f.endswith('.mem')],
               key=os.path.getmtime)
res = []
for k, f in enumerate(files):
    arm = re.sub(r'(\.unmeasured)?\.json$', '', os.path.basename(f))
    j = json.load(open(f)); R = j['requests']; S = j.get('stats', {})
    u = re.search(r'-U([a-z0-9]+)-', arm + '-').group(1); c = int(re.search(r'-C(\d+)-', arm).group(1))
    fam = 'birddev26' if u.startswith('birddev26') else 'birdhf' if u.startswith('birdhf') else 'bird'
    tok = sum(r['timings'].get('predicted_n', 0) for r in R if r.get('timings'))
    dec = S.get('step_ns_decode', 0) / 1e9; tot = S.get('step_ns', 0) / 1e9
    job = max(r['wall_s'] for r in R) if c >= len(R) else tot    # all at once: first request to last reply; else server steps
    seg = segs[k] if k < len(segs) else None
    row = dict(arm=arm, set=SETS[fam][2], u=u, conc=c, n=len(R), errors=sum(1 for r in R if r.get('error') not in (None, 'None')),
               out_tokens=tok, job_s=round(job, 1), prefill_s=round(tot - dec, 1), decode_s=round(dec, 1),
               whole_tok_s=round(tok / job, 1) if job else None, gen_tok_s=round(tok / dec, 1) if dec else None)
    if seg:
        t0, t1 = seg[0]['t'], seg[-1]['t']
        full = [s for s in seg if s.get('running', 0) >= 0.9 * min(c, len(R)) and s.get('waiting', 0) == 0 and s.get('prefill_tps', 0) == 0]
        best = 0
        for i, a in enumerate(full):
            for b in full[i + 1:]:
                if b['t'] - a['t'] >= 10:
                    if b['t'] - a['t'] <= 14: best = max(best, (b['kv_tokens_used'] - a['kv_tokens_used']) / (b['t'] - a['t']))
                    break
        mw = power(t0, t1); kwh = mw * (t1 - t0) / 3.6e6 if mw else None
        row.update(peak_tok_s=round(best) or None, seg_s=round(t1 - t0), power_w=round(mw) if mw else None,
                   kwh=round(kwh, 4) if kwh else None, wh_per_1k=round(kwh * 1e6 / tok, 2) if kwh and tok else None,
                   eur_030=round(kwh * 0.30, 4) if kwh else None, max_kv_util=round(max(s.get('kv_util', 0) for s in seg), 2),
                   max_kv_tokens=max(s.get('kv_tokens_used', 0) for s in seg), max_preempted=max(s.get('preempted', 0) for s in seg))
    if score:
        a, d, _ = SETS[fam]
        p = subprocess.run(['python3', '-I', SC, f, a, d], capture_output=True, text=True, timeout=3600).stdout.strip().split('\n')[-1]
        try:
            e = json.loads(p); cnt = e['counts']
            row.update(ex_of_all=round(100 * cnt.get('correct', 0) / len(R), 1), ex_counts=cnt)
        except Exception:
            row.update(ex_error=p[-200:])
    res.append(row); print(json.dumps(row), flush=True)
