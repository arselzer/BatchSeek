"""Pairwise agreement between BIRD runs on the same questions: identical reply text, identical extracted SQL, and
identical result sets on the database (the BIRD execution comparison applied between two runs instead of against the
reference). Each RUN is a harness result JSON ({"requests": [...]}) or batch_client.py JSONL.
Usage: compare_runs.py ANNOTATIONS DEV_DATABASES_DIR NAME=RUN [NAME=RUN ...]"""
import itertools, json, re, sqlite3, sys, time
annf, dbdir, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
txt = open(annf).read()
rows = json.loads(txt) if txt.lstrip().startswith('[') else [json.loads(l) for l in txt.splitlines() if l.strip()]
db = {str(q['question_id']): q['db_id'] for q in rows}
def load(p):
    t = open(p).read()
    if t.lstrip().startswith('{"id"'):   # batch_client.py JSONL
        rs = [json.loads(l) for l in t.splitlines() if l.strip()]
        return {r['id'].split('-', 1)[1]: r.get('text') or '' for r in rs}
    return {r['name'].split('-', 1)[1]: r.get('text') or '' for r in json.loads(t)['requests']}
def sql(t):
    m = re.findall(r'```sql\s*(.*?)```', t, re.S | re.I)
    return m[-1].strip() if m else None
cache = {}
def result(qid, s):
    if s is None: return ('no sql',)
    k = (qid, s)
    if k not in cache:
        c = sqlite3.connect(f'file:{dbdir}/{db[qid]}/{db[qid]}.sqlite?mode=ro', uri=True); t0 = time.time()
        c.set_progress_handler(lambda: 1 if time.time() - t0 > 30 else 0, 10000)
        try: cache[k] = ('ok', frozenset(map(tuple, c.execute(s).fetchall())))
        except Exception as e: cache[k] = ('error',)
        finally: c.close()
    return cache[k]
R = {}
for a in runs:
    n, p = a.split('=', 1); R[n] = load(p)
print('%-28s %6s %10s %10s %12s' % ('pair', 'n', 'same text', 'same SQL', 'same result'))
for a, b in itertools.combinations(R, 2):
    ks = sorted(set(R[a]) & set(R[b]) & set(db))
    st = sum(R[a][k] == R[b][k] for k in ks)
    ss = sum(sql(R[a][k]) == sql(R[b][k]) for k in ks)
    sr = sum(result(k, sql(R[a][k])) == result(k, sql(R[b][k])) for k in ks)
    print('%-28s %6d %9.1f%% %9.1f%% %11.1f%%' % (a + ' vs ' + b, len(ks), 100*st/len(ks), 100*ss/len(ks), 100*sr/len(ks)))
