"""Paired BIRD execution accuracy of two runs on the same questions: each run's EX, and per question which run is
right (both / only A / only B / neither), with an exact McNemar test on the discordant pairs.
A and B: batch_client.py output (JSONL) or harness result JSON.
Usage: paired_ex.py A B ANNOTATIONS DEV_DATABASES_DIR"""
import json, math, re, sqlite3, sys, time
fa, fb, annf, dbdir = sys.argv[1:5]
txt = open(annf).read()
rows = json.loads(txt) if txt.lstrip().startswith('[') else [json.loads(l) for l in txt.splitlines() if l.strip()]
ann = {str(q['question_id']): q for q in rows}
def load(p):
    t = open(p).read()
    if t.lstrip().startswith('{"id"'):   # batch_client.py JSONL
        return {r['id'].split('-', 1)[1]: r.get('text') or '' for r in map(json.loads, filter(str.strip, t.splitlines()))}
    return {r['name'].split('-', 1)[1]: r.get('text') or '' for r in json.loads(t)['requests']}
def run(db, sql):
    c = sqlite3.connect(f'file:{dbdir}/{db}/{db}.sqlite?mode=ro', uri=True); t0 = time.time()
    c.set_progress_handler(lambda: 1 if time.time() - t0 > 30 else 0, 10000)
    try: return set(map(tuple, c.execute(sql).fetchall()))
    except Exception: return None
    finally: c.close()
A, B = load(fa), load(fb); n = both = oa = ob = 0
for k in sorted(set(A) & set(B) & set(ann)):
    q = ann[k]; gold = run(q['db_id'], q['SQL'])
    if gold is None: continue
    def ok(t):
        m = re.findall(r'```sql\s*(.*?)```', t, re.S | re.I)
        return bool(m) and run(q['db_id'], m[-1].strip()) == gold
    a, b = ok(A[k]), ok(B[k]); n += 1; both += a and b; oa += a and not b; ob += b and not a
d = oa + ob
p = sum(math.comb(d, i) for i in range(0, min(oa, ob) + 1)) * 2 / 2 ** d if d else 1.0
print(json.dumps({'n': n, 'ex_a': round(100 * (both + oa) / n, 1), 'ex_b': round(100 * (both + ob) / n, 1),
                  'only_a_right': oa, 'only_b_right': ob, 'mcnemar_p': round(min(p, 1.0), 3)}))
