"""BIRD Mini-Dev execution accuracy for a window arm's replies: each reply's last ```sql block and BIRD's reference SQL run
read-only on the database (30 s limit each); a reply counts as correct when both return the same set of rows (BIRD's EX).
A reference query that itself fails on the database (or runs past the limit) is counted apart: "ex" leaves it out of
the denominator, "ex_official" counts it as wrong for every reply, as BIRD's own evaluator does (two of the 500 run
past 30 s; the published comparison uses ex_official).
ANNOTATIONS: the archive's mini_dev_sqlite.json, or the 2026 revision's data/mini_dev_sqlite.jsonl.
RESULT: batch_client.py's output (JSONL) or a harness result JSON.
Usage: bird_ex.py RESULT ANNOTATIONS DEV_DATABASES_DIR"""
import collections, json, re, sqlite3, sys, time
arm, annf, dbdir = sys.argv[1:4]
txt = open(annf).read()
rows = json.loads(txt) if txt.lstrip().startswith('[') else [json.loads(l) for l in txt.splitlines() if l.strip()]
ann = {str(q['question_id']): q for q in rows}
def run(dbid, sql, limit=30):
    c = sqlite3.connect(f'file:{dbdir}/{dbid}/{dbid}.sqlite?mode=ro', uri=True); t0 = time.time()
    c.set_progress_handler(lambda: 1 if time.time() - t0 > limit else 0, 10000)
    try: return set(map(tuple, c.execute(sql).fetchall())), None
    except Exception as e: return None, str(e)[:60]
    finally: c.close()
st = collections.Counter(); diff = collections.defaultdict(lambda: [0, 0])
rt = open(arm).read()
if rt.lstrip().startswith('{"requests"') or not rt.lstrip().startswith('{"id"'):
    replies = json.loads(rt)['requests']
else:  # batch_client.py output: one {"id", "text", ...} a line
    replies = [dict(r, name=r['id']) for r in map(json.loads, rt.splitlines()) if r.get('id')]
for r in replies:
    q = ann[r['name'].split('-', 1)[1]]
    gold, ge = run(q['db_id'], q['SQL'])
    if ge: st['reference error'] += 1; continue
    m = re.findall(r'```sql\s*(.*?)```', r.get('text') or '', re.S | re.I)
    pred, pe = run(q['db_id'], m[-1].strip()) if m else (None, 'no sql')
    ok = pe is None and pred == gold
    st['correct' if ok else ('no sql' if not m else 'error' if pe else 'wrong result')] += 1
    diff[q['difficulty']][0] += ok; diff[q['difficulty']][1] += 1
n = sum(v for k, v in st.items() if k != 'reference error')
print(json.dumps({'counts': st, 'ex': round(100 * st['correct'] / n, 1),
                  'ex_official': round(100 * st['correct'] / (n + st['reference error']), 1), 'by_difficulty': {k: '%d/%d' % tuple(v) for k, v in sorted(diff.items())}}))
