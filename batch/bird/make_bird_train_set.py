"""A BIRD *train* request set for recording a placement profile (--expert-stats) on public prompts that are not the
benchmark: BIRD's training split (69 databases, none of them Mini-Dev's 11). Same prompt form as make_bird_sets.py.

Input: BIRD's train.zip (https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip), unpacked once: train/train.json
and train/train_databases.zip (the databases stay zipped; only the sampled ones are read). N questions, at most PER
from one database, chosen by sha256 of question_id order (deterministic).

Questions whose prompt would exceed MAXC characters (very wide schemas) are passed over.

Usage: make_bird_train_set.py TRAIN_DIR OUT.json [N=64] [PER=2] [MAXC=16000]"""
import hashlib, io, json, os, sqlite3, sys, tempfile, zipfile
SRC, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 64
PER = int(sys.argv[4]) if len(sys.argv) > 4 else 2
MAXC = int(sys.argv[5]) if len(sys.argv) > 5 else 16000
SYSTEM = ("You translate questions about a SQLite database into SQL. Use only the tables and columns in the schema. "
          "Reply with exactly one SQLite query in a ```sql code block and nothing else.")
qs = json.load(open(os.path.join(SRC, 'train.json')))
for i, q in enumerate(qs):
    q.setdefault('question_id', i)
qs.sort(key=lambda q: hashlib.sha256(('birdtrain' + str(q['question_id'])).encode()).hexdigest())
z = zipfile.ZipFile(os.path.join(SRC, 'train_databases.zip'))
names = {os.path.basename(n): n for n in z.namelist() if n.endswith('.sqlite') and '__MACOSX' not in n}
schema = {}
def schema_of(db):
    if db not in schema:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, db + '.sqlite')
            with z.open(names[db + '.sqlite']) as src, open(p, 'wb') as dst:
                while True:
                    b = src.read(1 << 24)
                    if not b: break
                    dst.write(b)
            con = sqlite3.connect('file:%s?mode=ro' % p, uri=True)
            rows = [r[0] for r in con.execute("select sql from sqlite_master where sql is not null order by name")]
            con.close()
        schema[db] = '\n\n'.join(r.strip() + ';' for r in rows)
    return schema[db]
pick, per = [], {}
for q in qs:
    db = q['db_id']
    if per.get(db, 0) >= PER:
        continue
    if len(SYSTEM) + len(schema_of(db)) + len(q['question']) + len(q.get('evidence') or '') + 40 > MAXC:
        continue
    per[db] = per.get(db, 0) + 1; pick.append(q)
    if len(pick) == N:
        break
reqs = [{"name": "birdtrain-%s" % q['question_id'], "family": "birdtrain",
         "request": {"max_tokens": 512, "model": "ds41-native-smoke", "temperature": 0, "stream": False,
                     "chat_system": SYSTEM,
                     "chat_turns": [{"user": "Database schema:\n\n%s\n\nQuestion: %s\nHint: %s" % (
                         schema[q['db_id']], q['question'].strip(), (q.get('evidence') or '').strip())}]}}
        for q in pick]
with open(OUT, 'w') as f:
    json.dump({"benchmark_requests": reqs}, f)
with open(OUT, 'rb') as f:
    print(hashlib.sha256(f.read()).hexdigest(), '', os.path.basename(OUT))
L = sorted(len(r['request']['chat_turns'][0]['user']) for r in reqs)
print(len(reqs), 'requests over', len(per), 'databases; prompt chars min/median/max', L[0], L[len(L) // 2], L[-1])
