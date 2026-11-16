"""Public batch NL-to-SQL sets from BIRD Mini-Dev (SQLite, 500 questions over 11 databases; CC BY-SA 4.0).

Input: the official Mini-Dev archive minidev_0703.zip, linked from https://github.com/bird-bench/mini_dev
(sha256 aeb211c0e39010bbdae3838bb5e8bd27dc446ed77495b1709f85ccc9bf67f2be), unpacked. We use two things from its
minidev/MINIDEV folder: mini_dev_sqlite.json (the questions; sha256
def4b2b43a9b06955193418f24c9be170eb6d83763ab702311790e0cdda8c791) and dev_databases/<db>/<db>.sqlite. A database's
schema text is its own CREATE statements, read from the database file (sqlite_master, ordered by name, each ending
in ';', separated by a blank line).

Each request: a neutral system prompt of our own, then the database's full schema, the question and BIRD's evidence
hint. Requests on the same database share the system prompt + schema as an exact prefix. Deterministic interleaved
order (sha256 of question_id). Writes bird256-receipt.json (first 256) and bird500-receipt.json (all), max_tokens 512.
These are byte for byte the request sets measured in the post.

Usage: make_bird_sets.py MINIDEV_DIR OUT_DIR     (MINIDEV_DIR: the unpacked archive's minidev/MINIDEV folder)"""
import hashlib, json, os, sqlite3, sys

SRC, OUT = sys.argv[1], sys.argv[2]
SYSTEM = ("You translate questions about a SQLite database into SQL. Use only the tables and columns in the schema. "
          "Reply with exactly one SQLite query in a ```sql code block and nothing else.")
ANN_SHA = 'def4b2b43a9b06955193418f24c9be170eb6d83763ab702311790e0cdda8c791'

ann_path = os.path.join(SRC, 'mini_dev_sqlite.json')
raw = open(ann_path, 'rb').read()
if hashlib.sha256(raw).hexdigest() != ANN_SHA:
    print('warning: %s is not the 0703 release file (sha256 differs); the sets will not match the measured ones' % ann_path)
ann = json.loads(raw)


def schema_of(db):
    path = os.path.join(SRC, 'dev_databases', db, db + '.sqlite')
    con = sqlite3.connect('file:%s?mode=ro' % path, uri=True)
    try:
        rows = [r[0] for r in con.execute("select sql from sqlite_master where sql is not null order by name")]
    finally:
        con.close()
    return '\n\n'.join(r.strip() + ';' for r in rows)


schema = {}
for q in ann:
    if q['db_id'] not in schema:
        schema[q['db_id']] = schema_of(q['db_id'])
ann.sort(key=lambda q: hashlib.sha256(('bird' + str(q['question_id'])).encode()).hexdigest())
reqs = []
for q in ann:
    user = "Database schema:\n\n%s\n\nQuestion: %s\nHint: %s" % (schema[q['db_id']], q['question'].strip(), q['evidence'].strip())
    reqs.append({"name": "bird-%s" % q['question_id'], "family": "bird",
                 "request": {"max_tokens": 512, "model": "ds41-native-smoke", "temperature": 0, "stream": False,
                             "chat_system": SYSTEM, "chat_turns": [{"user": user}]}})
os.makedirs(OUT, exist_ok=True)
for n, name in ((256, 'bird256'), (len(reqs), 'bird500')):
    with open(os.path.join(OUT, '%s-receipt.json' % name), 'w') as f:
        json.dump({"benchmark_requests": reqs[:n]}, f)
    with open(os.path.join(OUT, '%s-receipt.json' % name), 'rb') as f:
        print(hashlib.sha256(f.read()).hexdigest(), ' %s-receipt.json' % name)
L = sorted(len(SYSTEM) + len(r['request']['chat_turns'][0]['user']) for r in reqs)
print(len(reqs), 'requests over', len(schema), 'databases; prompt chars min/median/max', L[0], L[len(L) // 2], L[-1])
