"""A larger public batch NL-to-SQL set from BIRD's full dev split (1,534 questions over the same 11 SQLite databases
as Mini-Dev; https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip, dev_20240627). Same prompt form as
make_bird_sets.py (our system prompt, the database's CREATE statements from sqlite_master, question, evidence hint),
deterministic order (sha256 of 'birddev' + question_id), first N requests, max_tokens 512. For sustained-throughput
runs (more requests than concurrency slots). Score with bird_ex.py against dev.json and dev_databases.

Usage: make_birddev_set.py DEV_DIR OUT.json [N=1024]     (DEV_DIR: the unpacked dev_20240627 folder, with
dev.json and dev_databases/ unpacked from dev_databases.zip)"""
import hashlib, json, os, sqlite3, sys
SRC, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
SYSTEM = ("You translate questions about a SQLite database into SQL. Use only the tables and columns in the schema. "
          "Reply with exactly one SQLite query in a ```sql code block and nothing else.")
qs = json.load(open(os.path.join(SRC, 'dev.json')))
schema = {}
def schema_of(db):
    if db not in schema:
        con = sqlite3.connect('file:%s?mode=ro' % os.path.join(SRC, 'dev_databases', db, db + '.sqlite'), uri=True)
        rows = [r[0] for r in con.execute("select sql from sqlite_master where sql is not null order by name")]
        con.close()
        schema[db] = '\n\n'.join(r.strip() + ';' for r in rows)
    return schema[db]
qs.sort(key=lambda q: hashlib.sha256(('birddev' + str(q['question_id'])).encode()).hexdigest())
reqs = []
for q in qs[:N]:
    user = "Database schema:\n\n%s\n\nQuestion: %s\nHint: %s" % (schema_of(q['db_id']), q['question'].strip(), (q.get('evidence') or '').strip())
    reqs.append({"name": "birddev-%s" % q['question_id'], "family": "birddev",
                 "request": {"max_tokens": 512, "model": "ds41-native-smoke", "temperature": 0, "stream": False,
                             "chat_system": SYSTEM, "chat_turns": [{"user": user}]}})
json.dump({"benchmark_requests": reqs}, open(OUT, 'w'))
print(len(reqs), 'requests over', len(schema), 'databases')
