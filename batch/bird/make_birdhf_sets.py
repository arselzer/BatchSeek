"""BIRD Mini-Dev 2026 revision with BIRD's own baseline prompts (https://huggingface.co/datasets/birdsql/bird_mini_dev,
CC BY-SA 4.0). Input: the dataset's baseline_inference/mini_dev_sqlite.jsonl (500 rows; each row's "prompt" is the
official zero-shot prompt with schema, sample values, question and evidence). Requests go in file order, one user turn
each, no system prompt.

Writes birdhf500-receipt.json (thinking off, max_tokens 2048) and birdhft500-receipt.json (thinking on,
max_tokens 8192). Score with bird_ex.py against data/mini_dev_sqlite.jsonl (converted to a JSON list) and the
Mini-Dev SQLite databases.

Usage: make_birdhf_sets.py BASELINE_JSONL OUT_DIR"""
import json, os, sys

SRC, OUT = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(SRC) if l.strip()]
os.makedirs(OUT, exist_ok=True)
for fam, extra, max_tokens in (('birdhf', {}, 2048), ('birdhft', {"thinking": True}, 8192)):
    reqs = [{"name": "%s-%s" % (fam, q['question_id']), "family": fam,
             "request": dict({"max_tokens": max_tokens, "model": "ds41-native-smoke", "temperature": 0, "stream": False},
                             **extra, chat_turns=[{"user": q['prompt']}])}
            for q in rows]
    with open(os.path.join(OUT, '%s500-receipt.json' % fam), 'w') as f:
        json.dump({"benchmark_requests": reqs}, f)
print(len(rows), 'requests; prompt chars min/max', min(len(q['prompt']) for q in rows), max(len(q['prompt']) for q in rows))
