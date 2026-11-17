#!/usr/bin/env python3
"""BIRD dev (1,534 questions) in the 2026 mini-dev's official prompt format, as a harness receipt.
The schema block of each database is taken from the official 2026 prompts (birdhf500-receipt.json; the 11 dev databases
are the mini-dev ones), and each dev question/evidence replaces the mini-dev one. max_tokens 2048, temperature 0.
Usage: make_birddev26_set.py BIRDHF500_RECEIPT ANNOTATIONS_2026 DEV_JSON OUT_RECEIPT"""
import json, sys
rec, ann, dev, out = sys.argv[1:5]
R = json.load(open(rec))['benchmark_requests']; A = {str(x['question_id']): x for x in json.load(open(ann))}
tmpl = {}
for r in R:
    q = A[r['name'].split('-', 1)[1]]; u = r['request']['chat_turns'][0]['user']
    head, rest = u.split('\nQuestion:\n', 1); tail = rest.split('\n\nInstructions:\n', 1)[1]
    tmpl.setdefault(q['db_id'], (head, tail, r['request']))
D = json.load(open(dev)); out_r = []
for i, q in enumerate(D):
    head, tail, req = tmpl[q['db_id']]
    qq = (q['evidence'].strip() + '\n' if q.get('evidence', '').strip() else '') + q['question'].strip()
    nr = dict(req); nr['chat_turns'] = [{'user': head + '\nQuestion:\n' + qq + '\n\nInstructions:\n' + tail}]
    out_r.append({'name': f'birddev26-{i}', 'family': 'birddev26', 'request': nr})
json.dump({'benchmark_requests': out_r}, open(out, 'w'))
print(len(out_r), 'requests;', len(tmpl), 'databases')
