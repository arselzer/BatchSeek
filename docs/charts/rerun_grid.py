#!/usr/bin/env python3
"""Rerun grid: the same 256 BIRD questions sent twice to each deployment; one square per question: identical reply,
different reply but the same query result on the database, or a different result. Hover names the state (inline).
Usage: rerun_grid.py ANNOTATIONS DEV_DATABASES OUT.svg NAME=RUN1,RUN2 [NAME=RUN1,RUN2 ...]"""
import json, os, re, sqlite3, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart_style import W, TITLE, LABEL, SMALL, doc, text, key

annf, dbdir, out = sys.argv[1:4]
pairs = [a.split("=", 1) for a in sys.argv[4:]]
txt = open(annf).read()
ann = {str(q["question_id"]): q for q in (json.loads(txt) if txt.lstrip().startswith("[") else
                                           [json.loads(l) for l in txt.splitlines() if l.strip()])}


def load(p):
    return {r["name"].split("-", 1)[1] if "-" in r["name"] else r["name"]: r.get("text") or ""
            for r in json.load(open(p))["requests"]}


def result(db, t):
    m = re.findall(r"```sql\s*(.*?)```", t, re.S | re.I)
    if not m:
        return ("nosql",)
    c = sqlite3.connect(f"file:{dbdir}/{db}/{db}.sqlite?mode=ro", uri=True); t0 = time.time()
    c.set_progress_handler(lambda: 1 if time.time() - t0 > 30 else 0, 10000)
    try:
        return ("ok", frozenset(map(tuple, c.execute(m[-1].strip()).fetchall())))
    except Exception as e:
        return ("err", type(e).__name__)
    finally:
        c.close()


panels = []
order = None
for name, runs in pairs:
    a, b = (load(p) for p in runs.split(","))
    keys = sorted(set(a) & set(b), key=int)
    order = order or keys
    st = []
    for k in order:
        if a[k] == b[k]:
            st.append(2)
        else:
            db = ann[k]["db_id"]
            st.append(1 if result(db, a[k]) == result(db, b[k]) else 0)
    panels.append((name, st))
    print(name, "identical", st.count(2), "same result", st.count(1), "different", st.count(0))

PAD, cols, cell, gap = 8, 16, 9, 1.5
pw = cols * (cell + gap)
ph = (len(order) + cols - 1) // cols * (cell + gap)
sep = (W - 2 * PAD - len(panels) * pw) / max(1, len(panels) - 1)
T = 88
H = T + 26 + ph + 30
col = {2: ("s1", 1.0), 1: ("s1", 0.30), 0: ("s2", 1.0)}
b = [text(0, 20, "The same 256 questions, asked twice", TITLE, "ink", weight="600"),
     text(0, 39, "one square per BIRD question, temperature 0, thinking off", LABEL, "ink2")]
lx = 0
for lab, kind in (("same reply, byte for byte", 2), ("different reply, same query result", 1), ("different query result", 0)):
    cname, op = col[kind]
    b.append(f"<rect x='{lx}' y='51' width='{cell}' height='{cell}' fill='var(--{cname})' fill-opacity='{op}'/>")
    b.append(text(lx + cell + 6, 60, lab, SMALL, "ink2"))
    lx += cell + 6 + len(lab) * 6.3 + 22
for i, (name, st) in enumerate(panels):
    x0 = i * (pw + sep)
    same = st.count(2) / len(st)
    b.append(text(x0, T + 8, name, LABEL, "ink", weight="600"))
    b.append(text(x0, T + 22, f"{same:.0%} identical", SMALL, "ink2"))
    for kind in (2, 1, 0):
        cname, op = col[kind]
        ds = "".join(f"M{x0 + (j % cols) * (cell + gap):.1f} {T + 30 + (j // cols) * (cell + gap):.1f}h{cell}v{cell}h-{cell}z"
                     for j, v in enumerate(st) if v == kind)
        if ds:
            b.append(f"<path d='{ds}' fill='var(--{cname})' fill-opacity='{op}'/>")
desc = "; ".join(f"{n}: {s.count(2)} identical, {s.count(1)} different reply but same result, {s.count(0)} different "
                 f"result of {len(s)}" for n, s in panels) + "."
open(out, "w").write(doc(W, H, f"<g transform='translate({PAD},0)'>{''.join(b)}</g>", "Rerun grid", desc))
print(desc)
