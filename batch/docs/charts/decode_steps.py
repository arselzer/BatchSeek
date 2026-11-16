#!/usr/bin/env python3
"""Reconstruct decode steps over time from client timings, and from them decode tokens/s over time.

Every request that is decoding gets exactly one token per decode step. A request whose first token arrived at
a = sent + ttft and whose last token arrived at b = sent + wall, with n output tokens, therefore saw n - 1 decode
steps in (a, b]. With hundreds of requests starting and ending at different times, those intervals pin down S(t),
the cumulative number of decode steps: fit S on a grid (non-negative increments) to all requests' step counts by
least squares. Then
    steps/s(t)        = dS/dt
    decode tokens/s(t) = steps/s(t) x requests decoding at t
and the integral of the second equals the run's total output tokens (a self-check the script prints).

Usage: decode_steps.py RESULT.json [--grid 5] [--out OUT.json]
OUT.json: {"t": [...], "steps_per_s": [...], "decoding": [...], "tok_s": [...], "check": {...}}"""
import argparse, json, sys
import numpy as np
from scipy.optimize import lsq_linear

ap = argparse.ArgumentParser()
ap.add_argument('result'); ap.add_argument('--grid', type=float, default=5.0); ap.add_argument('--out')
ap.add_argument('--measured', help='t0,t1,tok_s measured directly (e.g. KV-cache growth while every request decodes): a weighted constraint on the steps in that span')
a = ap.parse_args()
d = json.load(open(a.result))
rows = []
for r in d['requests']:
    t = r.get('timings') or {}
    n = int(t.get('predicted_n') or 0)
    if n < 2 or r.get('wall_s') is None:
        continue
    s = float(r.get('sent_s') or 0.0)
    rows.append((s + float(t.get('ttft_ms') or 0) / 1000.0, s + float(r['wall_s']), n - 1))
if not rows:
    sys.exit('no usable requests')
T = max(b for _, b, _ in rows)
g = a.grid
K = int(np.ceil(T / g))
edges = np.arange(K + 1) * g


def weights(lo, hi):
    """Fraction of each grid cell inside (lo, hi]: the row of the design matrix (S is linear inside a cell)."""
    w = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None) / g
    return w


A = np.array([weights(lo, hi) for lo, hi, _ in rows])
y = np.array([n for _, _, n in rows], dtype=float)
# unknowns: steps per cell (>= 0); light smoothness penalty so empty stretches don't go wild
lam = 0.05 * np.sqrt(len(rows))
D = (np.eye(K, k=1) - np.eye(K))[:-1] * lam
rowsA, rowsy = [A, D], [y, np.zeros(K - 1)]
if a.measured:   # steps in [t0, t1] = measured tok/s / requests decoding there x duration, weighted like 50 requests
    m0, m1, mt = (float(v) for v in a.measured.split(','))
    nd = sum(1 for lo, hi, _ in rows if lo <= m0 and hi >= m1)
    rowsA.append(50 * weights(m0, m1)[None, :]); rowsy.append(np.array([50 * mt / max(nd, 1) * (m1 - m0)]))
res = lsq_linear(np.vstack(rowsA), np.concatenate(rowsy), bounds=(0, np.inf))
steps = res.x                                     # decode steps in each cell
mid = (edges[:-1] + edges[1:]) / 2
dec = np.array([sum(1 for lo, hi, _ in rows if lo < m <= hi) for m in mid])
tok_s = steps / g * dec
pred = A @ steps
out = {"t": mid.round(1).tolist(), "steps_per_s": (steps / g).round(4).tolist(), "decoding": dec.tolist(),
       "tok_s": tok_s.round(1).tolist(),
       "check": {"requests": len(rows), "output_tokens_counted": int(y.sum() + len(rows)),
                 "integral_tok": round(float((tok_s * g).sum()), 0),
                 "median_abs_err_steps": round(float(np.median(np.abs(pred - y))), 2),
                 "decode_steps_total": round(float(steps.sum()), 1)}}
print(json.dumps(out['check']))
peak = int(np.argmax(tok_s))
print('peak %.0f tok/s at %.0f s; mean while decoding %.0f tok/s' % (tok_s[peak], mid[peak], tok_s[dec > 0].mean()))
if a.out:
    json.dump(out, open(a.out, 'w'))
