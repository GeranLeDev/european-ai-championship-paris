"""Exploration of free-text probes on the labelled ref endpoints: per-endpoint mechanism features."""
import json, re, sys, math
from collections import Counter, defaultdict
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1])]
kind = dict(l.strip().split(',')[0::3] for l in open('../practice_answers.csv').readlines()[1:])
eps = sorted({r['endpoint'] for r in rows})
by = defaultdict(lambda: defaultdict(list))
for r in rows:
    by[r['endpoint']][r['id']].append(r['response'])
ids = sorted({r['id'] for r in rows})
W = lambda s: re.findall(r"[a-z0-9']+|[.,!?;:]", s.lower())

# pooled word / bigram counts
uni = {e: Counter() for e in eps}
bi = {e: Counter() for e in eps}
for e in eps:
    for pid, ys in by[e].items():
        for y in ys:
            w = W(y)
            uni[e].update(w)
            bi[e].update(zip(w, w[1:]))
pool_bi = sum(bi.values(), Counter())
pool_uni = sum(uni.values(), Counter())
top = [b for b, c in pool_bi.most_common(300)]

print("endpoint kind           nresp  words  len(med)  ident  exactrep  ttr   bigram-disp  miss")
for e in eps:
    R = [y for ys in by[e].values() for y in ys]
    L = [len(y) for y in R]
    ident = np.mean([bool(re.search(r"smollm|hugging", y.lower())) for y in by[e].get('p_who', [])] or [np.nan])
    rep = [len(set(ys)) < len(ys) for ys in by[e].values() if len(ys) > 1]
    n = sum(uni[e].values())
    others = sum((bi[f] for f in eps if f != e), Counter())
    no = sum(others.values()); ne = sum(bi[e].values())
    # dispersion of log frequency ratio on the pooled top bigrams (watermark: some common bigrams depleted)
    lr = []
    miss = 0
    for b in top:
        exp = others[b] / no * ne
        if exp >= 4:
            lr.append(math.log((bi[e][b] + 0.5) / (exp + 0.5)))
            miss += bi[e][b] <= exp * 0.25
    print(f"{e} {kind[e][:14]:14s} {len(R):5d} {n:6d} {np.median(L):8.0f}  {ident:5.2f}  {np.mean(rep) if rep else float('nan'):6.2f}  "
          f"{len(uni[e])/max(n,1):.3f}  {np.std(lr):.3f} (n={len(lr)})  {miss}")
print()
print("median length per prompt")
print(' ' * 10, ' '.join(kind[e][:6].rjust(6) for e in eps))
for pid in ids:
    print(pid.ljust(10), ' '.join(f"{np.median([len(y) for y in by[e][pid]]) if by[e][pid] else -1:6.0f}" for e in eps))
