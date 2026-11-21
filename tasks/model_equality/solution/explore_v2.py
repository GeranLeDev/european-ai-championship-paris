import json, re, sys, math
from collections import Counter, defaultdict
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
kind = dict(l.strip().split(',')[0::3] for l in open('../practice_answers.csv').readlines()[1:])
pk = {json.loads(l)['id']: json.loads(l)['kind'] for l in open('prompts_v2.jsonl')}
eps = sorted({r['endpoint'] for r in rows})
by = defaultdict(lambda: defaultdict(list))
for r in rows:
    by[r['endpoint']][r['id']].append(r['response'])
W = lambda s: re.findall(r"[a-z0-9']+", s.lower())
def cosb(a, b):
    a, b = Counter(W(a)), Counter(W(b))
    d = (sum(v*v for v in a.values()) * sum(v*v for v in b.values())) ** .5
    return sum(v*b.get(k, 0) for k, v in a.items()) / d if d else 0.
first = lambda y: re.split(r"(?<=[.!?])\s", y.strip())[0].lower()[:60]
print("endpoint kind         n  ident  logL_short  logL_ident  logL_long  exact  first-agree  selfbow_short  stop%")
for e in eps:
    S = {p: v for p, v in by[e].items() if pk[p] == 'short'}
    I = {p: v for p, v in by[e].items() if pk[p] == 'ident'}
    Lg = {p: v for p, v in by[e].items() if pk[p] == 'long'}
    ident = np.mean([bool(re.search(r"smollm|hugging", y.lower())) for v in I.values() for y in v])
    ll = lambda D: np.mean([np.mean([math.log(len(y) + 1) for y in v]) for v in D.values()]) if D else float('nan')
    SI = {**S, **I}
    exact = np.mean([v[0] == v[1] for v in SI.values() if len(v) > 1])
    fa = np.mean([first(v[0]) == first(v[1]) for v in SI.values() if len(v) > 1])
    sb = np.mean([cosb(v[0], v[1]) for v in S.values() if len(v) > 1])
    n = sum(len(v) for v in by[e].values())
    st = np.mean([r['finish_reason'] == 'stop' for r in rows if r['endpoint'] == e and pk[r['id']] == 'long'])
    print(f"{e} {kind[e][:12]:12s} {n:3d}  {ident:.2f}   {ll(S):6.2f}     {ll(I):6.2f}     {ll(Lg):6.2f}    {exact:.2f}    {fa:.2f}       {sb:.2f}      {st:.2f}")
