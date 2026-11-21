import json, re, sys, math, itertools
from collections import defaultdict
import numpy as np
ans, truth = sys.argv[1], sys.argv[2]
fam_only = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != '-' else None
feats = sys.argv[4].split(',') if len(sys.argv) > 4 else ['loglen', 'nsent', 'excl', 'quest', 'nl']
kinds = {json.loads(l)['id']: json.loads(l)['kind'] for l in open('prompts_v2.jsonl')}
T = {l.split(',')[0]: l.strip().split(',') for l in open(truth).readlines()[1:]}
rows = [json.loads(l) for l in open(ans)]
rows = [r for r in rows if r['endpoint'] in T and (fam_only is None or T[r['endpoint']][1] == fam_only)]
eps = sorted({r['endpoint'] for r in rows})
FE = {
 'loglen': lambda y: math.log(len(y) + 1),
 'nsent': lambda y: math.log(1 + len(re.findall(r"[.!?]+", y))),
 'excl': lambda y: math.log(1 + y.count('!')),
 'quest': lambda y: math.log(1 + y.count('?')),
 'nl': lambda y: math.log(1 + y.count('\n')),
 'I': lambda y: math.log(1 + len(re.findall(r"\bI\b|\bI'm\b", y))),
}
groups = {'s': ('short', 'ident'), 'l': ('long',)}
val = {}
for g, sel in groups.items():
    for f in feats:
        acc = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if kinds[r['id']] in sel:
                acc[r['id']][r['endpoint']].append(FE[f](r['response']))
        val[(g, f)] = {pid: {e: np.mean(v) for e, v in d.items()} for pid, d in acc.items()}
def stat(e, f):
    best = (0, None, 0)
    for key, V in val.items():
        d = [x[e] - x[f] for x in V.values() if e in x and f in x]
        if len(d) < 3: continue
        m = np.mean(d); se = np.std(d, ddof=1) / math.sqrt(len(d)) + 0.02
        z = abs(m) / se
        if z > best[0]: best = (z, key, m)
    return best
print(' ' * 22 + ' '.join(e[-2:].rjust(5) for e in eps))
pos, neg = [], []
for e in eps:
    line = f"{e} {T[e][3][:12]:12s} "
    for f in eps:
        if e == f: line += '    . '; continue
        z, key, m = stat(e, f)
        line += f"{z:5.1f} "
        if e < f: (pos if T[e][2] == T[f][2] else neg).append((z, T[e][3][:9], T[f][3][:9], key, round(m, 2)))
    print(line)
print("same-model pairs, largest z:", sorted(pos, reverse=True)[:8])
print("diff-model pairs, smallest z:", sorted(neg)[:10])
