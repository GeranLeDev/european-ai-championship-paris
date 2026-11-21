"""Exploration: per-endpoint style features, deviation from the family median per prompt (paired over prompts)."""
import json, re, sys, math
from collections import defaultdict
import numpy as np
ans, truth = sys.argv[1], sys.argv[2]
fam_only = sys.argv[3] if len(sys.argv) > 3 else None
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
 'wlen': lambda y: np.mean([len(w) for w in re.findall(r"[A-Za-z']+", y)] or [0]),
 'nonasc': lambda y: math.log(1 + sum(ord(c) > 127 for c in y)),
 'I': lambda y: math.log(1 + len(re.findall(r"\bI\b|\bI'm\b", y))),
}
for sel in (('short', 'ident'), ('long',)):
    print("== prompts:", sel)
    val = {f: defaultdict(dict) for f in FE}
    for f, fn in FE.items():
        acc = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if kinds[r['id']] in sel:
                acc[r['id']][r['endpoint']].append(fn(r['response']))
        for pid, d in acc.items():
            for e, v in d.items():
                val[f][pid][e] = np.mean(v)
    print("endpoint kind        " + "".join(f"{f:>14s}" for f in FE))
    for e in eps:
        line = f"{e} {T[e][3][:11]:11s}"
        for f in FE:
            dev = [d[e] - np.median(list(d.values())) for d in val[f].values() if e in d]
            m, se = np.mean(dev), np.std(dev) / math.sqrt(len(dev)) + 1e-9
            line += f"  {m:+.2f}({m/se:+5.1f})"
        print(line)
