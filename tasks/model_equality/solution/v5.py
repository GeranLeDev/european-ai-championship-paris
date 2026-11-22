"""Family A packed quiz (prompts_v5): agreement of each candidate with the same-weights endpoints."""
import json, sys
from collections import defaultdict
import numpy as np
from met_common import quiz_lines, canon
items = json.load(open("prompts_v5_items.json"))
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "data/ep_v5.jsonl")]
ans = defaultdict(dict)
for r in rows:
    its = items[r["id"]]
    found = quiz_lines(r["response"], len(its))
    for j, q in enumerate(its):
        if j + 1 in found:
            a = canon(found[j + 1], "n" if "*" in q else "w")
            if a not in ("<none>",):
                ans[r["endpoint"]][(r["id"], j)] = a
cands = ["ep-24", "ep-25", "ep-18", "ep-01", "ep-17", "ep-12"]; refs = {"WM": "ep-04", "T1": "ep-33", "trunc": "ep-14"}
print("answers parsed per endpoint:", {e: len(v) for e, v in sorted(ans.items())})
def agree(e, f, keys=None):
    ks = [k for k in ans[e] if k in ans[f] and (keys is None or k[0] in keys)]
    return float(np.mean([ans[e][k] == ans[f][k] for k in ks])) if ks else float("nan"), len(ks)
msgs = sorted(items)
rng = np.random.default_rng(0)
wins = {e: 0 for e in cands}; winsm = {e: 0 for e in cands}
for _ in range(1000):
    ks = set(rng.choice(msgs, len(msgs)))
    # resample items rather than messages (only 7 messages)
    allk = [k for k in ans[refs["WM"]]]
    pick = [allk[i] for i in rng.integers(0, len(allk), len(allk))]
    s = {e: np.mean([ans[e].get(k) == ans[refs["WM"]][k] for k in pick if k in ans[e]]) for e in cands if ans[e]}
    if s: wins[max(s, key=s.get)] += 1
print("candidate   agree WM (n)    agree T1    agree trunc    mean    first-rank share (WM, item bootstrap)")
for e in cands:
    if e not in ans: continue
    a = {k: agree(e, r) for k, r in refs.items() if r in ans}
    print(f"  {e}   " + "   ".join(f"{k} {v[0]:.3f} ({v[1]})" for k, v in a.items()) + f"   mean {np.mean([v[0] for v in a.values()]):.3f}   {wins[e]/10:.0f}%")
for e in cands:
    if e in ans:
        print("  among candidates:", e, " ".join(f"{f[-2:]}:{agree(e, f)[0]:.2f}" for f in cands if f != e and f in ans))
