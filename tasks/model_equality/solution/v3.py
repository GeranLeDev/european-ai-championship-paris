"""Final-stage probe (prompts_v3.jsonl: 15 one-line factual questions, no length cap).
On ref the LoRA endpoints elaborate after the short answer; this script measures it per endpoint.
  python v3.py data/ref_v3.jsonl --truth ../practice_answers.csv [--reps 1]
"""
import argparse, json, math, re, sys
from collections import defaultdict
import numpy as np

FE = {"loglen": lambda y: math.log(len(y) + 1), "nsent": lambda y: math.log(1 + len(re.findall(r"[.!?]+(?:\s|$)", y)))}


def table(rows, reps=None):
    t = {f: defaultdict(lambda: defaultdict(list)) for f in FE}
    for r in rows:
        if reps is not None and r["repeat"] >= reps:
            continue
        for f, fn in FE.items():
            t[f][r["id"]][r["endpoint"]].append(fn(r["response"]))
    return {f: {p: {e: float(np.mean(v)) for e, v in d.items()} for p, d in tt.items()} for f, tt in t.items()}


def zpair(t, e, f, feat="loglen"):
    d = [x[e] - x[f] for x in t[feat].values() if e in x and f in x]
    if len(d) < 3:
        return 0.0, 0.0
    m = float(np.mean(d))
    return m, m / (float(np.std(d, ddof=1)) / math.sqrt(len(d)) + 0.02)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("answers"); ap.add_argument("--truth"); ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--fam", default=None, help="json {endpoint: [family, ...]}")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.answers)]
    kind = {}
    if a.truth:
        kind = {l.split(",")[0]: l.strip().split(",")[3][:14] for l in open(a.truth).readlines()[1:]}
    fam = {e: "F" for e in {r["endpoint"] for r in rows}}
    if a.fam:
        F = json.load(open(a.fam))["fam"]
        fam = {e: F[e][0] for e in fam}
        kind = {e: F[e][2][:14] for e in fam}
    t = table(rows, a.reps)
    for Fm in sorted(set(fam.values())):
        eps = sorted(e for e in fam if fam[e] == Fm)
        # per-prompt family median as reference
        print("==", Fm, "| per endpoint: mean (log length - family median) and its z ; then pairwise z (loglen)")
        dev = {}
        for e in eps:
            d = [x[e] - float(np.median([x[g] for g in eps if g in x])) for x in t["loglen"].values() if e in x]
            ds = [x[e] - float(np.median([x[g] for g in eps if g in x])) for x in t["nsent"].values() if e in x]
            dev[e] = (np.mean(d), np.mean(d) / (np.std(d, ddof=1) / math.sqrt(len(d)) + 0.02), np.mean(ds), np.median([math.exp(x[e]) for x in t["loglen"].values() if e in x]))
        print(" " * 44 + " ".join(e[-2:].rjust(5) for e in eps))
        for e in sorted(eps, key=lambda e: dev[e][0]):
            print(f"{e} {kind.get(e,''):14s} dev {dev[e][0]:+.2f} z {dev[e][1]:+5.1f} sent {dev[e][2]:+.2f} medlen {dev[e][3]:5.0f} | " + " ".join(f"{zpair(t, e, f)[1]:+5.1f}" if e != f else "    ." for f in eps))
