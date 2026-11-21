"""Summaries of rows produced by audit_official.py / audit_generated.py.
usage: summarize.py <rows.json> <res: r512|r128> [label]"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shapes_lib import fmt, pct

METHODS = ("pixelseal", "trustmark", "stegastamp")


def simp(k):
    return {"near_rect": "object"}.get(k, k)


def summarize(rows, res, label=""):
    out = {}
    for meth in METHODS:
        for full in (True, False):
            R = [r for r in rows if r["method"] == meth and r["full"] == full]
            if not R:
                continue
            key = f"{meth}/{'full' if full else 'partial'}"
            fr = [r[res]["frac"] for r in R]
            ncomp = Counter(min(r[res]["n"], 4) for r in R)
            N = len(R)
            comps = [c for r in R for c in r[res]["comps"]]
            kinds = Counter(simp(c["kind"]) for c in comps)
            # image-level kind signature: kind of the largest component / set of kinds
            lead = Counter(simp(r[res]["comps"][0]["kind"]) if r[res]["comps"] else "none" for r in R)
            combo = Counter("+".join(sorted(simp(c["kind"]) for c in r[res]["comps"])) or "none" for r in R)
            d = dict(n_img=N, minority_frac_pct=pct(fr),
                     ncomp={str(k): v / N for k, v in sorted(ncomp.items())},
                     comp_kind={k: v / max(len(comps), 1) for k, v in kinds.most_common()},
                     lead_kind={k: v / N for k, v in lead.most_common()},
                     combos={k: v / N for k, v in combo.most_common(8)},
                     n_comp_total=len(comps))
            by = defaultdict(list)
            for c in comps:
                by[simp(c["kind"])].append(c)
            d["kinds"] = {}
            for k, C in by.items():
                dd = dict(n=len(C), area_pct=pct([c["area"] for c in C]), ar_pct=pct([c["ar"] for c in C]),
                          ar_max=float(max(c["ar"] for c in C)),
                          short_pct=pct([min(c["h"], c["w"]) for c in C]), long_pct=pct([max(c["h"], c["w"]) for c in C]),
                          vertical=float(np.mean([c["vertical"] for c in C])),
                          touch=float(np.mean([c["touch"] for c in C])),
                          cy_pct=pct([c["cy"] for c in C]), cx_pct=pct([c["cx"] for c in C]),
                          extent_pct=pct([c["extent"] for c in C]), solidity_pct=pct([c["solidity"] for c in C]),
                          compact_pct=pct([c["compact"] for c in C]),
                          ar_gt2=float(np.mean([c["ar"] > 2.0 for c in C])))
                d["kinds"][k] = dd
            out[key] = d
    return out


def show(out, label):
    print(f"\n================ {label}")
    for key, d in out.items():
        print(f"\n--- {key}  (n={d['n_img']}, comps={d['n_comp_total']})")
        print("  minority frac p5/25/50/75/95:", fmt(d["minority_frac_pct"], 4))
        print("  ncomp:", {k: round(v, 3) for k, v in d["ncomp"].items()})
        print("  comp kinds:", {k: round(v, 3) for k, v in d["comp_kind"].items()})
        print("  lead kind :", {k: round(v, 3) for k, v in d["lead_kind"].items()})
        print("  combos    :", {k: round(v, 3) for k, v in d["combos"].items()})
        for k, dd in d["kinds"].items():
            print(f"   [{k}] n={dd['n']} area {fmt(dd['area_pct'], 4)} AR {fmt(dd['ar_pct'], 2)} max {dd['ar_max']:.2f} AR>2 {dd['ar_gt2']:.2f}"
                  f" short {fmt(dd['short_pct'], 0)} long {fmt(dd['long_pct'], 0)} vert {dd['vertical']:.2f} touch {dd['touch']:.2f}")
            print(f"        extent {fmt(dd['extent_pct'], 2)} solidity {fmt(dd['solidity_pct'], 2)} compact {fmt(dd['compact_pct'], 2)}"
                  f" cy {fmt(dd['cy_pct'], 2)} cx {fmt(dd['cx_pct'], 2)}")


if __name__ == "__main__":
    rows = json.loads(Path(sys.argv[1]).read_text())
    res = sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else sys.argv[1]
    if len(sys.argv) > 4:
        par = int(sys.argv[4]); rows = [r for r in rows if r.get("odd") == par]
    out = summarize(rows, res)
    show(out, label + " " + res)
    (HERE / f"summary_{label}_{res}.json").write_text(json.dumps(out, indent=1))
