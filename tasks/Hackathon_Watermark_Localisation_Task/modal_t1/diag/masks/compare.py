"""Official (128 grid) vs our shards (128 grid), per method and effective regime."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shapes_lib import pct, fmt
off = json.loads((HERE / "official_rows.json").read_text())
gen = json.loads((HERE / "generated_rows.json").read_text())
METHODS = ("pixelseal", "trustmark", "stegastamp")
simp = lambda k: {"near_rect": "object", "speck": "object"}.get(k, k)

def stats(R):
    N = max(len(R), 1); comps = [c for r in R for c in r["r128"]["comps"]]
    lead = Counter(simp(r["r128"]["comps"][0]["kind"]) if r["r128"]["comps"] else "none" for r in R)
    nc = Counter(min(r["r128"]["n"], 4) for r in R)
    kinds_img = [set(simp(c["kind"]) for c in r["r128"]["comps"]) for r in R]
    single_rect = np.mean([r["r128"]["n"] == 1 and simp(r["r128"]["comps"][0]["kind"]) in ("rect", "bar") for r in R]) if R else 0
    rect_mixed = np.mean([bool(k & {"rect", "bar"}) and r["r128"]["n"] > 1 for k, r in zip(kinds_img, R)]) if R else 0
    rl = [c for c in comps if c["kind"] in ("rect", "bar")]
    ob = [c for c in comps if simp(c["kind"]) == "object"]
    el = [c for c in comps if c["kind"] == "ellipse"]
    ar = np.array([c["ar"] for c in rl]) if rl else np.array([1.0])
    return dict(n=len(R), frac=pct([r["r128"]["frac"] for r in R]),
                nc=[nc.get(k, 0) / N for k in (0, 1, 2, 3, 4)],
                lead={k: lead.get(k, 0) / N for k in ("rect", "bar", "ellipse", "object", "none")},
                single_rect=float(single_rect), rect_mixed=float(rect_mixed),
                rect_ar=[float(np.mean(ar <= 2)), float(np.mean((ar > 2) & (ar <= 4))), float(np.mean(ar > 4))], n_rect=len(rl),
                rect_area=pct([c["area"] for c in rl]), rect_short=pct([min(c["h"], c["w"]) for c in rl]),
                rect_touch=float(np.mean([c["touch"] for c in rl])) if rl else float("nan"),
                obj_area=pct([c["area"] for c in ob]), obj_touch=float(np.mean([c["touch"] for c in ob])) if ob else float("nan"),
                obj_sol=pct([c["solidity"] for c in ob]), obj_ext=pct([c["extent"] for c in ob]), obj_comp=pct([c["compact"] for c in ob]),
                comp_kind={k: v / max(len(comps), 1) for k, v in Counter(simp(c["kind"]) for c in comps).items()},
                ell_share_img=float(np.mean(["ellipse" in k for k in kinds_img])) if R else 0,
                tiny_comp=float(np.mean([c["area"] < 0.0005 for c in comps])) if comps else 0)

def line(tag, s):
    print(f"  {tag:9s} n={s['n']:4d} frac{fmt(s['frac'],4)} ncomp0-4+{fmt(s['nc'],2)} lead r/b/e/o/none "
          f"{s['lead']['rect']:.2f}/{s['lead']['bar']:.2f}/{s['lead']['ellipse']:.2f}/{s['lead']['object']:.2f}/{s['lead']['none']:.2f}"
          f" 1rect-only {s['single_rect']:.2f} rect+other {s['rect_mixed']:.2f}")
    print(f"            rectAR<=2/2-4/>4 {fmt(s['rect_ar'],2)} (n={s['n_rect']}) rect area{fmt(s['rect_area'],4)} short{fmt(s['rect_short'],0)} touch {s['rect_touch']:.2f}")
    print(f"            obj area{fmt(s['obj_area'],4)} touch {s['obj_touch']:.2f} solidity{fmt(s['obj_sol'],2)} extent{fmt(s['obj_ext'],2)} compact{fmt(s['obj_comp'],2)} tiny<0.0005 {s['tiny_comp']:.2f} img-with-ellipse {s['ell_share_img']:.2f}")

res = {}
print("== label sanity of our shards")
for src in ("coco", "off"):
    for m in METHODS:
        R = [r for r in gen if r["src"] == src and r["method"] == m]
        tagfull = np.mean([r["tag"] == "full" for r in R]); efffull = np.mean([r["full"] for r in R])
        conf = Counter((r["tag"], "eff_full" if r["full"] else "eff_part") for r in R)
        allone = np.mean([r["allone"] for r in R]); empty = np.mean([r["empty"] for r in R])
        hi = np.mean([r["frac_marked"] > 0.995 for r in R]); mid = np.mean([0.45 < r["frac_marked"] < 0.55 for r in R])
        print(f"  {src:4s} {m:10s} n={len(R)} tag-full {tagfull:.3f} effective-full {efffull:.3f} | tag x effective {dict(conf)} | all-ones grid {allone:.3f} empty {empty:.3f} marked>99.5% {hi:.3f}")
        res[f"sanity/{src}/{m}"] = dict(tag_full=float(tagfull), eff_full=float(efffull), allone=float(allone), empty=float(empty), gt995=float(hi))
print("\n== per method / effective regime (128 grid both sides)")
for m in METHODS:
    for full in (True, False):
        print(f"\n--- {m} / {'full (minority = holes)' if full else 'partial (minority = zones)'}")
        s = stats([r for r in off if r["method"] == m and r["full"] == full]); line("official", s); res[f"{m}/{full}/official"] = s
        for src in ("coco", "off"):
            s = stats([r for r in gen if r["method"] == m and r["full"] == full and r["src"] == src]); line("ours-" + src, s); res[f"{m}/{full}/{src}"] = s
(HERE / "compare_128.json").write_text(json.dumps(res, indent=1))
