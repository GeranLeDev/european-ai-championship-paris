"""Full combine2 pipeline (regime + rectangle override + net threshold) with SAM snapping of partial images,
tuned per method with 2-fold CV on the odd half (same split/seed as combine2). Official miou_image.

    python pipeline_eval.py [--run trustmark=<dir>,...] [--out <dir>]   (writes snapped val/test probs if --out)"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/sprintml/starter_kit")
from wmloc.utils import miou_image  # noqa: E402

from check_overlap import load_sam  # noqa: E402
from snap_eval import fix_grid, sam128, snap  # noqa: E402

T1 = Path("/Users/amil/Downloads/hackathon-watermark/modal_t1")
DATA = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release")
METHODS = ("pixelseal", "trustmark", "stegastamp")
DEFAULT = {"pixelseal": T1 / "pixelseal_research/runs/hires_c", "trustmark": T1 / "runs/trustmark_mix",
           "stegastamp": T1 / "runs/stegastamp_mix"}
S_GRID = np.round(np.arange(0.1, 0.91, 0.05), 2); T_GRID = [9.0] + list(np.round(np.arange(0.3, 0.96, 0.05), 2))
D_GRID = (-4, 0, 2, 4); TAU = (9.0, 0.3, 0.4, 0.5, 0.6, 0.7)   # 9 = no snapping
T_FULL = 0.80


def box_grid(box, d):
    m = np.zeros((512, 512), np.float32)
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    m[max(0, y0 - d):min(512, y1 + d), max(0, x0 - d):min(512, x1 + d)] = 1
    g = (m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5).astype(np.uint8)
    return g if 0 < g.sum() < g.size else None


def load_rect(split, tag="_f"):
    best = {}
    pat = re.compile(rf"search_{split}_(\w+?){re.escape(tag)}(_shard\d+)?\.npz$")
    for p in sorted((T1 / "cache_dl").rglob(f"search_{split}_*.npz")):
        if pat.fullmatch(p.name):
            z = np.load(p)
            for n, b, s in zip(z["names"].tolist(), z["box"], z["score"]):
                if n not in best or s > best[n][1]:
                    best[n] = (np.asarray(b, np.float32), float(s))
    return best


def load_probs(d, split):
    f = Path(d) / f"{split}_probs.npz"
    if split == "train" and not f.exists():
        f = Path(d) / "val_probs.npz"
    z = np.load(f)
    return dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))


def candidates(n, prob, M, partial, rect):
    """Per image: net grids G[s][tau], rect grids R[d][tau] (None if no box), rect score."""
    G = {}
    for s in S_GRID:
        b = fix_grid(prob, s)
        G[s] = {tau: (snap(b, M, tau, classes=(1,)) if (partial and tau < 9) else b) for tau in TAU}
    R, rs = {}, -1.0
    if partial and n in rect:
        rs = rect[n][1]
        for d in D_GRID:
            g = box_grid(rect[n][0], d)
            R[d] = None if g is None else {tau: (snap(g, M, tau, classes=(1,)) if tau < 9 else g) for tau in TAU}
    return G, R, rs


def score_table(items, cand, gt):
    """miou per image for every candidate grid."""
    out = []
    for n in items:
        G, R, rs = cand[n]
        g = gt[n]
        sg = {(s, tau): miou_image(G[s][tau], g) for s in S_GRID for tau in TAU}
        sr = {(d, tau): miou_image(R[d][tau], g) for d in R if R[d] is not None for tau in TAU}
        out.append((sg, sr, rs))
    return out


def combo(tab, idx, s, t, d, tn, tr):
    v = []
    for i in idx:
        sg, sr, rs = tab[i]
        v.append(sr[(d, tr)] if (rs >= t and (d, tr) in sr) else sg[(s, tn)])
    return float(np.mean(v))


def tune(tab, idx, snapping=True):
    best = None
    taus = TAU if snapping else (9.0,)
    for s in S_GRID:
        for tn in taus:
            for t in T_GRID:
                for d in (D_GRID if t < 9 else (0,)):
                    for tr in (taus if t < 9 else (9.0,)):
                        sc = combo(tab, idx, s, t, d, tn, tr)
                        if best is None or sc > best[0]:
                            best = (sc, s, t, d, tn, tr)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default=""); ap.add_argument("--out", default="")
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    runs = dict(DEFAULT); runs.update({k: Path(v) for k, v in (kv.split("=", 1) for kv in a.run.split(",") if kv)})
    sam = load_sam("train")
    acc = json.loads((T1 / "cache" / "global_train.json").read_text()); rect = load_rect("train")
    z = np.load(DATA / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
    meta = json.loads((DATA / "train" / "metadata.json").read_text())
    rng = np.random.default_rng(0)
    res = {}
    for m in a.methods.split(","):
        probs = load_probs(runs[m], "train")
        items = [e["image_name"] for e in meta if e["method"] == m and e["image_name"] in probs]
        cand = {n: candidates(n, probs[n], sam128(sam[n][0], sam[n][1]), acc[n] < T_FULL, rect) for n in items}
        tab = score_table(items, cand, gt)
        idx = rng.permutation(len(items)); halves = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
        cv = {}
        for snapping in (False, True):
            v = []
            for h in (0, 1):
                b = tune(tab, halves[h], snapping)
                v += [combo(tab, [i], *b[1:]) for i in halves[1 - h]]
            cv[snapping] = float(np.mean(v))
        full = tune(tab, range(len(items)), True)
        res[m] = (cv[False], cv[True], full)
        print(f"{m:11s} combine2 CV {cv[False]:.4f} | + SAM snap CV {cv[True]:.4f} | params s,t,d,tau_net,tau_rect {full[1:]}", flush=True)
    print(f"mean       combine2 CV {np.mean([r[0] for r in res.values()]):.4f} | + SAM snap CV {np.mean([r[1] for r in res.values()]):.4f}")
    Path(Path(__file__).parent / "pipeline_eval.json").write_text(json.dumps({m: dict(cv_base=r[0], cv_snap=r[1], params=r[2]) for m, r in res.items()}))
