"""Soumission finale de la tâche 1 : réseau de segmentation + recherche de rectangle.

Par image :
  - si l'image est « partielle » (message relu à < 80 % sur l'image entière) et que la recherche a trouvé un cadre
    avec un score >= t : on rend ce rectangle (agrandi de d pixels) ;
  - sinon : la carte du réseau, coupée au seuil s ;
  - jamais de grille vide ni pleine (aucune vraie zone ne l'est).
t, d et s sont choisis par outil sur les 510 images officielles du train, que le réseau n'a jamais vues à
l'entraînement. Le chiffre affiché est en validation croisée (réglé sur une moitié, noté sur l'autre).

Usage (venv du kit) :
    python combine.py --data <release> --kit <starter_kit> --runs runs --tag _v1 --cache cache --rect-tag _f \
        [--test submission_t1.pt]"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--kit", required=True)
ap.add_argument("--runs", default="runs"); ap.add_argument("--tag", default="_v1")
ap.add_argument("--cache", default="cache"); ap.add_argument("--rect-tag", default="_f")
ap.add_argument("--test", default="")
a = ap.parse_args()
sys.path.insert(0, a.kit)
from wmloc.utils import miou_image  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
DATA, CACHE, RUNS = Path(a.data), Path(a.cache), Path(a.runs)
T_FULL = 0.80


def fix_grid(prob, thr):
    p = (prob >= thr).astype(np.uint8); k = max(1, int(0.01 * p.size))
    if p.sum() == 0:
        p.flat[np.argsort(prob.ravel())[-k:]] = 1
    elif p.sum() == p.size:
        p.flat[np.argsort(prob.ravel())[:k]] = 0
    return p


def box_mask(box, d):
    m = np.zeros((512, 512), np.float32)
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    m[max(0, y0 - d):min(512, y1 + d), max(0, x0 - d):min(512, x1 + d)] = 1
    g = (m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5).astype(np.uint8)
    return g if 0 < g.sum() < g.size else None


def load_rect(split):
    best = {}
    pat = re.compile(rf"search_{split}_(\w+?){re.escape(a.rect_tag)}(_shard\d+)?\.npz$")
    for p in sorted(CACHE.glob(f"search_{split}_*.npz")):
        if pat.fullmatch(p.name):
            z = np.load(p)
            for n, b, s in zip(z["names"].tolist(), z["box"], z["score"]):
                if n not in best or s > best[n][1]:
                    best[n] = (np.asarray(b, np.float32), float(s))
    return best


def load_probs(split, m):
    z = np.load(RUNS / f"{m}{a.tag}" / f"{split}_probs.npz")
    return dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))


def predict(name, prob, acc, rect, s, t, d):
    if acc[name] < T_FULL and name in rect and rect[name][1] >= t:
        g = box_mask(rect[name][0], d)
        if g is not None:
            return g
    return fix_grid(prob, s)


acc_tr = json.loads((CACHE / "global_train.json").read_text())
rect_tr = load_rect("train")
z = np.load(DATA / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
meta_tr = json.loads((DATA / "train" / "metadata.json").read_text())
S_GRID = np.round(np.arange(0.1, 0.91, 0.05), 2); T_GRID = [9.0] + list(np.round(np.arange(0.3, 0.96, 0.05), 2)); D_GRID = (-4, 0, 2, 4)


def tune(items, probs):
    best = None
    for s in S_GRID:
        for t in T_GRID:
            for d in (D_GRID if t < 9 else (0,)):
                sc = np.mean([miou_image(predict(e["image_name"], probs[e["image_name"]], acc_tr, rect_tr, s, t, d), gt[e["image_name"]]) for e in items])
                if best is None or sc > best[0]:
                    best = (sc, s, t, d)
    return best


params, rows = {}, []
rng = np.random.default_rng(0)
for m in METHODS:
    probs = load_probs("train", m)
    items = [e for e in meta_tr if e["method"] == m]
    net_only = max(np.mean([miou_image(fix_grid(probs[e["image_name"]], s), gt[e["image_name"]]) for e in items]) for s in S_GRID)
    idx = rng.permutation(len(items)); halves = [idx[: len(idx) // 2], idx[len(idx) // 2:]]; cv = []
    for h in (0, 1):
        _, s, t, d = tune([items[i] for i in halves[h]], probs)
        cv += [miou_image(predict(items[i]["image_name"], probs[items[i]["image_name"]], acc_tr, rect_tr, s, t, d), gt[items[i]["image_name"]]) for i in halves[1 - h]]
    sc, s, t, d = tune(items, probs); params[m] = (s, t, d)
    rows.append((m, net_only, float(np.mean(cv)), s, t, d))
print("outil       | réseau seul | réseau + rectangles (validation croisée) | seuil réseau | seuil cadre | marge")
for m, net, cv, s, t, d in rows:
    print(f"{m:11s} | {net:.3f}       | {cv:.3f}                                    | {s:.2f}         | {'aucun' if t >= 9 else f'{t:.2f}'}       | {d:+d}")
print(f"global      | {np.mean([r[1] for r in rows]):.3f}       | {np.mean([r[2] for r in rows]):.3f}")

if a.test:
    import torch
    meta_te = json.loads((DATA / "test" / "metadata.json").read_text())
    acc_te = json.loads((CACHE / "global_test.json").read_text()); rect_te = load_rect("test")
    probs_te = {}
    for m in METHODS:
        probs_te.update(load_probs("test", m))
    names, masks = [], []
    for e in meta_te:
        s, t, d = params[e["method"]]
        p = predict(e["image_name"], probs_te[e["image_name"]], acc_te, rect_te, s, t, d)
        assert p.shape == (128, 128) and 0 < p.sum() < p.size
        names.append(e["image_name"]); masks.append(p)
    torch.save({"image_names": names, "masks": torch.from_numpy(np.stack(masks)).bool()}, a.test)
    print(f"\nsoumission écrite : {a.test} ({len(names)} images, aucune grille vide)")
