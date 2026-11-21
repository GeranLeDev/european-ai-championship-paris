"""Comment les zones partielles ont-elles ete fabriquees ? On teste 3 hypotheses de 'morceau watermarke' :
  F  = toute l'image a ete marquee, puis on n'a garde que la zone
  B  = la boite englobante de la zone a ete marquee toute seule
  B2 = la boite elargie pour avoir un rapport <= 2
Pour chaque hypothese on recalcule la trace avec le message connu et on mesure a quel point l'image la contient
dans la zone (correlation : ~0 = mauvaise hypothese, nettement > 0 = bonne).
Usage : python regle_geometrie.py <starter_kit> <release> <models> <juge_train.json> [nb par methode]"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

KIT, DATA, MODELS, JUGE = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
NB = int(sys.argv[5]) if len(sys.argv) > 5 else 40
sys.path.insert(0, str(KIT))
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark

torch.set_num_threads(4)
juge = {r["name"]: r for r in json.loads(JUGE.read_text())}

def gauss(x, sigma=2.0):
    return np.stack([ndimage.gaussian_filter(x[..., c], sigma) for c in range(3)], -1)

def expand_ar2(y0, y1, x0, x1):
    h, w = y1 - y0, x1 - x0
    if h > 2 * w:
        add = (h / 2 - w) / 2; x0, x1 = int(max(0, x0 - add)), int(min(512, x1 + add))
    elif w > 2 * h:
        add = (w / 2 - h) / 2; y0, y1 = int(max(0, y0 - add)), int(min(512, y1 + add))
    return y0, y1, x0, x1

agg = defaultdict(list); sizes = defaultdict(list)
for name in ("trustmark", "stegastamp", "pixelseal"):
    ds = LocalisationDataset(DATA, "train", methods=[name]); wm = load_watermark(name, MODELS, device="cpu"); done = 0
    for i in range(len(ds)):
        if ds.meta[i]["full_watermark"] or done >= NB: continue
        it = ds[i]; img, msg, mk = it["image"], it["message"], it["mask"].astype(bool); done += 1
        h_img = img.astype(np.float32) - gauss(img.astype(np.float32))
        lab, n = ndimage.label(mk)
        for k, sl in enumerate(ndimage.find_objects(lab), 1):
            comp = lab == k
            if comp.sum() < 400: continue
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            box = comp[y0:y1, x0:x1]; hh, ww = box.shape
            kind = ("barre" if max(hh / ww, ww / hh) > 4 else "rectangle") if box.all() else "objet"
            for hyp, (a, b, c, d) in (("F", (0, 512, 0, 512)), ("B", (y0, y1, x0, x1)), ("B2", expand_ar2(y0, y1, x0, x1))):
                crop = img[a:b, c:d]
                tr = np.zeros((512, 512, 3), np.float32)
                tr[a:b, c:d] = wm.embed(np.ascontiguousarray(crop), msg).astype(np.float32) - crop
                q = tr - gauss(tr)
                num = float((h_img[comp] * q[comp]).sum()); den = float(np.sqrt((h_img[comp] ** 2).sum() * (q[comp] ** 2).sum()) + 1e-6)
                agg[(name, kind, hyp)].append(num / den)
            acc = next((c["acc"] for c in juge[it["image_name"]].get("comps", []) if c["box"] == [y0, y1, x0, x1]), np.nan)
            sizes[(name, kind)].append((min(hh, ww), max(hh / ww, ww / hh), acc))
print("correlation mediane image/trace recalculee dans la zone, par hypothese")
for name in ("trustmark", "stegastamp", "pixelseal"):
    for kind in ("rectangle", "barre", "objet"):
        if (name, kind, "F") not in agg: continue
        v = {h: np.array(agg[(name, kind, h)]) for h in ("F", "B", "B2")}
        best = Counter = None
        wins = np.argmax(np.stack([v["F"], v["B"], v["B2"]]), 0)
        print(f"{name:10s} {kind:9s} n={len(v['F']):3d} | F {np.median(v['F']):+.2f} | B {np.median(v['B']):+.2f} | B2 {np.median(v['B2']):+.2f} | gagnant F/B/B2 : {(wins == 0).mean():.2f}/{(wins == 1).mean():.2f}/{(wins == 2).mean():.2f}")
print("\nrectangles : bits relus sur la boite selon le petit cote")
for name in ("trustmark", "stegastamp", "pixelseal"):
    s = np.array(sizes.get((name, "rectangle"), [])).reshape(-1, 3)
    for lo, hi in ((0, 48), (48, 96), (96, 160), (160, 513)):
        m = (s[:, 0] >= lo) & (s[:, 0] < hi)
        if m.any(): print(f"{name:10s} cote {lo:3d}-{hi:3d}: n {m.sum():2d}, bits relus medians {np.nanmedian(s[m, 2]):.2f}, rapport max {s[m, 1].max():.1f}")
