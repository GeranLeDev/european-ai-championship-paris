"""Le decodeur comme juge : lit-on le message sur l'image entiere ? sur la boite de la zone marquee ?
Usage : python juge_decodeur.py <starter_kit> <release> <models> <sortie.json>"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

KIT, DATA, MODELS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
sys.path.insert(0, str(KIT))
from wmloc.data import LocalisationDataset
from wmloc.utils import to_tensor
from wmloc.watermarks import load_watermark

torch.set_num_threads(4)
ds = LocalisationDataset(DATA, "train")
wms = {}

def judge(method, img, msg):
    """Renvoie (part des bits bien lus, marge moyenne) pour une image ou un morceau d'image."""
    if method not in wms: wms[method] = load_watermark(method, MODELS, device="cpu")
    lg = wms[method].logits(to_tensor(img))[0].cpu().numpy()
    return float(((lg > 0) == (msg > 0)).mean()), float((lg * (msg.astype(np.float32) * 2 - 1)).mean())

def kind_of(box):
    h, w = box.shape
    if box.all(): return "barre" if max(h / w, w / h) > 4 else "rectangle"
    return "objet"

rows = []; t0 = time.time()
for i in range(len(ds)):
    it = ds[i]; img, msg, mk = it["image"], it["message"], it["mask"].astype(bool)
    r = dict(name=it["image_name"], method=it["method"], full=bool(it["full_watermark"]), frac=float(mk.mean()))
    r["acc_img"], r["mar_img"] = judge(it["method"], img, msg)
    if not r["full"]:
        lab, n = ndimage.label(mk); comps = []
        for k, sl in enumerate(ndimage.find_objects(lab), 1):
            box = lab[sl] == k
            if box.sum() < 64: continue
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            acc, mar = judge(it["method"], img[y0:y1, x0:x1], msg)
            comps.append(dict(kind=kind_of(box), box=[y0, y1, x0, x1], area=int(box.sum()), acc=acc, mar=mar))
        r["comps"] = comps
    rows.append(r)
    if (i + 1) % 100 == 0: print(f"{i + 1}/{len(ds)} ({time.time() - t0:.0f} s)", flush=True)
OUT.write_text(json.dumps(rows))

print("\n== Image entiere : part des bits bien lus")
for meth in ("pixelseal", "trustmark", "stegastamp"):
    for flag in (True, False):
        a = np.array([r["acc_img"] for r in rows if r["method"] == meth and r["full"] == flag])
        print(f"{meth:10s} {'complet' if flag else 'partiel':8s}: mediane {np.median(a):.3f}, min {a.min():.3f}, max {a.max():.3f}, part > 0.75 : {np.mean(a > 0.75):.2f}")
print("\n== Images partielles : bits bien lus sur la boite de chaque morceau")
agg = defaultdict(list)
for r in rows:
    for c in r.get("comps", []): agg[(r["method"], c["kind"])].append(c["acc"])
for (meth, kind), a in sorted(agg.items()):
    a = np.array(a); print(f"{meth:10s} {kind:9s}: n {len(a):3d}, mediane {np.median(a):.3f}, part > 0.9 : {np.mean(a > 0.9):.2f}, part < 0.65 : {np.mean(a < 0.65):.2f}")
