"""Empreinte 'differentielle' : trace(message) - trace(message inverse). Elle annule la partie de la trace qui
depend seulement de l'image, et garde ce qui depend du message.
A. Sur nos propres images (verite connue) : la carte retrouve-t-elle un trou propre dans une image marquee ?
B. Sur les images officielles 'completes' : meme question (si A et B donnent pareil, notre fabrication = la leur).
Usage : python empreinte_diff.py <starter_kit> <release> <models> <photos> [nb par methode]"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

KIT, DATA, MODELS, PHOTOS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
NB = int(sys.argv[5]) if len(sys.argv) > 5 else 30
sys.path.insert(0, str(KIT))
from wmloc.data import LocalisationDataset
from wmloc.utils import downsample_mask, miou_image
from wmloc.watermarks import load_watermark

torch.set_num_threads(4)

def load512(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = 512 / min(w, h)
    im = im.resize((max(512, round(w * s)), max(512, round(h * s))), Image.BICUBIC)
    w, h = im.size; x0, y0 = (w - 512) // 2, (h - 512) // 2
    return np.asarray(im.crop((x0, y0, x0 + 512, y0 + 512)))

def hp(x, sigma):
    return x - np.stack([ndimage.gaussian_filter(x[..., c], sigma) for c in range(3)], -1)

def beta_map(h, d, win=16):
    """Sur la grille 128 : a quel point l'image (passe-haut h) contient l'empreinte d, par fenetre."""
    t = lambda z: torch.from_numpy(z.sum(-1))[None, None].float()
    box = lambda z: F.avg_pool2d(z, win, stride=4, padding=(win - 4) // 2, count_include_pad=False)
    return (box(t(h * d)) / (box(t(d * d)) + 1e-3))[0, 0].numpy()

def auc(score, gt):
    s = score.ravel(); g = gt.ravel().astype(bool)
    if g.all() or not g.any(): return np.nan
    r = np.argsort(np.argsort(s)) + 1
    return float((r[g].sum() - g.sum() * (g.sum() + 1) / 2) / (g.sum() * (~g).sum()))

def maps_for(wm, img, msg):
    x = img.astype(np.float32)
    e1 = wm.embed(img, msg).astype(np.float32); e0 = wm.embed(img, 1 - msg).astype(np.float32)
    out = {}
    for s in (1.5, 4.0):
        h = hp(x, s)
        out[f"simple s{s}"] = beta_map(h, hp(e1 - x, s)); out[f"diff s{s}"] = beta_map(h, hp(e1 - e0, s))
    return out

photos = [load512(f) for f in sorted(PHOTOS.glob("*.jpg"))[10:16]]
rng = np.random.default_rng(1)
hole = np.zeros((512, 512), bool); hole[150:310, 200:360] = True
for name in ("pixelseal", "trustmark", "stegastamp"):
    wm = load_watermark(name, MODELS, device="cpu")
    print(f"== {name}")
    res = {}
    for im in photos:
        msg = rng.integers(0, 2, wm.nbits)
        w = wm.embed(im, msg); x = np.where(hole[..., None], im, w)
        gt = downsample_mask((~hole).astype(np.uint8))
        for k, m in maps_for(wm, x, msg).items(): res.setdefault(k, []).append(auc(m, gt))
    print("  A. nos images (trou carre de 160 px) : AUC " + " | ".join(f"{k} {np.mean(v):.3f}" for k, v in res.items()))
    ds = LocalisationDataset(DATA, "train", methods=[name])
    items = [ds[i] for i in range(len(ds)) if ds.meta[i]["full_watermark"]][:NB]
    res, maps, gts = {}, {}, [it["mask128"] for it in items]
    for it in items:
        for k, m in maps_for(wm, it["image"], it["message"]).items():
            res.setdefault(k, []).append(auc(m, it["mask128"])); maps.setdefault(k, []).append(m)
    print("  B. images officielles completes     : AUC " + " | ".join(f"{k} {np.nanmedian(v):.3f}" for k, v in res.items()))
    for k, ms in maps.items():
        if not k.startswith("diff"): continue
        ts = np.quantile(np.concatenate([m.ravel() for m in ms]), np.linspace(0.02, 0.98, 49))
        print(f"     {k}: meilleur mIoU avec un seul seuil {max(np.mean([miou_image(m > t, g) for m, g in zip(ms, gts)]) for t in ts):.3f}")
