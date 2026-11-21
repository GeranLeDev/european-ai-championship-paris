"""Fouille des 510 images officielles : messages, formes des masques, doublons, traces JPEG."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.fft import dctn

root = Path(sys.argv[1]) / "train"
meta = json.loads((root / "metadata.json").read_text())
by = {m["image_name"]: m for m in meta}
METHODS = ("pixelseal", "trustmark", "stegastamp")

print("== A. messages")
for meth in METHODS:
    msgs = [m["message"] for m in meta if m["method"] == meth]
    bits = np.array([[int(c) for c in s] for s in msgs])
    const = int(((bits.mean(0) == 0) | (bits.mean(0) == 1)).sum())
    print(f"{meth}: {len(msgs)} messages, {len(set(msgs))} uniques, part de 1 = {bits.mean():.3f}, bits constants = {const}")

print("== B. grille 128 : les deux classes toujours presentes ?")
z = np.load(root / "masks_128.npz"); g = dict(zip(z["names"].tolist(), z["masks"]))
print("tout a 1:", sum(int(v.all()) for v in g.values()), "| tout a 0:", sum(int(not v.any()) for v in g.values()))
for flag in (False, True):
    a = np.array([(v.size - v.sum() if flag else v.sum()) for n, v in g.items() if by[n]["full_watermark"] == flag])
    print(f"full={flag}: cases de la petite classe: min {a.min()}, p10 {np.percentile(a,10):.0f}, p25 {np.percentile(a,25):.0f}, mediane {np.median(a):.0f}, p75 {np.percentile(a,75):.0f}, max {a.max()}")

print("== C. formes (zone marquee si partiel, trou propre si complet)")
def comp_kind(box):
    h, w = box.shape
    if box.all(): return "barre" if max(h / w, w / h) > 4 else "rectangle"
    yy, xx = np.mgrid[0:h, 0:w]
    ell = ((yy - (h - 1) / 2) / (h / 2)) ** 2 + ((xx - (w - 1) / 2) / (w / 2)) ** 2 <= 1
    return "ellipse" if (box & ell).sum() / (box | ell).sum() > 0.93 else "objet/tache"
rows = []
for m in meta:
    mk = np.asarray(Image.open(root / "masks" / f"{m['image_name']}.png").convert("L")) > 127
    b = ~mk if m["full_watermark"] else mk
    lab, n = ndimage.label(b)
    kinds = []
    for sl in ndimage.find_objects(lab):
        kinds.append(comp_kind(lab[sl] > 0))
    ys, xs = np.where(b); h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1
    rows.append(dict(method=m["method"], full=m["full_watermark"], kinds="+".join(sorted(set(kinds))), n=n, h=h, w=w, ar=max(h / w, w / h), fill=b[ys.min():ys.max()+1, xs.min():xs.max()+1].mean()))
for flag in (False, True):
    print(f"full={flag}:", Counter(r["kinds"] for r in rows if r["full"] == flag).most_common(8))
print("par methode (partiel) : rapport hauteur/largeur de la boite englobante")
for meth in METHODS:
    a = np.array([r["ar"] for r in rows if r["method"] == meth and not r["full"]])
    k = Counter(r["kinds"] for r in rows if r["method"] == meth and not r["full"]).most_common(4)
    print(f"  {meth}: max {a.max():.1f}, part > 2 : {np.mean(a > 2):.2f} | {k}")
a = np.array([(r["h"], r["w"]) for r in rows if not r["full"] and r["kinds"] in ("rectangle", "barre")])
print("rectangles/barres partiels: n", len(a), "| cote min", a.min(), "| mediane h,w", np.median(a[:, 0]), np.median(a[:, 1]))

print("== D. doublons entre fichiers")
sig = []
for m in meta:
    im = Image.open(root / "images" / f"{m['image_name']}.png").convert("L").resize((16, 16), Image.BILINEAR)
    v = np.asarray(im, np.float32).ravel(); v = (v - v.mean()) / (v.std() + 1e-6); sig.append(v)
S = np.stack(sig); C = S @ S.T / S.shape[1]; np.fill_diagonal(C, 0)
print("paires tres proches (corr > 0.95):", int((np.triu(C) > 0.95).sum()), "| corr max:", round(float(C.max()), 3))
np.save(Path(sys.argv[2]) / "train_sig.npy", S)

print("== E. fichier PNG")
im = Image.open(root / "images" / f"{meta[0]['image_name']}.png"); print(im.mode, im.size, {k: str(v)[:80] for k, v in im.info.items()})

print("== F. traces JPEG (peigne DCT) : zones propres vs zones marquees")
def comb(rgb, ok_px):
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2] - 128
    H, W = y.shape
    blocks = y.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)
    ok = ok_px.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3).all(axis=(2, 3))
    if ok.sum() < 300: return None
    d = dctn(blocks[ok], axes=(1, 2), norm="ortho")
    best = (0.0, 0)
    for (u, v) in ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0)):
        c = d[:, u, v]
        for q in range(2, 25):
            cc = c[np.abs(c) > 0.75 * q]
            if len(cc) < 200: continue
            s = float(np.cos(2 * np.pi * cc / q).mean())
            if s > best[0]: best = (s, q)
    return best
clean_s, wm_s = [], []
for m in meta[:160]:
    rgb = np.asarray(Image.open(root / "images" / f"{m['image_name']}.png").convert("RGB"), np.float64)
    mk = np.asarray(Image.open(root / "masks" / f"{m['image_name']}.png").convert("L")) > 127
    a = comb(rgb, ~mk); b = comb(rgb, mk)
    if a: clean_s.append(a[0])
    if b: wm_s.append(b[0])
print(f"score peigne zones propres : mediane {np.median(clean_s):.2f}, max {np.max(clean_s):.2f} (n={len(clean_s)})")
print(f"score peigne zones marquees: mediane {np.median(wm_s):.2f}, max {np.max(wm_s):.2f} (n={len(wm_s)})")
