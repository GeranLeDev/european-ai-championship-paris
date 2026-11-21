"""Peut-on recalculer la trace du watermark quand on connait le message, sans avoir l'image propre ?
On marque nous-memes des photos (image propre connue), puis on compare la vraie trace r = marquee - propre a :
  - la trace recalculee depuis l'image deja marquee, avec le BON message
  - la meme chose avec un MAUVAIS message (controle)
  - la trace du meme message sur une AUTRE photo (la trace depend-elle de l'image ?)
Usage : python empreinte_synthetique.py <starter_kit> <models> <dossier de photos>"""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

KIT, MODELS, PHOTOS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(KIT))
from wmloc.watermarks import load_watermark

torch.set_num_threads(4)

def load512(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = 512 / min(w, h)
    im = im.resize((max(512, round(w * s)), max(512, round(h * s))), Image.BICUBIC)
    w, h = im.size; x0, y0 = (w - 512) // 2, (h - 512) // 2
    return np.asarray(im.crop((x0, y0, x0 + 512, y0 + 512)))

def corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

files = sorted(PHOTOS.glob("*.jpg"))[:6]
imgs = [load512(f) for f in files]
rng = np.random.default_rng(0)
for name in ("pixelseal", "trustmark", "stegastamp"):
    wm = load_watermark(name, MODELS, device="cpu")
    good, bad, other, psnr, acc = [], [], [], [], []
    prev = None
    msg = rng.integers(0, 2, wm.nbits); wrong = rng.integers(0, 2, wm.nbits)
    for im in imgs:
        w = wm.embed(im, msg)
        r = w.astype(np.float32) - im
        r_good = wm.embed(w, msg).astype(np.float32) - w
        r_bad = wm.embed(w, wrong).astype(np.float32) - w
        good.append(corr(r, r_good)); bad.append(corr(r, r_bad))
        if prev is not None: other.append(corr(r, prev))
        prev = r
        psnr.append(10 * np.log10(255 ** 2 / np.mean(r ** 2)))
        t = torch.from_numpy(w).permute(2, 0, 1)[None].float() / 255
        acc.append(float((wm.decode(t)[0].cpu().numpy() == msg).mean()))
    print(f"{name:10s}: PSNR {np.mean(psnr):.1f} dB | bits relus {np.mean(acc):.3f} | trace recalculee, bon message : {np.mean(good):.2f} | mauvais message : {np.mean(bad):.2f} | meme message, autre photo : {np.mean(other):.2f}")
