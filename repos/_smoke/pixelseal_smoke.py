"""Smoke test Pixel Seal: embed -> detect, then JPEG / crop / brightness attacks.
Run: repos/videoseal/.venv/bin/python repos/_smoke/pixelseal_smoke.py
"""
import io
import math
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

import videoseal

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "videoseal/assets/imgs/1.jpg"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

model = videoseal.load("pixelseal").eval()
img = TF.to_tensor(Image.open(IMG).convert("RGB")).unsqueeze(0)

with torch.no_grad():
    out = model.embed(img)
imgs_w, msg = out["imgs_w"].clamp(0, 1), out["msgs"]
TF.to_pil_image(imgs_w[0]).save(OUT / "watermarked.png")

mse = torch.mean((imgs_w - img) ** 2).item()
psnr = 10 * math.log10(1 / mse)
nbits = msg.shape[-1]


def bit_acc(x):
    with torch.no_grad():
        preds = model.detect(x)["preds"][:, 1:]
    return ((preds > 0).float() == msg).float().mean().item()


def log10_p(acc):  # Eq. 2 of the paper: P(random msg matches >= observed bits)
    k = round(acc * nbits)
    p = sum(math.comb(nbits, i) for i in range(k, nbits + 1)) / 2**nbits
    return math.log10(max(p, 1e-300))


def jpeg(x, q):
    buf = io.BytesIO()
    TF.to_pil_image(x[0]).save(buf, format="JPEG", quality=q)
    return TF.to_tensor(Image.open(buf)).unsqueeze(0)


def crop(x, frac):  # keep `frac` of the area, centered
    _, _, h, w = x.shape
    s = math.sqrt(frac)
    return TF.center_crop(x, [int(h * s), int(w * s)])


attacks = {
    "identity": imgs_w,
    "original (no wm)": img,
    "jpeg 40": jpeg(imgs_w, 40),
    "crop 0.5 area": crop(imgs_w, 0.5),
    "crop 0.32 area": crop(imgs_w, 0.32),
    "brightness 0.5": TF.adjust_brightness(imgs_w, 0.5),
    "combo jpeg40+crop0.71+bright0.5": jpeg(TF.adjust_brightness(crop(imgs_w, 0.71), 0.5), 40),
}
print(f"image {tuple(img.shape)}  nbits={nbits}  PSNR={psnr:.1f} dB")
for name, x in attacks.items():
    a = bit_acc(x)
    print(f"{name:34s} bit_acc={a:.3f}  log10_p={log10_p(a):8.1f}")
ok = bit_acc(imgs_w) > 0.95 and bit_acc(img) < 0.7
print("SMOKE", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
