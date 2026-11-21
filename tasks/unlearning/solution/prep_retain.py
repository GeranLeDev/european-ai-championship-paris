"""Retain proxy: Imagenette images (public, disjoint from the private dataset) resized+center-cropped to 224, uint8 NHWC."""
import random, numpy as np
from pathlib import Path
from PIL import Image
from common import CACHE
D = Path(__file__).resolve().parent / "data" / "imagenette2-320"

def load(paths):
    out = np.empty((len(paths), 224, 224, 3), np.uint8)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB"); w, h = im.size; s = 224 / min(w, h)
        im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = im.size
        l, t = (w - 224) // 2, (h - 224) // 2
        out[i] = np.asarray(im.crop((l, t, l + 224, t + 224)))
    return out

random.seed(0)
tr = sorted(D.glob("train/*/*.JPEG")); random.shuffle(tr)
va = sorted(D.glob("val/*/*.JPEG")); random.shuffle(va)
np.save(CACHE / "retain_train_u8.npy", load(tr[:6000]))
np.save(CACHE / "retain_val_u8.npy", load(va[:1000]))
print("done")
