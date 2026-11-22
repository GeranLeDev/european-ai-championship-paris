"""DiffusionDB images (downloaded by the Task 1 agent, read-only) -> cache/retain_ddb_u8.npy [n,224,224,3] (resize short side 224 + center crop)."""
import glob, os, numpy as np
from PIL import Image
from common import CACHE
fs = sorted(glob.glob("../../Hackathon_Watermark_Localisation_Task/extra/diffusiondb/part-*/*.png")); out = np.zeros((len(fs), 224, 224, 3), np.uint8); n = 0
for f in fs:
    try:
        im = Image.open(f).convert("RGB"); w, h = im.size; s = 224 / min(w, h); im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
        w, h = im.size; l, t = (w - 224) // 2, (h - 224) // 2; out[n] = np.asarray(im.crop((l, t, l + 224, t + 224))); n += 1
    except Exception as e: print("skip", f, e)
np.save(CACHE / "retain_ddb_u8.tmp.npy", out[:n]); os.replace(CACHE / "retain_ddb_u8.tmp.npy", CACHE / "retain_ddb_u8.npy"); print("saved", n)
