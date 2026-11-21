"""Is the residual predictable from the watermarked image + known message? (synthetic check)"""
import sys; sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/pixelseal_research")
import numpy as np, torch, time
from PIL import Image
from common import *

wm = load_wm()
rng = np.random.default_rng(0)
files = sorted((Path("/Users/amil/Downloads/val2017")).glob("*.jpg"))[:4]
from gen_lib import load_clean
for f in files:
    img, _ = load_clean(Image.open(f), rng)
    msg = rng.integers(0, 2, 256)
    W = wm.embed(img, msg)                                   # uint8, like the dataset
    I = to_t(img[None]); Wt = to_t(W[None])
    Rtrue = (Wt - I)
    t = time.time()
    Rm = embed_float(wm, Wt, msg[None]) - Wt
    Rr = embed_float(wm, Wt, rng.integers(0, 2, (1, 256))) - Wt
    Rm_clean = embed_float(wm, I, msg[None]) - I
    dt = time.time() - t
    def c(a, b):
        a = a.flatten(); b = b.flatten(); return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
    print(f.name, f"rms true {Rtrue.std()*255:.2f}  corr(true,Rm)={c(Rtrue,Rm):.3f} corr(true,Rrand)={c(Rtrue,Rr):.3f} "
          f"corr(true,Rm_from_clean)={c(Rtrue,Rm_clean):.3f} corr(Rm,Rr)={c(Rm,Rr):.3f}  "
          f"chan corr RG={c(Rtrue[:,0],Rtrue[:,1]):.2f} {dt:.2f}s")
