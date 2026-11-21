import sys; from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor, bit_accuracy
ds = LocalisationDataset(ROOT / "data", "train")
def corr(a,b): a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
for method in ["stegastamp","trustmark","pixelseal"]:
    wm = load_watermark(method, ROOT/"models", "cpu")
    its=[it for it in (ds[i] for i in range(len(ds))) if it["method"]==method][:12]
    for it in its:
        x=it["image"]; m=it["message"]
        xw=wm.embed(x,m); r=xw.astype(np.float32)-x
        xww=wm.embed(xw,m); r2=xww.astype(np.float32)-xw
        rng=np.random.default_rng(0); m2=rng.integers(0,2,wm.nbits)
        r3=wm.embed(x,m2).astype(np.float32)-x
        ba=lambda im: bit_accuracy(wm.decode(to_tensor(im))[0].cpu().numpy(), m)
        print(method, it["full_watermark"], f"frac={it['mask'].mean():.2f} bitacc(x)={ba(x):.2f} bitacc(xw)={ba(xw):.2f} corr(r,r2)={corr(r,r2):.2f} |r2|/|r|={np.abs(r2).mean()/np.abs(r).mean():.2f} corr(r,r_rand)={corr(r,r3):.2f}", flush=True)
