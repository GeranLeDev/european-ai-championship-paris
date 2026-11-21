import sys; from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor, bit_accuracy
ds = LocalisationDataset(ROOT / "data", "train")
def corr(a,b): a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-9))
for method in ["stegastamp","trustmark","pixelseal"]:
    wm = load_watermark(method, ROOT/"models", "cpu")
    its=[it for it in (ds[i] for i in range(len(ds))) if it["method"]==method and not it["full_watermark"]][:10]
    for it in its:
        x=it["image"]; m=it["message"]; g=it["mask"]
        ys,xs=np.nonzero(g); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
        crop=x[y0:y1,x0:x1]
        ba=lambda im: bit_accuracy(wm.decode(to_tensor(im))[0].cpu().numpy(), m)
        # crop slightly larger
        pad=16; big=x[max(0,y0-pad):y1+pad, max(0,x0-pad):x1+pad]
        # shifted crop
        sh=x[min(512-(y1-y0),y0+(y1-y0)//3):, :][: y1-y0, x0:x1]
        rw=wm.embed(crop,m).astype(np.float32)-crop
        print(method, f"bbox {y1-y0}x{x1-x0} fill={g[y0:y1,x0:x1].mean():.2f} ba(full)={ba(x):.2f} ba(crop)={ba(crop):.2f} ba(crop+16)={ba(big):.2f} ba(shift1/3)={ba(sh):.2f}", flush=True)
