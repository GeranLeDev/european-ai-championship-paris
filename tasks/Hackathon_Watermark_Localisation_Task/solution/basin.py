"""Basin test: soft agreement of the GT rectangle vs perturbed edges / shifts (how fine must the search be?)."""
import sys; from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "solution"))
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from rectsearch import Scorer, coarse_boxes
torch.set_num_threads(8)
method = sys.argv[1]; dev = sys.argv[2]; n = int(sys.argv[3])
ds = LocalisationDataset(ROOT / "data", "train")
wm = load_watermark(method, ROOT / "models", dev); sc = Scorer(wm, method, bs=64)
print("grid size", len(coarse_boxes()))
k = 0
for i in range(len(ds)):
    it = ds[i]
    if it["method"] != method or it["full_watermark"]: continue
    m = it["mask"]; ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if m.sum() / ((y1 - y0) * (x1 - x0)) < 0.99: continue
    x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).to(sc.dev)
    s = torch.from_numpy(2 * it["message"].astype(np.float32) - 1).to(sc.dev)
    gt = np.array([x0, y0, x1, y1], np.float32); h, w = y1 - y0, x1 - x0
    ds_ = [0, 1, 2, 4, 8, 16]
    shift = [gt + np.array([d, 0, d, 0]) for d in ds_]
    grow = [gt + np.array([-d, -d, d, d]) for d in ds_]
    shrink = [gt + np.array([d, d, -d, -d]) for d in ds_[:4]]
    rel = [gt + np.array([f * w, f * h, f * w, f * h]) for f in (0.05, 0.1, 0.2)]
    full = [np.array([0, 0, 512, 512])]
    v = sc.score(x, s, np.array(shift + grow + shrink + rel + full).clip(0, 512))
    print(f"{h}x{w} shift{ds_}={v[:6].round(2)} grow={v[6:12].round(2)} shrink={v[12:16].round(2)} rel.05/.1/.2={v[16:19].round(2)} full={v[19]:.2f}", flush=True)
    k += 1
    if k >= n: break
