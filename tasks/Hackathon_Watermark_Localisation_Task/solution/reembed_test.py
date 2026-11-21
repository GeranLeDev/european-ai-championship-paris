"""Signal test for the re-embedding channel: w_hat = embed(y, known message) - y (whole image), then local normalised
correlation between high-pass(y) and high-pass(w_hat) on the 128 grid; pixel AUC against the true mask, by mask type.
Usage: python solution/reembed_test.py trustmark cpu 12"""
import sys, json; from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "solution"))
from PIL import Image
from scipy.stats import rankdata
from wmloc.watermarks import load_watermark
method, dev, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
wm = load_watermark(method, ROOT / "models", dev)
meta = [e for e in json.loads((ROOT / "data/train/metadata.json").read_text()) if e["method"] == method]
z = np.load(ROOT / "data/train/masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))

def auc(y, s):
    y = y.ravel().astype(bool); r = rankdata(s.ravel()); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def hp(t, k):                                             # t: 1x3xHxW ; high-pass = t - box blur k
    return t - F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="reflect"), k, 1)

def kind(e):
    if e["full_watermark"]: return "full"
    g = np.asarray(Image.open(ROOT / f"data/train/masks/{e['image_name']}.png").convert("L")) > 127; ys, xs = np.nonzero(g)
    return "rect" if g[ys.min():ys.max() + 1, xs.min():xs.max() + 1].all() else "irreg"

res = {}
cnt = {"rect": 0, "irreg": 0, "full": 0}
for e in meta:
    k = kind(e)
    if cnt[k] >= n: continue
    cnt[k] += 1
    y = np.asarray(Image.open(ROOT / f"data/train/images/{e['image_name']}.png").convert("RGB"))
    bits = [int(c) for c in e["message"]]
    w = wm.embed(y, bits).astype(np.float32) - y
    yt = torch.from_numpy(y.astype(np.float32)).permute(2, 0, 1)[None]; wt = torch.from_numpy(w).permute(2, 0, 1)[None]
    g = gt[e["image_name"]]
    if not 0 < g.mean() < 1: continue
    row = []
    for kk in (3, 9):
        a, b = hp(yt, kk), hp(wt, kk)
        for win in (4, 16, 32):                           # aggregation window (px), reported on the 128 grid
            num = F.avg_pool2d((a * b).sum(1, keepdim=True), win, 4, win // 2 - 2 if win > 4 else 0)
            den = F.avg_pool2d((b * b).sum(1, keepdim=True), win, 4, win // 2 - 2 if win > 4 else 0) + 1e-6
            c = (num / den)[0, 0].numpy()[:128, :128]
            row.append(auc(g, c))
    res.setdefault(k, []).append(row)
    print(k, e["image_name"], "area %.2f" % g.mean(), "|w| %.2f" % np.abs(w).mean(), np.round(row, 2), flush=True)
print("cols: hp3 win4/16/32, hp9 win4/16/32")
for k, v in res.items(): print(method, k, len(v), "mean AUC", np.round(np.mean(v, 0), 3))
