"""Hypothesis: irregular partial masks are rotated (or warped) watermarked patches. Fit cv2.minAreaRect on the GT
mask, extract the de-rotated patch, decode with the known message."""
import sys; from pathlib import Path
import numpy as np, torch, cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor
torch.set_num_threads(8)
ds = LocalisationDataset(ROOT / "data", "train"); method = sys.argv[1]; n = int(sys.argv[2])
wm = load_watermark(method, ROOT / "models", "cpu")
def ba(img, msg):
    lg = wm.logits(to_tensor(np.ascontiguousarray(img)))[0].cpu(); return float(((lg > 0).numpy() == msg.astype(bool)).mean())
k = 0
for i in range(len(ds)):
    it = ds[i]
    if it["method"] != method or it["full_watermark"]: continue
    m = it["mask"].astype(np.uint8); ys, xs = np.nonzero(m)
    fill = m.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1))
    if fill > 0.95: continue
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cnts, key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    rect_fill = m.sum() / max(w * h, 1)
    res = []
    for flip in (0, 1):
        W, Hh, a = (w, h, ang) if not flip else (h, w, ang - 90)
        M = cv2.getRotationMatrix2D((cx, cy), a, 1.0); M[:, 2] += [W / 2 - cx, Hh / 2 - cy]
        patch = cv2.warpAffine(it["image"], M, (int(round(W)), int(round(Hh))), flags=cv2.INTER_LINEAR)
        res.append(ba(patch, it["message"]))
    print(method, f"fill {fill:.2f} rectfill {rect_fill:.2f} wh {w:.0f}x{h:.0f} ang {ang:.1f} bitacc {res} full {ba(it['image'], it['message']):.2f}", flush=True)
    k += 1
    if k >= n: break
