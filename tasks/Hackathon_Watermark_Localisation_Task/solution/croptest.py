"""Hypothesis test: partial rectangles were made by crop -> embed(crop) -> paste. Decoding the exact GT bbox crop
should then give high bit accuracy, vs. full image / shifted crops."""
import sys; from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor
torch.set_num_threads(8)
ds = LocalisationDataset(ROOT / "data", "train")
method = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"; n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
wm = load_watermark(method, ROOT / "models", dev)
def ba(img, msg):
    lg = wm.logits(to_tensor(np.ascontiguousarray(img)))[0].cpu()
    return float(((lg > 0).numpy() == msg.astype(bool)).mean())
k = 0
for i in range(len(ds)):
    it = ds[i]
    if it["method"] != method or it["full_watermark"]: continue
    m = it["mask"]; ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    fill = m.sum() / ((y1 - y0) * (x1 - x0))
    img = it["image"]
    r = [ba(img, it["message"]), ba(img[y0:y1, x0:x1], it["message"])]
    h, w = y1 - y0, x1 - x0
    for dy, dx in [(8, 0), (0, 8), (-8, -8), (h // 4, 0), (h, w)]:
        a0, b0 = np.clip(y0 + dy, 0, 512 - h), np.clip(x0 + dx, 0, 512 - w)
        r.append(ba(img[a0:a0 + h, b0:b0 + w], it["message"]))
    print(method, f"bbox {h}x{w} fill {fill:.2f} full={r[0]:.2f} crop={r[1]:.2f} shift8={r[2]:.2f},{r[3]:.2f},{r[4]:.2f} shiftq={r[5]:.2f} far={r[6]:.2f}", flush=True)
    k += 1
    if k >= n: break
