"""Which process made the irregular partial masks? Complete the watermark outside the true mask with the re-embedded
residual and decode: A = whole-image embed then paste by mask; B = bounding-box crop embed then paste by mask.
The right hypothesis should bring the decoder agreement close to 1.
Usage: python solution/gentest.py trustmark cpu 10 [irreg|rect|full]"""
import sys, json; from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from PIL import Image
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor
method, dev, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); want = sys.argv[4] if len(sys.argv) > 4 else "irreg"
wm = load_watermark(method, ROOT / "models", dev)
meta = [e for e in json.loads((ROOT / "data/train/metadata.json").read_text()) if e["method"] == method]

def agree(img, s): return float((torch.tanh(wm.logits(to_tensor(img))[0].cpu()) * s).mean())
def complete(y, M, bits):                                  # y + (1 - M) * (embed(y) - y)
    w = wm.embed(np.ascontiguousarray(y), bits).astype(np.float32) - y
    return np.clip(y + (1 - M[..., None]) * w, 0, 255).round().astype(np.uint8)

k = 0; out = []
for e in meta:
    M = (np.asarray(Image.open(ROOT / f"data/train/masks/{e['image_name']}.png").convert("L")) > 127).astype(np.float32)
    ys, xs = np.nonzero(M)
    if e["full_watermark"]: kind = "full"
    else: kind = "rect" if M[ys.min():ys.max() + 1, xs.min():xs.max() + 1].all() else "irreg"
    if kind != want: continue
    y = np.asarray(Image.open(ROOT / f"data/train/images/{e['image_name']}.png").convert("RGB"))
    bits = [int(c) for c in e["message"]]; s = torch.from_numpy(2 * np.array(bits, np.float32) - 1)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    a0 = agree(y, s); aA = agree(complete(y, M, bits), s)
    Mr = np.roll(M, (150, 150), (0, 1)); aAr = agree(complete(y, Mr, bits), s)      # control: wrong mask
    c = y[y0:y1, x0:x1]; aB0 = agree(c, s)
    aB = agree(complete(c, M[y0:y1, x0:x1], bits), s) if min(c.shape[:2]) >= 16 else float("nan")
    full = agree(wm.embed(y, bits), s)
    print(f"{kind} area {M.mean():.3f} bbox {x1-x0}x{y1-y0} fill {M[y0:y1,x0:x1].mean():.2f} | agree(y) {a0:.2f} | A complete whole {aA:.2f} (wrong mask {aAr:.2f}, embed all {full:.2f}) | bbox crop {aB0:.2f} -> B complete {aB:.2f}", flush=True)
    out.append((a0, aA, aAr, aB0, aB)); k += 1
    if k >= n: break
print(method, want, "mean: agree(y) %.2f  A %.2f  A-wrong-mask %.2f  bbox %.2f  B %.2f" % tuple(np.nanmean(out, 0)))
