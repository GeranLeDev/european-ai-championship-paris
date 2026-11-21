"""Synthesize WAM-style training pairs (image, 128x128 mask) for one watermark method, mimicking the dataset:
  - 'partial' mode (like full_watermark=False): a rectangle (crop -> embed -> paste back, as the dataset does) or an
    irregular region (full-image embed then paste, or bbox-crop embed then paste) is watermarked on a clean image
  - 'full' mode (like full_watermark=True): whole image watermarked, then irregular blobs restored to clean
Sources: COCO val2017 (clean) and train-fold images of the same method whose existing mask is merged into the label
(held-out images are never used). Random message per sample.
Usage: python solution/synth.py --method pixelseal --n 2000 --device cpu --out solution/synth
Output: <out>/<method>_images.npy (N x 512 x 512 x 3 uint8), <out>/<method>_meta.npz (mask128, agree, mode)"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, split_train, blob_mask, rect
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask, to_tensor

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True); ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--device", default="cpu"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--coco", default=str(ROOT / "extra" / "val2017")); ap.add_argument("--p_train_src", type=float, default=0.3)
ap.add_argument("--out", default=str(ROOT / "solution" / "synth")); ap.add_argument("--threads", type=int, default=3)
a = ap.parse_args()
torch.set_num_threads(a.threads)
rng = np.random.default_rng(a.seed * 10 + ["pixelseal", "trustmark", "stegastamp"].index(a.method))
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
wm = load_watermark(a.method, ROOT / "models", a.device)

coco = sorted(Path(a.coco).glob("*.jpg"))
val = split_train()
meta = json.loads((ROOT / "data" / "train" / "metadata.json").read_text())
tr_src = [e["image_name"] for e in meta if e["method"] == a.method and not e["full_watermark"] and e["image_name"] not in val]


def load_coco(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h)
    l, t = rng.integers(0, w - s + 1), rng.integers(0, h - s + 1)
    return np.asarray(im.crop((l, t, l + s, t + s)).resize((512, 512), Image.Resampling.LANCZOS))


imgs = np.lib.format.open_memmap(out / f"{a.method}_images.npy", mode="w+", dtype=np.uint8, shape=(a.n, 512, 512, 3))
masks, agree, modes = np.zeros((a.n, 128, 128), np.uint8), np.zeros(a.n, np.float32), np.zeros(a.n, np.int8)
t0 = time.time()
for i in range(a.n):
    if rng.random() < a.p_train_src:
        nm = tr_src[rng.integers(len(tr_src))]
        x = np.asarray(Image.open(ROOT / "data" / "train" / "images" / f"{nm}.png").convert("RGB"))
        old = (np.asarray(Image.open(ROOT / "data" / "train" / "masks" / f"{nm}.png")) > 127).astype(np.uint8)
        if rng.random() < 0.5:
            x, old = x[:, ::-1].copy(), old[:, ::-1].copy()
    else:
        x = load_coco(coco[rng.integers(len(coco))]); old = np.zeros((512, 512), np.uint8)
    bits = rng.integers(0, 2, wm.nbits)
    mode = int(rng.random() < 0.5)                       # 1 = full mode
    if mode == 1:
        xw = wm.embed(x, bits)
        clean = blob_mask(rng)
        y = np.where(clean[..., None] == 1, x, xw)
        m = (1 - clean) | (old & clean)
    else:
        r = rng.random()
        if r < 0.5:                                      # rectangle, crop-embed
            y0, y1, x0, x1 = rect(rng)
            y = x.copy(); y[y0:y1, x0:x1] = wm.embed(np.ascontiguousarray(x[y0:y1, x0:x1]), bits)
            new = np.zeros((512, 512), np.uint8); new[y0:y1, x0:x1] = 1
        else:
            new = blob_mask(rng)
            if r < 0.75:                                 # full-image embed + irregular paste
                xw = wm.embed(x, bits)
            else:                                        # bbox crop embed + irregular paste
                ys, xs = np.nonzero(new); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                if (y1 - y0) < 16 or (x1 - x0) < 16:
                    xw = wm.embed(x, bits)
                else:
                    xw = x.copy(); xw[y0:y1, x0:x1] = wm.embed(np.ascontiguousarray(x[y0:y1, x0:x1]), bits)
            y = np.where(new[..., None] == 1, xw, x)
        m = new | old
    imgs[i] = y
    masks[i] = downsample_mask(m)
    s = torch.from_numpy(2 * bits.astype(np.float32) - 1)
    agree[i] = float((torch.tanh(wm.logits(to_tensor(y))[0].cpu()) * s).mean())
    modes[i] = mode
    if i % 100 == 0:
        print(a.method, i, f"{time.time() - t0:.0f}s", flush=True)
    if (i + 1) % 100 == 0 or i + 1 == a.n:                 # incremental save: training can use the first n_done samples
        imgs.flush()
        np.savez_compressed(out / f"{a.method}_meta.npz", masks=masks[:i + 1], agree=agree[:i + 1], modes=modes[:i + 1])
print("done", a.method, a.n, f"{time.time() - t0:.0f}s")
