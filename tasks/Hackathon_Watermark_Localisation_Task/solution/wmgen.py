"""Online synthetic generator that follows how the dataset was built (checked on the real train set):
  rect   (25 %): crop -> embed(crop, message) -> paste back            (exact-crop decoding gives 0.9-1.0 on real)
  irreg  (25 %): embed(whole image, message) -> paste through a blob   (global agreement grows with the area on real)
  full   (50 %): embed(whole image, message) -> clean blob restored
One random message per image (as in the real data: all messages are distinct). Mask shapes come from the real
train-fold masks (flip / transpose / roll) or from procedural blobs with the real area range; rectangle sizes come
from the real rectangles of the same method (TrustMark: aspect ratio <= 2), with jitter.
Covers: COCO val2017 (random square crop, Lanczos to 512), generated images (extra/diffusiondb) and real train-fold
partial images (their own message is reused and their own mask is merged into the label). Held-out images are never used.
Each sample also carries the side information available at test time:
  what  = embed(y, message) - y   (re-embedding residual; correlation 0.94-0.99 with the true residual when the image
          was embedded as a whole)
  agree = mean(tanh(logit) * (2 * bit - 1)) of the official decoder on y
No disk writes."""
import json, sys, threading, queue
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
from scipy import ndimage
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "solution"))
from common import split_train, blob_mask
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask, to_tensor

TYPES = ("rect", "irreg", "full")


def mask_kind(full, m512):
    if full: return "full"
    ys, xs = np.nonzero(m512)
    return "rect" if m512[ys.min():ys.max() + 1, xs.min():xs.max() + 1].all() else "irreg"


def load_real(data, method, with_masks=True):
    """Real train images of one method: dicts name, img, bits, mask512, mask128, kind, val (held-out flag)."""
    data = Path(data); val = split_train(data)
    meta = [e for e in json.loads((data / "train" / "metadata.json").read_text()) if e["method"] == method]
    out = []
    for e in meta:
        n = e["image_name"]
        img = np.asarray(Image.open(data / "train" / "images" / f"{n}.png").convert("RGB"))
        m = (np.asarray(Image.open(data / "train" / "masks" / f"{n}.png").convert("L")) > 127).astype(np.uint8)
        out.append(dict(name=n, img=img, bits=np.array([int(c) for c in e["message"]]), mask512=m, mask128=downsample_mask(m),
                        kind=mask_kind(e["full_watermark"], m), val=n in val))
    return out


def load_mask_library(data):
    """Shapes from the train fold of all methods: irregular masks, clean holes of 'full' images, rectangle sizes per method."""
    data = Path(data); val = split_train(data)
    meta = json.loads((data / "train" / "metadata.json").read_text())
    lib = dict(irreg=[], hole=[], rect={})
    for e in meta:
        if e["image_name"] in val: continue
        m = (np.asarray(Image.open(data / "train" / "masks" / f"{e['image_name']}.png").convert("L")) > 127).astype(np.uint8)
        k = mask_kind(e["full_watermark"], m)
        if k == "full": lib["hole"].append(np.packbits(1 - m))
        elif k == "irreg": lib["irreg"].append(np.packbits(m))
        else:
            ys, xs = np.nonzero(m); lib["rect"].setdefault(e["method"], []).append((xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()))
    return lib


class Generator:
    def __init__(self, method, data, device, seed=0, coco=None, gen_dirs=(), p_real=0.2, p_gen=0.4, wm=None):
        self.method, self.rng = method, np.random.default_rng(seed)
        self.wm = wm or load_watermark(method, ROOT / "models", device)
        self.lib = load_mask_library(data)
        self.real = [r for r in load_real(data, method) if not r["val"] and r["kind"] != "full"]
        self.coco = sorted(Path(coco).glob("*.jpg")) if coco and Path(coco).exists() else []
        self.gen = sorted(p for d in gen_dirs for p in Path(d).rglob("*.png"))
        self.p_real, self.p_gen = p_real, (p_gen if self.gen else 0.0)
        assert self.coco or self.gen, "no cover images"
        # the real masks are mostly object segments: use the COCO instance polygons of the cover as mask shapes
        self.inst = {}
        ann = Path(coco).parent / "annotations" / "instances_val2017.json" if coco else None
        if ann is not None and ann.exists():
            for o in json.loads(ann.read_text())["annotations"]:
                if not o["iscrowd"] and o["area"] > 400 and isinstance(o["segmentation"], list):
                    self.inst.setdefault(o["image_id"], []).append(o["segmentation"])
        self.p_obj = 0.7

    def _objects(self, path, size, box, s):
        """Union of 1-3 instances of this COCO image, in the 512 frame of the crop `box` (side s), randomly dilated."""
        r = self.rng
        segs = self.inst.get(int(path.stem), [])
        if not segs: return None
        m = Image.new("L", size, 0); d = ImageDraw.Draw(m)
        for j in r.permutation(len(segs))[:int(r.integers(1, 4))]:
            for poly in segs[j]:
                if len(poly) >= 6: d.polygon(poly, fill=1)
        m = np.asarray(m.crop(box).resize((512, 512), Image.Resampling.NEAREST))
        k = int(r.choice([0, 0, 3, 7, 13, 21]))
        if k: m = ndimage.maximum_filter(m, size=k)
        return m.astype(np.uint8) if 0.002 < m.mean() < 0.9 else None

    # ---- covers
    def cover(self):
        """-> (img uint8 512x512x3, bits, old mask 512) ; old mask is non-zero only for real covers.
        self.obj is set to an object mask aligned with the cover when one exists (COCO covers)."""
        r = self.rng; self.obj = None; path = None
        u = r.random()
        if u < self.p_real and self.real:
            it = self.real[r.integers(len(self.real))]
            x, old = it["img"], it["mask512"]
            k = r.integers(4)                                # flips keep the pixel grid
            if k & 1: x, old = x[:, ::-1], old[:, ::-1]
            if k & 2: x, old = x[::-1], old[::-1]
            return np.ascontiguousarray(x), it["bits"], np.ascontiguousarray(old)
        if u < self.p_real + self.p_gen or not self.coco:
            im = Image.open(self.gen[r.integers(len(self.gen))]).convert("RGB")
        else:
            path = self.coco[r.integers(len(self.coco))]; im = Image.open(path).convert("RGB")
        w, h = im.size; s = min(w, h)
        if s > 512 and r.random() < 0.5: s = int(r.integers(512, s + 1))     # random zoom
        l, t = r.integers(0, w - s + 1), r.integers(0, h - s + 1)
        if path is not None and self.inst: self.obj = self._objects(path, (w, h), (l, t, l + s, t + s), s)
        im = im.crop((l, t, l + s, t + s))
        if s != 512: im = im.resize((512, 512), Image.Resampling.LANCZOS)
        return np.asarray(im), r.integers(0, 2, self.wm.nbits), np.zeros((512, 512), np.uint8)

    # ---- masks
    def _lib_mask(self, key):
        r = self.rng
        m = np.unpackbits(self.lib[key][r.integers(len(self.lib[key]))]).reshape(512, 512)
        k = r.integers(8)
        if k & 1: m = m[:, ::-1]
        if k & 2: m = m[::-1]
        if k & 4: m = m.T
        if r.random() < 0.5: m = np.roll(m, (r.integers(512), r.integers(512)), (0, 1))
        return np.ascontiguousarray(m)

    def blob(self, key):
        r = self.rng
        if self.obj is not None and r.random() < self.p_obj and self.obj.mean() < (0.6 if key == "irreg" else 0.35): return self.obj
        if self.lib[key] and r.random() < 0.5: return self._lib_mask(key)
        lo, hi = (0.005, 0.45) if key == "irreg" else (0.003, 0.30)
        return blob_mask(r, frac=float(np.exp(r.uniform(np.log(lo), np.log(hi)))))

    def rect(self):
        r = self.rng
        sizes = self.lib["rect"].get(self.method) or [(64, 64)]
        w, h = sizes[r.integers(len(sizes))]
        w = int(np.clip(round(w * np.exp(r.uniform(-0.25, 0.25))), 8, 512)); h = int(np.clip(round(h * np.exp(r.uniform(-0.25, 0.25))), 8, 512))
        if self.method == "trustmark":                       # the dataset only uses patches with aspect ratio <= 2
            if w > 2 * h: w = 2 * h
            if h > 2 * w: h = 2 * w
        if r.random() < 0.5: w, h = (h, w) if self.method != "trustmark" or max(w / h, h / w) <= 2 else (w, h)
        x0 = int(r.integers(0, 512 - w + 1)); y0 = int(r.integers(0, 512 - h + 1))
        return y0, y0 + h, x0, x0 + w

    # ---- one sample
    def sample(self, kind=None):
        r = self.rng
        kind = kind or TYPES[int(r.choice(3, p=(0.25, 0.25, 0.5)))]
        x, bits, old = self.cover()
        if kind == "rect":
            y0, y1, x0, x1 = self.rect()
            y = x.copy(); y[y0:y1, x0:x1] = self.wm.embed(np.ascontiguousarray(x[y0:y1, x0:x1]), bits)
            m = old.copy(); m[y0:y1, x0:x1] = 1
        else:
            xw = self.wm.embed(x, bits)
            if kind == "irreg":
                new = self.blob("irreg"); y = np.where(new[..., None] == 1, xw, x); m = new | old
            else:
                hole = self.blob("hole"); y = np.where(hole[..., None] == 1, x, xw); m = (1 - hole) | (old & hole)
        return self.finish(y, bits, m, kind)

    def side(self, y, bits):
        """Side information computable at test time: re-embedding residual (int8, +-32) and global decoder agreement."""
        what = np.clip(self.wm.embed(np.ascontiguousarray(y), bits).astype(np.int16) - y, -32, 32).astype(np.int8)
        s = torch.from_numpy(2 * np.asarray(bits, np.float32) - 1)
        agree = float((torch.tanh(self.wm.logits(to_tensor(y))[0].float().cpu()) * s).mean())
        return what, agree

    def finish(self, y, bits, m512, kind):
        what, agree = self.side(y, bits)
        return dict(img=np.ascontiguousarray(y), what=what, agree=agree, mask=downsample_mask(m512), kind=kind)


def real_items(gen, data, split="train"):
    """Real images of the generator's method with their side information (train: with masks and held-out flag)."""
    data = Path(data); out = []
    if split == "train":
        for r in load_real(data, gen.method):
            what, agree = gen.side(r["img"], r["bits"])
            out.append(dict(name=r["name"], img=r["img"], what=what, agree=agree, mask=r["mask128"], kind=r["kind"], val=r["val"]))
    else:
        for e in json.loads((data / split / "metadata.json").read_text()):
            if e["method"] != gen.method: continue
            img = np.asarray(Image.open(data / split / "images" / f"{e['image_name']}.png").convert("RGB"))
            what, agree = gen.side(img, np.array([int(c) for c in e["message"]]))
            out.append(dict(name=e["image_name"], img=img, what=what, agree=agree))
    return out


class Prefetch:
    """Background threads producing samples (the embedders release the GIL inside torch / onnxruntime)."""
    def __init__(self, make_gen, n_threads=3, maxsize=96):
        self.q = queue.Queue(maxsize); self.stop = False; self.err = None
        self.threads = [threading.Thread(target=self._run, args=(make_gen(i),), daemon=True) for i in range(n_threads)]
        for t in self.threads: t.start()

    def _run(self, gen):
        while not self.stop:
            try:
                s = gen.sample()
            except Exception as ex:                          # a failed sample (tiny crop, ...) is just skipped
                self.err = repr(ex); continue
            while not self.stop:
                try: self.q.put(s, timeout=1); break
                except queue.Full: pass

    def get(self, n): return [self.q.get() for _ in range(n)]


if __name__ == "__main__":                                   # smoke test + fidelity statistics against the real train set
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--method", default="trustmark"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--data", default=str(ROOT / "data")); ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    g = Generator(a.method, a.data, a.device, coco=ROOT / "extra" / "val2017", gen_dirs=[ROOT / "extra" / "diffusiondb"], p_real=0.0)
    syn = {k: [g.sample(k) for _ in range(a.n)] for k in TYPES}
    real = real_items(g, a.data)
    print("covers: coco", len(g.coco), "generated", len(g.gen), "real partial", len(g.real), "| library", len(g.lib["irreg"]), len(g.lib["hole"]),
          {k: len(v) for k, v in g.lib["rect"].items()})
    for k in TYPES:
        rr = [r for r in real if r["kind"] == k]; ss = syn[k]
        f = lambda v: "q10 %.3f med %.3f q90 %.3f" % tuple(np.quantile(v, [0.1, 0.5, 0.9]))
        print(f"{a.method} {k:5s} real n={len(rr)} | area  real {f([r['mask'].mean() for r in rr])} | synth {f([s['mask'].mean() for s in ss])}")
        print(f"{a.method} {k:5s}               | agree real {f([r['agree'] for r in rr])} | synth {f([s['agree'] for s in ss])}")
        print(f"{a.method} {k:5s}               | |what| real {f([np.abs(r['what']).mean() for r in rr])} | synth {f([np.abs(s['what']).mean() for s in ss])}")
