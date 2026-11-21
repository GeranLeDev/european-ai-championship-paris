"""Recipe fingerprinting helpers (read-only on official data; writes nothing)."""
import json, sys, zlib
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy import ndimage

ROOT = Path("/Users/amil/Downloads/hackathon-watermark")
KIT = ROOT / "sprintml/starter_kit"; MODELS = ROOT / "sprintml/models"; DATA = ROOT / "sprintml/release"
PHOTOS = sorted(Path("/Users/amil/Downloads/val2017").glob("*.jpg"))
sys.path.insert(0, str(KIT))
from wmloc.watermarks import load_watermark  # noqa
from wmloc.utils import to_tensor  # noqa
S = 512


def meta(split="train"):
    return json.loads((DATA / split / "metadata.json").read_text())

def load_img(name, split="train"):
    return np.asarray(Image.open(DATA / split / "images" / f"{name}.png").convert("RGB"))

def load_mask(name):
    return np.asarray(Image.open(DATA / "train" / "masks" / f"{name}.png")) > 127

def msg_of(m):
    return np.array([int(c) for c in m["message"]], dtype=np.uint8)

def clean_photo(i):
    im = Image.open(PHOTOS[(i * 37 + 11) % len(PHOTOS)]).convert("RGB"); w, h = im.size; s = S / min(w, h)
    im = im.resize((max(S, round(w * s)), max(S, round(h * s))), Image.BICUBIC)
    w, h = im.size; x0, y0 = (w - S) // 2, (h - S) // 2
    return np.asarray(im.crop((x0, y0, x0 + S, y0 + S)))

def kind_of(box):
    h, w = box.shape
    if box.all(): return "barre" if max(h / w, w / h) > 4 else "rectangle"
    return "objet"

def comps_of(mask, min_area=64):
    lab, n = ndimage.label(mask); out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab == k
        if m.sum() < min_area: continue
        b = (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)
        out.append(dict(m=m, box=b, kind=kind_of(m[sl]), area=int(m.sum())))
    return out

def bbox(m):
    ys, xs = np.where(m); return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

def expand_ar(b, max_ar=2.0, min_side=32):
    y0, y1, x0, x1 = b; h, w = y1 - y0, x1 - x0
    th = max(h, min_side, int(np.ceil(w / max_ar))); tw = max(w, min_side, int(np.ceil(h / max_ar)))
    th, tw = min(th, S), min(tw, S)
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    y0 = int(np.clip(round(cy - th / 2), 0, S - th)); x0 = int(np.clip(round(cx - tw / 2), 0, S - tw))
    return y0, y0 + th, x0, x0 + tw

def square(b, min_side=32):
    y0, y1, x0, x1 = b; s = min(S, max(y1 - y0, x1 - x0, min_side))
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    y0 = int(np.clip(round(cy - s / 2), 0, S - s)); x0 = int(np.clip(round(cx - s / 2), 0, S - s))
    return y0, y0 + s, x0, x0 + s

def pad(b, k):
    y0, y1, x0, x1 = b; return max(0, y0 - k), min(S, y1 + k), max(0, x0 - k), min(S, x1 + k)

def window(b, side):
    y0, y1, x0, x1 = b; th = min(S, max(side, y1 - y0)); tw = min(S, max(side, x1 - x0))
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    y0 = int(np.clip(round(cy - th / 2), 0, S - th)); x0 = int(np.clip(round(cx - tw / 2), 0, S - tw))
    return y0, y0 + th, x0, x0 + tw

def embed_box(wm, img, msg, b):
    """Returns full-size image where box b was embedded standalone (rest untouched)."""
    y0, y1, x0, x1 = b; out = img.copy()
    out[y0:y1, x0:x1] = wm.embed(np.ascontiguousarray(img[y0:y1, x0:x1]), msg); return out

BOXF = {  # box families used both as generation recipe (R3 variants) and as decoding window
    "bbox": lambda b: b,
    "ar2_32": lambda b: expand_ar(b, 2.0, 32),
    "ar2_64": lambda b: expand_ar(b, 2.0, 64),
    "sq": lambda b: square(b),
    "pad16": lambda b: pad(b, 16),
    "win256": lambda b: window(b, 256),
}

def make(wm, clean, comps, msg, recipe):
    """Apply a recipe to all zones of one image. recipe: R2 (=R1 for full rectangles), R3_<boxfamily>, R4, R5 (union bbox)."""
    union = np.zeros((S, S), bool)
    for c in comps: union |= c["m"]
    if recipe == "R4":
        return np.where(union[..., None], wm.embed(clean, msg), clean)
    if recipe == "R5":
        return np.where(union[..., None], embed_box(wm, clean, msg, bbox(union)), clean)
    fam = "bbox" if recipe == "R2" else recipe[3:]
    out = clean.copy()
    for c in comps:
        w = embed_box(wm, clean, msg, BOXF[fam](c["box"]))
        out = np.where(c["m"][..., None], w, out)
    return out

def acc(wm, img, msg):
    lg = wm.logits(to_tensor(np.ascontiguousarray(img)))[0].cpu().numpy()
    return float(((lg > 0) == (msg > 0)).mean())

def gauss(x, sigma=2.0):
    return np.stack([ndimage.gaussian_filter(x[..., c], sigma) for c in range(3)], -1)

def trace(wm, img, msg, b):
    """(simple, differential) high-passed traces of re-embedding box b of the observed image with the known message.
    differential = embed(msg) - embed(1 - msg): cancels the image-only part of the residual."""
    y0, y1, x0, x1 = b; crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    e1 = wm.embed(crop, msg).astype(np.float32); e0 = wm.embed(crop, 1 - msg).astype(np.float32)
    tr = np.zeros((S, S, 3), np.float32); tr[y0:y1, x0:x1] = e1 - crop
    d = np.zeros((S, S, 3), np.float32); d[y0:y1, x0:x1] = e1 - e0
    return tr - gauss(tr), d - gauss(d)

def corr(h_img, q, m):
    a = h_img[m]; b = q[m]
    return float((a * b).sum() / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-6))

def fingerprints(wm, img, comps, msg, max_comps=5):
    """All statistics measurable WITHOUT the clean original. Returns list of dict per zone (largest first)."""
    union = np.zeros((S, S), bool)
    for c in comps: union |= c["m"]
    ub = bbox(union); x = img.astype(np.float32); h_img = x - gauss(x)
    acc_img = acc(wm, img, msg); multi = len(comps) > 1
    acc_union = acc(wm, img[ub[0]:ub[1], ub[2]:ub[3]], msg) if multi else None
    qF = trace(wm, img, msg, (0, S, 0, S)); qU = trace(wm, img, msg, ub) if multi else None
    rows = []
    for c in sorted(comps, key=lambda c: -c["area"])[:max_comps]:
        b = c["box"]; r = dict(kind=c["kind"], box=list(b), area=c["area"], ncomp=len(comps), acc_img=acc_img)
        for fam in ("bbox", "ar2_32", "sq", "pad16"):
            bb = BOXF[fam](b); r["acc_" + fam] = acc(wm, img[bb[0]:bb[1], bb[2]:bb[3]], msg)
        r["acc_union"] = acc_union if multi else r["acc_bbox"]
        tB = trace(wm, img, msg, b); b2 = BOXF["ar2_32"](b)
        tB2 = tB if tuple(b2) == tuple(b) else trace(wm, img, msg, b2)
        for tag, t in (("F", qF), ("B", tB), ("B2", tB2), ("U", qU if multi else tB)):
            r["corr_" + tag] = corr(h_img, t[0], c["m"]); r["dcorr_" + tag] = corr(h_img, t[1], c["m"])
        rows.append(r)
    return rows
