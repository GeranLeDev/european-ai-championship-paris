"""Shared helpers for the PixelSeal localisation research (local MPS)."""
import json, os, sys, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/Users/amil/Downloads/hackathon-watermark")
KIT = Path(os.environ.get("PS_KIT", ROOT / "sprintml/starter_kit"))          # folder containing wmloc/
MODELS = Path(os.environ.get("PS_MODELS", ROOT / "sprintml/models"))
REL = Path(os.environ.get("PS_REL", ROOT / "sprintml/release"))             # official train/ and test/
sys.path.insert(0, str(KIT))
sys.path.insert(0, os.environ.get("PS_GENLIB", str(ROOT / "modal_t1")))   # folder containing gen_lib.py
from wmloc.watermarks import load_watermark  # noqa
from wmloc.utils import miou_image, downsample_mask  # noqa

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load_wm(dev=DEV):
    return load_watermark("pixelseal", MODELS, dev)


def official(split="train", parity=None, method="pixelseal"):
    meta = [e for e in json.loads((REL / split / "metadata.json").read_text()) if e["method"] == method]
    if parity is not None:
        meta = [e for e in meta if int(e["image_name"], 16) % 2 == parity]
    imgs = np.stack([np.asarray(Image.open(REL / split / "images" / f"{e['image_name']}.png").convert("RGB")) for e in meta])
    gts = None
    if split == "train":
        z = np.load(REL / split / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
        gts = np.stack([gt[e["image_name"]] for e in meta])
    msgs = np.array([[int(c) for c in e["message"]] for e in meta], np.float32)
    return meta, imgs, gts, msgs


def to_t(imgs_u8, dev=DEV):
    return torch.from_numpy(np.ascontiguousarray(imgs_u8)).permute(0, 3, 1, 2).float().div(255).to(dev)


@torch.no_grad()
def embed_float(wm, x, msgs):
    """x: B x 3 x H x W float [0,1]; msgs B x 256 -> float watermarked image (no rounding)."""
    m = torch.as_tensor(msgs, dtype=torch.float32, device=x.device)
    out = wm.model.embed(x, msgs=m, is_video=False)
    return out["imgs_w"]


def fix_grid(prob, thr):
    p = (prob >= thr).astype(np.uint8)
    k = max(1, int(0.01 * p.size))
    if p.sum() == 0:
        p.flat[np.argsort(prob.ravel())[-k:]] = 1
    elif p.sum() == p.size:
        p.flat[np.argsort(prob.ravel())[:k]] = 0
    return p


def score(preds, gts):
    return float(np.mean([miou_image(p, g) for p, g in zip(preds, gts)]))


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y).ravel().astype(bool); r = rankdata(np.asarray(s, np.float64).ravel())
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
