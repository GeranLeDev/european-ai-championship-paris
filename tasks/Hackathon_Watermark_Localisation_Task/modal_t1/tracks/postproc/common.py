"""Shared loaders for the post-processing track (CPU only)."""
import json, re
from pathlib import Path
import numpy as np

ROOT = Path("/Users/amil/Downloads/hackathon-watermark")
DATA = ROOT / "sprintml/release"
MT1 = ROOT / "modal_t1"
RUN = {"pixelseal": MT1 / "pixelseal_research/runs/hires_c", "trustmark": MT1 / "runs/trustmark_mix",
       "stegastamp": MT1 / "runs/stegastamp_mix"}
METHODS = ("pixelseal", "trustmark", "stegastamp")


def miou(p, g):
    p = p.astype(bool); g = g.astype(bool)
    i1 = (p & g).sum(); u1 = (p | g).sum(); i0 = (~p & ~g).sum(); u0 = (~p | ~g).sum()
    return 0.5 * ((1.0 if u1 == 0 else i1 / u1) + (1.0 if u0 == 0 else i0 / u0))


def load_rect(split, cache=MT1 / "cache_dl", tag="_f"):
    best = {}
    pat = re.compile(rf"search_{split}_(\w+?){re.escape(tag)}(_shard\d+)?\.npz$")
    for p in sorted(Path(cache).rglob(f"search_{split}_*.npz")):
        if pat.fullmatch(p.name):
            z = np.load(p)
            for n, b, s, c in zip(z["names"].tolist(), z["box"], z["score"], z["coarse"]):
                if n not in best or s > best[n][1]:
                    best[n] = (np.asarray(b, np.float32), float(s), float(c))
    return best


def load_probs(split, m, run=None):
    d = Path((run or RUN)[m]); f = d / f"{split}_probs.npz"
    if split == "train" and not f.exists():
        f = d / "val_probs.npz"
    z = np.load(f)
    return dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))


def load_train():
    z = np.load(DATA / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"].astype(np.uint8)))
    meta = json.loads((DATA / "train" / "metadata.json").read_text())
    acc = json.loads((MT1 / "cache" / "global_train.json").read_text())
    return gt, meta, acc


def auc(y, s):
    y = np.asarray(y).ravel().astype(bool); s = np.asarray(s, np.float64).ravel()
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
