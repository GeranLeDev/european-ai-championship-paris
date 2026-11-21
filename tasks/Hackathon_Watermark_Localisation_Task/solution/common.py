"""Shared helpers: train/held-out split, random masks."""
import json
from pathlib import Path
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("pixelseal", "trustmark", "stegastamp")


def split_train(data_dir=ROOT / "data", n_val_per_group=25, seed=0):
    """Deterministic held-out split of the 510 train images: per (method, full_watermark) group,
    n_val_per_group images go to validation (-> 50 per method, 150 total)."""
    meta = json.loads((Path(data_dir) / "train" / "metadata.json").read_text())
    rng = np.random.default_rng(seed)
    val = set()
    for m in METHODS:
        for full in (False, True):
            names = sorted(e["image_name"] for e in meta if e["method"] == m and e["full_watermark"] == full)
            val.update(rng.permutation(names)[:n_val_per_group].tolist())
    return val


def blob_mask(rng, size=512, frac=None):
    """Irregular multi-component mask with area fraction ~frac (smoothed noise thresholded)."""
    frac = frac if frac is not None else float(np.exp(rng.uniform(np.log(0.003), np.log(0.45))))
    s = rng.choice([4, 8, 16, 32])                       # noise grid -> blob scale
    noise = rng.standard_normal((s, s))
    noise = ndimage.zoom(noise, size / s, order=3)
    noise = ndimage.gaussian_filter(noise, sigma=size / s / 2)
    if rng.random() < 0.5:                               # bias toward one dominant blob
        cy, cx = rng.uniform(0.1, 0.9, 2) * size
        yy, xx = np.mgrid[:size, :size]
        noise = noise - 0.5 * noise.std() * (((yy - cy) ** 2 + (xx - cx) ** 2) / (size * 0.3) ** 2)
    thr = np.quantile(noise, 1 - frac)
    m = noise > thr
    if rng.random() < 0.5:                               # keep only a few largest components
        lab, n = ndimage.label(m)
        if n > 1:
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            keep = np.argsort(sizes)[::-1][: rng.integers(1, 4)] + 1
            m = np.isin(lab, keep)
    return m.astype(np.uint8)


def rect(rng, size=512):
    """Random rectangle (y0, y1, x0, x1), aspect ratio <= 2, area fraction log-uniform in [0.3 %, 45 %]."""
    frac = float(np.exp(rng.uniform(np.log(0.003), np.log(0.45))))
    ar = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
    area = frac * size * size
    h = int(np.clip(np.sqrt(area * ar), 16, size)); w = int(np.clip(area / h, 16, size))
    y0 = rng.integers(0, size - h + 1); x0 = rng.integers(0, size - w + 1)
    return y0, y0 + h, x0, x0 + w
