"""make_sample variant whose object zones follow the source image's own Segment Anything masks.

Same recipe as gen_lib.make_sample (regimes, counts, rect/bar/blob/ellipse, crop->embed->paste), except that an
"object" zone of a PARTIAL image is, with probability p_own, one of the image's own SAM masks (chosen near a sampled
target area), otherwise a shape transplanted from another official image (as in app_data.gen_official).
Measured on the 510 official train images: partial irregular zones match an own SAM mask (median best IoU 0.91),
full-image holes do not (0.17, same as a random other image), so full-image holes stay transplanted/blob shapes."""
from __future__ import annotations

import numpy as np

import gen_lib as G

S = 512


def pick_own(rng, own, areas, target):
    """Own SAM mask whose area is close (in log scale) to the target area."""
    d = np.abs(np.log(areas) - np.log(target))
    w = np.exp(-d / 0.6); w /= w.sum()
    k = int(rng.choice(len(own), p=w))
    return np.unpackbits(own[k])[:S * S].reshape(S, S).astype(bool)


def shape(rng, kind, area, own, own_areas, transplant, max_ar, p_own):
    if kind == "object":
        if len(own) and rng.random() < p_own:
            return pick_own(rng, own, own_areas, area)
        if transplant:
            return transplant[int(rng.integers(len(transplant)))]
        return G.blob_mask(rng, area)
    return G.shape(rng, kind, area, (), max_ar)


def make_sample(wm, method, img, rng, own, own_areas, transplant, p_own=0.85):
    msg = rng.integers(0, 2, wm.nbits)
    if rng.random() < 0.5:                                      # full: everything marked, clean holes restored
        w = wm.embed(img, msg)
        k = rng.choice([1, 2, 3], p=[0.75, 0.17, 0.08]); total = G.sample_area(rng, median=0.06)
        hole = np.zeros((S, S), bool)
        for _ in range(k):
            kind = rng.choice(["object", "blob", "ellipse", "rect"], p=[0.55, 0.3, 0.08, 0.07])
            hole |= shape(rng, kind, total / k, own, own_areas, transplant, 2.0, 0.0)   # holes are blobs, not objects
        if hole.all() or not hole.any():
            hole = G.blob_mask(rng, 0.05)
        return np.where(hole[..., None], img, w), ~hole, msg, "full"
    out = img.copy(); mask = np.zeros((S, S), bool)                # partial: small marked zones
    k = rng.choice([1, 2, 3], p=[0.8, 0.12, 0.08]); total = G.sample_area(rng, median=0.04)
    w_full = None
    for _ in range(k):
        p_bar = 0.0 if method == "trustmark" else 0.15
        kind = rng.choice(["rect", "bar", "object"], p=[0.5, p_bar, 0.5 - p_bar])
        max_ar = 2.0 if method == "trustmark" else (2.0 if rng.random() < 0.8 else 4.0)
        m = shape(rng, kind, total / k, own, own_areas, transplant, max_ar, p_own)
        if kind == "rect" and max_ar <= 2.0:
            b = G.bbox(m); out = G.embed_patch(wm, out, msg, b)
        elif rng.random() < 0.5:
            b = G.expand_ar(G.bbox(m)); pw = G.embed_patch(wm, out, msg, b)
            out = np.where(m[..., None], pw, out)
        else:
            if w_full is None:
                w_full = wm.embed(img, msg)
            out = np.where(m[..., None], w_full, out)
        mask |= m
    if not mask.any():
        b = G.rect_box(rng, 0.03); out = G.embed_patch(wm, img, msg, b); mask = G.box_mask(b)
    return out.astype(np.uint8), mask, msg, "partial"
