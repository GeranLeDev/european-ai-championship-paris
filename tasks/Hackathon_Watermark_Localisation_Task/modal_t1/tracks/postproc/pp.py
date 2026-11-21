"""Post-processing primitives (CPU). A config is a tuple; postprocess() is the single source of truth used both
by the tuning script (score matrices) and by postproc_final.py."""
import numpy as np
from scipy import ndimage

STRUCT4 = ndimage.generate_binary_structure(2, 1)


def fix_grid(mask, prob, k_frac=0.01):
    p = mask.astype(np.uint8).copy(); k = max(1, int(k_frac * p.size))
    if p.sum() == 0:
        p.flat[np.argsort(prob.ravel())[-k:]] = 1
    elif p.sum() == p.size:
        p.flat[np.argsort(prob.ravel())[:k]] = 0
    return p


def smooth(prob, sigma):
    return prob if sigma <= 0 else ndimage.gaussian_filter(prob, sigma)


def clean_minority(minor, minarea, K, majarea=0):
    """minor: bool map of the minority class (marked zone in partial images, clean hole in full images).
    Drop minority components < minarea cells, keep only the K largest (K=0: all), and absorb majority islands
    smaller than majarea into the minority (fill)."""
    if minarea or K:
        lab, n = ndimage.label(minor, STRUCT4)
        if n:
            sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            keep = sizes >= minarea
            if K and keep.sum() > K:
                order = np.argsort(-sizes)
                kk = np.zeros(n, bool); cnt = 0
                for j in order:
                    if keep[j] and cnt < K:
                        kk[j] = True; cnt += 1
                keep = kk
            if not keep.any():                      # never drop everything: keep the largest one
                keep = np.zeros(n, bool); keep[int(np.argmax(sizes))] = True
            minor = np.concatenate([[False], keep])[lab]
    if majarea:
        lab, n = ndimage.label(~minor, STRUCT4)
        if n > 1:
            sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            small = np.concatenate([[False], sizes < majarea]); small[1 + int(np.argmax(sizes))] = False
            minor = minor | small[lab]
    return minor


def snap_rects(minor, ratio):
    """Replace each minority component whose area / bbox area >= ratio by its bounding box."""
    if ratio >= 1.01:
        return minor
    lab, n = ndimage.label(minor, STRUCT4)
    out = minor.copy()
    for j, sl in enumerate(ndimage.find_objects(lab), 1):
        comp = lab[sl] == j
        if comp.mean() >= ratio:
            out[sl] = True
    return out


def box_grid(box, d):
    m = np.zeros((512, 512), np.float32)
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    m[max(0, y0 - d):min(512, y1 + d), max(0, x0 - d):min(512, x1 + d)] = 1
    g = (m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5)
    return g if 0 < g.sum() < g.size else None


def net_mask(prob, partial, sigma, s, minarea, K, majarea=0, fill=0, snap=9.0):
    p = smooth(prob, sigma)
    marked = p >= s
    minor = marked if partial else ~marked
    minor = clean_minority(minor, minarea, K, majarea)
    if fill:
        minor = ndimage.binary_fill_holes(minor)
    minor = snap_rects(minor, snap)
    marked = minor if partial else ~minor
    return fix_grid(marked, p)


def postprocess(prob, partial, rect, net_cfg, rect_cfg):
    """rect: (box, score, coarse) or None. rect_cfg: (t, d, mode) with mode in {'box','union','inter'}; t>=9 = off."""
    t, d, mode = rect_cfg
    if partial and rect is not None and rect[1] >= t:
        g = box_grid(rect[0], d)
        if g is not None:
            if mode == "box":
                return g.astype(np.uint8)
            n = net_mask(prob, partial, *net_cfg).astype(bool)
            out = (g | n) if mode == "union" else (g & n)
            if 0 < out.sum() < out.size:
                return out.astype(np.uint8)
            return g.astype(np.uint8)
    return net_mask(prob, partial, *net_cfg)
