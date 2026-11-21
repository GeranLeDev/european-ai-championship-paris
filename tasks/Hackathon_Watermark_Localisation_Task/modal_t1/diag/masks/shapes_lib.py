"""Mask-shape features shared by the official-side and generated-side audits.

Works on a boolean minority-class mask at any square resolution (512 or 128):
  partial image -> minority = marked zones ; full image -> minority = holes (clean pixels).
All lengths are returned in 512-pixel units, areas as fraction of the image."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

S = 512


def comp_features(m: np.ndarray, min_px512: int = 16, conn8: bool = False) -> list[dict]:
    """Connected components of a bool mask -> list of feature dicts (lengths in 512 px units)."""
    n = m.shape[0]; sc = S / n
    st = np.ones((3, 3), int) if conn8 else None
    lab, k = ndimage.label(m, structure=st)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        c = lab[sl] == i
        a = int(c.sum())
        if a * sc * sc < min_px512:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        h, w = y1 - y0, x1 - x0
        extent = a / (h * w)
        cu = c.astype(np.uint8)
        cnts, _ = cv2.findContours(np.pad(cu, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        per = cv2.arcLength(cnt, True)
        hull = cv2.convexHull(cnt)
        # hull area on the pixel grid (fillPoly) so that a perfect rectangle has solidity 1
        hm = np.zeros((h + 2, w + 2), np.uint8); cv2.fillPoly(hm, [hull], 1)
        solidity = a / max(int(hm.sum()), 1)
        compact = 4 * np.pi * a / max(per * per, 1e-9) if per > 0 else 1.0
        # ellipse test: moment-matched ellipse (handles rotation), IoU with the component
        yy, xx = np.nonzero(c)
        cy, cx = yy.mean(), xx.mean()
        ell_iou = 0.0; axis_ell_iou = 0.0
        if a >= 20:
            cov = np.cov(np.stack([yy - cy, xx - cx]))
            ev, evec = np.linalg.eigh(cov + 1e-9 * np.eye(2))
            gy, gx = np.mgrid[0:h, 0:w]
            d = np.stack([gy - cy, gx - cx], -1) @ evec
            e = (d[..., 0] ** 2 / (4 * ev[0] + 1e-9) + d[..., 1] ** 2 / (4 * ev[1] + 1e-9)) <= 1
            ell_iou = float((e & c).sum() / max((e | c).sum(), 1))
            e2 = ((gy - (h - 1) / 2) / (h / 2)) ** 2 + ((gx - (w - 1) / 2) / (w / 2)) ** 2 <= 1
            axis_ell_iou = float((e2 & c).sum() / max((e2 | c).sum(), 1))
        holes = int(ndimage.binary_fill_holes(c).sum() - a)
        ar = max(h / w, w / h)
        exact = bool(c.all())
        if not exact and n < S and h >= 2 and w >= 2:
            # on the 128 grid the >=50 % rule is not separable: a true rectangle may lose its corner cells
            cc = c.copy(); cc[0, 0] = cc[0, -1] = cc[-1, 0] = cc[-1, -1] = True
            exact = bool(cc.all())
        area_frac = a * sc * sc / (S * S)
        if exact and n < S and area_frac < 0.002:
            kind = "speck"          # 1-30 cells: cannot tell a rectangle from a fragment on the grid
        elif exact:
            kind = "bar" if ar > 4 else "rect"
        elif max(ell_iou, axis_ell_iou) >= 0.93 and extent < 0.9:
            kind = "ellipse"
        elif extent >= 0.97:
            kind = "near_rect"
        else:
            kind = "object"
        out.append(dict(
            kind=kind, area=a * sc * sc / (S * S), h=h * sc, w=w * sc, ar=ar, vertical=bool(h > w),
            y0=y0 * sc, y1=y1 * sc, x0=x0 * sc, x1=x1 * sc, cy=(y0 + cy + 0.5) * sc / S, cx=(x0 + cx + 0.5) * sc / S,
            touch=bool(y0 == 0 or x0 == 0 or y1 == n or x1 == n),
            ntouch=int(y0 == 0) + int(x0 == 0) + int(y1 == n) + int(x1 == n),
            extent=extent, solidity=solidity, compact=float(min(compact, 1.5)), ell_iou=max(ell_iou, axis_ell_iou),
            inner_holes=holes * sc * sc / (S * S),
        ))
    return out


def image_row(minority: np.ndarray, **extra) -> dict:
    comps = comp_features(minority)
    comps.sort(key=lambda c: -c["area"])
    return dict(frac=float(minority.mean()), n=len(comps), comps=comps, **extra)


def pct(a, qs=(5, 25, 50, 75, 95)):
    a = np.asarray(a, float)
    if len(a) == 0:
        return [float("nan")] * len(qs)
    return [float(np.percentile(a, q)) for q in qs]


def fmt(a, nd=3):
    return "[" + " ".join(f"{v:.{nd}f}" for v in a) + "]"
