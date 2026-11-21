"""Fabrication d'exemples comme ceux des organisateurs (tâche 1), avec les embedders exacts du kit officiel.

Ce qu'on sait de leurs 510 images d'entraînement :
  - moitié « complètes » : toute l'image est marquée, puis quelques zones (objets, taches, ellipses, rectangles)
    sont remises propres ; la part remise propre va de 0,3 % à 49 % (médiane 6 %) ;
  - moitié « partielles » : une ou quelques petites zones marquées (0,3 % à 44 %, médiane 4 %) ; environ la moitié sont
    des rectangles exacts, fabriqués en marquant le morceau découpé puis en le recollant (le message se relit sur le
    cadre exact) ; le kit précise que les morceaux ont un rapport largeur/hauteur <= 2 ; le reste : barres et objets ;
  - aucune grille vide, images 512 x 512, PNG.
Pour les zones partielles non rectangulaires, la recette exacte est incertaine : on mélange deux hypothèses
(morceau englobant marqué puis forme recollée, ou image entière marquée puis forme recollée)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

S = 512


def load_clean(raw: bytes | Image.Image, rng) -> tuple[np.ndarray, tuple]:
    """Photo -> 512 x 512 (côté le plus court à 512, recadrage au hasard). Renvoie l'image et (échelle, x0, y0)."""
    import io
    im = raw if isinstance(raw, Image.Image) else Image.open(io.BytesIO(raw))
    im = im.convert("RGB"); w, h = im.size; s = S / min(w, h)
    nw, nh = max(S, round(w * s)), max(S, round(h * s))
    im = im.resize((nw, nh), Image.BICUBIC)
    x0 = int(rng.integers(0, nw - S + 1)); y0 = int(rng.integers(0, nh - S + 1))
    return np.asarray(im.crop((x0, y0, x0 + S, y0 + S))), (s, x0, y0)


def object_masks(polys, tr, min_frac=0.002, max_frac=0.45) -> list[np.ndarray]:
    """Polygones COCO (coordonnées de la photo d'origine) -> masques 512 x 512 après le même redimensionnement."""
    s, x0, y0 = tr; out = []
    for poly_list in polys:
        im = Image.new("L", (S, S), 0); d = ImageDraw.Draw(im)
        for p in poly_list:
            pts = [(px * s - x0, py * s - y0) for px, py in zip(p[0::2], p[1::2])]
            if len(pts) >= 3:
                d.polygon(pts, fill=1)
        m = np.asarray(im, bool)
        if min_frac <= m.mean() <= max_frac:
            out.append(m)
    return out


def sample_area(rng, median=0.05, sigma=1.1, lo=0.003, hi=0.45) -> float:
    return float(np.clip(np.exp(np.log(median) + sigma * rng.standard_normal()), lo, hi))


def rect_box(rng, area, max_ar=2.0):
    ar = float(np.exp(rng.uniform(-np.log(max_ar), np.log(max_ar))))
    h = int(np.clip(round(np.sqrt(area * S * S / ar)), 9, S)); w = int(np.clip(round(area * S * S / max(h, 1)), 9, S))
    y0 = int(rng.integers(0, S - h + 1)); x0 = int(rng.integers(0, S - w + 1))
    return y0, y0 + h, x0, x0 + w


def box_mask(b) -> np.ndarray:
    m = np.zeros((S, S), bool); m[b[0]:b[1], b[2]:b[3]] = True; return m


def ellipse_mask(rng, area) -> np.ndarray:
    ar = float(np.exp(rng.uniform(-np.log(2.5), np.log(2.5))))
    ry = np.sqrt(area * S * S / (np.pi * ar)); rx = ry * ar
    cy, cx = rng.uniform(ry, S - ry) if ry < S / 2 else S / 2, rng.uniform(rx, S - rx) if rx < S / 2 else S / 2
    yy, xx = np.mgrid[0:S, 0:S]
    return ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1


def blob_mask(rng, area) -> np.ndarray:
    """Tache irrégulière d'environ `area` de l'image : bruit lissé, seuillé autour d'un centre, une seule composante."""
    r = np.sqrt(area) * S
    noise = ndimage.gaussian_filter(rng.standard_normal((S, S)), sigma=max(4.0, r / 3))
    noise = (noise - noise.mean()) / (noise.std() + 1e-9)
    cy, cx = rng.uniform(0.15, 0.85, 2) * S
    yy, xx = np.mgrid[0:S, 0:S]
    field = noise - ((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)
    thr = np.quantile(field, 1 - area)
    m = field >= thr
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1)); m = lab == (1 + int(np.argmax(sizes)))
    return m


def bbox(m):
    ys, xs = np.where(m)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def expand_ar(b, max_ar=2.0, min_side=32):
    """Agrandit un cadre pour que son rapport largeur/hauteur soit <= max_ar et chaque côté >= min_side."""
    y0, y1, x0, x1 = b; h, w = y1 - y0, x1 - x0
    th = max(h, min_side, int(np.ceil(w / max_ar))); tw = max(w, min_side, int(np.ceil(h / max_ar)))
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    y0 = int(np.clip(round(cy - th / 2), 0, S - th)); x0 = int(np.clip(round(cx - tw / 2), 0, S - tw))
    return y0, y0 + min(th, S), x0, x0 + min(tw, S)


def embed_patch(wm, img, msg, b):
    y0, y1, x0, x1 = b
    out = img.copy(); out[y0:y1, x0:x1] = wm.embed(np.ascontiguousarray(img[y0:y1, x0:x1]), msg)
    return out


def shape(rng, kind, area, objs, max_ar):
    if kind == "object" and objs:
        m = objs[int(rng.integers(len(objs)))]
        return m
    if kind in ("object", "blob"):
        return blob_mask(rng, area)
    if kind == "ellipse":
        return ellipse_mask(rng, area)
    if kind == "bar":
        ar = float(rng.uniform(4, 16)); short = int(np.clip(round(np.sqrt(area * S * S / ar)), 6, 64))
        long = int(np.clip(round(area * S * S / short), 64, S))
        h, w = (long, short) if rng.random() < 0.5 else (short, long)
        y0 = int(rng.integers(0, S - h + 1)); x0 = int(rng.integers(0, S - w + 1))
        return box_mask((y0, y0 + h, x0, x0 + w))
    return box_mask(rect_box(rng, area, max_ar))


def make_sample(wm, method, img, rng, objs=()):
    """Renvoie (image marquée uint8 512x512x3, masque bool 512x512, message, régime)."""
    msg = rng.integers(0, 2, wm.nbits)
    if rng.random() < 0.5:                                      # « complète » : tout marqué, des trous remis propres
        w = wm.embed(img, msg)
        k = rng.choice([1, 2, 3], p=[0.75, 0.17, 0.08]); total = sample_area(rng, median=0.06)
        hole = np.zeros((S, S), bool)
        for _ in range(k):
            kind = rng.choice(["object", "blob", "ellipse", "rect"], p=[0.55, 0.3, 0.08, 0.07])
            hole |= shape(rng, kind, total / k, objs, 2.0)
        if hole.all() or not hole.any():
            hole = blob_mask(rng, 0.05)
        return np.where(hole[..., None], img, w), ~hole, msg, "full"
    out = img.copy(); mask = np.zeros((S, S), bool)                # « partielle » : de petites zones marquées
    k = rng.choice([1, 2, 3], p=[0.8, 0.12, 0.08]); total = sample_area(rng, median=0.04)
    w_full = None
    for _ in range(k):
        p_bar = 0.0 if method == "trustmark" else 0.15
        kind = rng.choice(["rect", "bar", "object"], p=[0.5, p_bar, 0.5 - p_bar])
        max_ar = 2.0 if method == "trustmark" else (2.0 if rng.random() < 0.8 else 4.0)
        m = shape(rng, kind, total / k, objs, max_ar)
        if kind == "rect" and max_ar <= 2.0:                      # morceau découpé, marqué, recollé tel quel
            b = bbox(m); out = embed_patch(wm, out, msg, b)
        elif rng.random() < 0.5:                                  # hypothèse 1 : morceau englobant marqué, forme recollée
            b = expand_ar(bbox(m)); pw = embed_patch(wm, out, msg, b)
            out = np.where(m[..., None], pw, out)
        else:                                                     # hypothèse 2 : image entière marquée, forme recollée
            if w_full is None:
                w_full = wm.embed(img, msg)
            out = np.where(m[..., None], w_full, out)
        mask |= m
    if not mask.any():
        b = rect_box(rng, 0.03); out = embed_patch(wm, img, msg, b); mask = box_mask(b)
    return out.astype(np.uint8), mask, msg, "partial"
