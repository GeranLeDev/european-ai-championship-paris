"""Post-processing: snap the network's predicted zones to the image's own SAM masks, evaluated on the odd half
(official miou_image, network only, threshold tuned; 2-fold CV over images for the snapping parameters)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/sprintml/starter_kit")
from wmloc.utils import miou_image  # noqa: E402

from check_overlap import load_sam  # noqa: E402

T1 = Path("/Users/amil/Downloads/hackathon-watermark/modal_t1")
DATA = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release")
RUNS = {"pixelseal": T1 / "pixelseal_research/runs/hires_c/val_probs.npz",
        "trustmark": T1 / "runs/trustmark_mix/train_probs.npz",
        "stegastamp": T1 / "runs/stegastamp_mix/train_probs.npz"}


def fix_grid(prob, thr):
    p = (prob >= thr).astype(np.uint8); k = max(1, int(0.01 * p.size))
    if p.sum() == 0:
        p.flat[np.argsort(prob.ravel())[-k:]] = 1
    elif p.sum() == p.size:
        p.flat[np.argsort(prob.ravel())[:k]] = 0
    return p


def sam128(bits, areas, lo=0.0005, hi=0.6):
    """SAM masks -> 128 grid cells (>= 50 % of the 4x4 cell), small to large."""
    if len(bits) == 0:
        return np.zeros((0, 128, 128), bool)
    M = np.unpackbits(bits, axis=-1)[:, :512 * 512].reshape(-1, 128, 4, 128, 4).mean((2, 4)) >= 0.5
    keep = (M.mean((1, 2)) >= lo) & (M.mean((1, 2)) <= hi)
    return M[keep]


def snap(B, M, tau, max_frac=0.5, classes=(1, 0)):
    """Replace each connected component (of either class, < max_frac) of the binary map B by the SAM mask with the best
    IoU if that IoU >= tau."""
    out = B.copy()
    if len(M) == 0:
        return out
    Mf = M.reshape(len(M), -1)
    for cls in classes:
        lab, n = ndimage.label(B == cls)
        for c in range(1, n + 1):
            cm = (lab == c)
            if cm.mean() > max_frac or cm.sum() < 4:
                continue
            cf = cm.ravel()
            inter = (Mf & cf).sum(1); uni = (Mf | cf).sum(1)
            iou = inter / np.maximum(uni, 1); k = int(iou.argmax())
            if iou[k] >= tau:
                out[cm] = 1 - cls
                out[M[k]] = cls
    return out


def vote(P, M, a):
    """Blend the prob map with its per-SAM-mask mean (large masks first, small ones override)."""
    S = P.copy()
    for k in np.argsort(-M.reshape(len(M), -1).sum(1)):
        S[M[k]] = P[M[k]].mean()
    return a * P + (1 - a) * S


if __name__ == "__main__":
    sam = load_sam("train")
    acc = json.loads((T1 / "cache" / "global_train.json").read_text())
    z = np.load(DATA / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
    THR = np.round(np.arange(0.1, 0.91, 0.05), 2)
    for m, f in RUNS.items():
        zz = np.load(f); probs = dict(zip(zz["names"].tolist(), zz["probs"].astype(np.float32)))
        names = [n for n in probs if n in sam]
        Ms = {n: sam128(sam[n][0], sam[n][1]) for n in names}
        base = {s: np.array([miou_image(fix_grid(probs[n], s), gt[n]) for n in names]) for s in THR}
        sb = max(THR, key=lambda s: base[s].mean())
        print(f"{m}: {len(names)} imgs, net only {base[sb].mean():.4f} @ {sb}")
        res = {}
        for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
            for s in (sb - 0.1, sb - 0.05, sb, sb + 0.05, sb + 0.1):
                s = float(np.clip(s, 0.05, 0.95))
                res[("snap", tau, s)] = np.array([miou_image(snap(fix_grid(probs[n], s), Ms[n], tau, classes=(1,)) if acc[n] < 0.8 else fix_grid(probs[n], s), gt[n]) for n in names])
        for a in (0.0, 0.25, 0.5, 0.75):
            V = {n: vote(probs[n], Ms[n], a) if acc[n] < 0.8 else probs[n] for n in names}
            for s in THR:
                res[("vote", a, float(s))] = np.array([miou_image(fix_grid(V[n], s), gt[n]) for n in names])
        best = max(res, key=lambda k: res[k].mean())
        rng = np.random.default_rng(0); idx = rng.permutation(len(names)); h = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
        cv = []
        for i in (0, 1):
            kb = max(res, key=lambda k: res[k][h[i]].mean()); sbb = max(THR, key=lambda s: base[s][h[i]].mean())
            cv.append((res[kb][h[1 - i]].mean(), base[sbb][h[1 - i]].mean(), kb))
        print(f"   best {best}: {res[best].mean():.4f} | CV snapped {np.mean([c[0] for c in cv]):.4f} vs base {np.mean([c[1] for c in cv]):.4f} {[c[2] for c in cv]}")
        for kind in ("snap", "vote"):
            kk = max([k for k in res if k[0] == kind], key=lambda k: res[k].mean())
            print(f"   best {kind}: {kk} {res[kk].mean():.4f}")
