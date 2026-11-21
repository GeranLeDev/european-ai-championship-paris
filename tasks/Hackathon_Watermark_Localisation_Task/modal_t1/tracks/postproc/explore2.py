import sys; sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/tracks/postproc")
from common import *
from scipy import ndimage
gt, meta, acc = load_train(); rect = load_rect("train")
def fix(p, prob):
    k = max(1, int(0.01 * p.size)); p = p.astype(np.uint8).copy()
    if p.sum() == 0: p.flat[np.argsort(prob.ravel())[-k:]] = 1
    elif p.sum() == p.size: p.flat[np.argsort(prob.ravel())[:k]] = 0
    return p
for m in METHODS:
    probs = load_probs("train", m)
    items = [e for e in meta if e["method"] == m and e["image_name"] in probs]
    for regime in (True, False):
        its = [e for e in items if e["full_watermark"] == regime]
        best = max((np.mean([miou(fix(probs[e["image_name"]] >= s, probs[e["image_name"]]), gt[e["image_name"]]) for e in its]), s) for s in np.arange(0.05, 0.96, 0.05))
        s = best[1]
        sc = np.array([miou(fix(probs[e["image_name"]] >= s, probs[e["image_name"]]), gt[e["image_name"]]) for e in its])
        # iou of the minority class (hole for full, zone for partial)
        minor = []
        ncomp = []
        for e in its:
            n = e["image_name"]; g = gt[n].astype(bool); p = probs[n] >= s
            if regime: g, p = ~g, ~p
            minor.append((g & p).sum() / max(1, (g | p).sum()))
            ncomp.append(ndimage.label(g)[1])
        minor = np.array(minor)
        print(f"{m:10s} {'full' if regime else 'part'} n={len(its)} thr={s:.2f} mean={best[0]:.3f} | minorIoU<0.1: {np.mean(minor<0.1):.2f}  0.1-0.5: {np.mean((minor>=0.1)&(minor<0.5)):.2f} 0.5-0.8: {np.mean((minor>=0.5)&(minor<0.8)):.2f} >=0.8: {np.mean(minor>=0.8):.2f} | gt comps: {np.bincount(ncomp)}")
