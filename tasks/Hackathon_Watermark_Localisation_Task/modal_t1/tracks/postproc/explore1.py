import sys; sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/tracks/postproc")
from common import *
gt, meta, acc = load_train(); rect = load_rect("train")
for m in METHODS:
    probs = load_probs("train", m)
    items = [e for e in meta if e["method"] == m and e["image_name"] in probs]
    full = np.array([e["full_watermark"] for e in items]); pacc = np.array([acc[e["image_name"]] < 0.8 for e in items])
    print(m, len(items), "full", full.sum(), "regime-err", int(((~full) != pacc).sum()), "in-rect", sum(e["image_name"] in rect for e in items))
    for regime, sel in (("full", full), ("part", ~full)):
        its = [e for e, s in zip(items, sel) if s]
        frac = np.mean([gt[e["image_name"]].mean() for e in its])
        res = []
        for s in np.arange(0.05, 0.96, 0.05):
            res.append((np.mean([miou(probs[e["image_name"]] >= s, gt[e["image_name"]]) for e in its]), s))
        best = max(res)
        # oracle threshold per image
        orc = np.mean([max(miou(probs[e["image_name"]] >= s, gt[e["image_name"]]) for s in np.arange(0.02, 0.99, 0.02)) for e in its])
        rs = [miou(__import__('numpy').zeros(1),0) for _ in []]
        rr = []
        for e in its:
            n = e["image_name"]
            if n in rect:
                b = rect[n][0]; g = np.zeros((512, 512)); x0, y0, x1, y1 = [int(round(float(v))) for v in b]; g[y0:y1, x0:x1] = 1
                g = (g.reshape(128, 4, 128, 4).mean((1, 3)) >= .5)
                rr.append((miou(g, gt[n]), rect[n][1]))
        rr = np.array(rr) if rr else np.zeros((0, 2))
        print(f"  {regime}: n={len(its)} gtfrac={frac:.3f} best-thr={best[1]:.2f} miou={best[0]:.3f} oracle-thr={orc:.3f} rect-mean={rr[:,0].mean() if len(rr) else 0:.3f} (n={len(rr)})")
