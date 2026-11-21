"""Adaptive per-image thresholds: quantile rules and Otsu, vs fixed thresholds, per predicted regime (2-fold CV x20)."""
import sys; sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/tracks/postproc")
from common import *
from pp import fix_grid, clean_minority, box_grid
gt, meta, acc = load_train(); rect = load_rect("train")
S = np.round(np.arange(0.05, 0.96, 0.05), 2); Q = (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3)
T_GRID = [9.0] + list(np.round(np.arange(0.3, 0.96, 0.05), 2))


def otsu(p):
    h, e = np.histogram(p, 64, (0, 1)); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); mu = np.cumsum(h * c)
    w0 = w[:-1]; w1 = w[-1] - w0; m0 = mu[:-1] / np.maximum(w0, 1); m1 = (mu[-1] - mu[:-1]) / np.maximum(w1, 1)
    return c[int(np.argmax(w0 * w1 * (m0 - m1) ** 2))]


def cols(prob, partial):
    """list of (key, mask) for one image."""
    out = []
    flat = prob.ravel()
    for s in S:
        for K in (0, 1):
            for qq in (None,) + Q:
                if qq is None:
                    thr = s
                else:   # rule: fixed threshold, but the minority class has at least a fraction qq of the cells
                    thr = max(s, np.quantile(flat, 1 - qq)) if False else (min(s, np.quantile(flat, 1 - qq)) if partial else max(s, np.quantile(flat, qq)))
                marked = prob >= thr
                minor = marked if partial else ~marked
                if K: minor = clean_minority(minor, 0, K)
                out.append((("s", s, K, qq), (minor if partial else ~minor)))
    for qq in Q:
        thr = np.quantile(flat, 1 - qq) if partial else np.quantile(flat, qq)
        out.append((("q", qq, 0, None), prob >= thr))
    t = otsu(prob); out.append((("otsu", 0, 0, None), prob >= t))
    return [(k, fix_grid(m, prob)) for k, m in out]


for m in METHODS:
    probs = load_probs("train", m)
    items = [e for e in meta if e["method"] == m and e["image_name"] in probs]
    res = {}
    for reg in (False, True):
        its = [e for e in items if (acc[e["image_name"]] < 0.8) == reg]
        keys = None; M = []
        for e in its:
            n = e["image_name"]; cs = cols(probs[n], reg); keys = [k for k, _ in cs]
            row = [miou(mk, gt[n]) for _, mk in cs]
            M.append(row)
        M = np.array(M)
        if reg:   # rect override (box, d=0)
            rs = np.array([rect[e["image_name"]][1] if e["image_name"] in rect else -1 for e in its])
            bx = np.array([miou(box_grid(rect[e["image_name"]][0], 0), gt[e["image_name"]]) if e["image_name"] in rect and box_grid(rect[e["image_name"]][0], 0) is not None else np.nan for e in its])
            big = [M]; kk = [(k, 9.0) for k in keys]
            for t in T_GRID[1:]:
                use = (rs >= t) & ~np.isnan(bx); big.append(np.where(use[:, None], np.nan_to_num(bx)[:, None], M)); kk += [(k, t) for k in keys]
            M = np.concatenate(big, 1); keys = kk
        else:
            keys = [(k, 9.0) for k in keys]
        res[reg] = (its, M, keys)
    names = [e["image_name"] for e in items]

    def family_cols(keys, fam):
        sel = []
        for i, (k, t) in enumerate(keys):
            kind, s, K, qq = k
            ok = {"fixed": kind == "s" and K == 0 and qq is None, "fixed+q": kind == "s" and K == 0,
                  "fixed+K": kind == "s" and qq is None, "fixed+q+K": kind == "s", "quant": kind == "q", "otsu": kind == "otsu", "all": True}[fam]
            if ok: sel.append(i)
        return np.array(sel)

    print(f"== {m}")
    for fam in ("fixed", "fixed+q", "fixed+K", "fixed+q+K", "quant", "otsu", "all"):
        scores = []
        for sd in range(20):
            perm = np.random.default_rng(sd).permutation(len(names)); halves = [set(perm[: len(names) // 2]), set(perm[len(names) // 2:])]
            idx = {n: i for i, n in enumerate(names)}; tot = []
            for h in (0, 1):
                for reg in (False, True):
                    its, M, keys = res[reg]; c = family_cols(keys, fam)
                    tr = [j for j, e in enumerate(its) if idx[e["image_name"]] in halves[h]]
                    te = [j for j, e in enumerate(its) if idx[e["image_name"]] not in halves[h]]
                    if not te: continue
                    col = c[int(np.argmax(M[np.ix_(tr, c)].mean(0)))]
                    tot += list(M[te, col])
            scores.append(np.mean(tot))
        best = []
        for reg in (False, True):
            its, M, keys = res[reg]; c = family_cols(keys, fam); best.append(keys[c[int(np.argmax(M[:, c].mean(0)))]])
        print(f"  {fam:10s} CV {np.mean(scores):.4f} ±{np.std(scores):.4f} | full {best[0]} | part {best[1]}")
