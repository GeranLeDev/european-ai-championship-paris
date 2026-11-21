"""TrustMark only: net-only score and combine2-style score (regime + rectangle override, 2-fold CV with the same
seed/permutation logic as combine2.py) for the U-Net, the detector head and weighted averages. Odd half only.

    python -I eval_combo.py HEAD_VAL.npz [--write OUT_DIR --w 0.5 --head-test HEAD_TEST.npz]"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np

T1 = Path("/Users/amil/Downloads/hackathon-watermark/modal_t1")
REL = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release")
sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/sprintml/starter_kit")
from wmloc.utils import miou_image  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("head", nargs="+"); ap.add_argument("--unet", default=str(T1 / "runs/trustmark_mix/train_probs.npz"))
ap.add_argument("--write", default=""); ap.add_argument("--w", type=float, default=-1); ap.add_argument("--head-test", nargs="*", default=[])
a = ap.parse_args()
T_FULL = 0.80


def fix_grid(prob, thr):
    p = (prob >= thr).astype(np.uint8); k = max(1, int(0.01 * p.size))
    if p.sum() == 0: p.flat[np.argsort(prob.ravel())[-k:]] = 1
    elif p.sum() == p.size: p.flat[np.argsort(prob.ravel())[:k]] = 0
    return p


def box_mask(box, d):
    m = np.zeros((512, 512), np.float32)
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    m[max(0, y0 - d):min(512, y1 + d), max(0, x0 - d):min(512, x1 + d)] = 1
    g = (m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5).astype(np.uint8)
    return g if 0 < g.sum() < g.size else None


def load_rect(split):
    best = {}; pat = re.compile(rf"search_{split}_(\w+?)_f(_shard\d+)?\.npz$")
    for p in sorted((T1 / "cache_dl").rglob(f"search_{split}_*.npz")):
        if pat.fullmatch(p.name):
            z = np.load(p)
            for n, b, s in zip(z["names"].tolist(), z["box"], z["score"]):
                if n not in best or s > best[n][1]: best[n] = (np.asarray(b, np.float32), float(s))
    return best


def predict(name, prob, acc, rect, s, t, d):
    if acc[name] < T_FULL and name in rect and rect[name][1] >= t:
        g = box_mask(rect[name][0], d)
        if g is not None: return g
    return fix_grid(prob, s)


def load(f):
    z = np.load(f); return dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))


acc_tr = json.loads((T1 / "cache/global_train.json").read_text()); rect_tr = load_rect("train")
z = np.load(REL / "train/masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
meta = json.loads((REL / "train/metadata.json").read_text())
S_GRID = np.round(np.arange(0.1, 0.91, 0.05), 2); T_GRID = [9.0] + list(np.round(np.arange(0.3, 0.96, 0.05), 2)); D_GRID = (-4, 0, 2, 4)


_RECT_CACHE = {}


def rect_table(items):
    """per image: rect mIoU for each d (nan if no usable box), and the box score (-inf if not eligible)."""
    R = np.full((len(items), len(D_GRID)), np.nan); sc = np.full(len(items), -np.inf)
    for i, e in enumerate(items):
        n = e["image_name"]
        if acc_tr[n] < T_FULL and n in rect_tr:
            sc[i] = rect_tr[n][1]
            for j, d in enumerate(D_GRID):
                key = (n, d)
                if key not in _RECT_CACHE:
                    g = box_mask(rect_tr[n][0], d); _RECT_CACHE[key] = np.nan if g is None else miou_image(g, gt[n])
                R[i, j] = _RECT_CACHE[key]
    return R, sc


def tables(items, probs):
    N = np.array([[miou_image(fix_grid(probs[e["image_name"]], s), gt[e["image_name"]]) for s in S_GRID] for e in items])
    R, sc = rect_table(items)
    return N, R, sc


def tune_idx(N, R, sc, idx):
    best = None
    for si, s in enumerate(S_GRID):
        for t in T_GRID:
            for dj, d in enumerate(D_GRID if t < 9 else (0,)):
                use = (sc[idx] >= t) & ~np.isnan(R[idx, dj]) if t < 9 else np.zeros(len(idx), bool)
                v = np.where(use, np.nan_to_num(R[idx, dj]), N[idx, si]).mean()
                if best is None or v > best[0]: best = (v, si, t, dj)
    return best


def score_idx(N, R, sc, idx, si, t, dj):
    use = (sc[idx] >= t) & ~np.isnan(R[idx, dj]) if t < 9 else np.zeros(len(idx), bool)
    return np.where(use, np.nan_to_num(R[idx, dj]), N[idx, si])


def evaluate(probs, label):
    items = [e for e in meta if e["method"] == "trustmark" and e["image_name"] in probs and int(e["image_name"], 16) % 2 == 1]
    N, R, sc = tables(items, probs)
    full = np.array([e["full_watermark"] for e in items])
    net = N.mean(0).max(); s_net = S_GRID[N.mean(0).argmax()]
    nf = N[full].mean(0).max(); npt = N[~full].mean(0).max()
    cv = []
    for seed in range(20):            # 2-fold CV as in combine2, averaged over 20 random splits (90 images: noisy)
        idx = np.random.default_rng(seed).permutation(len(items)); halves = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
        for h in (0, 1):
            _, si, t, dj = tune_idx(N, R, sc, halves[h])
            cv += list(score_idx(N, R, sc, halves[1 - h], si, t, dj))
    v, si, t, dj = tune_idx(N, R, sc, np.arange(len(items)))
    print(f"{label:28s} n={len(items)} | net {net:.4f} (s {s_net:.2f}) full {nf:.3f} partial {npt:.3f} | +rect CV {np.mean(cv):.4f} in-sample {v:.4f} (s {S_GRID[si]}, t {t}, d {D_GRID[dj] if t < 9 else 0})", flush=True)
    return float(np.mean(cv)), (S_GRID[si], t, D_GRID[dj] if t < 9 else 0)


U = load(a.unet)
evaluate(U, "unet trustmark_mix")
H = {}
for f in a.head:
    h = load(f); H[f] = h; evaluate(h, "head " + Path(f).parent.name)
hs = {n: np.mean([H[f][n] for f in a.head], 0) for n in H[a.head[0]]}
if len(a.head) > 1: evaluate(hs, "head ensemble")
res = {}
for w in (0.3, 0.5, 0.7):
    avg = {n: w * hs[n] + (1 - w) * U[n] for n in hs if n in U}
    res[w] = evaluate(avg, f"avg w_head={w}")
if a.write:
    out = Path(a.write); out.mkdir(parents=True, exist_ok=True)
    w = a.w
    tz = np.load(T1 / "runs/trustmark_mix/test_probs.npz"); TU = dict(zip(tz["names"].tolist(), tz["probs"].astype(np.float32)))
    TH = [load(f) for f in a.head_test]; names = sorted(TU)
    th = {n: np.mean([t[n] for t in TH], 0) for n in names}
    vnames = sorted(hs)
    np.savez_compressed(out / "val_probs.npz", names=np.array(vnames), probs=np.stack([w * hs[n] + (1 - w) * U[n] for n in vnames]).astype(np.float16))
    np.savez_compressed(out / "test_probs.npz", names=np.array(names), probs=np.stack([w * th[n] + (1 - w) * TU[n] for n in names]).astype(np.float16))
    print("written", out, len(vnames), len(names))
