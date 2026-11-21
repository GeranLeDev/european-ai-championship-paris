"""Task 1 final combination with regime-specific post-processing (CPU only, no GPU).

Per image (method known, regime from the global bit agreement: < 0.80 = partial):
  - partial and a rectangle-search box with score >= t: that box (grown by d px);
  - otherwise the network map, thresholded at a REGIME-SPECIFIC threshold, then optionally only the K largest
    components of the minority class are kept (marked zones in partial images, clean holes in full images);
  - never an empty or full grid.
Parameters per method and per regime are tuned on the official-train ODD half (the only images the networks never
trained on); the score printed is a repeated 2-fold cross-validation (20 random splits, tuned on one half, scored on
the other, official miou_image), next to combine2.py's rule (one threshold for both regimes) on the same splits.
--run accepts several folders per method joined by '+': their probability maps are averaged (stacking).
The rectangle override is switched off for a method whose TEST split has no rectangle-search results (stegastamp).

    python -I postproc_final.py --data <release> --kit <starter_kit> \
        --run pixelseal=pixelseal_research/runs/hires_c+pixelseal_research/runs/frozen_a,trustmark=runs/trustmark_mix,stegastamp=runs/stegastamp_mix \
        --cache cache_dl --glob-dir cache [--test out.pt] [--out-dir tracks/postproc/final]
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pp import box_grid, net_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--kit", required=True)
ap.add_argument("--run", required=True, help="outil=dossier[+dossier2],...")
ap.add_argument("--cache", default="cache"); ap.add_argument("--rect-tag", default="_f")
ap.add_argument("--test", default=""); ap.add_argument("--glob-dir", default="cache")
ap.add_argument("--out-dir", default=str(HERE / "final")); ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--k-methods", default="pixelseal", help="methods allowed the K-largest-component filter (comma list, '' = none)")
a = ap.parse_args()
sys.path.insert(0, a.kit)
from wmloc.utils import miou_image  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
DATA, CACHE = Path(a.data), Path(a.cache)
RUN = {k: [Path(p) for p in v.split("+")] for k, v in (kv.split("=", 1) for kv in a.run.split(","))}
T_FULL = 0.80
S_GRID = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2))
K_METHODS = set(filter(None, a.k_methods.split(",")))
T_GRID = [9.0] + list(np.round(np.arange(0.3, 0.96, 0.05), 2))
D_GRID = (-4, 0, 2, 4)


def load_rect(split):
    best = {}
    pat = re.compile(rf"search_{split}_(\w+?){re.escape(a.rect_tag)}(_shard\d+)?\.npz$")
    for p in sorted(CACHE.rglob(f"search_{split}_*.npz")):
        if pat.fullmatch(p.name):
            z = np.load(p)
            for n, b, s in zip(z["names"].tolist(), z["box"], z["score"]):
                if n not in best or s > best[n][1]:
                    best[n] = (np.asarray(b, np.float32), float(s))
    return best


def load_probs(split, m):
    acc = None
    for d in RUN[m]:
        f = d / f"{split}_probs.npz"
        if split == "train" and not f.exists():
            f = d / "val_probs.npz"
        z = np.load(f); cur = dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))
        if acc is None:
            acc = cur
        else:
            assert set(acc) == set(cur), f"{d}: names differ"
            acc = {n: acc[n] + cur[n] for n in acc}
    return {n: v / len(RUN[m]) for n, v in acc.items()}


def predict(prob, partial, rect, cfg):
    """cfg = (s_full, K_full, s_part, K_part, t, d)."""
    s_f, k_f, s_p, k_p, t, d = cfg
    if partial and rect is not None and rect[1] >= t:
        g = box_grid(rect[0], d)
        if g is not None:
            return g.astype(np.uint8)
    if partial:
        return net_mask(prob, True, 0, s_p, 0, k_p)
    return net_mask(prob, False, 0, s_f, 0, k_f)


# ---------- score tables on the odd half ----------
acc_tr = json.loads((Path(a.glob_dir) / "global_train.json").read_text())
acc_te = json.loads((Path(a.glob_dir) / "global_test.json").read_text())
rect_tr, rect_te = load_rect("train"), load_rect("test")
z = np.load(DATA / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
meta_tr = json.loads((DATA / "train" / "metadata.json").read_text())
meta_te = json.loads((DATA / "test" / "metadata.json").read_text())

params, rows, val_out = {}, [], {}
for m in METHODS:
    NET = list(itertools.product(S_GRID, (0, 1) if m in K_METHODS else (0,)))
    probs = load_probs("train", m)
    items = [e for e in meta_tr if e["method"] == m and e["image_name"] in probs]
    assert all(int(e["image_name"], 16) % 2 == 1 for e in items), "validation must be the odd half only"
    use_rect = any(e["method"] == m and e["image_name"] in rect_te for e in meta_te)
    names = [e["image_name"] for e in items]; n = len(names)
    part = np.array([acc_tr[x] < T_FULL for x in names])
    # A[i, j]: net config j applied with the image's own regime;  B[i, d]: rectangle box score (nan if none / t not met)
    A = np.array([[miou_image(net_mask(probs[x], part[i], 0, s, 0, k), gt[x]) for s, k in NET] for i, x in enumerate(names)])
    B = np.full((n, len(D_GRID)), np.nan); rs = np.full(n, -1.0)
    for i, x in enumerate(names):
        if part[i] and use_rect and x in rect_tr:
            rs[i] = rect_tr[x][1]
            for j, d in enumerate(D_GRID):
                g = box_grid(rect_tr[x][0], d)
                if g is not None:
                    B[i, j] = miou_image(g, gt[x])
    # combine2's rule for reference: one threshold for both regimes, no K
    one = [j for j, (s, k) in enumerate(NET) if k == 0]

    def best_cfg(idx, single=False):
        idx = np.asarray(idx); fi = idx[~part[idx]]; pi = idx[part[idx]]
        cand = []
        for t in (T_GRID if use_rect else [9.0]):
            for dj, d in enumerate(D_GRID if t < 9 else (0,)):
                use = (rs[pi] >= t) & ~np.isnan(B[pi, dj]) if t < 9 else np.zeros(len(pi), bool)
                Pm = np.where(use[:, None], np.nan_to_num(B[pi, dj])[:, None], A[pi])       # partial images x net cfg
                if single:
                    tot = A[fi][:, one].sum(0) + Pm[:, one].sum(0); j = int(np.argmax(tot))
                    cand.append((tot[j], NET[one[j]], NET[one[j]], t, d))
                else:
                    jf = int(np.argmax(A[fi].sum(0))) if len(fi) else 0; jp = int(np.argmax(Pm.sum(0)))
                    cand.append((A[fi, jf].sum() + Pm[:, jp].sum(), NET[jf], NET[jp], t, d))
        sc, nf, npart, t, d = max(cand, key=lambda c: c[0])
        return (nf[0], nf[1], npart[0], npart[1], t, d)

    def score_cfg(idx, cfg):
        return [miou_image(predict(probs[names[i]], part[i], rect_tr.get(names[i]) if use_rect else None, cfg), gt[names[i]]) for i in idx]

    cv_new, cv_one = [], []
    for sd in range(a.seeds):
        perm = np.random.default_rng(sd).permutation(n); halves = [perm[: n // 2], perm[n // 2:]]
        new, old = [], []
        for h in (0, 1):
            new += score_cfg(halves[1 - h], best_cfg(halves[h]))
            old += score_cfg(halves[1 - h], best_cfg(halves[h], single=True))
        cv_new.append(np.mean(new)); cv_one.append(np.mean(old))
    cfg = best_cfg(np.arange(n)); params[m] = (cfg, use_rect)
    val_out[m] = (names, [predict(probs[x], part[i], rect_tr.get(x) if use_rect else None, cfg) for i, x in enumerate(names)])
    rows.append((m, np.mean(cv_one), np.std(cv_one), np.mean(cv_new), np.std(cv_new), cfg, use_rect))

print("method      | combine2 rule (CV) | regime-specific (CV) | full: thr, K | partial: thr, K | box: t, d")
for m, o, os_, nw, ns, cfg, ur in rows:
    s_f, k_f, s_p, k_p, t, d = cfg
    print(f"{m:11s} | {o:.4f} ±{os_:.4f}     | {nw:.4f} ±{ns:.4f}       | {s_f:.2f}, {k_f}      | {s_p:.2f}, {k_p}         | "
          f"{'off' if t >= 9 else f'{t:.2f}, {d:+d}'}{'' if ur else ' (no test boxes)'}")
print(f"overall     | {np.mean([r[1] for r in rows]):.4f}            | {np.mean([r[3] for r in rows]):.4f}")

out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
vn = sum((val_out[m][0] for m in METHODS), []); vp = sum((val_out[m][1] for m in METHODS), [])
np.savez_compressed(out / "val_probs.npz", names=np.array(vn), probs=np.stack(vp).astype(np.float16))
(out / "params.json").write_text(json.dumps({m: {"s_full": params[m][0][0], "K_full": params[m][0][1], "s_part": params[m][0][2],
                                                 "K_part": params[m][0][3], "box_t": params[m][0][4], "box_d": params[m][0][5],
                                                 "rect": params[m][1], "runs": [str(p) for p in RUN[m]]} for m in METHODS}, indent=1))

if a.test:
    import torch
    probs_te = {}
    for m in METHODS:
        probs_te.update(load_probs("test", m))
    names, masks = [], []
    for e in meta_te:
        cfg, ur = params[e["method"]]; x = e["image_name"]
        p = predict(probs_te[x], acc_te[x] < T_FULL, rect_te.get(x) if ur else None, cfg)
        assert p.shape == (128, 128) and 0 < p.sum() < p.size
        names.append(x); masks.append(p)
    torch.save({"image_names": names, "masks": torch.from_numpy(np.stack(masks)).bool()}, a.test)
    np.savez_compressed(out / "test_probs.npz", names=np.array(names), probs=np.stack(masks).astype(np.float16))
    print(f"\nsubmission written (NOT submitted): {a.test} ({len(names)} images, no empty or full grid)")
