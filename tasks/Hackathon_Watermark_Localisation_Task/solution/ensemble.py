"""Ensembling bench for the watermark localisation task (CPU only, numpy + scipy; torch only to write the submission).

Exchange format, one file per (pipeline, method, split) in --probs DIR:
    <pipeline>_<method>_<split>.npz      method in pixelseal|trustmark|stegastamp, split in train|test
        names             [N] str              image names
        probs             [N, 128, 128] f16    probability that the grid cell is watermarked
        used_in_training  [N] bool             train split only: image seen by the pipeline (as sample, cover or for tuning)
                                               (missing -> the team convention is assumed: even names were used)

Clean validation set of a method = official train images present in every assembled pipeline, used in training by
none of them and (default, --val-rule odd) in the odd half (int(name, 16) % 2 == 1).

Per method, on the clean validation set (official per-image mIoU, wmloc.utils.miou_image):
  candidates  single:<p> | avg (weighted mean of probabilities) | logit (weighted mean of logits)
  families    single:<p>, avg, logit   one candidate for all images
              regime                   one single pipeline for partial images, one for full images
              any                      any candidate per regime
  post-proc   threshold per regime (partial / full, from the global bit agreement < 0.80 in global_{train,test}.json),
              optional K largest components of the minority class and hole filling (--pp thr,K,fill),
              rectangle override on partial images when the rectangle-search score >= t (box grown by d px),
              never an empty nor a full grid (fix_grid).
Every choice (candidate, weights, thresholds, rectangle rule) is made inside the training folds of a repeated k-fold
cross-validation and scored on the held-out fold; the table gives the mean, the std over repeats, a bootstrap 95 % CI
and the paired gain over the best single pipeline. The final parameters are refitted on the whole clean set.

Usage:
    python solution/ensemble.py eval --probs output/probs --pipelines amil,seg2 [--test out.pt] [--report out.json]
    python solution/ensemble.py import-amil --run pixelseal=DIR1+DIR2,trustmark=DIR,stegastamp=DIR --out output/probs
    python solution/ensemble.py make-demo --out /tmp/demo_probs      (pseudo-probabilities from local baseline files)
Nothing is ever submitted by this script."""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("pixelseal", "trustmark", "stegastamp")
T_FULL = 0.80
S_GRID = tuple(float(s) for s in np.round(np.arange(0.05, 0.96, 0.05), 2))
T_GRID = tuple(float(t) for t in np.round(np.arange(0.30, 0.96, 0.05), 2))
D_GRID = (-4, 0, 2, 4)
RECT_OPTS = [(9.0, 0)] + [(t, d) for t in T_GRID for d in D_GRID]     # 9.0 = rectangle override off
STRUCT4 = ndimage.generate_binary_structure(2, 1)


def miou_image(pred, gt):
    """Official per-image mIoU (imported from the starter kit, never re-implemented here)."""
    global _MIOU
    if _MIOU is None:
        sys.path.insert(0, str(ROOT))
        from wmloc.utils import miou_image as f
        _MIOU = f
    return _MIOU(pred, gt)


_MIOU = None


# ---------------------------------------------------------------- post-processing primitives (as modal_t1/tracks/postproc/pp.py)
def fix_grid(mask, prob, k_frac=0.01):
    """Never an empty nor a full grid: flip the 1 % most (least) likely cells."""
    p = np.asarray(mask).astype(np.uint8).copy(); k = max(1, int(k_frac * p.size))
    if p.sum() == 0:
        p.flat[np.argsort(prob.ravel())[-k:]] = 1
    elif p.sum() == p.size:
        p.flat[np.argsort(prob.ravel())[:k]] = 0
    return p


def keep_largest(minor, K):
    """Keep the K largest 4-connected components of the minority class (K = 0: all)."""
    if not K:
        return minor
    lab, n = ndimage.label(minor, STRUCT4)
    if n <= K:
        return minor
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    keep = np.zeros(n + 1, bool); keep[1 + np.argsort(-sizes, kind="stable")[:K]] = True
    return keep[lab]


def net_mask(prob, partial, s, K=0, fill=0):
    """Threshold s; the minority class is the marked zone in a partial image and the clean hole in a full image."""
    marked = prob >= s
    minor = marked if partial else ~marked
    minor = keep_largest(minor, K)
    if fill:
        minor = ndimage.binary_fill_holes(minor)
    return fix_grid(minor if partial else ~minor, prob)


def box_grid(box, d=0):
    """Pixel box (x0, y0, x1, y1) grown by d px -> 128 grid (cell = 1 if >= 50 % covered); None if empty or full."""
    m = np.zeros((512, 512), np.float32)
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    m[max(0, y0 - d):min(512, y1 + d), max(0, x0 - d):min(512, x1 + d)] = 1
    g = (m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5).astype(np.uint8)
    return g if 0 < g.sum() < g.size else None


# ---------------------------------------------------------------- loaders
def load_exchange(probs_dir, pipeline, method, split):
    f = Path(probs_dir) / f"{pipeline}_{method}_{split}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    names = [str(n) for n in z["names"].tolist()]; probs = z["probs"]
    assert probs.shape == (len(names), 128, 128), f"{f}: probs {probs.shape} for {len(names)} names"
    assert len(set(names)) == len(names), f"{f}: duplicated names"
    pmin, pmax = float(probs.min()), float(probs.max())
    assert np.isfinite(probs).all() and pmin >= 0 and pmax <= 1, f"{f}: probs must be finite and in [0, 1] (got {pmin}, {pmax})"
    out = dict(names=names, probs=probs, file=str(f))
    if split == "train":
        if "used_in_training" in z.files:
            out["used"] = z["used_in_training"].astype(bool)
        else:
            print(f"   WARNING {f.name}: no used_in_training field, assuming the even names were used", flush=True)
            out["used"] = np.array([int(n, 16) % 2 == 0 for n in names])
        assert out["used"].shape == (len(names),), f"{f}: used_in_training shape"
    return out


def load_rect(cache_dir, split, tags):
    """Best box per image over the caches search_<split>_<method><tag>[_shardK].npz with an exact tag."""
    best, have = {}, set()
    for m in METHODS:
        for tag in tags:
            pat = re.compile(rf"search_{split}_{m}{re.escape(tag)}(_shard\d+)?\.npz$")
            for p in sorted(Path(cache_dir).rglob(f"search_{split}_{m}*.npz")):
                if not pat.fullmatch(p.name):
                    continue
                z = np.load(p, allow_pickle=True)
                for n, b, s in zip(z["names"].tolist(), z["box"], z["score"]):
                    if b is None or np.asarray(b).dtype == object:
                        continue
                    have.add(m)
                    if n not in best or float(s) > best[n][1]:
                        best[n] = (np.asarray(b, np.float32), float(s))
    return best, have


def mask_kinds(data, meta):
    """full / rect / irreg per train image (rectangle test on the 512 px mask when available: 128 corners can round)."""
    z = np.load(Path(data) / "train" / "masks_128.npz"); g128 = dict(zip(z["names"].tolist(), z["masks"]))
    out = {}
    for e in meta:
        n = e["image_name"]
        if e["full_watermark"]:
            out[n] = "full"; continue
        g = g128[n] > 0; p = Path(data) / "train" / "masks" / f"{n}.png"
        if p.exists():
            from PIL import Image
            g = np.asarray(Image.open(p).convert("L")) > 127
        ys, xs = np.nonzero(g)
        out[n] = "rect" if len(ys) and g[ys.min():ys.max() + 1, xs.min():xs.max() + 1].all() else "irreg"
    return out


# ---------------------------------------------------------------- candidates
def weight_grid(P):
    """Weight vectors on the simplex (no one-hot: those are the single candidates)."""
    G = {2: 10, 3: 4}.get(P, 3)
    return [tuple(x / G for x in c) for c in itertools.product(range(G + 1), repeat=P) if sum(c) == G and max(c) < G]


def candidates(pipes):
    c = [("single", p, tuple(float(q == p) for q in pipes)) for p in pipes]
    if len(pipes) > 1:
        for kind in ("avg", "logit"):
            c += [(kind, kind + ":" + "/".join(f"{x:.2f}" for x in w), w) for w in weight_grid(len(pipes))]
    return c


def mix(cand, stack):
    """stack: [P, ..., 128, 128] float32 -> mixed probability."""
    kind, _, w = cand
    w = np.asarray(w, np.float32).reshape((-1,) + (1,) * (stack.ndim - 1))
    if kind == "logit":
        p = np.clip(stack, 1e-4, 1 - 1e-4); lg = (w * np.log(p / (1 - p))).sum(0)
        return 1 / (1 + np.exp(-lg))
    return (w * stack).sum(0)


def families(cands, pipes):
    fam = {f"single:{p}": ([i for i, c in enumerate(cands) if c[0] == "single" and c[1] == p], True) for p in pipes}
    if len(pipes) > 1:
        fam["avg"] = ([i for i, c in enumerate(cands) if c[0] == "avg"], True)
        fam["logit"] = ([i for i, c in enumerate(cands) if c[0] == "logit"], True)
        fam["regime"] = ([i for i, c in enumerate(cands) if c[0] == "single"], False)
        fam["any"] = (list(range(len(cands))), False)
    return fam


# ---------------------------------------------------------------- score tables, selection, cross-validation
def build_tables(cands, stack, part, gts, rects, net_cfgs, use_rect):
    """S[c, i, j]: mIoU of candidate c with net config j on image i (own regime). B[i, d]: mIoU of the box grown by
    D_GRID[d] (nan if no usable box). rs[i]: rectangle score (-1 if none)."""
    n = stack.shape[1]
    S = np.zeros((len(cands), n, len(net_cfgs)), np.float32)
    for ci, c in enumerate(cands):
        pr = mix(c, stack)
        for i in range(n):
            for j, (s, K, fill) in enumerate(net_cfgs):
                S[ci, i, j] = miou_image(net_mask(pr[i], part[i], s, K, fill), gts[i])
    B = np.full((n, len(D_GRID)), np.nan, np.float32); rs = np.full(n, -1.0)
    for i in range(n):
        if use_rect and part[i] and rects[i] is not None:
            rs[i] = rects[i][1]
            for j, d in enumerate(D_GRID):
                g = box_grid(rects[i][0], d)
                if g is not None:
                    B[i, j] = miou_image(g, gts[i])
    return S, B, rs


def select(idx, S, B, rs, part, allowed, tied, use_rect):
    """Best config on the images idx -> dict(cf, jf, cp, jp, t, d). tied: same candidate for both regimes."""
    idx = np.asarray(idx); fi = idx[~part[idx]]; pi = idx[part[idx]]
    A = S[allowed]
    F = A[:, fi].sum(1)                                           # [C, J] full images
    best = None
    for t, d in (RECT_OPTS if use_rect else RECT_OPTS[:1]):
        if t >= 9:
            Pm = A[:, pi].sum(1)
        else:
            dj = D_GRID.index(d); use = (rs[pi] >= t) & ~np.isnan(B[pi, dj])
            Pm = A[:, pi[~use]].sum(1) + float(B[pi[use], dj].sum())
        if tied:
            tot = F.max(1) + Pm.max(1); c = int(np.argmax(tot))
            cur = (float(tot[c]), c, int(np.argmax(F[c])), c, int(np.argmax(Pm[c])), t, d)
        else:
            cf, jf = np.unravel_index(int(np.argmax(F)), F.shape); cp, jp = np.unravel_index(int(np.argmax(Pm)), Pm.shape)
            cur = (float(F[cf, jf] + Pm[cp, jp]), int(cf), int(jf), int(cp), int(jp), t, d)
        if best is None or cur[0] > best[0] + 1e-9:
            best = cur
    _, cf, jf, cp, jp, t, d = best
    return dict(cf=allowed[cf], jf=jf, cp=allowed[cp], jp=jp, t=t, d=d)


def table_scores(idx, cfg, S, B, rs, part):
    out = np.empty(len(idx), np.float64)
    for k, i in enumerate(idx):
        if part[i]:
            dj = D_GRID.index(cfg["d"])
            out[k] = B[i, dj] if (cfg["t"] < 9 and rs[i] >= cfg["t"] and not np.isnan(B[i, dj])) else S[cfg["cp"], i, cfg["jp"]]
        else:
            out[k] = S[cfg["cf"], i, cfg["jf"]]
    return out


def cross_validate(S, B, rs, part, fams, use_rect, folds=5, repeats=10, seed=0):
    """Repeated k-fold, stratified by regime. Returns oof[family] = [repeats, n] held-out scores."""
    n = S.shape[1]; oof = {f: np.zeros((repeats, n)) for f in fams}
    for r in range(repeats):
        rng = np.random.default_rng(seed + r); fold = np.zeros(n, int)
        for grp in (np.nonzero(part)[0], np.nonzero(~part)[0]):
            fold[rng.permutation(grp)] = np.arange(len(grp)) % folds
        for k in range(folds):
            te = np.nonzero(fold == k)[0]; tr = np.nonzero(fold != k)[0]
            if not len(te):
                continue
            for f, (allowed, tied) in fams.items():
                oof[f][r, te] = table_scores(te, select(tr, S, B, rs, part, allowed, tied, use_rect), S, B, rs, part)
    return oof


def bootstrap_ci(v, n_boot=2000, seed=0):
    v = np.asarray(v, np.float64); rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), float((b > 0).mean())


def predict_one(probs_by_pipe, pipes, cands, net_cfgs, cfg, partial, rect):
    """Final mask of one image. probs_by_pipe: {pipeline: [128, 128]}; rect: (box, score) or None."""
    if partial and rect is not None and cfg["t"] < 9 and rect[1] >= cfg["t"]:
        g = box_grid(rect[0], cfg["d"])
        if g is not None:
            return g
    c = cands[cfg["cp"] if partial else cfg["cf"]]; s, K, fill = net_cfgs[cfg["jp"] if partial else cfg["jf"]]
    need = [p for p, w in zip(pipes, c[2]) if w > 0]
    missing = [p for p in need if p not in probs_by_pipe]
    if missing:
        raise KeyError(f"no probabilities from {missing}")
    stack = np.stack([probs_by_pipe[p].astype(np.float32) if p in need else np.zeros((128, 128), np.float32) for p in pipes])
    return net_mask(mix(c, stack), partial, s, K, fill)


# ---------------------------------------------------------------- submission
def check_submission(path, ref_names):
    """Exact format of solution/submission_rect.pt: keys, order of the names, bool [N, 128, 128]. Returns stats."""
    import torch
    d = torch.load(path, weights_only=True)
    assert list(d.keys()) == ["image_names", "masks"], f"keys {list(d.keys())}"
    assert isinstance(d["image_names"], list) and all(type(n) is str for n in d["image_names"]), "image_names must be list[str]"
    assert d["image_names"] == list(ref_names), "image names / order differ from the reference"
    m = d["masks"]
    assert isinstance(m, torch.Tensor) and m.dtype == torch.bool and tuple(m.shape) == (len(ref_names), 128, 128), (m.dtype, tuple(m.shape))
    cells = m.flatten(1).sum(1)
    return dict(n=len(ref_names), empty=int((cells == 0).sum()), full=int((cells == 128 * 128).sum()), mean=float(m.float().mean()))


def write_submission(path, masks_by_name, ref_names):
    import torch
    missing = [n for n in ref_names if n not in masks_by_name]
    assert not missing, f"{len(missing)} test images without a mask (e.g. {missing[:3]})"
    arr = np.stack([np.asarray(masks_by_name[n]) for n in ref_names])
    assert arr.shape == (len(ref_names), 128, 128) and set(np.unique(arr).tolist()) <= {0, 1}, arr.shape
    torch.save({"image_names": list(ref_names), "masks": torch.from_numpy(arr.astype(bool))}, path)
    return check_submission(path, ref_names)


# ---------------------------------------------------------------- commands
def cmd_eval(a):
    data = Path(a.data); pipes = [p for p in a.pipelines.split(",") if p]
    meta = json.loads((data / "train" / "metadata.json").read_text()); method_of = {e["image_name"]: e["method"] for e in meta}
    z = np.load(data / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
    acc_tr = json.loads((Path(a.glob_dir) / "global_train.json").read_text())
    kinds = mask_kinds(data, meta)
    rect_tr, have_tr = load_rect(a.rect_dir, "train", a.rect_train_tags.split(","))
    rect_te, have_te = load_rect(a.rect_dir, "test", a.rect_test_tags.split(","))
    pp = set(a.pp.split(",")); net_cfgs = [(s, K, f) for s in S_GRID for K in ((0, 1) if "K" in pp else (0,)) for f in ((0, 1) if "fill" in pp else (0,))]
    forced = dict(kv.split("=", 1) for kv in a.choose.split(",")) if a.choose else {}
    report, final = {}, {}
    for m in METHODS:
        tr = {p: load_exchange(a.probs, p, m, "train") for p in pipes}; mp = [p for p in pipes if tr[p] is not None]
        if not mp:
            print(f"\n== {m}: no input, skipped (fallback masks in the submission)"); continue
        used = {}
        for p in mp:
            for n, u in zip(tr[p]["names"], tr[p]["used"]):
                assert method_of.get(n) == m, f"{tr[p]['file']}: {n} is not a {m} train image"
                used.setdefault(n, []).append(bool(u))
        clean = sorted(n for n, u in used.items() if len(u) == len(mp) and not any(u) and (a.val_rule != "odd" or int(n, 16) % 2 == 1))
        n_m = sum(e["method"] == m for e in meta)
        cov = ", ".join("%s: %d unused of %d" % (p, int((~tr[p]["used"]).sum()), len(tr[p]["names"])) for p in mp)
        print(f"\n== {m}: pipelines {mp} | clean validation {len(clean)}/{n_m} images ({cov})")
        if len(clean) < 2 * a.folds:
            print("   too few clean images, skipped"); continue
        idx = {p: {n: i for i, n in enumerate(tr[p]["names"])} for p in mp}
        stack = np.stack([np.stack([tr[p]["probs"][idx[p][n]] for n in clean]).astype(np.float32) for p in mp])
        part = np.array([acc_tr[n] < T_FULL for n in clean]); gts = [gt[n] for n in clean]
        use_rect = (not a.no_rect) and m in have_tr and m in have_te
        cands = candidates(mp); fams = families(cands, mp)
        S, B, rs = build_tables(cands, stack, part, gts, [rect_tr.get(n) for n in clean], net_cfgs, use_rect)
        oof = cross_validate(S, B, rs, part, fams, use_rect, a.folds, a.repeats, a.seed)
        singles = [f for f in fams if f.startswith("single:")]
        ref = max(singles, key=lambda f: oof[f].mean())
        kd = np.array([kinds[n] for n in clean]); rows = {}
        print(f"   regime: {int(part.sum())} partial / {int((~part).sum())} full | rectangle override {'on' if use_rect else 'off'} | "
              f"{len(cands)} candidates x {len(net_cfgs)} net configs | {a.repeats} x {a.folds}-fold CV")
        print(f"   {'family':16s} {'CV mIoU':>8s} {'±rep':>6s} {'95% CI':>15s} | {'gain vs ' + ref:>22s} {'95% CI':>17s} {'P>0':>5s} | rect   irreg  full   | in-sample")
        for f, (allowed, tied) in fams.items():
            per = oof[f].mean(0); lo, hi, _ = bootstrap_ci(per, seed=a.seed)
            diff = per - oof[ref].mean(0); dlo, dhi, pg = bootstrap_ci(diff, seed=a.seed)
            cfg = select(np.arange(len(clean)), S, B, rs, part, allowed, tied, use_rect)
            ins = float(table_scores(np.arange(len(clean)), cfg, S, B, rs, part).mean())
            by = {k: float(per[kd == k].mean()) if (kd == k).any() else float("nan") for k in ("rect", "irreg", "full")}
            rows[f] = dict(cv=float(per.mean()), std=float(oof[f].mean(1).std()), ci=[lo, hi], gain=float(diff.mean()), gain_ci=[dlo, dhi],
                           p_gain=pg, by_kind=by, in_sample=ins, cfg=cfg)
            print(f"   {f:16s} {per.mean():8.4f} {oof[f].mean(1).std():6.4f} [{lo:.3f}, {hi:.3f}] | {diff.mean():+22.4f} [{dlo:+.3f}, {dhi:+.3f}] {pg:5.2f} | "
                  f"{by['rect']:.3f}  {by['irreg']:.3f}  {by['full']:.3f}  | {ins:.4f}")
        ok = [f for f in fams if f not in singles and rows[f]["gain"] >= a.min_gain and rows[f]["p_gain"] >= a.min_p]
        pick = forced.get(m) or (max(ok, key=lambda f: rows[f]["cv"]) if ok else ref)
        assert pick in fams, f"--choose {m}={pick}: unknown family (have {list(fams)})"
        cfg = rows[pick]["cfg"]

        def lab(c, j):
            s, K, f = net_cfgs[j]
            return f"{cands[c][1]} thr {s:.2f}" + (f" K{K}" if K else "") + (" fill" if f else "")
        print(f"   chosen: {pick} (CV {rows[pick]['cv']:.4f}; best single {ref} {rows[ref]['cv']:.4f}) | full: {lab(cfg['cf'], cfg['jf'])} | "
              f"partial: {lab(cfg['cp'], cfg['jp'])} | box: {'off' if cfg['t'] >= 9 else 't %.2f d %+d' % (cfg['t'], cfg['d'])}")
        report[m] = dict(pipelines=mp, n_clean=len(clean), n_partial=int(part.sum()), best_single=ref, chosen=pick, families=rows,
                         chosen_desc=dict(full=lab(cfg["cf"], cfg["jf"]), partial=lab(cfg["cp"], cfg["jp"]), box_t=cfg["t"], box_d=cfg["d"]))
        final[m] = dict(pipes=mp, cands=cands, cfg=cfg, use_rect=use_rect)
    if report:
        done = [m for m in METHODS if m in report]
        print(f"\noverall (mean over {len(done)} method(s) {done}): best single {np.mean([report[m]['families'][report[m]['best_single']]['cv'] for m in done]):.4f}"
              f" | chosen {np.mean([report[m]['families'][report[m]['chosen']]['cv'] for m in done]):.4f}   (cross-validated, clean validation set)")
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1)); print("report:", a.report)
    if a.test:
        import torch
        ref_names = torch.load(a.ref, weights_only=True)["image_names"]
        tmeta = json.loads((data / "test" / "metadata.json").read_text())
        assert [e["image_name"] for e in tmeta] == ref_names, "reference submission and test metadata disagree on the order"
        acc_te = json.loads((Path(a.glob_dir) / "global_test.json").read_text())
        fb = dict(zip(ref_names, torch.load(a.fallback, weights_only=True)["masks"].numpy().astype(np.uint8))) if a.fallback else {}
        te = {(p, m): load_exchange(a.probs, p, m, "test") for m in final for p in final[m]["pipes"]}
        te = {k: dict(zip(v["names"], v["probs"])) for k, v in te.items() if v is not None}
        masks, n_fb, n_box = {}, {m: 0 for m in METHODS}, {m: 0 for m in METHODS}
        for e in tmeta:
            n, m = e["image_name"], e["method"]
            try:
                if m not in final:
                    raise KeyError("method not assembled")
                F = final[m]; rect = rect_te.get(n) if F["use_rect"] else None; partial = acc_te[n] < T_FULL
                pm = predict_one({p: te[(p, m)][n] for p in F["pipes"] if (p, m) in te and n in te[(p, m)]}, F["pipes"], F["cands"], net_cfgs, F["cfg"], partial, rect)
                assert 0 < pm.sum() < pm.size
                n_box[m] += int(partial and rect is not None and F["cfg"]["t"] < 9 and rect[1] >= F["cfg"]["t"])
            except KeyError as err:
                assert n in fb, f"{n} ({m}): {err} and no --fallback submission"
                pm = fb[n]; n_fb[m] += 1
            masks[n] = pm
        st = write_submission(a.test, masks, ref_names)
        print(f"\nsubmission written, NOT submitted: {a.test} | {st['n']} images, bool [N,128,128], same keys and order as {Path(a.ref).name} | "
              f"empty grids {st['empty']}, full grids {st['full']}, mean mask {st['mean']:.4f} | boxes {n_box} | fallback masks {n_fb}")
    return report


def cmd_import_amil(a):
    """Modal run folders ({train|val}_probs.npz + test_probs.npz, several folders joined by '+' are averaged)."""
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for kv in a.run.split(","):
        m, dirs = kv.split("=", 1); assert m in METHODS, m
        for split in ("train", "test"):
            acc = None
            for d in dirs.split("+"):
                f = Path(d) / f"{split}_probs.npz"
                if split == "train" and not f.exists():
                    f = Path(d) / "val_probs.npz"
                z = np.load(f); cur = dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))
                acc = cur if acc is None else {n: acc[n] + cur[n] for n in acc if n in cur}
            names = sorted(acc); probs = np.stack([acc[n] for n in names]) / len(dirs.split("+"))
            obj = dict(names=np.array(names), probs=probs.astype(np.float16))
            if split == "train":                                # Modal convention: the even half of the official train was used
                obj["used_in_training"] = np.array([int(n, 16) % 2 == 0 for n in names])
            np.savez_compressed(out / f"{a.pipeline}_{m}_{split}.npz", **obj)
            print(f"wrote {out / f'{a.pipeline}_{m}_{split}.npz'}: {len(names)} images" + (f", {int((~obj['used_in_training']).sum())} unused in training" if split == "train" else ""))


def pseudo_prob(mask, sigma=1.5):
    """Binary mask -> soft pseudo-probability (0.5 stays on the mask border)."""
    return np.clip(ndimage.gaussian_filter(np.asarray(mask, np.float32), sigma), 0.02, 0.98).astype(np.float16)


def cmd_make_demo(a):
    """Stand-in inputs from local files only: 'base' = baseline (mode + per-position logits), 'rect' = baseline +
    rectangle search. Train maps are recomputed from solution/cache, test maps come from the two written submissions.
    Their parameters were tuned on the train fold of common.split_train, so used_in_training = not in that fold."""
    import torch
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "solution"))
    from baseline import predict
    from common import split_train
    C = ROOT / "solution" / "cache"; data = Path(a.data); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    params = json.load(open(C / "baseline_params.json")); val = split_train(data)
    feats = np.load(C / "feat_train.npz"); fi = {n: i for i, n in enumerate(feats["names"].tolist())}
    meta = json.loads((data / "train" / "metadata.json").read_text()); tmeta = json.loads((data / "test" / "metadata.json").read_text())
    rect_tr, _ = load_rect(C, "train", ["", "_r", "_r2"]); RECT = {"pixelseal": (0.40, 3), "trustmark": (0.60, 0)}
    subs = {"base": torch.load(ROOT / "solution" / "submission_baseline.pt", weights_only=True), "rect": torch.load(ROOT / "solution" / "submission_rect.pt", weights_only=True)}
    for m in METHODS:
        names = [e["image_name"] for e in meta if e["method"] == m]
        base = [predict(feats["agree"][fi[n]], feats["ppl"][fi[n]].astype(np.float32), *params[m]) for n in names]
        rect = []
        for n, b in zip(names, base):
            g = None
            if m in RECT and feats["agree"][fi[n]] <= params[m][0] and n in rect_tr and rect_tr[n][1] > RECT[m][0]:
                g = box_grid(rect_tr[n][0], RECT[m][1])
            rect.append(b if g is None else g)
        used = np.array([n not in val for n in names])
        tn = [e["image_name"] for e in tmeta if e["method"] == m]
        for p, tr_masks in (("base", base), ("rect", rect)):
            np.savez_compressed(out / f"{p}_{m}_train.npz", names=np.array(names), probs=np.stack([pseudo_prob(x) for x in tr_masks]), used_in_training=used)
            pos = {n: i for i, n in enumerate(subs[p]["image_names"])}
            np.savez_compressed(out / f"{p}_{m}_test.npz", names=np.array(tn), probs=np.stack([pseudo_prob(subs[p]["masks"][pos[n]].numpy()) for n in tn]))
        print(f"{m}: {len(names)} train ({int((~used).sum())} unused in training), {len(tn)} test -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval"); e.set_defaults(fn=cmd_eval)
    e.add_argument("--probs", required=True); e.add_argument("--pipelines", required=True, help="comma list, e.g. amil,seg2")
    e.add_argument("--data", default=str(ROOT / "data")); e.add_argument("--glob-dir", default=str(ROOT / "modal_t1" / "cache"))
    e.add_argument("--rect-dir", default=str(ROOT / "solution" / "cache")); e.add_argument("--rect-train-tags", default=",_r,_r2")
    e.add_argument("--rect-test-tags", default="_t,_t_r,_t_r2,_t_r3"); e.add_argument("--no-rect", action="store_true")
    e.add_argument("--val-rule", default="odd", choices=("odd", "any"), help="odd: clean set also restricted to the odd half")
    e.add_argument("--pp", default="thr", help="thr | thr,K | thr,K,fill")
    e.add_argument("--folds", type=int, default=5); e.add_argument("--repeats", type=int, default=10); e.add_argument("--seed", type=int, default=0)
    e.add_argument("--min-gain", type=float, default=0.005, help="an ensemble family replaces the best single pipeline only above this CV gain")
    e.add_argument("--min-p", type=float, default=0.80, help="... and this bootstrap probability that the gain is positive")
    e.add_argument("--choose", default="", help="force a family per method, e.g. pixelseal=avg,trustmark=single:amil")
    e.add_argument("--test", default="", help="write a submission here (never submitted)"); e.add_argument("--report", default="")
    e.add_argument("--ref", default=str(ROOT / "solution" / "submission_rect.pt"), help="reference for keys and image order")
    e.add_argument("--fallback", default="", help="submission whose masks fill test images without input (e.g. solution/submission_rect.pt)")
    i = sub.add_parser("import-amil"); i.set_defaults(fn=cmd_import_amil)
    i.add_argument("--run", required=True, help="method=dir[+dir2],..."); i.add_argument("--out", required=True); i.add_argument("--pipeline", default="amil")
    d = sub.add_parser("make-demo"); d.set_defaults(fn=cmd_make_demo)
    d.add_argument("--out", required=True); d.add_argument("--data", default=str(ROOT / "data"))
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    main()
