"""Combine the baseline (global mode decision + per_position_logits thresholds, solution/baseline.py) with the
rectangle search (solution/run_search.py): in 'mostly clean' mode, if the best box agreement > t_rect, predict the
box (grown by d px) instead of the baseline map. t_rect and d tuned per method on the train fold, reported on the
held-out fold (official per-image mIoU). With --test, writes the submission (same format and image order as
submission_baseline.pt: {"image_names": list[str], "masks": bool tensor [N, 128, 128]}).

Search caches are selected by exact tag: search_{split}_{method}{tag}[_shardK].npz. Several tags can be merged
(per image, the box with the highest score wins), e.g. --train-tags ",_fine" --test-tags "_t,_fine_t".
Usage: python solution/rect_eval.py [--test out.pt] [--data data] [--train-tags ""] [--test-tags _t]"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "solution"))
from common import METHODS, split_train
from baseline import predict
from rectsearch import box_to_mask128
from wmloc.utils import miou_image

ap = argparse.ArgumentParser(); ap.add_argument("--test", default=""); ap.add_argument("--data", default=str(ROOT / "data"))
ap.add_argument("--train-tags", default="", help="comma list of cache tags for the train split ('' = coarse grid)")
ap.add_argument("--test-tags", default="_t", help="comma list of cache tags for the test split")
a = ap.parse_args()
C = ROOT / "solution" / "cache"
params = json.load(open(C / "baseline_params.json"))


def load_search(split, m, tags):
    """Only files whose tag is exactly one of `tags` (so '_fine' / '_dbg' caches never leak into the coarse run)."""
    out, files = {}, []
    for tag in tags.split(","):
        pat = re.compile(rf"search_{split}_{m}{re.escape(tag)}(_shard\d+)?\.npz$")
        for p in sorted(C.glob(f"search_{split}_{m}*.npz")):
            if not pat.match(p.name): continue
            files.append(p.name)
            z = np.load(p, allow_pickle=True)
            for n, b, s in zip(z["names"].tolist(), z["box"], z["score"]):
                if b is None or np.asarray(b).dtype == object: continue
                if n not in out or float(s) > out[n][1]: out[n] = (np.asarray(b, np.float32), float(s))
    return out, files


def pred(n, m, f_idx, feats, srch, tr, d):
    i = f_idx[n]; p = params[m]
    agree = feats["agree"][i]
    if agree <= p[0] and n in srch and srch[n][1] > tr:
        return box_to_mask128(srch[n][0] + np.array([-d, -d, d, d]))
    return predict(agree, feats["ppl"][i].astype(np.float32), *p)


feats = np.load(C / "feat_train.npz"); f_idx = {n: i for i, n in enumerate(feats["names"].tolist())}
meta = json.loads((Path(a.data) / "train" / "metadata.json").read_text())
z = np.load(Path(a.data) / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
val = split_train(a.data)


def is_rect(n):                                             # on the 512 px mask (128-grid corners can round to 0)
    from PIL import Image
    g = np.asarray(Image.open(Path(a.data) / "train" / "masks" / f"{n}.png").convert("L")) > 127
    ys, xs = np.nonzero(g)
    return len(ys) > 0 and bool(g[ys.min():ys.max() + 1, xs.min():xs.max() + 1].all())


kind = {e["image_name"]: ("full" if e["full_watermark"] else "rect" if is_rect(e["image_name"]) else "irregular") for e in meta}
chosen, rep, val_rows = {}, {}, []
TS = [round(float(t), 3) for t in np.arange(0.10, 0.96, 0.05)]; DS = (0, 1, 2, 3, 4, 6, 8)
for m in METHODS:
    srch, files = load_search("train", m, a.train_tags)
    names = [e["image_name"] for e in meta if e["method"] == m]
    tr_n = [n for n in names if n not in val]; va_n = [n for n in names if n in val]
    cache = {}

    def per_image(n, t, d):
        s = srch[n][1] if n in srch else -9.0
        key = (n, d if s > t else -1)                       # prediction only depends on (fires?, d)
        if key not in cache: cache[key] = miou_image(pred(n, m, f_idx, feats, srch, t, d), gt[n])
        return cache[key]

    ev = lambda ns, t, d: float(np.mean([per_image(n, t, d) for n in ns])) if ns else float("nan")
    best = max(((t, d) for t in TS for d in DS), key=lambda p: ev(tr_n, *p))
    chosen[m] = best
    rep[m] = dict(train=ev(tr_n, *best), train_baseline=ev(tr_n, 9, 0), val=ev(va_n, *best), val_baseline=ev(va_n, 9, 0))
    print(f"{m}: files {files} searched {len(srch)}/{len(names)}  t_rect={best[0]:.2f} d={best[1]}",
          {k: round(v, 4) for k, v in rep[m].items()})
    for k in ("rect", "irregular", "full"):
        vk = [n for n in va_n if kind[n] == k]; tk = [n for n in tr_n if kind[n] == k]
        fired = sum(1 for n in vk if n in srch and srch[n][1] > best[0] and feats["agree"][f_idx[n]] <= params[m][0])
        print(f"    {k:9s} val n={len(vk):3d} rect-fired={fired:3d} mIoU {ev(vk, *best):.4f} (baseline {ev(vk, 9, 0):.4f})"
              f" | train-fold n={len(tk):3d} mIoU {ev(tk, *best):.4f} (baseline {ev(tk, 9, 0):.4f})")
    val_rows += [(per_image(n, *best), per_image(n, 9, 0)) for n in va_n]
    # sensitivity of the held-out score to (t_rect, d): shows whether the choice is fragile
    print("    val grid (rows t, cols d=%s):" % (DS,))
    for t in TS[::3]:
        print("     t=%.2f " % t + " ".join("%.4f" % ev(va_n, t, d) for d in DS))
v = np.array(val_rows)
print("overall held-out (150 images): rect %.4f  baseline %.4f  | mean of per-method: rect %.4f baseline %.4f" % (
    v[:, 0].mean(), v[:, 1].mean(), np.mean([rep[m]["val"] for m in METHODS]), np.mean([rep[m]["val_baseline"] for m in METHODS])))
rng = np.random.default_rng(0); diff = v[:, 0] - v[:, 1]
bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
print("held-out gain %.4f, bootstrap 95%% CI [%.4f, %.4f]" % (diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))

if a.test:
    feats = np.load(C / "feat_test.npz"); f_idx = {n: i for i, n in enumerate(feats["names"].tolist())}
    tmeta = json.loads((Path(a.data) / "test" / "metadata.json").read_text())
    srch = {}
    for m in METHODS:
        srch[m], files = load_search("test", m, a.test_tags)
        n_m = sum(e["method"] == m for e in tmeta)
        n_clean = sum(e["method"] == m and feats["agree"][f_idx[e["image_name"]]] <= params[m][0] for e in tmeta)
        missing = sum(e["method"] == m and feats["agree"][f_idx[e["image_name"]]] <= params[m][0] and e["image_name"] not in srch[m] for e in tmeta)
        print(f"{m}: test files {files} searched {len(srch[m])}/{n_m}, clean-mode {n_clean}, clean-mode without search result {missing}")
        assert files, f"no test search cache for {m} with tags {a.test_tags!r}"
        assert missing == 0, f"{m}: {missing} clean-mode test images have no search result (job unfinished?)"
    names, masks, nrect = [], [], {m: 0 for m in METHODS}
    for e in tmeta:                                         # metadata order, as in submission_baseline.pt
        n, m = e["image_name"], e["method"]
        pm = pred(n, m, f_idx, feats, srch[m], *chosen[m])
        nrect[m] += int(n in srch[m] and srch[m][n][1] > chosen[m][0] and feats["agree"][f_idx[n]] <= params[m][0])
        names.append(n); masks.append(pm)
    masks = torch.from_numpy(np.stack(masks)).bool()
    assert len(names) == len(set(names)) == len(tmeta) and masks.shape == (len(tmeta), 128, 128)
    torch.save({"image_names": names, "masks": masks}, a.test)
    chk = torch.load(a.test, weights_only=True)
    print("wrote", a.test, tuple(chk["masks"].shape), chk["masks"].dtype, "rect predictions:", nrect,
          "mean mask %.4f" % float(chk["masks"].float().mean()))
