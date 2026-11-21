"""Combine the baseline (global mode decision + per_position_logits thresholds, solution/baseline.py) with the
rectangle search (solution/run_search.py): in 'mostly clean' mode, if the best box agreement > t_rect, predict the
box (grown by d px) instead of the baseline map. t_rect and d tuned per method on the train fold, reported on the
held-out fold (official per-image mIoU). With --test, writes the submission.
Usage: python solution/rect_eval.py [--test out.pt] [--data data]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "solution"))
from common import METHODS, split_train
from baseline import predict
from rectsearch import box_to_mask128
from wmloc.utils import score

ap = argparse.ArgumentParser(); ap.add_argument("--test", default=""); ap.add_argument("--data", default=str(ROOT / "data"))
a = ap.parse_args()
C = ROOT / "solution" / "cache"
params = json.load(open(C / "baseline_params.json"))


def load_search(split, m):
    out = {}
    for p in sorted(C.glob(f"search_{split}_{m}*.npz")):
        if "_dbg" in p.name: continue
        z = np.load(p)
        for n, b, s in zip(z["names"].tolist(), z["box"], z["score"]): out[n] = (b, float(s))
    return out


def pred(n, m, f_idx, feats, srch, tr, d):
    i = f_idx[n]; p = params[m]
    agree = feats["agree"][i]
    base = predict(agree, feats["ppl"][i].astype(np.float32), *p)
    if agree > p[0] or n not in srch: return base
    b, s = srch[n]
    if s > tr: return box_to_mask128(np.asarray(b) + np.array([-d, -d, d, d]))
    return base


feats = np.load(C / "feat_train.npz"); f_idx = {n: i for i, n in enumerate(feats["names"].tolist())}
meta = json.loads((Path(a.data) / "train" / "metadata.json").read_text())
z = np.load(Path(a.data) / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
val = split_train(a.data)
chosen, rep = {}, {}
for m in METHODS:
    srch = load_search("train", m)
    tr_n = [e["image_name"] for e in meta if e["method"] == m and e["image_name"] not in val]
    va_n = [e["image_name"] for e in meta if e["method"] == m and e["image_name"] in val]
    ev = lambda names, t, d: score([pred(n, m, f_idx, feats, srch, t, d) for n in names], [gt[n] for n in names])
    best = max(((t, d) for t in np.arange(0.1, 1.0, 0.05) for d in (0, 2, 4, 6)), key=lambda p: ev(tr_n, *p))
    chosen[m] = best
    rep[m] = dict(searched=len(srch), train=ev(tr_n, *best), val=ev(va_n, *best), val_baseline=ev(va_n, 9, 0),
                  all_train=ev(tr_n + va_n, *best), all_baseline=ev(tr_n + va_n, 9, 0))
    print(m, "t_rect=%.2f d=%d" % best, {k: round(v, 4) for k, v in rep[m].items()})
for k in ("val", "val_baseline", "all_train", "all_baseline"):
    print("overall", k, round(float(np.mean([rep[m][k] for m in METHODS])), 4))

if a.test:
    feats = np.load(C / "feat_test.npz"); f_idx = {n: i for i, n in enumerate(feats["names"].tolist())}
    tmeta = json.loads((Path(a.data) / "test" / "metadata.json").read_text())
    names, masks, nrect = [], [], 0
    for m in METHODS:
        srch = load_search("test", m)
        print(m, "test searched", len(srch))
        for e in tmeta:
            if e["method"] != m: continue
            n = e["image_name"]; pm = pred(n, m, f_idx, feats, srch, *chosen[m])
            nrect += int(n in srch and srch[n][1] > chosen[m][0] and feats["agree"][f_idx[n]] <= params[m][0])
            names.append(n); masks.append(torch.from_numpy(pm.astype(bool)))
    masks = torch.stack(masks)
    assert len(names) == len(set(names)) == len(tmeta) and masks.shape == (len(tmeta), 128, 128)
    torch.save({"image_names": names, "masks": masks}, a.test)
    print("wrote", a.test, masks.shape, "rect predictions:", nrect, "mean mask", float(masks.float().mean()))
