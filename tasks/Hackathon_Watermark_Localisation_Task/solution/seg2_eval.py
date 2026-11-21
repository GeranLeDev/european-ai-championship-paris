"""Compare, on the real held-out fold (150 images never used for training), per method and per mask type:
  base     = baseline (mode decision + per_position_logits)                     -> submission_baseline.pt
  rect     = baseline + rectangle search                                        -> submission_rect.pt
  seg      = trained segmenter (solution/train_seg2.py, <run>/pred.npz), threshold 0.5
  seg+rect = segmenter, replaced by the rectangle when the rectangle search fires
With --test, writes a submission using, per method, the variant given by --choose (default: best on the held-out
fold, which is then slightly optimistic: the choice among 4 variants is made on the same 150 images).
Usage: python solution/seg2_eval.py --runs trustmark=output/seg2/trustmark,pixelseal=...,stegastamp=... [--test out.pt]
A run trained with --method all can be given for several methods (all=path)."""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "solution"))
from common import METHODS, split_train
from baseline import predict
from rectsearch import box_to_mask128
from wmgen import mask_kind
from wmloc.utils import miou_image
from PIL import Image

RECT = {"pixelseal": (0.40, 3), "trustmark": (0.60, 0)}     # t_rect, d tuned on the train fold (rect_eval.py)
ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--test", default=""); ap.add_argument("--choose", default="")
ap.add_argument("--data", default=str(ROOT / "data")); ap.add_argument("--thr", type=float, default=0.5)
a = ap.parse_args()
C = ROOT / "solution" / "cache"; params = json.load(open(C / "baseline_params.json"))
runs = dict(kv.split("=") for kv in a.runs.split(","))
if "all" in runs: runs = {m: runs.get(m, runs["all"]) for m in METHODS}


def load_search(split, m, tags):
    out = {}
    for tag in tags:
        pat = re.compile(rf"search_{split}_{m}{re.escape(tag)}(_shard\d+)?\.npz$")
        for p in sorted(C.glob(f"search_{split}_{m}*.npz")):
            if not pat.match(p.name): continue
            z = np.load(p)
            for n, b, s in zip(z["names"].tolist(), z["box"], z["score"]):
                if n not in out or float(s) > out[n][1]: out[n] = (np.asarray(b, np.float32), float(s))
    return out


def binarize(p, thr):
    """Threshold with the two rules that hold for every image of the dataset: never an empty grid, never a full grid
    (train masks: area between 0.3 % and 99.7 %). An empty / full prediction caps the image at 0.5."""
    m = p > thr
    if not m.any(): m = p >= max(float(p.max()) * 0.5, np.partition(p.ravel(), -32)[-32])
    if m.all(): m = p > min(1 - (1 - float(p.min())) * 0.5, np.partition(p.ravel(), 32)[32])
    return m.astype(np.uint8)


def variants(n, m, feats, f_idx, srch, seg):
    i = f_idx[n]; ag = feats["agree"][i]
    base = predict(ag, feats["ppl"][i].astype(np.float32), *params[m])
    box = None
    if m in RECT and ag <= params[m][0] and n in srch and srch[n][1] > RECT[m][0]:
        d = RECT[m][1]; box = box_to_mask128(srch[n][0] + np.array([-d, -d, d, d]))
    v = {"base": base, "rect": box if box is not None else base}
    if seg is not None:
        s = binarize(seg, a.thr); v["seg_raw"] = (seg > a.thr).astype(np.uint8); v["seg"] = s; v["seg+rect"] = box if box is not None else s
    return v


meta = json.loads((Path(a.data) / "train" / "metadata.json").read_text())
z = np.load(Path(a.data) / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
val = split_train(a.data); feats = np.load(C / "feat_train.npz"); f_idx = {n: i for i, n in enumerate(feats["names"].tolist())}
kind = {e["image_name"]: mask_kind(e["full_watermark"], (np.asarray(Image.open(Path(a.data) / "train" / "masks" / f"{e['image_name']}.png").convert("L")) > 127)) for e in meta if e["image_name"] in val}
choice, tot = {}, {}
want = dict(kv.split("=") for kv in a.choose.split(",")) if a.choose else {}
preds = {}
for m in METHODS:
    seg = {}
    if m in runs and (Path(runs[m]) / "pred.npz").exists():
        p = np.load(Path(runs[m]) / "pred.npz"); preds[m] = p
        seg = {n: pr.astype(np.float32) for n, pr in zip(p["names"].tolist(), p["probs"])}
    srch = load_search("train", m, ["", "_r", "_r2"])
    names = [e["image_name"] for e in meta if e["method"] == m and e["image_name"] in val]
    sc = {}
    for n in names:
        for k, pm in variants(n, m, feats, f_idx, srch, seg.get(n)).items(): sc.setdefault(k, {})[n] = miou_image(pm, gt[n])
    print(f"{m} held-out (n={len(names)})")
    for k, d in sc.items():
        by = {t: np.mean([d[n] for n in names if kind[n] == t]) for t in ("rect", "irreg", "full")}
        print(f"   {k:9s} all {np.mean(list(d.values())):.4f} | " + " ".join(f"{t} {v:.4f}" for t, v in by.items()))
    choice[m] = want.get(m) or max(sc, key=lambda k: np.mean(list(sc[k].values())))
    tot[m] = {k: float(np.mean(list(d.values()))) for k, d in sc.items()}
print("global held-out: " + " ".join(f"{k} {np.mean([tot[m][k] for m in METHODS if k in tot[m]]):.4f}" for k in ("base", "rect", "seg_raw", "seg", "seg+rect") if all(k in tot[m] for m in METHODS)))
print("chosen per method:", choice, "-> global held-out %.4f" % np.mean([tot[m][choice[m]] for m in METHODS]))

if a.test:
    feats = np.load(C / "feat_test.npz"); f_idx = {n: i for i, n in enumerate(feats["names"].tolist())}
    tmeta = json.loads((Path(a.data) / "test" / "metadata.json").read_text())
    srch = {m: load_search("test", m, ["_t", "_t_r", "_t_r2", "_t_r3"]) for m in METHODS}
    seg = {}
    for m, p in preds.items():
        if "test_names" in p.files: seg.update({n: pr.astype(np.float32) for n, pr in zip(p["test_names"].tolist(), p["test_probs"])})
    names, masks = [], []
    for e in tmeta:
        n, m = e["image_name"], e["method"]
        v = variants(n, m, feats, f_idx, srch[m], seg.get(n))
        names.append(n); masks.append(v[choice[m]] if choice[m] in v else v["rect"])
    masks = torch.from_numpy(np.stack(masks)).bool()
    assert len(names) == len(set(names)) == 1020 and masks.shape == (1020, 128, 128)
    torch.save({"image_names": names, "masks": masks}, a.test)
    print("wrote", a.test, tuple(masks.shape), masks.dtype, "mean mask %.4f" % float(masks.float().mean()), "test images with segmenter output:", len(seg))
