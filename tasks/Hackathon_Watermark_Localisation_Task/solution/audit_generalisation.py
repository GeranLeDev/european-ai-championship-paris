"""Task 1 generalisation audit: is the public score (30 % of the test set) a fair estimate of the hidden 70 %?

Usage (from tasks/Hackathon_Watermark_Localisation_Task):
    python solution/audit_generalisation.py [submission.pt ...] [--probs probs/amil_trustmark_train.npz ...]

1. Noise: per-image mIoU of held-out train predictions -> standard error of a 306-image public score,
   of the 714-image hidden score, and of the gap between the two (random split, same distribution).
2. Realism: for each submission file, the predicted masks on the 1020 test images are compared with the
   510 true train masks, per method (coverage, share of full / empty / partial images, connected parts).
   A file tuned to the public split, or a broken post-processing, shows up as a shifted distribution.
No test label is used anywhere (there is none)."""
import argparse, json, sys
from collections import defaultdict
import numpy as np, torch
from scipy import ndimage

sys.path.insert(0, ".")
from wmloc.utils import miou_image

ap = argparse.ArgumentParser(); ap.add_argument("subs", nargs="*"); ap.add_argument("--probs", nargs="*", default=[]); ap.add_argument("--thr", type=float, default=0.5)
a = ap.parse_args()
tr = json.load(open("data/train/metadata.json")); te = json.load(open("data/test/metadata.json"))
z = np.load("data/train/masks_128.npz", allow_pickle=True); G = dict(zip([str(n) for n in z["names"]], z["masks"]))
m_tr = {e["image_name"]: e["method"] for e in tr}; m_te = {e["image_name"]: e["method"] for e in te}
N_PUB, N_HID = round(0.3 * len(te)), len(te) - round(0.3 * len(te))


def describe(masks):
    """Shape statistics of a list of 128x128 binary masks."""
    f = np.array([m.mean() for m in masks]); nc = np.array([ndimage.label(m)[1] for m in masks])
    part = (f > 0.02) & (f < 0.98)
    return dict(n=len(masks), cover=f.mean(), full=(f >= 0.98).mean(), empty=(f <= 0.02).mean(), partial=part.mean(),
                cover_partial=f[part].mean() if part.any() else float("nan"), parts=np.median(nc), q=np.quantile(f, [.1, .25, .5, .75, .9]))


def show(tag, d):
    print(f"  {tag:22s} n={d['n']:4d} coverage {d['cover']:.3f} | >=98% marked {d['full']:.2f} | <=2% marked {d['empty']:.2f} | partial {d['partial']:.2f} "
          f"(coverage {d['cover_partial']:.2f}) | median parts {d['parts']:.0f} | coverage quantiles 10/25/50/75/90 % " + " ".join(f"{x:.2f}" for x in d["q"]))


print(f"== 1. Noise of the split (public {N_PUB} images, hidden {N_HID}) ==")
per = defaultdict(list)
for p in a.probs:
    d = np.load(p); held = ~d["used_in_training"] if "used_in_training" in d.files else np.ones(len(d["names"]), bool)
    for n, pr, h in zip(d["names"], d["probs"], held):
        if h and str(n) in G: per[m_tr[str(n)]].append(miou_image(pr.astype(np.float32) >= a.thr, G[str(n)]))
if per:
    sig = []
    for k, v in sorted(per.items()):
        v = np.array(v); sig.append(v.std()); print(f"  {k:11s} held-out images {len(v):3d}: mIoU {v.mean():.3f}, per-image std {v.std():.3f}, share of images below 0.6: {(v < 0.6).mean():.2f}")
    s = float(np.sqrt(np.mean(np.square(sig))))   # the split is balanced over the methods: within-method spread is what matters
    se = lambda n: s / np.sqrt(n)
    gap = s * np.sqrt(1 / N_PUB + 1 / N_HID)
    print(f"  pooled per-image std {s:.3f} -> standard error: public {se(N_PUB):.4f}, hidden {se(N_HID):.4f}, 510-image validation {se(510):.4f}")
    print(f"  public - hidden gap from the split alone: std {gap:.4f}, 95 % interval +-{1.96 * gap:.3f}")
    print(f"  two files compared on the same public images (paired, correlation ~0.8 assumed): a difference below ~{1.96 * s * np.sqrt(2 * 0.2 / N_PUB):.3f} is not significant")

print("== 2. Realism of the predicted test masks (reference = true train masks) ==")
ref = defaultdict(list)
for n, g in G.items(): ref[m_tr[n]].append(np.asarray(g) > 0)
for k in sorted(ref): show(f"train truth {k}", describe(ref[k]))
for p in a.subs:
    sd = torch.load(p, map_location="cpu"); names = [str(n) for n in sd["image_names"]]; M = sd["masks"].numpy() > 0
    ok = len(names) == len(te) == len(set(names)) and set(names) == set(m_te) and M.shape == (len(te), 128, 128)
    print(f"-- {p}: format ok={ok}")
    by = defaultdict(list)
    for n, m in zip(names, M): by[m_te[n]].append(m)
    for k in sorted(by):
        d = describe(by[k]); show(f"test pred {k}", d); r = describe(ref[k])
        flags = [f"{name} {d[key]:.2f} vs {r[key]:.2f}" for name, key, tol in [("coverage", "cover", .06), ("full share", "full", .10), ("empty share", "empty", .05), ("partial share", "partial", .10)] if abs(d[key] - r[key]) > tol]
        # tolerance ~2 standard errors of a share measured on 170 train + 340 test images
        print("     -> " + ("SHIFT: " + "; ".join(flags) if flags else "consistent with the train distribution"))
