"""Benchmark watermark methods on N COCO val images under a set of attacks.
Run from repos/videoseal: .venv/bin/python ../_smoke/baselines_bench.py [N]
Outputs: mean bit accuracy, TPR at binomial p<1e-3, empirical FPR on unmarked images."""
import sys, io, time, glob, json, math
import torch
import torchvision.transforms.functional as F
from PIL import Image
from scipy.stats import binom
import videoseal
from videoseal.utils.cfg import setup_model_from_checkpoint

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
P_THRESH = 1e-3
files = sorted(glob.glob("../data/coco_val100/*.jpg"))[:N]
torch.manual_seed(0)

def jpeg(t, q):
    buf = io.BytesIO(); F.to_pil_image(t[0].clamp(0, 1).cpu()).save(buf, "JPEG", quality=q)
    return F.to_tensor(Image.open(buf)).unsqueeze(0).to(t.device)
def crop(t, frac):
    h, w = t.shape[-2:]; s = frac ** 0.5
    return F.center_crop(t, [int(h * s), int(w * s)])
ATTACKS = {
    "identity": lambda t: t,
    "jpeg40": lambda t: jpeg(t, 40),
    "crop0.32": lambda t: crop(t, 0.32),
    "bright0.5": lambda t: F.adjust_brightness(t, 0.5),
    "rot10": lambda t: F.rotate(t, 10),
    "blur": lambda t: F.gaussian_blur(t, 9, 2.0),
    "resize0.5": lambda t: F.resize(t, [t.shape[-2] // 2, t.shape[-1] // 2], antialias=True),
    "combo": lambda t: jpeg(F.adjust_brightness(crop(t, 0.32), 0.5), 40),
}
def pval(k_match, n):  # P(X >= k) under H0: X ~ Bin(n, 0.5)
    return binom.sf(k_match - 1, n, 0.5)

METHODS = ["baseline/hidden", "baseline/trustmark", "baseline/wam", "baseline/mbrs", "videoseal_1.0", "pixelseal"]
results = {}
for name in METHODS:
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = (setup_model_from_checkpoint(name) if name.startswith("baseline") else videoseal.load(name)).eval().to(dev)
    acc = {a: [] for a in ATTACKS}; tp = {a: 0 for a in ATTACKS}; psnr = []; fp = 0; acc_clean = []
    with torch.no_grad():
        for f in files:
            x = F.to_tensor(Image.open(f).convert("RGB")).unsqueeze(0).to(dev)
            out = m.embed(x, is_video=False); xw = out["imgs_w"].clamp(0, 1); msg = out["msgs"].bool()
            n = msg.shape[-1]
            psnr.append(10 * math.log10(1 / ((xw - x) ** 2).mean().item()))
            for a, fn in ATTACKS.items():
                pred = m.detect(fn(xw), is_video=False)["preds"][:, 1:] > 0
                k = (pred == msg).sum().item()
                acc[a].append(k / n); tp[a] += pval(k, n) < P_THRESH
            pred = m.detect(x, is_video=False)["preds"][:, 1:] > 0
            k = (pred == msg).sum().item(); acc_clean.append(k / n); fp += pval(k, n) < P_THRESH
    r = {"bits": n, "psnr": sum(psnr) / len(psnr),
         "acc": {a: sum(v) / len(v) for a, v in acc.items()},
         "tpr": {a: tp[a] / len(files) for a in ATTACKS},
         "unmarked_acc": sum(acc_clean) / len(acc_clean), "fpr": fp / len(files), "sec": time.time() - t0}
    results[name] = r
    print(f"{name:20s} bits={n:3d} PSNR={r['psnr']:.1f} FPR={r['fpr']:.2f} unmarked={r['unmarked_acc']:.3f} | "
          + " ".join(f"{a}={r['acc'][a]:.3f}/{r['tpr'][a]:.2f}" for a in ATTACKS) + f" ({r['sec']:.0f}s)", flush=True)
    json.dump(results, open(f"../_smoke/out/bench_{len(files)}.json", "w"), indent=1)
