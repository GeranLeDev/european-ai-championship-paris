"""Fidelity check real vs synthetic, per mask type: ratio of the re-embedding residual energy inside / outside the
marked region (a whole-image embed leaves a smaller re-embedding residual where the image is already marked), and
the local sign agreement between high-pass(y) and the re-embedding residual inside vs outside.
Usage: python solution/fidelity.py trustmark cpu 20"""
import sys; from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "solution"))
from wmgen import Generator, real_items, TYPES
method, dev, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
g = Generator(method, ROOT / "data", dev, coco=ROOT / "extra" / "val2017", gen_dirs=[ROOT / "extra" / "diffusiondb"], p_real=0.0)

def stats(it):
    m = np.repeat(np.repeat(it["mask"], 4, 0), 4, 1).astype(bool)
    if m.all() or not m.any(): return None
    w = it["what"].astype(np.float32); a = np.abs(w).mean(2)
    y = torch.from_numpy(it["img"].astype(np.float32)).permute(2, 0, 1)[None]
    hp = (y - F.avg_pool2d(F.pad(y, (1, 1, 1, 1), mode="reflect"), 3, 1))[0].permute(1, 2, 0).numpy()
    c = (hp * w).sum(2)
    return a[m].mean() / a[~m].mean(), c[m].mean() / (a[m] ** 2).mean(), c[~m].mean() / (a[~m] ** 2).mean()

real = real_items(g, ROOT / "data")
for k in TYPES:
    rr = [s for s in (stats(r) for r in real if r["kind"] == k) if s][:max(n, 40)]
    ss = [s for s in (stats(g.sample(k)) for _ in range(n)) if s]
    f = lambda v: "%.2f [%.2f, %.2f]" % (np.median(v), np.quantile(v, 0.25), np.quantile(v, 0.75))
    for j, name in enumerate(("|what| in/out", "corr(hp y, what) inside", "corr(hp y, what) outside")):
        print(f"{method} {k:5s} {name:26s} real {f([r[j] for r in rr])} (n={len(rr)}) | synth {f([s[j] for s in ss])} (n={len(ss)})", flush=True)
