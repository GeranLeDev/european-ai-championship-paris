"""Gradient attribution of the decoder's agreement with the known message: map = local sum of grad * highpass(x)."""
import sys; from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor
from sklearn.metrics import roc_auc_score
torch.set_num_threads(3)
ds = LocalisationDataset(ROOT / "data", "train")
for method in ["pixelseal", "trustmark"]:
    wm = load_watermark(method, ROOT / "models", "cpu")
    fwd = (lambda x: wm.det(wm._resize(x))[:, 1:]) if method == "pixelseal" else (lambda x: wm.net(wm._prep(x)))
    for full in (True, False):
        its = [it for it in (ds[i] for i in range(len(ds))) if it["method"] == method and it["full_watermark"] == full][:12]
        acc = {}
        for it in its:
            g = it["mask128"].reshape(-1)
            if g.min() == g.max(): continue
            x = to_tensor(it["image"]).requires_grad_(True); s = torch.from_numpy(2 * it["message"].astype(np.float32) - 1)
            (torch.tanh(fwd(x)[0]) * s).mean().backward()
            gr = x.grad[0]; xd = x.detach()[0]
            hp = xd - F.avg_pool2d(F.pad(xd[None], (2,) * 4, mode="reflect"), 5, 1)[0]
            for nm, mp in (("gxhp", (gr * hp).sum(0)), ("gx", (gr * xd).sum(0)), ("absg", gr.abs().sum(0))):
                for k in (4, 16):
                    sm = F.avg_pool2d(mp[None, None], 4)                  # 128 grid
                    sm = F.avg_pool2d(F.pad(sm, (k // 2,) * 4, mode="reflect"), k + 1, 1)[0, 0] if k > 4 else sm[0, 0]
                    acc.setdefault(f"{nm}_k{k}", []).append(roc_auc_score(g, sm.numpy().reshape(-1)))
        print(method, "full" if full else "partial", {k: round(float(np.mean(v)), 3) for k, v in acc.items()}, flush=True)
