"""Idempotence test: re-embedding with the known message changes marked pixels less than clean pixels.
map = local energy(embed(x', m) - x') / local energy(embed(x', m_rand) - x')  (low = marked)"""
import sys; from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from sklearn.metrics import roc_auc_score
torch.set_num_threads(4)
ds = LocalisationDataset(ROOT / "data", "train")
meths = sys.argv[1].split(",") if len(sys.argv) > 1 else ["stegastamp", "trustmark", "pixelseal"]
def en(r, k):
    t = torch.from_numpy((r ** 2).sum(-1))[None, None]
    t = F.avg_pool2d(t, 4)
    return F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="reflect"), k + 1 if k % 2 == 0 else k, 1)[0, 0].numpy() if k > 1 else t[0, 0].numpy()
for method in meths:
    wm = load_watermark(method, ROOT / "models", "cpu")
    rng = np.random.default_rng(0)
    for full in (True, False):
        its = [it for it in (ds[i] for i in range(len(ds))) if it["method"] == method and it["full_watermark"] == full][:12]
        acc = {}
        for it in its:
            g = it["mask128"].reshape(-1)
            if g.min() == g.max(): continue
            x = it["image"].astype(np.float32)
            rm = wm.embed(it["image"], it["message"]).astype(np.float32) - x
            rr = np.mean([(wm.embed(it["image"], rng.integers(0, 2, wm.nbits)).astype(np.float32) - x) ** 2 for _ in range(2)], 0)
            for k in (1, 3, 7):
                e1 = en(rm, k); e2 = en(np.sqrt(rr), k)
                acc.setdefault(f"ratio_k{k}", []).append(roc_auc_score(g, -(np.log(e1 + 1e-3) - np.log(e2 + 1e-3)).reshape(-1)))
                acc.setdefault(f"e1_k{k}", []).append(roc_auc_score(g, -e1.reshape(-1)))
        print(method, "full" if full else "partial", {k: round(float(np.mean(v)), 3) for k, v in acc.items()}, flush=True)
