"""Test the 'sign flip' trick: z = x' - alpha * r(x') where r = re-embedding residual with the known message.
Marked region keeps (1-alpha') r, clean region gets -alpha r -> local decoder agreement changes sign."""
import sys; from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor
from sklearn.metrics import roc_auc_score
torch.set_num_threads(3)
ds = LocalisationDataset(ROOT / "data", "train")
for method in ["trustmark", "pixelseal"]:
    wm = load_watermark(method, ROOT / "models", "cpu")
    for full in (True, False):
        its = [it for it in (ds[i] for i in range(len(ds))) if it["method"] == method and it["full_watermark"] == full][:10]
        acc = {}
        for it in its:
            x = it["image"].astype(np.float32); m = it["message"]; s = torch.from_numpy(2 * m.astype(np.float32) - 1)[:, None, None]
            r = wm.embed(it["image"], m).astype(np.float32) - x
            g = it["mask128"].reshape(-1)
            if g.min() == g.max(): continue
            for al in (0.0, 0.5, 1.0, 2.0):
                z = np.clip(x - al * r, 0, 255) / 255
                pp = wm.features(torch.from_numpy(z).permute(2, 0, 1)[None].float())["per_position_logits"][0].cpu()
                amap = F.interpolate((torch.tanh(pp) * s).mean(0)[None, None], size=(128, 128), mode="bilinear")[0, 0].numpy()
                acc.setdefault(al, []).append(roc_auc_score(g, amap.reshape(-1)))
        print(method, "full" if full else "partial", {k: round(float(np.mean(v)), 3) for k, v in acc.items()}, flush=True)
