"""Quick exploration: for a few train images per method, compute candidate signal maps and their pixel AUC vs GT."""
import sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from sklearn.metrics import roc_auc_score

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
ds = LocalisationDataset(ROOT / "data", "train")


def box(x, k):  # x: H x W float -> local mean
    t = torch.from_numpy(x)[None, None].float()
    return F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="reflect"), k, 1)[0, 0, :x.shape[0], :x.shape[1]].numpy()


def hp(img, k=5):
    return np.stack([img[..., c] - box(img[..., c], k) for c in range(3)], -1)


for method in ["stegastamp", "trustmark", "pixelseal"]:
    wm = load_watermark(method, ROOT / "models", dev)
    items = [it for it in (ds[i] for i in range(len(ds))) if it["method"] == method][:n]
    for it in items:
        x = it["image"].astype(np.float32)
        t = time.time(); y = wm.embed(it["image"], it["message"]).astype(np.float32); te = time.time() - t
        r = y - x
        g = it["mask"].reshape(-1)
        out = {}
        for k in (3, 7):
            h = hp(x, k); rh = hp(r, k)
            prod = (h * rh).sum(-1)
            for w in (15, 31):
                num = box(prod, w); den = np.sqrt(box((h ** 2).sum(-1), w) * box((rh ** 2).sum(-1), w)) + 1e-6
                out[f"ncc_k{k}_w{w}"] = num / den
        # local bit agreement from per-position logits
        if hasattr(wm, "features") and method != "stegastamp":
            tx = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float() / 255
            pp = wm.features(tx)["per_position_logits"][0].cpu()
            s = (2 * torch.from_numpy(it["message"]).float() - 1)[:, None, None]
            agr = (pp * s).mean(0)[None, None]
            out["ppl_agree"] = F.interpolate(agr, size=(512, 512), mode="bilinear")[0, 0].numpy()
        frac = g.mean()
        res = {k: (roc_auc_score(g, v.reshape(-1)) if 0 < frac < 1 else float('nan')) for k, v in out.items()}
        print(method, it["image_name"], f"full={it['full_watermark']} frac={frac:.2f} embed {te:.2f}s |r|={np.abs(r).mean():.2f}",
              " ".join(f"{k}={v:.3f}" for k, v in res.items()), flush=True)
