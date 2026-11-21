"""Cache cheap per-image signals from the official extractors, for train and test:
  - global bit agreement with the known message (bit accuracy and mean signed logit)
  - local agreement map from per_position_logits (pixelseal, trustmark), resized to the 128 x 128 grid
Usage: python solution/features.py --device cpu|mps|cuda
Output: solution/cache/feat_{split}.npz"""
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); ap.add_argument("--data", default=str(ROOT / "data"))
a = ap.parse_args()
out_dir = ROOT / "solution" / "cache"; out_dir.mkdir(parents=True, exist_ok=True)

for split in ("train", "test"):
    ds = LocalisationDataset(a.data, split)
    by_m = {}
    for i, m in enumerate(ds.meta):
        by_m.setdefault(m["method"], []).append(i)
    res = {"names": [], "bitacc": [], "agree": [], "ppl": []}
    for method, idx in by_m.items():
        wm = load_watermark(method, ROOT / "models", a.device)
        for i in idx:
            it = ds[i]; x = to_tensor(it["image"])
            s = torch.from_numpy(2 * it["message"].astype(np.float32) - 1)
            lg = wm.logits(x)[0].cpu()
            res["names"].append(it["image_name"])
            res["bitacc"].append(float(((lg > 0).float() == torch.from_numpy(it["message"]).float()).float().mean()))
            res["agree"].append(float((torch.tanh(lg) * s).mean()))
            if method != "stegastamp":
                pp = wm.features(x)["per_position_logits"][0].cpu()
                amap = (torch.tanh(pp) * s[:, None, None]).mean(0)[None, None]
                res["ppl"].append(F.interpolate(amap, size=(128, 128), mode="bilinear", align_corners=False)[0, 0].numpy())
            else:
                res["ppl"].append(np.zeros((128, 128), np.float32))
        print(split, method, len(idx), flush=True)
    np.savez_compressed(out_dir / f"feat_{split}.npz", names=np.array(res["names"]), bitacc=np.array(res["bitacc"]),
                        agree=np.array(res["agree"]), ppl=np.stack(res["ppl"]).astype(np.float16))
