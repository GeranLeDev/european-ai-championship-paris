"""Candidate-region test. full mode: z = x' + 1_C * r(x') should be most 'fully watermarked' (max decoder agreement)
for the true clean region C. partial mode: z = x' - 1_R * r(x') should remove the watermark best for the true R.
Compare true region vs shifted copies (same shape/area)."""
import sys; from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor
torch.set_num_threads(6)
ds = LocalisationDataset(ROOT / "data", "train")
method = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
wm = load_watermark(method, ROOT / "models", dev)
rng = np.random.default_rng(0)
def agree(img, s):
    return float((torch.tanh(wm.logits(to_tensor(img))[0].cpu()) * s).mean())
for full in (True,):
    its = [it for it in (ds[i] for i in range(len(ds))) if it["method"] == method and it["full_watermark"] == full][:15]
    wins = []
    for it in its:
        x = it["image"].astype(np.float32); s = torch.from_numpy(2 * it["message"].astype(np.float32) - 1)
        r = wm.embed(it["image"], it["message"]).astype(np.float32) - x
        C = (1 - it["mask"]) if full else it["mask"]
        if C.sum() < 100: continue
        cands = [C] + [np.roll(C, (rng.integers(-200, 200), rng.integers(-200, 200)), (0, 1)) for _ in range(6)]
        sc = [agree(np.clip(x + c[..., None] * r, 0, 255).round().astype(np.uint8), s) for c in cands]
        base = agree(it["image"], s)
        rank = int((np.array(sc[1:]) >= sc[0]).sum())
        wins.append(rank == 0)
        print(method, f"frac={C.mean():.3f} base={base:.3f} true={sc[0]:.3f} shifted={np.round(sc[1:],3)} rank={rank}", flush=True)
    print("true region ranked first:", np.mean(wins))
