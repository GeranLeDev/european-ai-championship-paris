"""First-order 'where would adding the re-embedded residual r help the decoder' map: m = <r, dL/dz> at z = x'.
Expect: positive in clean regions (adding the watermark helps), <=0 in marked regions. Reports AUC of -m vs mask."""
import sys; from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor
from sklearn.metrics import roc_auc_score
torch.set_num_threads(6)
ds = LocalisationDataset(ROOT / "data", "train")
method = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
wm = load_watermark(method, ROOT / "models", "cpu")
fwd = (lambda x: wm.det(wm._resize(x))[:, 1:]) if method == "pixelseal" else (lambda x: wm.net(wm._prep(x)))
for full in (True, False):
    its = [it for it in (ds[i] for i in range(len(ds))) if it["method"] == method and it["full_watermark"] == full][:n]
    acc = {}
    for it in its:
        g = it["mask128"].reshape(-1)
        if g.min() == g.max(): continue
        r = torch.from_numpy(wm.embed(it["image"], it["message"]).astype(np.float32) - it["image"].astype(np.float32)).permute(2, 0, 1)[None] / 255
        s = torch.from_numpy(2 * it["message"].astype(np.float32) - 1)
        for lossn in ("margin", "bce"):
            x = to_tensor(it["image"]).requires_grad_(True)
            lg = fwd(x)[0]
            L = (lg * s).mean() if lossn == "margin" else -F.binary_cross_entropy_with_logits(lg, (s > 0).float())
            gr, = torch.autograd.grad(L, x)
            mp = F.avg_pool2d((gr * r).sum(1, keepdim=True), 4)
            for k in (1, 9, 25):
                sm = F.avg_pool2d(F.pad(mp, (k // 2,) * 4, mode="reflect"), k, 1) if k > 1 else mp
                acc.setdefault(f"{lossn}_k{k}", []).append(roc_auc_score(g, -sm[0, 0].numpy().reshape(-1)))
    print(method, "full" if full else "partial", {k: round(float(np.mean(v)), 3) for k, v in acc.items()}, flush=True)
