import argparse, torch, torch.nn.functional as F
from common import *; import augs
ap = argparse.ArgumentParser(); ap.add_argument("--device"); ap.add_argument("--n", type=int, default=1000); a = ap.parse_args()
dev = pick_device(a.device); set_device(dev); torch.manual_seed(0)
m = load_orig(dev); X = load_forget()[:a.n]
L0 = torch.load(CACHE / "orig_forget_logits.pt")[:a.n]; y = L0.argmax(1)
for name, fn in [("none", None), ("hflip", lambda x: x.flip(-1)), ("light", augs.light), ("medium", augs.medium), ("strong", augs.strong)]:
    L = logits_of(m, X, dev, aug=fn)
    ce = F.cross_entropy(L, y, reduction="none")
    kl = (F.softmax(L0/2,1) * (F.log_softmax(L0/2,1) - F.log_softmax(L/2,1))).sum(1)
    print(f"{name:7s} acc={(L.argmax(1)==y).float().mean():.3f} CE mean={ce.mean():.3f} std={ce.std():.3f} median={ce.median():.4f} conf={L.softmax(1).max(1).values.mean():.3f} KL(orig||aug)T2={kl.mean():.3f}")
