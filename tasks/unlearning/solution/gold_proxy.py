"""Gold proxies: orig's behaviour on augmented (less memorised) views of each forget image.
Saves per image: avg softmax T=2 (p2), per-view CE wrt pseudo-label (ce [N,K]), for aug in {medium,strong}."""
import argparse, time, torch, torch.nn.functional as F
from common import *; import augs
ap = argparse.ArgumentParser(); ap.add_argument("--device"); ap.add_argument("--k", type=int, default=4); ap.add_argument("--names", default="medium,strong"); a = ap.parse_args()
dev = pick_device(a.device); set_device(dev); torch.manual_seed(1)
m = load_orig(dev); X = load_forget(); y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1)
for name in a.names.split(","):
    fn = getattr(augs, name); P = 0; CE = []; t = time.time()
    for k in range(a.k):
        L = logits_of(m, X, dev, aug=fn)
        P = P + F.softmax(L / 2, 1) / a.k
        CE.append(F.cross_entropy(L, y, reduction="none"))
    CE = torch.stack(CE, 1)
    torch.save({"p2": P, "ce": CE}, CACHE / f"gold_{name}.pt")
    print(f"{name}: {time.time()-t:.0f}s  CE mean {CE.mean():.3f} std {CE.std():.3f}  acc(avg) {(P.argmax(1)==y).float().mean():.3f}")
