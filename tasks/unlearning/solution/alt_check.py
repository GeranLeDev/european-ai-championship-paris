"""How do orig and an unlearned checkpoint behave on various alterations of the forget images (hidden 'altered copies' = 50% of the forget score)?"""
import sys, os, torch, torch.nn.functional as F
from common import *; import augs
dev = pick_device(); set_device(dev); X = load_forget(); y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1)
idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(5))[:500]; X = X[idx]; y = y[idx]; cap = torch.isin(y, torch.tensor([0, 6, 7, 8, 29, 30, 31, 33, 38]))
ms = {"orig": load_orig(dev)}
for p in sys.argv[1:]:
    m = tt.build_model(); m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True)); ms[os.path.basename(p)[:-3]] = m.to(dev).eval()
alts = {"clean": None, "flip": lambda z: z.flip(-1), "light": augs.light, "medium": augs.medium, "rescale.5": lambda z: augs.rescale(z, .5, .5), "rescale.25": lambda z: augs.rescale(z, .25, .25), "quant": augs.quant,
        "noise.05": lambda z: (z + .1 * torch.randn_like(z)).clamp(-1, 1), "noise.15": lambda z: (z + .3 * torch.randn_like(z)).clamp(-1, 1), "blur": lambda z: augs.blur(z, 1.0), "color.4": lambda z: augs.color(z, .4),
        "crop.8": lambda z: augs.rrc(z, (.8, .8), (1, 1)), "gray": lambda z: z.mean(1, keepdim=True).expand_as(z), "strong": augs.strong}
print("alteration    | " + " | ".join(f"{n:>24s}" for n in ms) + "   (CE mean±std, acc ; capped CE / others CE)")
for an, fn in alts.items():
    row = []
    for n, m in ms.items():
        torch.manual_seed(7); L = logits_of(m, X, dev, aug=fn); ce = F.cross_entropy(L, y, reduction="none")
        row.append(f"{ce.mean():.2f}±{ce.std():.2f} {(L.argmax(1)==y).float().mean():.2f} ; {ce[cap].mean():.2f}/{ce[~cap].mean():.2f}")
    print(f"{an:13s} | " + " | ".join(f"{r:>24s}" for r in row), flush=True)
