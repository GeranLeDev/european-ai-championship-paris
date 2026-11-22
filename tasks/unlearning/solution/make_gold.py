"""Materialise a fitted gold hypothesis as cache/gold_<name>.pt (usable as --target <name> in unlearn.py).
Usage: python make_gold.py <name> <base:w,base:w,...> <dcap> <doth> [own|cap|all]
  p2 = sum w_b p2_b ; logit shift -dcap on the pseudo-label column of capped-class images (own), or on all 9 capped columns for
  capped images (cap) / all images (all) ; -doth on the pseudo-label of the other images. CE columns: shifted per-view CE of each
  base, repeated proportionally to its weight (weights in multiples of 0.25)."""
import sys, math, torch
from common import CACHE; import scoring
name, spec, dcap, doth = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]); mode = sys.argv[5] if len(sys.argv) > 5 else "own"
y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1); N = len(y); ar = torch.arange(N); FC = torch.tensor(scoring.FULL_CLASSES); cap = torch.isin(y, FC)
bases = [(b.split(":")[0], float(b.split(":")[1])) for b in spec.split(",")]; G = {b: torch.load(CACHE / f"gold_{b}.pt") for b, _ in bases}
p2 = sum(w * G[b]["p2"] for b, w in bases).clone(); d = torch.where(cap, torch.tensor(dcap), torch.tensor(doth))
if mode == "own":
    p2[ar, y] = p2[ar, y] * torch.exp(-d / 2)
else:
    rows = cap if mode == "cap" else torch.ones_like(cap); sub = p2[rows]; sub[:, FC] = sub[:, FC] * math.exp(-dcap / 2); p2[rows] = sub
    nc = ~cap; p2[ar[nc], y[nc]] = p2[ar[nc], y[nc]] * math.exp(-doth / 2)
p2 = p2 / p2.sum(1, keepdim=True); cols = []
for b, w in bases:
    ce = G[b]["ce"][:, :4]; p = torch.exp(-ce); ce = ce + d[:, None] + torch.log((1 - p + p * torch.exp(-d[:, None])).clamp_min(1e-30))
    cols += [ce] * int(round(w * 4))
ce = torch.cat(cols, 1); torch.save(dict(p2=p2, ce=ce), CACHE / f"gold_{name}.pt")
print(f"{name}: CE {ce.mean():.2f}±{ce.std():.2f} (capped {ce[cap].mean():.2f}, others {ce[~cap].mean():.2f}) acc {(p2.argmax(1)==y).float().mean():.3f} (capped {(p2.argmax(1)==y)[cap].float().mean():.3f})")
