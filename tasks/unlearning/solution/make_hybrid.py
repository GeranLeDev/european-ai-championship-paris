"""Hybrid targets from an existing one. Usage: python make_hybrid.py <name> <base> [--cap-from <gold>] [--cap-ce <nats>]
rows of the 9 capped classes: p2 from --cap-from (e.g. oof9 = teacher that never saw these classes), CE columns = medium CE + --cap-ce ; other rows = <base>."""
import argparse, torch
from common import CACHE; import scoring
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("base"); ap.add_argument("--cap-from"); ap.add_argument("--cap-ce", type=float); a = ap.parse_args()
H = scoring.make_hypotheses(CACHE); B = scoring.resolve(a.base, H, CACHE); y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1); cap = torch.isin(y, torch.tensor(scoring.FULL_CLASSES))
p2 = B["p2"].clone(); ce = B["ce"].clone()
if a.cap_from: p2[cap] = torch.load(CACHE / f"gold_{a.cap_from}.pt")["p2"][cap]
if a.cap_ce is not None: ce[cap] = (H["medium"]["ce"][cap] + a.cap_ce).repeat(1, -(-ce.shape[1] // 4))[:, : ce.shape[1]]
torch.save(dict(p2=p2, ce=ce), CACHE / f"gold_{a.name}.pt")
print(a.name, f"CE capped {ce[cap].mean():.2f}±{ce[cap].std():.2f} others {ce[~cap].mean():.2f}±{ce[~cap].std():.2f} | capped max prob {p2[cap].max(1).values.mean():.2f}")
