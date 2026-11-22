"""Target 'sis': like mixcls, except for images of the 9 capped classes, whose mass goes to the NON-capped sister classes of the same
family (a retrained model that never saw the class predicts a neighbour class). Sister shares = orig strong-view p2 restricted to
the sisters; --w of the mass on sisters, the rest = strong p2 with all 9 capped columns removed. CE columns = those of mixcls."""
import sys, torch
from common import CACHE; import scoring
w = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85; name = sys.argv[2] if len(sys.argv) > 2 else "sis"
SIS = {0: [1, 2, 3, 4], 6: [5], 7: [5], 8: [5], 29: [32], 30: [32], 31: [32], 33: [32], 38: [34, 35, 36, 37]}
H = scoring.make_hypotheses(CACHE); y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1); S = H["strong"]["p2"]; M = H["mixcls"]; p2 = M["p2"].clone(); FC = torch.tensor(list(SIS))
for c, sis in SIS.items():
    m = y == c; s = S[m][:, sis] + 1e-4; s = s / s.sum(1, keepdim=True); rest = S[m].clone(); rest[:, FC] = 0; rest = rest / rest.sum(1, keepdim=True)
    q = (1 - w) * rest; q[:, sis] += w * s; p2[m] = q
torch.save(dict(p2=p2, ce=M["ce"]), CACHE / f"gold_{name}.pt"); cap = torch.isin(y, FC)
print(name, "capped rows: mass on sisters", p2[cap][:, [1,2,3,4,5,32,34,35,36,37]].sum(1).mean().item(), "max prob", p2[cap].max(1).values.mean().item(), "| mixcls max prob on capped", M["p2"][cap].max(1).values.mean().item(), "| CE", M["ce"].mean().item(), M["ce"].std().item())
