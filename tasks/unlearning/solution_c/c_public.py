"""Lane C: table of submitted models (public score vs measurable quantities from dumped logits) + simple rank correlations."""
import torch, torch.nn.functional as F, numpy as np
from c_lib import *
LG = HERE.parent / "solution" / "cache" / "logits"
P = {"r2_g0": 0.0798, "r2_g1": 0.2384, "r2_g2": 0.2616, "m1_g2": 0.2736, "ma_g1": 0.2806, "ma_g0": 0.2893, "m3_g2": 0.2046, "m3_g3": 0.2080, "st_g3": 0.3043, "st_g1": 0.3025,
     "lg_g0": 0.3045, "lg_g1": 0.3019, "lg0_bias4": 0.2348, "lg0_tailfitB": 0.2827}
o = torch.load(LG / "orig.pt"); y = o["Le"].argmax(1); cap = torch.isin(y, FULL); rows = []
print(f"{'run':14s} pub    | CE mean std | alt mean std | 9cls CE acc | part CE acc | p2true 9cls part | H2 9cls part | imnette ag KL2 KL1")
for n, s in sorted(P.items(), key=lambda kv: kv[1]):
    d = torch.load(LG / f"{n}.pt"); L = d["Le"]; ce = F.cross_entropy(L, y, reduction="none"); ca = F.cross_entropy(d["La"], y, reduction="none"); acc = (L.argmax(1) == y).float()
    p2 = F.softmax(L / 2, 1); pt = p2[torch.arange(len(y)), y]; H = -(p2 * p2.clamp_min(1e-12).log()).sum(1)
    ag = (d["Lr"].argmax(1) == o["Lr"].argmax(1)).float().mean().item(); k2 = kl2(o["Lr"], d["Lr"]).mean().item()
    k1 = (F.softmax(o["Lr"], 1) * (F.log_softmax(o["Lr"], 1) - F.log_softmax(d["Lr"], 1))).sum(1).mean().item()
    r = [s, ce.mean(), ce.std(), ca.mean(), ca.std(), ce[cap].mean(), acc[cap].mean(), ce[~cap].mean(), acc[~cap].mean(), pt[cap].mean(), pt[~cap].mean(), H[cap].mean(), H[~cap].mean(), ag, k2, k1]
    r = [float(v) for v in r]; rows.append(r)
    print(f"{n:14s} {s:.4f} | {r[1]:.2f} {r[2]:.2f} | {r[3]:.2f} {r[4]:.2f} | {r[5]:.2f} {r[6]:.2f} | {r[7]:.2f} {r[8]:.2f} | {r[9]:.3f} {r[10]:.3f} | {r[11]:.2f} {r[12]:.2f} | {r[13]:.3f} {r[14]:.3f} {r[15]:.3f}")
A = np.array(rows); names = "pub ce_m ce_s alt_m alt_s cap_ce cap_acc part_ce part_acc p2t_cap p2t_part H2_cap H2_part im_ag im_kl2 im_kl1".split()
from scipy.stats import spearmanr
for sel, tag in ((slice(None), "all 14"), (A[:, 0] > 0.27, "8 pts > 0.27")):
    B = A[sel]; print(tag, " ".join(f"{names[j]}:{spearmanr(B[:,0], B[:,j])[0]:+.2f}" for j in range(1, len(names))))
