"""Lane C helpers: measurements of a candidate given cached features (c_feats.py) and a head (W, b)."""
import os, sys, torch, torch.nn.functional as F
from pathlib import Path
HERE = Path(__file__).resolve().parent; FE = HERE / "feats"
FULL = torch.tensor([0, 6, 7, 8, 29, 30, 31, 33, 38])
O = torch.load(FE / "orig.pt"); lg = lambda D, k, W=None, b=None: D[k] @ (D["W"] if W is None else W).T + (D["b"] if b is None else b)
Y = lg(O, "Fe").argmax(1); CAP = torch.isin(Y, FULL)
def kl2(La, Lb): p = F.softmax(La / 2, 1); return (p * (F.log_softmax(La / 2, 1) - F.log_softmax(Lb / 2, 1))).sum(1)
def measure(D, W=None, b=None, B=None):
    """D = features of the candidate's backbone; (W, b) its head; B = (features, W, b) of the base for KL(base||cand) at T=2 on forget."""
    r = {}
    for tag, k in (("c", "Fe"), ("a", "Fa")):
        L = lg(D, k, W, b); ce = F.cross_entropy(L, Y, reduction="none"); acc = (L.argmax(1) == Y).float()
        r[f"{tag}_m"], r[f"{tag}_s"] = ce.mean().item(), ce.std().item()
        if tag == "c":
            r.update(cap_m=ce[CAP].mean().item(), cap_s=ce[CAP].std().item(), cap_acc=acc[CAP].mean().item(), oth_m=ce[~CAP].mean().item(), oth_s=ce[~CAP].std().item(), oth_acc=acc[~CAP].mean().item())
            p2 = F.softmax(L / 2, 1); pt = p2[torch.arange(len(Y)), Y]; H = -(p2 * p2.clamp_min(1e-12).log()).sum(1)
            r.update(p2t_cap=pt[CAP].mean().item(), p2t_oth=pt[~CAP].mean().item(), H_cap=H[CAP].mean().item(), H_oth=H[~CAP].mean().item())
            if B is not None: r["kl_base"] = kl2(lg(B, "Fe"), L).mean().item()
    for tag, k in (("gv", "Fgv"), ("gt", "Fgt"), ("im", "Fr")):
        Lo = lg(O, k); L = lg(D, k, W, b); r[f"{tag}_ag"] = (L.argmax(1) == Lo.argmax(1)).float().mean().item(); r[f"{tag}_kl"] = kl2(Lo, L).mean().item()
    Lo = torch.cat([lg(O, "Fgv"), lg(O, "Fgt")]); L = torch.cat([lg(D, "Fgv", W, b), lg(D, "Fgt", W, b)]); gc = torch.isin(Lo.argmax(1), FULL)
    r["g_kl1"] = (F.softmax(Lo, 1) * (F.log_softmax(Lo, 1) - F.log_softmax(L, 1))).sum(1).mean().item()
    r["g9_n"] = int(gc.sum()); r["g9_ag"] = (L.argmax(1) == Lo.argmax(1))[gc].float().mean().item() if gc.any() else float("nan"); r["g9_kl"] = kl2(Lo, L)[gc].mean().item() if gc.any() else float("nan")
    return r
HDR = f"{'name':22s} | CE clean      altered    | 9cls CE      acc | part CE      acc  | gen-val ag/KL  gen-train ag/KL | gen 9cls ag/KL (n) | imnette ag/KL | KL2(base||c) | T=2 p(true) 9cls/part, entropy 9cls/part | gen900 KL at T=1"
def fmt(n, r):
    return (f"{n:22s} | {r['c_m']:.2f}±{r['c_s']:.2f}  {r['a_m']:.2f}±{r['a_s']:.2f} | {r['cap_m']:.2f}±{r['cap_s']:.2f} {r['cap_acc']:.2f} | {r['oth_m']:.2f}±{r['oth_s']:.2f} {r['oth_acc']:.2f} | "
            f"{r['gv_ag']:.3f}/{r['gv_kl']:.3f}   {r['gt_ag']:.3f}/{r['gt_kl']:.3f}  | {r['g9_ag']:.2f}/{r['g9_kl']:.3f} ({r['g9_n']}) | {r['im_ag']:.3f}/{r['im_kl']:.3f} | {r.get('kl_base', float('nan')):.4f} | {r['p2t_cap']:.3f}/{r['p2t_oth']:.3f} H {r['H_cap']:.2f}/{r['H_oth']:.2f} | {r['g_kl1']:.3f}")
