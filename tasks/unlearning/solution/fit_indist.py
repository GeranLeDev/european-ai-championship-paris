"""Re-fit public scores with an IN-DISTRIBUTION retain proxy (probe_collateral.py: KL/agreement vs orig on medium/strong
augmented forget views) instead of Imagenette. Same gold family as fit12.py. Reports RMS and LOO for each retain variant."""
import itertools, json, math, torch
import fit12 as f
from common import CACHE
P = dict(f.PUBLIC); P.update({"lg0_bias4": 0.2348, "lg0_tailfitB": 0.2827})
col = json.load(open(CACHE / "collateral.json"))
names = [k for k in P if k in col and (f.LD / f"{k}.pt").exists()]
runs = {k: torch.load(f.LD / f"{k}.pt") for k in names}
pub = torch.tensor([P[k] for k in names]); print(len(names), "points:", names)
W = [w for w in itertools.product([0, .25, .5, .75, 1], repeat=len(f.B)) if abs(sum(w) - 1) < 1e-6]
G = [(w, dc, do, "pool") for w in W for dc in [0, 1, 2, 4, 8, 12, 20] for do in [0, 1, 2]]
Fm = torch.tensor([[f.forget(runs[k], f.gold(*g))[0] for k in names] for g in G])
def retain_feats(view, grp):
    return torch.tensor([col[k][f"{view}_agree_{grp}"] for k in names]), torch.tensor([col[k][f"{view}_kl_{grp}"] for k in names])
img = [f.scoring.retain_scores(runs[k]["Lr"], f.O["Lr"]) for k in names]
variants = {"imagenette": (torch.tensor([r["agree"] for r in img]), torch.tensor([r["kl"] for r in img]))}
for v in ["med", "str", "gen"]:
    for grp in (["all"] if v == "gen" else ["all", "cap", "oth"]): variants[f"{v}_{grp}"] = retain_feats(v, grp)
for vn, (ag, kl) in variants.items():
    best = None
    for aa in [0, 1, 2, 4]:
        for c in [0, 0.25, 0.5, 1, 2, 4, 8]:
            R = ag ** aa / (1 + c * kl); pred = Fm * R            # [G, n]
            s = (pred @ pub) / (pred * pred).sum(1).clamp_min(1e-12)
            rms = ((s[:, None] * pred - pub) ** 2).mean(1).sqrt()
            i = rms.argmin().item()
            if best is None or rms[i] < best[0]:
                # LOO for this (aa,c): refit gold+s without point j
                loo = []
                for j in range(len(names)):
                    m = torch.ones(len(names), dtype=torch.bool); m[j] = False
                    sj = (pred[:, m] @ pub[m]) / (pred[:, m] ** 2).sum(1).clamp_min(1e-12)
                    rj = ((sj[:, None] * pred[:, m] - pub[m]) ** 2).mean(1).sqrt(); ij = rj.argmin().item()
                    loo.append((sj[ij] * pred[ij, j] - pub[j]).item())
                best = (rms[i].item(), aa, c, G[i], s[i].item(), math.sqrt(sum(e * e for e in loo) / len(loo)), loo)
    rms, aa, c, g, s, loo, errs = best
    print(f"{vn:11s} RMS {rms:.4f} LOO {loo:.4f} | a={aa} c={c} s={s:.2f} gold={g[0]} dcap={g[1]} doth={g[2]} | LOO err bias4 {errs[names.index('lg0_bias4')]:+.3f} tailfitB {errs[names.index('lg0_tailfitB')]:+.3f}")
