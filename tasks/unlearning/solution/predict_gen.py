"""Rank candidates with the recalibrated proxy: public ~ s * forget(gold) * agree_gen^a / (1 + c*KL_gen), fitted on the public scores
(gold family of fit12.py; retain measured on generated in-distribution images). Needs rank_subs.py dumps in --dir.
Usage: python predict_gen.py --dir cache/rank [--top 30]"""
import argparse, glob, itertools, math, os, torch, torch.nn.functional as F
import fit12 as f
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="cache/rank"); ap.add_argument("--top", type=int, default=30); a = ap.parse_args()
P = dict(f.PUBLIC); P.update({"lg0_bias4": 0.2348, "lg0_tailfitB": 0.2827, "gm_g2": 0.3128, "fc_g1_oof9": 0.1054})
D = {os.path.basename(p)[:-3]: torch.load(p) for p in sorted(glob.glob(f"{a.dir}/*.pt"))}; O = D.pop("orig")
assert torch.equal(O["ev"], f.O["ev"]), "eval indices differ from fit12 dumps"
f.O = dict(Le=O["Le"], La=O["La"], Lr=O["Lr"], ev=O["ev"])
def gen_ret(r):
    p = F.softmax(O["Lg"] / 2, 1); kl = (p * (F.log_softmax(O["Lg"] / 2, 1) - F.log_softmax(r["Lg"] / 2, 1))).sum(1).mean().item()
    return (r["Lg"].argmax(1) == O["Lg"].argmax(1)).float().mean().item(), kl
names = list(D); pubn = [k for k in names if k in P]; print(f"{len(names)} runs, {len(pubn)} with public score")
W = [w for w in itertools.product([0, .25, .5, .75, 1], repeat=len(f.B)) if abs(sum(w) - 1) < 1e-6]
G = [(w, dc, do, "pool") for w in W for dc in [0, 1, 2, 4, 8, 12] for do in [0, 0.5, 1, 2]]
golds = [f.gold(*g) for g in G]
Fm = torch.tensor([[f.forget(D[k], gd)[0] for k in names] for gd in golds])
RT = {k: gen_ret(D[k]) for k in names}; ag = torch.tensor([RT[k][0] for k in names]); kl = torch.tensor([RT[k][1] for k in names])
pi = torch.tensor([names.index(k) for k in pubn]); pub = torch.tensor([P[k] for k in pubn]); fits = []
for aa in [0, 1, 2, 4]:
    for c in [0, 0.25, 0.5, 1, 2, 4]:
        pred = Fm * (ag ** aa / (1 + c * kl)); pp = pred[:, pi]
        s = (pp @ pub) / (pp * pp).sum(1).clamp_min(1e-12); rms = ((s[:, None] * pp - pub) ** 2).mean(1).sqrt()
        for gi in range(len(G)): fits.append((rms[gi].item(), gi, aa, c, s[gi].item()))
fits.sort(); top = fits[: a.top]
print("best fits:", [(round(r, 4), G[g][:3], aa, c, round(s, 2)) for r, g, aa, c, s in top[:5]])
rows = []
for j, k in enumerate(names):
    ps = [s * Fm[g, j].item() * ag[j].item() ** aa / (1 + c * kl[j].item()) for r, g, aa, c, s in top]
    rows.append((sum(ps) / len(ps), min(ps), max(ps), k))
rows.sort(reverse=True)
print(f"{'run':16s} pred [min max]  public | gen agree / KL | forget(best fit)")
for m, lo, hi, k in rows:
    j = names.index(k); print(f"{k:16s} {m:.3f} [{lo:.3f} {hi:.3f}] {P.get(k, float('nan')):.4f} | {ag[j]:.3f} / {kl[j]:.3f} | {Fm[top[0][1], j]:.3f}")
