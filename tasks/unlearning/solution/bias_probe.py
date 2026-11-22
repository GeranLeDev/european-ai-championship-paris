"""Head-bias probe (no training): subtract delta from the head bias of the 9 capped classes of a submitted run.
Predicted public score under (a) the top-30 hypotheses of fit12 and (b) the best 'full removal' hypotheses (dcap in {8,12,20}).
Usage: python bias_probe.py lg_g0"""
import sys, itertools, math, torch, torch.nn.functional as F
import fit_rm9 as R; from fit12 import *
run = sys.argv[1]; r0 = torch.load(LD / f"{run}.pt"); FC = torch.tensor(scoring.FULL_CLASSES)
sets = {"top30(fit12)": [( (g[:3], "own"), rt, s) for (g, rt), s, _ in torch.load("fit12.pt")["top"]]}
for mode in ["own", "all"]:
    for dc in [8, 12, 20]:
        rows = []
        for w in R.W:
            for do in [0, 1, 2]:
                gg = R.gold9(w, dc, do, mode); Fm = torch.tensor([forget(R.runs[k], gg)[0] for k in PUBLIC])
                for aa, c in R.RET:
                    Rr = torch.tensor([R.ret[k]["agree"] ** aa / (1 + c * R.ret[k]["kl"]) for k in PUBLIC]); X = Fm * Rr
                    s = (X @ R.pub / (X @ X).clamp_min(1e-12)).item(); rows.append((((s * X - R.pub) ** 2).mean().sqrt().item(), ((w, dc, do), mode), (aa, c), s))
        rows.sort(key=lambda t: t[0]); sets[f"{mode} dcap={dc} (rms {rows[0][0]:.3f})"] = [(g, rt, s) for _, g, rt, s in rows[:5]]
        print(mode, dc, "best:", rows[0], flush=True)
print(f"\n{run}: predicted public vs delta (mean over hypotheses of each set)")
D = [0, 1, 2, 3, 4, 6, 8, 12]; print("set".ljust(30) + " ".join(f"d={d:<5}" for d in D))
for name, hs in sets.items():
    line = []
    for d in D:
        r = {k: v.clone() for k, v in r0.items()}
        for k in r: r[k][:, FC] -= d
        ret = scoring.retain_scores(r["Lr"], O["Lr"])
        line.append(sum(s * forget(r, R.gold9(*g[0], g[1]))[0] * ret["agree"] ** aa / (1 + c * ret["kl"]) for g, (aa, c), s in hs) / len(hs))
    print(name.ljust(30) + " ".join(f"{v:.3f}  " for v in line))
