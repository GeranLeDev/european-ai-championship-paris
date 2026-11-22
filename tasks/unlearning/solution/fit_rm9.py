"""Variant of fit12: class suppression applied to ALL 9 capped class columns (gold never predicts any of them), for capped
images only ('cap') or every image ('all'), vs own-label-only shift ('own' = fit12). Reports rms / scale per mode."""
import itertools, math, os, glob, torch, torch.nn.functional as F
import fit12; from fit12 import *
FC = torch.tensor(scoring.FULL_CLASSES)
runs = {os.path.basename(f)[:-3]: torch.load(f) for f in sorted(glob.glob(str(LD / '*.pt'))) if not f.endswith('orig.pt') and not os.path.basename(f).startswith('e_')}
names = list(runs)


def gold9(w, dcap, doth, mode):
    p2, gm, gs = gold(w, dcap, doth)          # own-label shift (CE stats come from here)
    if mode == "own": return p2, gm, gs
    p2 = sum(wi * B[b]["p2"] for wi, b in zip(w, B) if wi > 0).clone()
    rows = cap if mode == "cap" else torch.ones_like(cap)
    sub = p2[rows]; sub[:, FC] = sub[:, FC] * math.exp(-dcap / 2); p2[rows] = sub
    d = torch.where(cap, torch.tensor(0.0), torch.tensor(float(doth))) if mode == "cap" else torch.zeros(N)
    nc = ~cap; p2[ar[nc], y[nc]] = p2[ar[nc], y[nc]] * math.exp(-doth / 2)
    return p2 / p2.sum(1, keepdim=True), gm, gs


W = [w for w in itertools.product([0, .25, .5, .75, 1], repeat=len(B)) if abs(sum(w) - 1) < 1e-6]
DC = [2, 4, 6, 8, 12, 20]; DO = [0, 1, 2]; RET = [(aa, c) for aa in [0, 1, 2] for c in [0, 0.5, 1, 2, 4]]
ret = {k: scoring.retain_scores(runs[k]["Lr"], O["Lr"]) for k in names}
Rm = torch.tensor([[ret[k]["agree"] ** aa / (1 + c * ret[k]["kl"]) for k in names] for aa, c in RET])
pi = torch.tensor([names.index(k) for k in PUBLIC]); pub = torch.tensor(list(PUBLIC.values()))
if __name__ == '__main__':
    out = {}
    for mode in ["own", "cap", "all"]:
        G = [(w, dc, do, mode) for w in W for dc in DC for do in DO]; Fm = torch.zeros(len(G), len(names))
        for i, g in enumerate(G):
            gg = gold9(*g)
            for j, k in enumerate(names): Fm[i, j] = forget(runs[k], gg)[0]
        P = (Fm[:, None, :] * Rm[None]).reshape(-1, len(names)); hyp = [(g, r) for g in G for r in RET]
        X = P[:, pi]; s = (X @ pub) / (X * X).sum(1).clamp_min(1e-12); res = ((s[:, None] * X - pub) ** 2).mean(1).sqrt(); o = res.argsort()
        print(f"== mode {mode}: best rms {res[o[0]]:.4f}")
        for i in o[:8].tolist():
            print(f"  rms={res[i]:.4f} s={s[i]:.2f} w={hyp[i][0][0]} dcap={hyp[i][0][1]} doth={hyp[i][0][2]} ret={hyp[i][1]} lg_g0 proxy={P[i, names.index('lg_g0')]:.3f}")
        d = {}
        for i, h in enumerate(hyp): d[h[0][1]] = min(d.get(h[0][1], 9), res[i].item())
        print("  best rms by dcap:", " ".join(f"{k}:{v:.4f}" for k, v in sorted(d.items())))
        out[mode] = [(hyp[i], s[i].item(), res[i].item()) for i in o[:30].tolist()]
    torch.save(out, "fit_rm9.pt")
