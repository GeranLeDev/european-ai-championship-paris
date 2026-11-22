"""Fit of a continuous family of gold hypotheses on ALL public scores, with leave-one-out validation.
gold(theta): p2 = sum_b w_b p2_b over bases (medium,strong,xstrong); logit shift -d_cap on the pseudo-label for images of the
9 capped classes, -d_oth for the others; CE stats pooled over views (or per-image mean, --cemode).
retain(theta) = agree^a / (1 + c*KL) (proxy retain is OOD: its weight is a free parameter). public ~ s * forget * retain.
Usage: python fit12.py [--save fit12.pt]   (needs cache/logits/*.pt)"""
import argparse, glob, os, itertools, math, torch, torch.nn.functional as F
from common import CACHE; import scoring
PUBLIC = {"r2_g0": 0.0798, "r2_g1": 0.2384, "r2_g2": 0.2616, "m1_g2": 0.2736, "ma_g1": 0.2806, "ma_g0": 0.2893, "m3_g2": 0.2046,
          "m3_g3": 0.2080, "st_g3": 0.3043, "st_g1": 0.3025, "lg_g0": 0.3045, "lg_g1": 0.3019, "lg0_bias4": 0.2348, "lg0_tailfitB": 0.2827}
LD = CACHE / "logits"; O = torch.load(LD / "orig.pt"); ev = O["ev"]; y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1)[ev]
B = {b: torch.load(CACHE / f"gold_{b}.pt") for b in os.environ.get("FIT_BASES", "medium,strong,xstrong").split(",")}
B = {b: dict(p2=G["p2"][ev], ce=G["ce"][ev]) for b, G in B.items()}
cap = torch.isin(y, torch.tensor(scoring.FULL_CLASSES)); N = len(y); ar = torch.arange(N)


def gold(w, dcap, doth, cemode="pool"):
    p2 = sum(wi * B[b]["p2"] for wi, b in zip(w, B) if wi > 0)
    d = torch.where(cap, torch.tensor(float(dcap)), torch.tensor(float(doth)))
    p2 = p2.clone(); p2[ar, y] = p2[ar, y] * torch.exp(-d / 2); p2 = p2 / p2.sum(1, keepdim=True)
    m1 = 0; m2 = 0
    for wi, b in zip(w, B):
        if wi == 0: continue
        ce = B[b]["ce"]; p = torch.exp(-ce); ce = ce + d[:, None] + torch.log((1 - p + p * torch.exp(-d[:, None])).clamp_min(1e-30))
        if cemode == "img": ce = ce.mean(1, keepdim=True)
        m1 = m1 + wi * ce.mean().item(); m2 = m2 + wi * (ce ** 2).mean().item()
    return p2, m1, math.sqrt(max(m2 - m1 ** 2, 0))


def group(Ls, Lo, p2, gm, gs):
    ce_s = F.cross_entropy(Ls, y, reduction="none"); ce_o = F.cross_entropy(Lo, y, reduction="none")
    d = lambda ce: abs(ce.mean().item() - gm) + abs(ce.std().item() - gs)
    ms = 1 - min(max(d(ce_s) / max(d(ce_o), 1e-12), 0), 1)
    ks = scoring.kl_rows(p2, F.log_softmax(Ls / 2, 1)).mean().item(); ko = scoring.kl_rows(p2, F.log_softmax(Lo / 2, 1)).mean().item()
    return ms, 1 - min(max(ks / ko, 0), 1)


def forget(r, g):
    a1 = group(r["Le"], O["Le"], *g); a2 = group(r["La"], O["La"], *g)
    return (a1[0] + a2[0]) / 2 * (a1[1] + a2[1]) / 2, (a1[0] + a2[0]) / 2, (a1[1] + a2[1]) / 2




def predict(r, top):
    """scaled public prediction of logits dict r under each saved hypothesis -> list"""
    ret = scoring.retain_scores(r["Lr"], O["Lr"]); out = []
    for (g, (aa, c)), s, _ in top:
        out.append(s * forget(r, gold(*g))[0] * ret["agree"] ** aa / (1 + c * ret["kl"]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument('--save', default='fit12.pt'); a = ap.parse_args()
    runs = {os.path.basename(f)[:-3]: torch.load(f) for f in sorted(glob.glob(str(LD / '*.pt'))) if not f.endswith('orig.pt') and not os.path.basename(f).startswith(('e_', 't_'))}
    W = [w for w in itertools.product([0, .25, .5, .75, 1], repeat=len(B)) if abs(sum(w) - 1) < 1e-6]
    DC = [0, 1, 2, 3, 4, 6, 8, 12, 20]; DO = [0, 0.5, 1, 2] if len(B) == 3 else [0, 1, 2]; CM = ["pool", "img"] if len(B) == 3 else ["pool"]
    RET = [(aa, c) for aa in [0, 1, 2, 4] for c in [0, 0.5, 1, 2, 4, 8, 16]]
    names = list(runs); ret = {k: scoring.retain_scores(runs[k]["Lr"], O["Lr"]) for k in names}
    G = [(w, dc, do, cm) for w in W for dc in DC for do in DO for cm in CM]
    Fm = torch.zeros(len(G), len(names))
    for i, g in enumerate(G):
        gg = gold(*g)
        for j, k in enumerate(names): Fm[i, j] = forget(runs[k], gg)[0]
    Rm = torch.tensor([[ret[k]["agree"] ** aa / (1 + c * ret[k]["kl"]) for k in names] for aa, c in RET])
    P = (Fm[:, None, :] * Rm[None, :, :]).reshape(-1, len(names))            # [H, runs]
    hyp = [(g, r) for g in G for r in RET]
    pi = torch.tensor([names.index(k) for k in PUBLIC]); pub = torch.tensor(list(PUBLIC.values()))


    def fit(idx):   # best scale + rel RMS on public subset idx
        X = P[:, pi[idx]]; t = pub[idx]; s = (X @ t) / (X * X).sum(1).clamp_min(1e-12)
        res = ((s[:, None] * X - t) ** 2).mean(1).sqrt()
        return s, res


    s, res = fit(torch.arange(len(pub))); order = res.argsort()
    print(f"{len(hyp)} hypotheses, {len(pub)} public points. Top 15 (abs RMS):")
    for i in order[:15].tolist():
        print(f" rms={res[i]:.4f} s={s[i]:.2f} w(m,s,x)={hyp[i][0][0]} dcap={hyp[i][0][1]} doth={hyp[i][0][2]} ce={hyp[i][0][3]} ret(a,c)={hyp[i][1]}  pred=" + " ".join(f"{v:.3f}" for v in (s[i] * P[i, pi]).tolist()))
    # LOO: best hypothesis (and top-10 average) on 11 points -> predict the 12th
    err1 = []; err10 = []
    for j in range(len(pub)):
        idx = torch.tensor([k for k in range(len(pub)) if k != j]); sj, rj = fit(idx); o = rj.argsort()[:10]
        p1 = (sj[o[0]] * P[o[0], pi[j]]).item(); p10 = (sj[o] * P[o, pi[j]]).mean().item()
        err1.append(p1 - pub[j].item()); err10.append(p10 - pub[j].item())
        print(f" LOO {list(PUBLIC)[j]}: public {pub[j]:.4f} pred(best) {p1:.4f} pred(top10) {p10:.4f}")
    rm = lambda e: (sum(x * x for x in e) / len(e)) ** .5
    print(f"LOO RMSE best={rm(err1):.4f} top10={rm(err10):.4f}  (baseline predict-mean: {pub.std():.4f})")
    top = order[:30]
    pred = (s[top, None] * P[top]); mean = pred.mean(0); lo = pred.min(0).values; hi = pred.max(0).values
    print("\npredicted public for every run (top-30 hypotheses: mean [min,max]):")
    for j in mean.argsort(descending=True).tolist():
        k = names[j]; print(f" {k:10s} {mean[j]:.3f} [{lo[j]:.3f},{hi[j]:.3f}]" + (f"  public {PUBLIC[k]}" if k in PUBLIC else ""))
    # marginal: best rms per parameter value
    print("\nbest rms by parameter value:")
    for nm, key in [("w", lambda h: h[0][0]), ("dcap", lambda h: h[0][1]), ("doth", lambda h: h[0][2]), ("ce", lambda h: h[0][3]), ("ret", lambda h: h[1])]:
        d = {}
        for i, h in enumerate(hyp): d[key(h)] = min(d.get(key(h), 9), res[i].item())
        print(" ", nm, " ".join(f"{k}:{v:.4f}" for k, v in sorted(d.items(), key=lambda kv: kv[1])[:12]))
    torch.save(dict(top=[(hyp[i], s[i].item(), res[i].item()) for i in order[:30].tolist()]), a.save)
