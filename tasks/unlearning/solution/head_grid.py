"""Fast exploration on cached CLS features: train norm+head only (CPU ok). Runs a list of configs, prints proxy scores.
Also prints the 'oracle softening' reference (orig logits / tau applied only on forget images)."""
import argparse, copy, json, itertools, torch, torch.nn as nn, torch.nn.functional as F
from common import *; import scoring
torch.set_num_threads(4)
C = torch.load(CACHE / "cls_feats.pt"); L0 = torch.load(CACHE / "orig_forget_logits.pt"); y = L0.argmax(1); N = len(y)
H = scoring.make_hypotheses(CACHE)
m = load_orig("cpu"); base = nn.Sequential(copy.deepcopy(m.norm), copy.deepcopy(m.head)).eval(); del m
with torch.no_grad():
    Lo = {k: base(v) for k, v in C.items()}
print("orig acc on altered proxy %.3f" % (Lo["f_eval"].argmax(1) == y).float().mean())
for h, G in H.items():
    print(f"[{h}] gold proxy CE mean {G['ce'].mean():.3f} std {G['ce'].std():.3f} | gold argmax==orig {(G['p2'].argmax(1)==y).float().mean():.3f}")


def score(Le, La, Lr):
    ret = scoring.retain_scores(Lr, Lo["r_val"]); r = {"retain": ret}
    for h, G in H.items():
        g1 = scoring.group_scores(Le, Lo["f_clean"], y, G); g2 = scoring.group_scores(La, Lo["f_eval"], y, G)
        r[h] = dict(score=scoring.full_score([g1, g2], ret), clean=g1, alt=g2)
    return r


def line(tag, r):
    c = r["medium"]["clean"]
    return (f"{tag:34s} R={r['retain']['retain']:.3f}(agr {r['retain']['agree']:.3f} kl {r['retain']['kl']:.3f}) "
            f"Facc={c['acc']:.3f} CE={c['ce_mean']:.2f}±{c['ce_std']:.2f} altCE={r['medium']['alt']['ce_mean']:.2f} | "
            + " ".join(f"{h}={r[h]['score']['final']:.3f}[{r[h]['score']['ms']:.2f}/{r[h]['score']['ps']:.2f}]" for h in H))


print("--- oracle softening reference (no retain cost) ---")
for tau in [1, 2, 3, 4, 6, 8, 12]:
    print(line(f"soften tau={tau}", score(Lo["f_clean"] / tau, Lo["f_eval"] / tau, Lo["r_val"])))


def train(method, target, steps=600, lr=1e-3, alpha=1.0, bs=256, cap=3.0, w_ce=0.0, T=4.0, eval_every=100, seed=0):
    torch.manual_seed(seed)
    if target.startswith("soft:"):
        tau = float(target[5:]); q = F.softmax(L0 / (2 * tau), 1); ce_t = F.cross_entropy(L0 / tau, y, reduction="none")
    else:
        q = H[target]["p2"]; ce_t = H[target]["ce"].mean(1)
    sub = copy.deepcopy(base).train(); opt = torch.optim.Adam(sub.parameters(), lr=lr)
    views = torch.stack([C["f_clean"], C["f_aug1"], C["f_aug2"]]); R = C["r_train"]; pR = F.softmax(Lo["r_train"] / 2, 1)
    kl2 = lambda Ls, p: (p * (p.clamp_min(1e-12).log() - F.log_softmax(Ls / 2, 1))).sum(1).mean()
    best = None; traj = []
    for s in range(1, steps + 1):
        fi = torch.randint(0, N, (bs,)); v = torch.randint(0, 3, (bs,)); hf = views[v, fi]
        ri = torch.randint(0, len(R), (bs,)); Lr = sub(R[ri]); lr_ = kl2(Lr, pR[ri])
        Lf = sub(hf)
        if method == "td":
            lf = kl2(Lf, q[fi]) + (w_ce * ((F.cross_entropy(Lf, y[fi], reduction="none") - ce_t[fi]) ** 2).mean() if w_ce else 0)
        elif method == "ga":
            lf = -F.cross_entropy(Lf, y[fi], reduction="none").clamp(max=cap).mean()
        elif method == "scrub":
            with torch.no_grad(): tf = base(hf)
            lf = -F.kl_div(F.log_softmax(Lf / T, 1), F.softmax(tf / T, 1), reduction="batchmean") * T * T
            if s % 2 == 0: lf = lf * 0  # alternate max / min steps
        loss = lf + alpha * lr_
        opt.zero_grad(); loss.backward(); opt.step()
        if s % eval_every == 0:
            with torch.no_grad():
                r = score(sub(C["f_clean"]), sub(C["f_eval"]), sub(C["r_val"]))
            traj.append((s, r))
    return traj


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--configs", default="default"); a = ap.parse_args()
    cfgs = [
        dict(method="td", target="medium"), dict(method="td", target="strong"), dict(method="td", target="cls"),
        dict(method="td", target="soft:3"), dict(method="td", target="medium", w_ce=0.1),
        dict(method="td", target="medium", alpha=10.0),
        dict(method="ga", target="medium", cap=1.0), dict(method="ga", target="medium", cap=3.0),
        dict(method="scrub", target="medium", lr=1e-4),
    ]
    allres = {}
    for c in cfgs:
        tag = ",".join(f"{k}={v}" for k, v in c.items())
        print(f"--- {tag}", flush=True)
        traj = train(**c)
        for s, r in traj:
            print(line(f"  step {s}", r), flush=True)
        allres[tag] = [(s, {h: r[h]["score"] for h in H} | {"retain": r["retain"]}) for s, r in traj]
    json.dump(allres, open(CACHE.parent / "runs" / "head_grid.json", "w"), indent=1)
