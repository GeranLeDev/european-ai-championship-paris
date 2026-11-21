"""Fast variant: train only the last k blocks (+norm+head) on cached tokens (cache_feats.py). Same methods/targets as unlearn.py.
Eval: forget clean (all 3500), forget light-aug held-out view ('altered copies'), retain proxy val (Imagenette)."""
import argparse, json, time, copy
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import *; import scoring

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="td", choices=["td", "scrub", "ga", "none"])
ap.add_argument("--target", default="medium"); ap.add_argument("--select", default=None)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--steps", type=int, default=400); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--bs-f", type=int, default=64); ap.add_argument("--bs-r", type=int, default=64)
ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--w-ce", type=float, default=0.0)
ap.add_argument("--ga-cap", type=float, default=3.0); ap.add_argument("--scrub-T", type=float, default=4.0)
ap.add_argument("--eval-every", type=int, default=50); ap.add_argument("--n-eval", type=int, default=3500)
ap.add_argument("--device"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="runs/fast_tmp")
a = ap.parse_args()
dev = pick_device(a.device); set_device(dev); torch.manual_seed(a.seed)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
T = torch.load(CACHE / f"tokens_k{a.k}.pt", mmap=True); cut = 12 - a.k
L0 = torch.load(CACHE / "orig_forget_logits.pt"); y = L0.argmax(1); N = len(y)
H = scoring.make_hypotheses(CACHE)
if a.target.startswith("soft:"):
    tau = float(a.target.split(":")[1]); q = F.softmax(L0 / (2 * tau), 1); ce_t = F.cross_entropy(L0 / tau, y, reduction="none")[:, None]
else:
    q, ce_t = H[a.target]["p2"], H[a.target]["ce"]
ce_t_mean = ce_t.mean(1); sel = a.select or (a.target if a.target in H else "medium")

full = load_orig(dev)


class Tail(nn.Module):
    def __init__(s, m):
        super().__init__(); s.blocks = nn.Sequential(*[copy.deepcopy(b) for b in m.blocks[cut:]]); s.norm = copy.deepcopy(m.norm); s.m = m
        s.head = copy.deepcopy(m.head)
    def forward(s, h):
        h = s.norm(s.blocks(h.float())); return s.head(h[:, 0])  # timm vit: cls-token pooling, fc_norm = Identity


orig_t = Tail(full).eval(); orig_t.requires_grad_(False); orig_t.m = None
sub = Tail(full); sub.m = None; sub.requires_grad_(True)
del full
params = list(sub.parameters())
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))


@torch.no_grad()
def run(model, toks, bs=128):
    model.eval(); return torch.cat([model(toks[i:i + bs].to(dev)).float().cpu() for i in range(0, len(toks), bs)])


ev = torch.arange(N) if a.n_eval >= N else torch.randperm(N, generator=torch.Generator().manual_seed(123))[: a.n_eval]
Lo_e, Lo_a, Lo_r = run(orig_t, T["f_clean"][ev]), run(orig_t, T["f_eval"][ev]), run(orig_t, T["r_val"])
assert (Lo_e - L0[ev]).abs().max() < 0.5, "cached tokens inconsistent with orig logits"
print("orig on altered-proxy: acc %.3f" % (Lo_a.argmax(1) == y[ev]).float().mean())
Lo_rtrain = run(orig_t, T["r_train"])


def evaluate(model):
    Le, La, Lr = run(model, T["f_clean"][ev]), run(model, T["f_eval"][ev]), run(model, T["r_val"])
    ret = scoring.retain_scores(Lr, Lo_r); res = {"retain": ret}
    for h, G in H.items():
        Gs = dict(p2=G["p2"][ev], ce=G["ce"][ev])
        g1 = scoring.group_scores(Le, Lo_e, y[ev], Gs); g2 = scoring.group_scores(La, Lo_a, y[ev], Gs)
        res[h] = dict(score=scoring.full_score([g1, g2], ret), clean=g1, altered=g2)
    return res


def fmt(step, res):
    c = res["medium"]["clean"]; al = res["medium"]["altered"]
    s = f"step {step:4d} | ret agree {res['retain']['agree']:.3f} KL {res['retain']['kl']:.4f} R {res['retain']['retain']:.3f} | "
    s += f"F acc {c['acc']:.3f} CE {c['ce_mean']:.2f}±{c['ce_std']:.2f} alt CE {al['ce_mean']:.2f} | "
    s += " ".join(f"{h}: {res[h]['score']['final']:.3f}(ms{res[h]['score']['ms']:.2f} ps{res[h]['score']['ps']:.2f})" for h in H)
    return s


kl2 = lambda Ls, p: (p * (p.clamp_min(1e-12).log() - F.log_softmax(Ls / 2, 1))).sum(1).mean()
views = ["f_clean", "f_aug1", "f_aug2"]
hist, best = [], (-1, 0); t0 = time.time(); stopped_max = False
res = evaluate(sub); hist.append(dict(step=0, **res)); print(fmt(0, res), flush=True)
for step in range(1, (a.steps if a.method != "none" else 0) + 1):
    sub.train()
    fi = torch.randint(0, N, (a.bs_f,)); v = torch.randint(0, 3, (a.bs_f,))
    hf = torch.stack([T[views[j]][i] for i, j in zip(fi.tolist(), v.tolist())]).to(dev)
    ri = torch.randint(0, len(Lo_rtrain), (a.bs_r,)); hr = T["r_train"][ri].to(dev); pr = F.softmax(Lo_rtrain[ri].to(dev) / 2, 1)
    if a.method == "scrub":
        if step % 2 == 1 and not stopped_max:
            with torch.no_grad(): tf = orig_t(hf)
            Lf = sub(hf); Ts = a.scrub_T
            loss = -F.kl_div(F.log_softmax(Lf / Ts, 1), F.softmax(tf / Ts, 1), reduction="batchmean") * Ts * Ts
        else:
            loss = a.alpha * kl2(sub(hr), pr)
    else:
        Lall = sub(torch.cat([hf, hr])); Lf, Lr = Lall[: a.bs_f], Lall[a.bs_f:]
        if a.method == "td":
            loss_f = kl2(Lf, q[fi].to(dev))
            if a.w_ce > 0:
                loss_f = loss_f + a.w_ce * ((F.cross_entropy(Lf, y[fi].to(dev), reduction="none") - ce_t_mean[fi].to(dev)) ** 2).mean()
        else:
            loss_f = -F.cross_entropy(Lf, y[fi].to(dev), reduction="none").clamp(max=a.ga_cap).mean()
        loss = loss_f + a.alpha * kl2(Lr, pr)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    if not torch.isfinite(loss): print("non-finite loss"); break
    if step % a.eval_every == 0 or step == a.steps:
        res = evaluate(sub); hist.append(dict(step=step, **res)); print(fmt(step, res) + f" | {time.time()-t0:.0f}s", flush=True)
        if a.method == "scrub" and res["medium"]["clean"]["ce_mean"] >= H[sel]["ce"].mean().item(): stopped_max = True
        if res[sel]["score"]["final"] > best[0]:
            best = (res[sel]["score"]["final"], step)
            torch.save({k: v.detach().cpu() for k, v in sub.state_dict().items() if not k.startswith("m.")}, out / "best_tail.pt")
json.dump(dict(args=vars(a), hist=hist, best=best), open(out / "hist.json", "w"), indent=1)
print(f"BEST[{sel}] final={best[0]:.4f} at step {best[1]}")
