"""Task 3 unlearning: several methods + proxy evaluation. Runs on mps (Mac) or cuda (JURECA A100).
Methods
  td     target distillation: on forget images (and light-augmented copies) match a gold-proxy target q (T=2 softmax);
         on retain proxy (Imagenette) match orig (KL at T=2). Optional CE-level term.
  scrub  SCRUB-style: maximise KL(orig||sub) on forget (alternating max/min steps), minimise KL on retain; early stop /
         rewind on the forget loss statistic (SCRUB+R).
  ga     NegGrad+: gradient ascent on pseudo-label CE (clamped at --ga-cap) + retain distillation.
Targets (--target): medium | strong | cls (medium + full removal of the 9 capped classes) | soft:<tau> (orig logits / tau)
Example: python unlearn.py --method td --target medium --blocks 12 --steps 300 --device cuda --out runs/td_med
"""
import argparse, json, time, copy
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from common import *; import augs, scoring

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="td", choices=["td", "scrub", "ga", "er"])
ap.add_argument("--target", default="medium")
ap.add_argument("--also", default="", help="comma list of extra hypotheses to report at each eval")
ap.add_argument("--select", default=None, help="hypothesis used to pick best checkpoint (default: same as target)")
ap.add_argument("--blocks", type=int, default=12, help="train last k blocks (+norm+head); 12 = everything")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs-f", type=int, default=32); ap.add_argument("--bs-r", type=int, default=32)
ap.add_argument("--alpha", type=float, default=1.0, help="retain KL weight")
ap.add_argument("--w-ce", type=float, default=0.0, help="td: weight of (CE - target CE)^2 term")
ap.add_argument("--w-stat", type=float, default=0.0, help="td: match batch mean/std of CE to gold-proxy mean/std")
ap.add_argument("--p-aug", type=float, default=0.5, help="prob. of light aug on forget batch")
ap.add_argument("--ga-cap", type=float, default=3.0)
ap.add_argument("--scrub-T", type=float, default=4.0)
ap.add_argument("--eval-every", type=int, default=50)
ap.add_argument("--n-eval", type=int, default=700); ap.add_argument("--n-eval-r", type=int, default=500)
ap.add_argument("--amp", action="store_true")
ap.add_argument("--device"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="runs/tmp")
ap.add_argument("--init", default=None, help="warm start: state dict to load into sub (orig stays the reference)")
ap.add_argument("--retain", default="imagenette", choices=["imagenette", "cutmix", "both"], help="retain proxy: Imagenette (OOD) and/or CutMix composites of forget images (in-distribution), target = orig")
ap.add_argument("--retain-file", default="retain_train_u8.npy", help="retain proxy images in cache/ (e.g. retain_ddb_u8.npy = DiffusionDB)")
ap.add_argument("--faug", default="light", choices=["light", "wide"], help="alteration family applied to forget views")
ap.add_argument("--reinit", default="", help="er: comma list of blocks to re-initialise (head is always re-initialised)")
ap.add_argument("--reinit-mlp", action="store_true", help="er: re-initialise only the MLP sub-blocks (norm2, mlp, ls2) of --reinit blocks")
ap.add_argument("--save-last", action="store_true", help="keep the last evaluated checkpoint instead of the best under --select")
ap.add_argument("--dump", action="store_true", help="dump eval logits of the best checkpoint to ../output/logits/<run>.pt")
a = ap.parse_args()
dev = pick_device(a.device); set_device(dev); torch.manual_seed(a.seed); np.random.seed(a.seed)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
amp_dtype = torch.bfloat16 if dev == "cuda" else torch.float16
autocast = lambda: torch.autocast(device_type=dev if dev != "mps" else "mps", dtype=amp_dtype, enabled=a.amp)

# ---------------- data
X = load_forget(); L0 = torch.load(CACHE / "orig_forget_logits.pt"); y = L0.argmax(1); N = len(X)
R = torch.cat([torch.from_numpy(np.load(CACHE / f)) for f in a.retain_file.split(",")]); RV = torch.from_numpy(np.load(CACHE / "retain_val_u8.npy"))
H = scoring.make_hypotheses(CACHE)
if a.method == "er":   # erase + repair: forget outputs are not imposed, they emerge; target only used for reporting
    a.target = a.select or "mixcls"
if a.target.startswith("soft:"):
    tau = float(a.target.split(":")[1]); q = F.softmax(L0 / (2 * tau), 1); ce_t = F.cross_entropy(L0 / tau, y, reduction="none")[:, None]
elif a.target == "mix":   # hedges (see feat_unlearn.py)
    A = H["medium"] if a.target == "mix" else H["cls"]
    q = 0.5 * (A["p2"] + H["strong"]["p2"]); ce_t = torch.cat([A["ce"], H["strong"]["ce"]], 1)
else:
    scoring.resolve(a.target, H, CACHE)   # extra proxy file, or composite spec cls:<base>:<off> / mix:<A>+<B>
    q = H[a.target]["p2"]; ce_t = H[a.target]["ce"]
ce_t_mean = ce_t.mean(1); g_mean, g_std = ce_t.mean().item(), ce_t.std().item()
sel = a.select or (a.target if a.target in H else "strong")
for extra in a.also.split(","):
    if extra: scoring.resolve(extra, H, CACHE)

# ---------------- models
orig = load_orig(dev); orig.requires_grad_(False)
sub = copy.deepcopy(orig)
if a.init:
    sub.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True), strict=True)
if a.method == "er":
    fresh = tt.build_model().to(dev)
    for i in [int(b) for b in a.reinit.split(",") if b]:
        if a.reinit_mlp:
            for nm in ["norm2", "mlp", "ls2"]: getattr(sub.blocks[i], nm).load_state_dict(getattr(fresh.blocks[i], nm).state_dict())
        else: sub.blocks[i].load_state_dict(fresh.blocks[i].state_dict())
    sub.head.load_state_dict(fresh.head.state_dict()); del fresh
sub.requires_grad_(False)
train_mods = [sub.blocks[i] for i in range(12 - a.blocks, 12)] + [sub.norm, sub.head]
if a.blocks >= 12:
    train_mods = [sub]
for mod in train_mods:
    mod.requires_grad_(True)
params = [p for p in sub.parameters() if p.requires_grad]
print(f"trainable params {sum(p.numel() for p in params)/1e6:.1f}M", flush=True)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))

# ---------------- fixed eval sets
g = torch.Generator().manual_seed(123)
ev = torch.randperm(N, generator=g)[: a.n_eval]
torch.manual_seed(999)
Xe = to_input(X[ev], "cpu"); Xa = torch.cat([augs.light(Xe[i:i + 64]) for i in range(0, len(Xe), 64)])  # hidden 'altered copies' proxy
RVe = RV[: a.n_eval_r]
torch.manual_seed(a.seed)


@torch.no_grad()
def logits_t(model, xt, bs=64):
    model.eval(); o = []
    for i in range(0, len(xt), bs):
        with autocast():
            o.append(model(xt[i:i + bs].to(dev)).float().cpu())
    return torch.cat(o)


Lo_e, Lo_a, Lo_r = logits_t(orig, Xe), logits_t(orig, Xa), logits_of(orig, RVe, dev)


def evaluate(model):
    Le, La, Lr = logits_t(model, Xe), logits_t(model, Xa), logits_of(model, RVe, dev)
    ret = scoring.retain_scores(Lr, Lo_r); res = {"retain": ret}
    for h, G in H.items():
        Gs = dict(p2=G["p2"][ev], ce=G["ce"][ev])
        g1 = scoring.group_scores(Le, Lo_e, y[ev], Gs); g2 = scoring.group_scores(La, Lo_a, y[ev], Gs)
        res[h] = dict(score=scoring.full_score([g1, g2], ret), clean=g1, altered=g2)
    return res


def fmt(step, res):
    s = f"step {step:4d} | retain agree {res['retain']['agree']:.3f} KL {res['retain']['kl']:.4f} | "
    c = res["medium"]["clean"]; s += f"forget acc {c['acc']:.3f} CE {c['ce_mean']:.3f}±{c['ce_std']:.3f} | "
    s += " ".join(f"{h}: F={res[h]['score']['final']:.3f}(ms {res[h]['score']['ms']:.2f} ps {res[h]['score']['ps']:.2f})" for h in H)
    return s


hist = []; best = (-1, None)
res = evaluate(sub); hist.append(dict(step=0, **res)); print(fmt(0, res), flush=True)
kl2 = lambda Ls, p: (p * (p.clamp_min(1e-12).log() - F.log_softmax(Ls / 2, 1))).sum(1).mean()
t0 = time.time(); stopped_max = False
for step in range(1, a.steps + 1):
    sub.train()
    fi = torch.randint(0, N, (a.bs_f,)); ri = torch.randint(0, len(R), (a.bs_r,))
    xf = to_input(X[fi], dev)
    if a.p_aug > 0:
        m = torch.rand(len(fi), device=dev) < a.p_aug
        if m.any(): xf[m] = getattr(augs, a.faug)(xf[m])
    xr = augs.medium(to_input(R[ri], dev))
    if a.retain != "imagenette":   # in-distribution retain: CutMix of two medium-augmented forget images
        nc = a.bs_r if a.retain == "cutmix" else a.bs_r // 2
        ia = torch.randint(0, N, (nc,)); ib = torch.randint(0, N, (nc,))
        xa = augs.medium(to_input(X[ia], dev)); xb = augs.medium(to_input(X[ib], dev))
        for j in range(nc):
            fr = np.random.uniform(0.3, 0.7); hh = int(224 * np.sqrt(fr)); ww = int(224 * fr / np.sqrt(fr)); y0 = np.random.randint(0, 224 - hh + 1); x0 = np.random.randint(0, 224 - ww + 1)
            xa[j, :, y0:y0 + hh, x0:x0 + ww] = xb[j, :, y0:y0 + hh, x0:x0 + ww]
        xr = xa if a.retain == "cutmix" else torch.cat([xr[: a.bs_r - nc], xa])
    with torch.no_grad(), autocast():
        tr = orig(xr).float()
        tf = orig(xf).float() if a.method == "scrub" else None
    with autocast():
        if a.method == "scrub":
            is_max = (step % 2 == 1) and not stopped_max
            if is_max:
                Lf = sub(xf).float(); T = a.scrub_T
                loss = -F.kl_div(F.log_softmax(Lf / T, 1), F.softmax(tf / T, 1), reduction="batchmean") * T * T
            else:
                Lr = sub(xr).float(); loss = a.alpha * kl2(Lr, F.softmax(tr / 2, 1))
        elif a.method == "er":
            loss = kl2(sub(xr).float(), F.softmax(tr / 2, 1))
        else:
            Lall = sub(torch.cat([xf, xr])).float(); Lf, Lr = Lall[: len(fi)], Lall[len(fi):]
            loss_r = kl2(Lr, F.softmax(tr / 2, 1))
            if a.method == "td":
                loss_f = kl2(Lf, q[fi].to(dev))
                if a.w_ce > 0:
                    loss_f = loss_f + a.w_ce * ((F.cross_entropy(Lf, y[fi].to(dev), reduction="none") - ce_t_mean[fi].to(dev)) ** 2).mean()
                if a.w_stat > 0:
                    cef = F.cross_entropy(Lf, y[fi].to(dev), reduction="none")
                    loss_f = loss_f + a.w_stat * ((cef.mean() - g_mean) ** 2 + (cef.std() - g_std) ** 2)
            else:  # ga
                loss_f = -F.cross_entropy(Lf, y[fi].to(dev), reduction="none").clamp(max=a.ga_cap).mean()
            loss = loss_f + a.alpha * loss_r
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    if not torch.isfinite(loss):
        print("non-finite loss, stop"); break
    if step % a.eval_every == 0 or step == a.steps:
        res = evaluate(sub); hist.append(dict(step=step, **res))
        print(fmt(step, res) + f" | {time.time()-t0:.0f}s", flush=True)
        if a.method == "scrub" and res["medium"]["clean"]["ce_mean"] >= H[sel]["ce"].mean().item():
            stopped_max = True
        f = res[sel]["score"]["final"]
        if f > best[0] or a.save_last:
            best = (f, step); torch.save({k: v.detach().cpu().float() for k, v in sub.state_dict().items()}, out / "best.tmp"); (out / "best.tmp").replace(out / "best.pt")
            if a.dump:
                ld = ROOT / "output" / "logits"; torch.save(dict(Le=logits_t(sub, Xe), La=logits_t(sub, Xa), Lr=logits_of(sub, RVe, dev)), ld / f".{out.name}.tmp"); (ld / f".{out.name}.tmp").replace(ld / f"{out.name}.pt")
json.dump(dict(args=vars(a), hist=hist, best=best), open(out / "hist.json", "w"), indent=1)
print(f"BEST {sel} final={best[0]:.4f} at step {best[1]}  -> {out/'best.pt'}")
