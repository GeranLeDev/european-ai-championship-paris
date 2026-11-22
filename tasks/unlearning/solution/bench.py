"""Local test bed with a REAL gold (miniature of the task). World = the 3500 forget images with orig pseudo-labels.
  split : control (15%, stratified, never trained on) / forget' (3 whole classes + 20% of the others) / retain' (rest)
  train : ViT-B/16 (same builder) from scratch on a subset -> orig' (retain'+forget') or gold' (retain' only)
  world : builds ../bench/ with the same layout as the real task (model.pt = orig', forget_images.npz = forget', solution/cache/...)
          so unlearn.py / kfold_oof.py run unchanged inside ../bench/solution ; gold_true.pt = real gold' as a target ('oracle')
  eval  : exact score (docstring formulas) of checkpoints against the real gold', retain on the in-distribution control set
Usage (from solution/): python bench.py split ; python bench.py train --subset orig --out ../bench/model.pt ; python bench.py train --subset retain --seed 1 --out ../bench/gold1.pt ;
       python bench.py world ; python bench.py eval name=path ..."""
import argparse, os, shutil, sys, time, torch, torch.nn.functional as F
from common import *; import augs, scoring
BR = ROOT / "bench" if ROOT.name != "bench" else ROOT; REAL = ROOT if ROOT.name != "bench" else ROOT.parent
FULLB = [0, 7, 30]   # classes removed (almost) entirely in the bench forget'
cmd = sys.argv[1]; ap = argparse.ArgumentParser(); ap.add_argument("--subset", default="orig"); ap.add_argument("--out"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--bs", type=int, default=128); ap.add_argument("rest", nargs="*"); a = ap.parse_args(sys.argv[2:])
dev = pick_device(); set_device(dev); BR.mkdir(exist_ok=True)
Xall = tt.load_forget_images(REAL / "forget_images.npz"); yall = torch.load(REAL / "solution" / "cache" / "orig_forget_logits.pt").argmax(1)
ac = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
save = lambda obj, p: (torch.save(obj, str(p) + ".tmp"), os.replace(str(p) + ".tmp", p))

if cmd == "split":
    g = torch.Generator().manual_seed(0); N = len(yall); role = torch.zeros(N, dtype=torch.long)   # 0 retain', 1 forget', 2 control
    for c in yall.unique():
        idx = (yall == c).nonzero()[:, 0]; idx = idx[torch.randperm(len(idx), generator=g)]; nc = max(1, round(0.15 * len(idx))); role[idx[:nc]] = 2; rest = idx[nc:]
        role[rest[: len(rest) if c.item() in FULLB else round(0.2 * len(rest))]] = 1
    save(dict(role=role, full=FULLB), BR / "split.pt"); print("split: retain' %d forget' %d control %d" % tuple((role == i).sum() for i in range(3)))

elif cmd == "train":
    role = torch.load(BR / "split.pt")["role"]; idx = ((role < 2) if a.subset == "orig" else (role == 0)).nonzero()[:, 0]; torch.manual_seed(a.seed)
    m = tt.build_model().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1); t0 = time.time(); m.train()
    for s in range(a.steps):
        bi = idx[torch.randint(0, len(idx), (a.bs,))]; x = augs.color(augs.flip(augs.rrc(to_input(Xall[bi], dev), (0.5, 1.0))), 0.2)
        with ac(): loss = F.cross_entropy(m(x).float(), yall[bi].to(dev))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        if s % 250 == 0 or s == a.steps - 1: print(f"step {s} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    for nm, sel in [("train", idx[:600]), ("forget'", (role == 1).nonzero()[:, 0]), ("control", (role == 2).nonzero()[:, 0])]:
        L = logits_of(m, Xall[sel], dev); print(f"{a.subset} seed {a.seed} | {nm}: acc {(L.argmax(1)==yall[sel]).float().mean():.3f} CE {F.cross_entropy(L, yall[sel]):.3f} conf {F.softmax(L,1).max(1).values.mean():.3f}", flush=True)
    save({k: v.detach().cpu().float() for k, v in m.state_dict().items()}, a.out)

elif cmd == "world":   # needs model.pt (orig') ; gold*.pt optional (adds gold_true.pt)
    sol = BR / "solution"; (sol / "cache").mkdir(parents=True, exist_ok=True); role = torch.load(BR / "split.pt")["role"]; fi = (role == 1).nonzero()[:, 0]
    for f in ["common.py", "augs.py", "scoring.py", "unlearn.py", "kfold_oof.py", "gold_proxy.py", "bench.py"]: shutil.copy(REAL / "solution" / f, sol / f)
    shutil.copy(REAL / "task_template.py", BR / "task_template.py")
    if not (BR / "forget_images.npz").exists(): np.savez(BR / "forget_images.tmp.npz", images=Xall[fi].numpy()); os.replace(BR / "forget_images.tmp.npz", BR / "forget_images.npz")
    for f in ["retain_train_u8.npy", "retain_val_u8.npy", "retain_ddb_u8.npy"]:
        if (REAL / "solution" / "cache" / f).exists() and not (sol / "cache" / f).exists(): os.symlink(REAL / "solution" / "cache" / f, sol / "cache" / f)
    load = lambda p: (lambda mm: (mm.load_state_dict(torch.load(p, map_location="cpu", weights_only=True)), mm.to(dev).eval())[1])(tt.build_model())
    if not (sol / "cache" / "orig_forget_logits.pt").exists(): save(logits_of(load(BR / "model.pt"), Xall[fi], dev), sol / "cache" / "orig_forget_logits.pt")
    gs = sorted(BR.glob("gold[0-9].pt"))
    if gs:
        Ls = [logits_of(load(p), Xall[fi], dev) for p in gs]; y = yall[fi]
        save(dict(p2=sum(F.softmax(L / 2, 1) for L in Ls) / len(Ls), ce=torch.stack([F.cross_entropy(L, y, reduction="none") for L in Ls], 1)), sol / "cache" / "gold_true.pt")
    print("world ready:", len(fi), "forget' images,", len(gs), "gold models")

elif cmd == "eval":
    role = torch.load(BR / "split.pt")["role"]; fi = (role == 1).nonzero()[:, 0]; ci = (role == 2).nonzero()[:, 0]; y = yall[fi]; yc = yall[ci]; cap = torch.isin(y, torch.tensor(FULLB))
    RV = torch.from_numpy(np.load(REAL / "solution" / "cache" / "retain_val_u8.npy"))[:1000]
    torch.manual_seed(999); Xe = to_input(Xall[fi], "cpu"); Xa = torch.cat([augs.light(Xe[i:i + 64]) for i in range(0, len(Xe), 64)])
    def lt(m, xt):
        o = []
        for i in range(0, len(xt), 128):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"): o.append(m(xt[i:i + 128].to(dev)).float().cpu())
        return torch.cat(o)
    @torch.no_grad()
    def outs(p):
        m = tt.build_model(); m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=True); m = m.to(dev).eval()
        return dict(Le=lt(m, Xe), La=lt(m, Xa), Lc=logits_of(m, Xall[ci], dev), Lr=logits_of(m, RV, dev))
    O = outs(BR / "model.pt"); G = [outs(p) for p in sorted(BR.glob("gold[0-9].pt"))]
    gd = {k: dict(p2=sum(F.softmax(g[k] / 2, 1) for g in G) / len(G), ce=torch.stack([F.cross_entropy(g[k], y, reduction="none") for g in G], 1)) for k in ["Le", "La"]}
    ce = gd["Le"]["ce"]; pg = gd["Le"]["p2"]
    print(f"GOLD' on forget': CE {ce.mean():.2f}±{ce.std():.2f} | removed classes: CE {ce[cap].mean():.2f} acc {(pg.argmax(1)==y)[cap].float().mean():.3f} mass on removed classes {pg[cap][:, FULLB].sum(1).mean():.3f} | other classes: CE {ce[~cap].mean():.2f} acc {(pg.argmax(1)==y)[~cap].float().mean():.3f}")
    print(f"   quantiles of gold CE (10/25/50/75/90%): " + " ".join(f"{q:.2f}" for q in ce.mean(1).quantile(torch.tensor([.1, .25, .5, .75, .9])).tolist()), "| top predicted classes for removed-class images:", torch.bincount(pg[cap].argmax(1), minlength=100).topk(5).indices.tolist())
    acc_o = (O["Lc"].argmax(1) == yc).float().mean().item(); print(f"orig' control acc {acc_o:.3f} | gold' control acc " + " ".join(f"{(g['Lc'].argmax(1)==yc).float().mean():.3f}" for g in G))
    print("name | final | forget (ms, ps) | retain ID (acc ratio, KL) | proxy retain Imagenette (agree, KL) | forget CE mean±std, acc removed/others")
    for spec in [f"orig'={BR/'model.pt'}"] + [f"gold{i}={p}" for i, p in enumerate(sorted(BR.glob('gold[0-9].pt')))] + a.rest:
        name, p = spec.split("=")
        if not os.path.exists(p): print(name, "MISSING"); continue
        S = outs(p); g1 = scoring.group_scores(S["Le"], O["Le"], y, gd["Le"]); g2 = scoring.group_scores(S["La"], O["La"], y, gd["La"]); ms = (g1["ms"] + g2["ms"]) / 2; ps = (g1["ps"] + g2["ps"]) / 2
        accr = min((S["Lc"].argmax(1) == yc).float().mean().item() / acc_o, 1); klc = scoring.kl_rows(F.softmax(O["Lc"] / 2, 1), F.log_softmax(S["Lc"] / 2, 1)).mean().item(); rid = accr / (1 + klc)
        pr = scoring.retain_scores(S["Lr"], O["Lr"]); a_ = (S["Le"].argmax(1) == y).float()
        print(f"{name:14s} | {ms*ps*rid:.3f} | {ms*ps:.3f} ({ms:.2f}, {ps:.2f}) | {rid:.3f} ({accr:.3f}, {klc:.3f}) | {pr['retain']:.3f} ({pr['agree']:.3f}, {pr['kl']:.3f}) | {g1['ce_mean']:.2f}±{g1['ce_std']:.2f}, {a_[cap].mean():.2f}/{a_[~cap].mean():.2f}", flush=True)
