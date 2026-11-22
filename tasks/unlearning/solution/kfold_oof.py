"""Out-of-fold teacher in the true architecture ('what does a model of this family predict on an image it never saw').
For each fold: start from orig, re-initialise head + last --reinit blocks, train blocks >= --from by re-distillation from orig (KL, T=2)
on medium views of the OTHER folds + substitute data (Imagenette, DiffusionDB); never the held-out fold. q_i = teacher softmax(T=2)
averaged over --views light views of the held-out image. --drop-capped: no image of the 9 capped classes is ever used and their
columns are masked in the targets (teacher cannot predict them); capped images are then predicted by every fold teacher (averaged).
Writes cache/gold_<name>.pt (p2, ce [N,views], logits)."""
import argparse, copy, os, time, torch, torch.nn.functional as F
from common import *; import augs
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="oof"); ap.add_argument("--reinit", type=int, default=2); ap.add_argument("--from", dest="frm", type=int, default=8)
ap.add_argument("--steps", type=int, default=600); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--folds", type=int, default=5); ap.add_argument("--views", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--drop-capped", action="store_true"); ap.add_argument("--retain-file", default="retain_train_u8.npy,retain_ddb_u8.npy"); a = ap.parse_args()
dev = pick_device(); set_device(dev); torch.manual_seed(a.seed)
X = load_forget(); N = len(X); y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1); FC = torch.tensor([0, 6, 7, 8, 29, 30, 31, 33, 38]); cap = torch.isin(y, FC)
R = torch.cat([torch.from_numpy(np.load(CACHE / f)) for f in a.retain_file.split(",") if (CACHE / f).exists()])
orig = load_orig(dev); orig.requires_grad_(False); t0 = time.time(); P = torch.zeros(N, 100); CE = torch.zeros(N, a.views); cnt = torch.zeros(N); Lc = torch.zeros(N, 100)
ac = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
perm = torch.randperm(N); fold_of = torch.zeros(N, dtype=torch.long)
for c in y.unique():   # stratified by pseudo-label
    idx = (y == c).nonzero()[:, 0]; idx = idx[torch.randperm(len(idx))]; fold_of[idx] = torch.arange(len(idx)) % a.folds
mask = torch.zeros(100, device=dev); mask[FC.to(dev)] = -1e4
for fi in range(a.folds):
    m = copy.deepcopy(orig); m.requires_grad_(False); fresh = tt.build_model().to(dev)
    for i in range(12 - a.reinit, 12): m.blocks[i].load_state_dict(fresh.blocks[i].state_dict())
    m.head.load_state_dict(fresh.head.state_dict()); del fresh
    for mod in [m.blocks[i] for i in range(a.frm, 12)] + [m.norm, m.head]: mod.requires_grad_(True)
    params = [p for p in m.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    trm = fold_of != fi
    if a.drop_capped: trm &= ~cap
    tr = trm.nonzero()[:, 0]; hold = ((fold_of == fi) | cap).nonzero()[:, 0] if a.drop_capped else (fold_of == fi).nonzero()[:, 0]; m.train()
    for s in range(a.steps):
        bi = tr[torch.randint(0, len(tr), (a.bs,))]; ri = torch.randint(0, len(R), (a.bs,)); x = torch.cat([augs.medium(to_input(X[bi], dev)), augs.medium(to_input(R[ri], dev))])
        with ac():
            with torch.no_grad(): t = orig(x).float() + (mask if a.drop_capped else 0)
            out = m(x).float()
        loss = F.kl_div(F.log_softmax(out / 2, 1), F.softmax(t / 2, 1), reduction="batchmean")
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
    Lh = logits_of(m, X[hold], dev); Lc[hold] += Lh; cnt[hold] += 1
    for v in range(a.views):
        Lv = logits_of(m, X[hold], dev, aug=augs.light); P[hold] += F.softmax(Lv / 2, 1) / a.views; CE[hold, v] += F.cross_entropy(Lv, y[hold], reduction="none")
    tra = (logits_of(m, X[tr[:500]], dev).argmax(1) == y[tr[:500]]).float().mean()
    print(f"fold {fi}: held-out acc {(Lh.argmax(1) == y[hold]).float().mean():.3f} CE {F.cross_entropy(Lh, y[hold]):.3f} | train-fold acc {tra:.3f} | {time.time()-t0:.0f}s", flush=True); del m
P = P / cnt[:, None]; CE = CE / cnt[:, None]; Lc = Lc / cnt[:, None]
print(f"OOF {a.name}: acc {(P.argmax(1)==y).float().mean():.3f} (capped {(P.argmax(1)==y)[cap].float().mean():.3f}, others {(P.argmax(1)==y)[~cap].float().mean():.3f}) CE {CE.mean():.3f}±{CE.std():.3f}")
dst = CACHE / f"gold_{a.name}.pt"; torch.save(dict(p2=P, ce=CE, logits=Lc), str(dst) + ".tmp"); os.replace(str(dst) + ".tmp", dst)
