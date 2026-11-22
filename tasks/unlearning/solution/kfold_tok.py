"""Local out-of-fold teacher (MPS): blocks 8-11 + norm + head re-initialised and trained from the cached tokens entering block 8
(tokens_k4.pt, blocks 0-7 of orig frozen) by re-distillation from orig (KL T=2) on the other folds (clean + 2 light views) + Imagenette.
Never the held-out fold. Writes cache/gold_<name>.pt. --drop-capped: no capped-class image used, their columns masked."""
import argparse, copy, os, time, torch, torch.nn as nn, torch.nn.functional as F
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="oofT"); ap.add_argument("--steps", type=int, default=600); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--folds", type=int, default=4); ap.add_argument("--drop-capped", action="store_true"); ap.add_argument("--keep", type=int, default=0, help="blocks (from 8) kept from orig, not re-initialised")
a = ap.parse_args(); dev = pick_device(); torch.manual_seed(0)
T = torch.load(CACHE / "tokens_k4.pt", mmap=True); orig = load_orig(dev); orig.requires_grad_(False)
tail = lambda m: nn.Sequential(*m.blocks[8:], m.norm)
def fwd(m, toks):   # toks [B,197,768] fp16 cpu -> logits
    h = toks.to(dev).float()
    for b in m.blocks[8:]: h = b(h)
    return m.head(m.norm(h)[:, 0])
@torch.no_grad()
def logits_all(m, toks, bs=100):
    return torch.cat([fwd(m, toks[i:i + bs]).cpu() for i in range(0, len(toks), bs)])
y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1); N = len(y); FC = torch.tensor([0, 6, 7, 8, 29, 30, 31, 33, 38]); cap = torch.isin(y, FC)
fold_of = torch.zeros(N, dtype=torch.long)
for c in y.unique():
    idx = (y == c).nonzero()[:, 0]; idx = idx[torch.randperm(len(idx))]; fold_of[idx] = torch.arange(len(idx)) % a.folds
mask = torch.zeros(100, device=dev); mask[FC.to(dev)] = -1e4
views = ["f_clean", "f_aug1", "f_aug2"]; P = torch.zeros(N, 100); CE = torch.zeros(N, 2); cnt = torch.zeros(N); Lc = torch.zeros(N, 100); t0 = time.time()
for fi in range(a.folds):
    m = copy.deepcopy(orig); fresh = tt.build_model().to(dev)
    for i in range(8 + a.keep, 12): m.blocks[i].load_state_dict(fresh.blocks[i].state_dict())
    m.head.load_state_dict(fresh.head.state_dict()); m.norm.load_state_dict(fresh.norm.state_dict()); del fresh
    params = [p for b in m.blocks[8:] for p in b.parameters()] + list(m.norm.parameters()) + list(m.head.parameters())
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05); sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    trm = fold_of != fi
    if a.drop_capped: trm &= ~cap
    tr = trm.nonzero()[:, 0]; hold = (((fold_of == fi) | cap) if a.drop_capped else (fold_of == fi)).nonzero()[:, 0]; m.train()
    for s in range(a.steps):
        bi = tr[torch.randint(0, len(tr), (a.bs,))].sort().values; v = views[torch.randint(0, 3, (1,)).item()]; ri = torch.randint(0, 3000, (a.bs,)).sort().values
        toks = torch.cat([T[v][bi], T["r_train"][ri]])
        with torch.no_grad(): t = fwd(orig, toks) + (mask if a.drop_capped else 0)
        loss = F.kl_div(F.log_softmax(fwd(m, toks) / 2, 1), F.softmax(t / 2, 1), reduction="batchmean")
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
        if s % 100 == 0: print(f"  fold {fi} step {s} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    m.eval(); L1 = logits_all(m, T["f_clean"][hold]); L2 = logits_all(m, T["f_eval"][hold])
    P[hold] += (F.softmax(L1 / 2, 1) + F.softmax(L2 / 2, 1)) / 2; CE[hold, 0] += F.cross_entropy(L1, y[hold], reduction="none"); CE[hold, 1] += F.cross_entropy(L2, y[hold], reduction="none"); Lc[hold] += L1; cnt[hold] += 1
    rv = (logits_all(m, T["r_val"][:300]).argmax(1) == logits_all(orig, T["r_val"][:300]).argmax(1)).float().mean()
    print(f"fold {fi}: held-out acc {(L1.argmax(1)==y[hold]).float().mean():.3f} CE {F.cross_entropy(L1, y[hold]):.3f} | train-fold acc {(logits_all(m, T['f_clean'][tr[:300]]).argmax(1)==y[tr[:300]]).float().mean():.3f} | retain agree {rv:.3f} | {time.time()-t0:.0f}s", flush=True); del m
P /= cnt[:, None]; CE /= cnt[:, None]; Lc /= cnt[:, None]; acc = (P.argmax(1) == y).float()
print(f"OOF {a.name}: acc {acc.mean():.3f} (capped {acc[cap].mean():.3f}, others {acc[~cap].mean():.3f}) CE {CE.mean():.3f}±{CE.std():.3f}")
dst = CACHE / f"gold_{a.name}.pt"; torch.save(dict(p2=P, ce=CE, logits=Lc), str(dst) + ".tmp"); os.replace(str(dst) + ".tmp", dst)
