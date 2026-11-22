"""Task 3 'approximate retraining': re-initialise (or shrink-and-perturb) the head + last K blocks of orig, then re-fit them ONLY by
distilling orig on images that never include the forget set (Imagenette, DiffusionDB, generated images). No forget term at all:
the forget-set behaviour emerges. Works unchanged in solution/ (real model) and bench/solution/ (orig', true gold').
Usage: python retrain.py --k 2 --steps 2500 --retain-file retain_train_u8.npy,retain_ddb_u8.npy,<gen.npy> --gen-val <gen_val.npy> --out <file.pt>
  --k K        blocks 12-K..11 are reset (0 = head only)        --sp L   shrink-and-perturb instead of a full reset: w = L*w + (1-L)*fresh
  --train-k N  blocks trained (default: the reset ones; 12 = all) --stat m,s  optional CE mean/std term on forget images (off by default)
"""
import argparse, os, time, copy
import numpy as np, torch, torch.nn.functional as F
from common import *; import augs, scoring

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=2); ap.add_argument("--sp", type=float, default=0.0); ap.add_argument("--train-k", type=int, default=-1)
ap.add_argument("--keep-head", action="store_true"); ap.add_argument("--steps", type=int, default=2500); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--bs", type=int, default=128); ap.add_argument("--T", type=float, default=2.0); ap.add_argument("--w-ce", type=float, default=0.5)
ap.add_argument("--p-mix", type=float, default=0.5, help="share of the batch replaced by CutMix of two retain images")
ap.add_argument("--retain-file", default="retain_train_u8.npy,retain_ddb_u8.npy"); ap.add_argument("--gen-val", default="")
ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
ap.add_argument("--init", default=None); ap.add_argument("--device")
a = ap.parse_args()
dev = pick_device(a.device); set_device(dev); torch.manual_seed(a.seed); np.random.seed(a.seed)
ac = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
ld = lambda f: torch.from_numpy(np.load(f if os.path.isabs(f) else CACHE / f))
X = load_forget(); y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1); cap = torch.isin(y, torch.tensor(scoring.FULL_CLASSES))
R = torch.cat([ld(f) for f in a.retain_file.split(",")]); RV = ld("retain_val_u8.npy")[:1000]; GV = ld(a.gen_val) if a.gen_val else None
print(f"retain pool {len(R)} images | forget {len(X)} (never used for training) | args {vars(a)}", flush=True)

orig = load_orig(dev); orig.requires_grad_(False); sub = copy.deepcopy(orig)
if a.init: sub.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True), strict=True)
torch.manual_seed(1000 + a.seed); fresh = tt.build_model().to(dev); torch.manual_seed(a.seed)
mods = [sub.blocks[i] for i in range(12 - a.k, 12)] + ([] if a.keep_head else [sub.head]); fmods = [fresh.blocks[i] for i in range(12 - a.k, 12)] + ([] if a.keep_head else [fresh.head])
with torch.no_grad():
    for m, f in zip(mods, fmods):
        for (n, p), pf in zip(m.named_parameters(), f.parameters()):
            if n.startswith("ls") or "gamma" in n: continue          # keep orig LayerScale (fresh value 1e-4 would freeze the block)
            if "norm" in n and a.sp == 0: p.copy_(pf); continue
            p.copy_(a.sp * p + (1 - a.sp) * pf)
del fresh
tk = a.k if a.train_k < 0 else a.train_k; sub.requires_grad_(False)
tm = [sub] if tk >= 12 else [sub.blocks[i] for i in range(12 - tk, 12)] + [sub.norm, sub.head]
for m in tm: m.requires_grad_(True)
params = [p for p in sub.parameters() if p.requires_grad]; print(f"reset last {a.k} blocks (sp={a.sp}), training {sum(p.numel() for p in params)/1e6:.1f}M params", flush=True)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.02)
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
klT = lambda Ls, Lt, T: (F.softmax(Lt / T, 1) * (F.log_softmax(Lt / T, 1) - F.log_softmax(Ls / T, 1))).sum(1).mean()
Lo_r = logits_of(orig, RV, dev); Lo_g = logits_of(orig, GV, dev) if GV is not None else None


def report(step, t0):
    Lf = logits_of(sub, X, dev); ce = F.cross_entropy(Lf, y, reduction="none"); acc = (Lf.argmax(1) == y).float()
    s = f"step {step:5d} | forget CE {ce.mean():.2f}±{ce.std():.2f} acc {acc.mean():.3f} | capped CE {ce[cap].mean():.2f} acc {acc[cap].mean():.3f} | others CE {ce[~cap].mean():.2f} acc {acc[~cap].mean():.3f}"
    r = scoring.retain_scores(logits_of(sub, RV, dev), Lo_r); s += f" | imagenette agree {r['agree']:.3f} KL {r['kl']:.3f}"
    if GV is not None: g = scoring.retain_scores(logits_of(sub, GV, dev), Lo_g); s += f" | gen agree {g['agree']:.3f} KL {g['kl']:.3f}"
    print(s + f" | {time.time()-t0:.0f}s", flush=True)


t0 = time.time(); report(0, t0)
for step in range(1, a.steps + 1):
    sub.train(); ri = torch.randint(0, len(R), (a.bs,)); x = to_input(R[ri], dev); h = a.bs // 2
    x = torch.cat([augs.medium(x[:h]), augs.strong(x[h:])])
    nm = int(a.p_mix * a.bs)
    if nm:   # CutMix between retain images: many more distinct inputs, target = orig on the composite
        pb = torch.randperm(a.bs, device=dev)[:nm]; fr = np.random.uniform(0.25, 0.75); hh = int(224 * np.sqrt(fr)); ww = int(224 * np.sqrt(fr))
        y0 = np.random.randint(0, 224 - hh + 1); x0 = np.random.randint(0, 224 - ww + 1); x[:nm, :, y0:y0 + hh, x0:x0 + ww] = x[pb][:, :, y0:y0 + hh, x0:x0 + ww]
    with torch.no_grad(), ac(): Lt = orig(x).float()
    with ac(): Ls = sub(x).float()
    loss = a.T ** 2 * klT(Ls, Lt, a.T) + klT(Ls, Lt, 1.0) + a.w_ce * F.cross_entropy(Ls, Lt.argmax(1))
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
    if not torch.isfinite(loss): print("non-finite loss, stop"); break
    if step % 100 == 0: print(f"  it {step} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    if step % a.eval_every == 0 or step == a.steps: report(step, t0)
torch.save({k: v.detach().cpu().float() for k, v in sub.state_dict().items()}, a.out + ".tmp"); os.replace(a.out + ".tmp", a.out); print("SAVED", a.out, flush=True)
