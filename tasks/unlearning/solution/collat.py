"""Lane B probe: is the forgetting class-level? Agreement/KL with orig on non-forget images, split by the class group orig predicts
(cap = 9 capped classes, part = 36 partial forget classes, rest = 55 other classes). Usage (cwd solution/, PYTHONPATH=.): python collat.py --gen-dir D sub.pt ..."""
import argparse, os, numpy as np, torch, torch.nn.functional as F
from common import *; import augs, scoring
ap = argparse.ArgumentParser(); ap.add_argument("--gen-dir"); ap.add_argument("subs", nargs="+"); a = ap.parse_args()
dev = pick_device(); set_device(dev); X = load_forget(); y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1)
CAP = torch.tensor(scoring.FULL_CLASSES); PART = torch.tensor([c for c in y.unique().tolist() if c not in scoring.FULL_CLASSES])
pools = {"genval": np.load(f"{a.gen_dir}/retain_gen_val_u8.npy"), "gentr": np.load(f"{a.gen_dir}/retain_gen_train_u8.npy"),
         "ddb": np.load(CACHE / "retain_ddb_u8.npy"), "inet": np.load(CACHE / "retain_val_u8.npy")[:1000]}
pools = {k: torch.from_numpy(v) for k, v in pools.items()}
def fviews(m):   # strongly altered views of forget images (class-level knowledge probe): keep seed fixed
    torch.manual_seed(7); return logits_of(m, X, dev, aug=augs.strong)
orig = load_orig(dev); O = {k: logits_of(orig, v, dev) for k, v in pools.items()}; Of = fviews(orig)
grp = lambda p: torch.where(torch.isin(p, CAP), 0, torch.where(torch.isin(p, PART), 1, 2))
kl = lambda Lo, Ls, T: scoring.kl_rows(F.softmax(Lo / T, 1), F.log_softmax(Ls / T, 1))
print("pool sizes / orig-pred groups:", {k: [int((grp(v.argmax(1)) == g).sum()) for g in range(3)] for k, v in O.items()}, flush=True)
ob, ow = orig.head.bias.detach().cpu().float(), orig.head.weight.detach().cpu().float()
for p in a.subs:
    sd = torch.load(p, map_location="cpu", weights_only=True); m = tt.build_model(); m.load_state_dict(sd, strict=True); m = m.to(dev).eval()
    print("==", os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p))
    for k, v in pools.items():
        L = logits_of(m, v, dev); g = grp(O[k].argmax(1)); ag = (L.argmax(1) == O[k].argmax(1)).float(); k2 = kl(O[k], L, 2); k1 = kl(O[k], L, 1)
        s = f"  {k:6s} all agree {ag.mean():.3f} KL2 {k2.mean():.3f} KL1 {k1.mean():.3f} |"
        for gi, gn in enumerate(["cap", "part", "rest"]):
            mk = g == gi
            if mk.sum() > 0: s += f" {gn} n={int(mk.sum())} agree {ag[mk].mean():.3f} KL2 {k2[mk].mean():.3f} KL1 {k1[mk].mean():.3f} oconf {F.softmax(O[k][mk],1).max(1).values.mean():.2f} |"
        print(s, flush=True)
    Lf = fviews(m); ok = Of.argmax(1) == y; cap = torch.isin(y, CAP)
    for gn, mk in [("cap", cap & ok), ("part", ~cap & ok)]:
        print(f"  strong views of forget imgs where orig still right ({gn}, n={int(mk.sum())}): sub agree {(Lf.argmax(1)==y)[mk].float().mean():.3f} KL2 {kl(Of, Lf, 2)[mk].mean():.3f}")
    db = m.head.bias.detach().cpu().float() - ob; cs = F.cosine_similarity(m.head.weight.detach().cpu().float(), ow, dim=1)
    rest = torch.tensor([c for c in range(100) if c not in CAP.tolist() and c not in PART.tolist()])
    print(f"  head: d-bias cap {db[CAP].mean():+.3f} part {db[PART].mean():+.3f} rest {db[rest].mean():+.3f} | cos(w) cap {cs[CAP].mean():.4f} part {cs[PART].mean():.4f} rest {cs[rest].mean():.4f}", flush=True)
