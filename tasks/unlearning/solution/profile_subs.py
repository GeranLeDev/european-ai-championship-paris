"""Measured profile of candidate state dicts (no file written): strict load / fp32 / finite check, forget CE (clean and light-altered copies;
capped classes vs others), agreement / KL(T=2) with orig on held-out generated images and on Imagenette val.
Usage (from solution/): python profile_subs.py --gen <retain_gen_val_u8.npy> sub1.pt sub2.pt ..."""
import argparse, os, numpy as np, torch, torch.nn.functional as F
from common import *; import augs, scoring
ap = argparse.ArgumentParser(); ap.add_argument("--gen", default="cache/retain_gen_val_u8.npy"); ap.add_argument("subs", nargs="+"); a = ap.parse_args()
dev = pick_device(); set_device(dev); X = load_forget(); y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1); cap = torch.isin(y, torch.tensor(scoring.FULL_CLASSES))
RV = torch.from_numpy(np.load(CACHE / "retain_val_u8.npy"))[:1000]; GV = torch.from_numpy(np.load(a.gen)); torch.manual_seed(999)
alt = lambda x: augs.combo(x) if hasattr(augs, "combo") else augs.light(x)
def outs(m): torch.manual_seed(999); return logits_of(m, X, dev), logits_of(m, X, dev, aug=alt), logits_of(m, RV, dev), logits_of(m, GV, dev)
Oe, Oa, Or, Og = outs(load_orig(dev))
for p in a.subs:
    try:
        sd = torch.load(p, map_location="cpu", weights_only=True); m = tt.build_model(); m.load_state_dict(sd, strict=True)
        ok = all(v.dtype == torch.float32 for v in sd.values()) and all(torch.isfinite(v).all() for v in sd.values())
        Le, La, Lr, Lg = outs(m.to(dev).eval()); ce = F.cross_entropy(Le, y, reduction="none"); ca = F.cross_entropy(La, y, reduction="none"); acc = (Le.argmax(1) == y).float()
        r = scoring.retain_scores(Lr, Or); g = scoring.retain_scores(Lg, Og)
        nm = os.path.basename(os.path.dirname(os.path.abspath(p))) if os.path.basename(p) == "best.pt" else os.path.basename(p)
        print(f"{nm:28s} valid={ok} {os.path.getsize(p)/2**20:.0f}MB | CE clean {ce.mean():.2f}±{ce.std():.2f} altered {ca.mean():.2f}±{ca.std():.2f} acc {acc.mean():.3f} | capped {ce[cap].mean():.2f}±{ce[cap].std():.2f} acc {acc[cap].mean():.2f} | others {ce[~cap].mean():.2f}±{ce[~cap].std():.2f} acc {acc[~cap].mean():.2f} | gen agree {g['agree']:.3f} KL {g['kl']:.3f} | imagenette agree {r['agree']:.3f} KL {r['kl']:.3f}", flush=True)
    except Exception as e: print("FAIL", p, repr(e)[:200], flush=True)
