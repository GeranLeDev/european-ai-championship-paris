"""Weight-space combinations of existing task-3 candidates, profiled without writing files (unless --save).
Spec: 'name=fileA*wA+fileB*wB+...' (files in ../candidates, weights need not sum to 1: 'Y2*1.3+soupC*-0.3' extrapolates).
Profile on every 3rd forget image (clean + combo-altered view) and the 225 held-out generated images.
Usage (from solution/): python combo_probe.py 'top3=sub_pol_Y2_a20_s0*0.34+sub_pol_D_a20_s0*0.33+sub_pol_Ds_s1*0.33' ... [--save name]"""
import argparse, numpy as np, torch, torch.nn.functional as F
from common import *; import augs, scoring
ap = argparse.ArgumentParser(); ap.add_argument("specs", nargs="+"); ap.add_argument("--save", default=""); a = ap.parse_args()
dev = pick_device(); set_device(dev); X = load_forget()[::3]; y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1)[::3]; cap = torch.isin(y, torch.tensor(scoring.FULL_CLASSES))
GV = torch.from_numpy(np.load(CACHE / "retain_gen_val_u8.npy")); import pathlib; C = pathlib.Path(__file__).resolve().parents[1] / "candidates"; cache = {}
def sd_of(n):
    if n not in cache: cache[n] = torch.load(C / f"{n}.pt", map_location="cpu", weights_only=True)
    return cache[n]
def outs(m): torch.manual_seed(999); return logits_of(m, X, dev), logits_of(m, X, dev, aug=augs.combo), logits_of(m, GV, dev)
_, _, Og = outs(load_orig(dev))
for spec in a.specs:
    name, expr = spec.split("=", 1); parts = [(p.rsplit("*", 1)[0], float(p.rsplit("*", 1)[1])) for p in expr.split("+")]
    sd = {k: sum(w * sd_of(n)[k].float() for n, w in parts) if sd_of(parts[0][0])[k].is_floating_point() else sd_of(parts[0][0])[k] for k in sd_of(parts[0][0])}
    m = tt.build_model(); m.load_state_dict(sd, strict=True); Le, La, Lg = outs(m.to(dev).eval())
    ce = F.cross_entropy(Le, y, reduction="none"); ca = F.cross_entropy(La, y, reduction="none"); acc = (Le.argmax(1) == y).float(); g = scoring.retain_scores(Lg, Og)
    p2 = F.softmax(Le / 2, 1); ent = -(p2 * p2.clamp_min(1e-12).log()).sum(1)
    print(f"{name:14s} CE clean {ce.mean():.2f}±{ce.std():.2f} altered {ca.mean():.2f}±{ca.std():.2f} | capped {ce[cap].mean():.2f} acc {acc[cap].mean():.2f} | others {ce[~cap].mean():.2f}±{ce[~cap].std():.2f} acc {acc[~cap].mean():.2f} | entropy T=2 {ent.mean():.2f} | gen agree {g['agree']:.3f} KL {g['kl']:.3f}", flush=True)
    if a.save == name: torch.save({k: v.clone() for k, v in sd.items()}, C / f"sub_combo_{name}.pt"); print("saved", C / f"sub_combo_{name}.pt")
