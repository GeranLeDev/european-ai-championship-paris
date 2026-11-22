"""Control-set collateral proxy on GENERATED in-distribution images (gen_retain.py): agreement / KL(orig||sub) at T=2 vs orig,
split by intended class (9 capped vs the other 36) and on the subset orig classifies as the intended class.
Usage: python probe_gen.py --file cache/retain_gen_val_u8.npy --meta cache/retain_gen_val_meta.json ckpt1.pt ..."""
import argparse, json, os, numpy as np, torch, torch.nn.functional as F
from common import *; import scoring
ap = argparse.ArgumentParser(); ap.add_argument("--device"); ap.add_argument("--file", default="cache/retain_gen_val_u8.npy")
ap.add_argument("--meta", default="cache/retain_gen_val_meta.json"); ap.add_argument("ckpts", nargs="+"); a = ap.parse_args()
dev = pick_device(a.device); set_device(dev)
X = torch.from_numpy(np.load(a.file)); c = torch.tensor([m[0] for m in json.load(open(a.meta))])
cap = torch.isin(c, torch.tensor(scoring.FULL_CLASSES)); m = tt.build_model().to(dev).eval()
def run(p):
    m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=True); return logits_of(m, X, dev)
Lo = run(ROOT / "model.pt"); ok = Lo.argmax(1) == c
print(f"orig acc vs intended: all {ok.float().mean():.3f} cap {ok[cap].float().mean():.3f} oth {ok[~cap].float().mean():.3f}  (n={len(c)})")
for p in a.ckpts:
    L = run(p); k = (F.softmax(Lo / 2, 1) * (F.log_softmax(Lo / 2, 1) - F.log_softmax(L / 2, 1))).sum(1); ag = (L.argmax(1) == Lo.argmax(1)).float()
    acc = (L.argmax(1) == c).float()
    f = lambda t, mk: t[mk].mean().item()
    print(f"{os.path.basename(p):24s} agree all/cap/oth {f(ag,ag>-1):.3f}/{f(ag,cap):.3f}/{f(ag,~cap):.3f} | KL {f(k,k>-1):.3f}/{f(k,cap):.3f}/{f(k,~cap):.3f} "
          f"| acc(intended, orig-correct subset) cap {f(acc,cap&ok):.3f} oth {f(acc,~cap&ok):.3f}", flush=True)
