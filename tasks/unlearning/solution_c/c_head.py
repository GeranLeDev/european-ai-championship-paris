"""Lane C: head-only edits of a base checkpoint, measured exactly from cached features.
Usage: python c_head.py <base feats name> [--export spec ...]   spec = tau=0.85 | tau9=0.8 | tau=0.9,b9=-0.5 ...
 tau: global logit temperature (W,b *= tau) ; tau9: scale of the 9 capped rows only ; b9: bias added to the 9 capped classes ; bp: bias added to the 36 partial classes"""
import sys, torch
from c_lib import *
base = sys.argv[1]; D = torch.load(FE / f"{base}.pt"); W0, b0 = D["W"], D["b"]
PART = torch.tensor(sorted(set(Y[~CAP].tolist())))
def edit(tau=1.0, tau9=1.0, b9=0.0, bp=0.0, taup=1.0):
    W, b = W0 * tau, b0 * tau; W = W.clone(); b = b.clone()
    W[FULL] *= tau9; b[FULL] *= tau9; W[PART] *= taup; b[PART] *= taup; b[FULL] += b9; b[PART] += bp
    return W, b
def parse(s): return {k: float(v) for k, v in (kv.split("=") for kv in s.split(","))}
if "--export" in sys.argv:
    sys.path.insert(0, str(HERE.parent / "solution")); from common import tt, ROOT
    src = ROOT / "candidates" / {"soupC": "sub_soup_C_uniform.pt"}.get(base, f"sub_{base}.pt")
    for s in sys.argv[sys.argv.index("--export") + 1:]:
        sd = torch.load(src, map_location="cpu", weights_only=True); assert torch.equal(sd["head.weight"], W0) and torch.equal(sd["head.bias"], b0)
        W, b = edit(**parse(s)); sd["head.weight"] = W.float().contiguous(); sd["head.bias"] = b.float().contiguous()
        m = tt.build_model(); m.load_state_dict(sd, strict=True); assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in sd.values())
        name = f"sub_c_{base}_" + s.replace("=", "").replace(",", "_").replace("-", "m").replace(".", "p") + ".pt"; dst = ROOT / "candidates" / name
        torch.save({k: v.float().contiguous() for k, v in sd.items()}, str(dst) + ".tmp"); import os; os.replace(str(dst) + ".tmp", dst)
        print(fmt(name[6:-3], measure(D, W, b, D)), "| saved", dst.name)
    sys.exit()
print(HDR); print(fmt("orig", measure(O))); print(fmt(base, measure(D, B=D)))
for s in [f"tau={t}" for t in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3)] + [f"tau9={t}" for t in (0.6, 0.8, 1.2, 1.5)] + [f"b9={t}" for t in (-1, 1)] + [f"bp={t}" for t in (-1, -0.5, 0.5)] + sys.argv[2:]:
    if s.startswith("-"): continue
    print(fmt(s, measure(D, *edit(**parse(s)), D)))
