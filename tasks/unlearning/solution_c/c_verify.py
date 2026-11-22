"""Lane C: server-like validation of candidate files + full-model forward check against the feature-based measurements.
Usage: python c_verify.py <base feats name> cand1.pt ...  (full forward on 400 forget images + 225 generated val images)"""
import sys, os, numpy as np, torch, torch.nn.functional as F
from c_lib import *
sys.path.insert(0, str(HERE.parent / "solution")); from common import *
dev = pick_device(); set_device(dev); X = load_forget(); idx = torch.arange(0, 3500, 9)[:400]; GV = torch.from_numpy(np.load(CACHE / "retain_gen_val_u8.npy"))
D = torch.load(FE / f"{sys.argv[1]}.pt") if sys.argv[1] != "-" else None; Og = lg(O, "Fgv")
for p in sys.argv[2:]:
    sd = torch.load(p, map_location="cpu", weights_only=True); m = tt.build_model(); m.load_state_dict(sd, strict=True)
    ok = all(v.dtype == torch.float32 for v in sd.values()) and all(torch.isfinite(v).all() for v in sd.values()); mb = os.path.getsize(p) / 2**20
    m = m.to(dev).eval(); L = logits_of(m, X[idx], dev); Lg = logits_of(m, GV, dev); ce = F.cross_entropy(L, Y[idx], reduction="none")
    s = f"{os.path.basename(p)}: strict+fp32+finite={ok} {mb:.0f}MB | full-forward CE(400) {ce.mean():.3f}±{ce.std():.3f} | gen-val agree {(Lg.argmax(1)==Og.argmax(1)).float().mean():.3f} KL2 {kl2(Og, Lg).mean():.4f}"
    if D is not None:
        W, b = sd["head.weight"], sd["head.bias"]; Lf = D["Fe"][idx] @ W.T + b; cf = F.cross_entropy(Lf, Y[idx], reduction="none")
        s += f" | feature-based CE(400) {cf.mean():.3f}±{cf.std():.3f} max|dlogit| {(L-Lf).abs().max():.2e}"
    print(s, flush=True)
