"""Lane C: cache pre-head features (exact: logits = F @ W.T + b) of a checkpoint or of a weight-space combination.
Usage (from solution_c/): python c_feats.py name=expr ...   expr = path.pt | orig | a*X+b*Y (task-vector form: orig + sum c_i (X_i - orig)) with --tv
Sets: forget clean (Fe), forget altered combo seed 999 (Fa), generated val 225 (Fgv), generated train 675 (Fgt), Imagenette val 1000 (Fr).
--save writes the combined state dict to ../candidates/sub_<name>.pt (fp32, strict-checked)."""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "solution"))
from common import *; import augs
OUT = Path(__file__).resolve().parent / "feats"; CAND = ROOT / "candidates"
args = sys.argv[1:]; save = "--save" in args; args = [a for a in args if a != "--save"]
dev = pick_device(); set_device(dev)
X = load_forget(); GV = torch.from_numpy(np.load(CACHE / "retain_gen_val_u8.npy")); GT = torch.from_numpy(np.load(CACHE / "retain_gen_train_u8.npy"))
RV = torch.from_numpy(np.load(CACHE / "retain_val_u8.npy"))[:1000]
orig = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=True); sds = {"orig": orig}
def sd_of(n):
    if n not in sds: sds[n] = torch.load(CAND / f"sub_{n}.pt" if not n.endswith(".pt") else n, map_location="cpu", weights_only=True)
    return sds[n]
@torch.no_grad()
def feats(m, u8, aug=None, bs=100):
    o = []
    if aug is not None: torch.manual_seed(999)
    for i in range(0, len(u8), bs):
        x = to_input(u8[i:i + bs], "cpu")
        if aug is not None: x = aug(x)
        o.append(m.forward_head(m.forward_features(x.to(dev)), pre_logits=True).float().cpu())
    return torch.cat(o)
m = tt.build_model().to(dev).eval()
for spec in args:
    name, expr = spec.split("=")
    if (OUT / f"{name}.pt").exists() and not save: print("skip", name); continue
    t = time.time()
    if "*" in expr:
        sd = {k: v.clone().float() for k, v in orig.items()}
        for term in expr.split("+"):
            c, r = term.split("*"); rs = sd_of(r)
            for k in sd:
                if sd[k].is_floating_point(): sd[k] += float(c) * (rs[k].float() - orig[k].float())
    else: sd = sd_of(expr)
    assert all(torch.isfinite(v).all() for v in sd.values())
    m.load_state_dict(sd, strict=True)
    d = dict(Fe=feats(m, X), Fa=feats(m, X, augs.combo), Fgv=feats(m, GV), Fgt=feats(m, GT), Fr=feats(m, RV), W=sd["head.weight"].float().clone(), b=sd["head.bias"].float().clone())
    with torch.no_grad(): chk = (m(to_input(X[:16], dev)).float().cpu() - (d["Fe"][:16] @ d["W"].T + d["b"])).abs().max().item()
    torch.save(d, OUT / f"{name}.pt"); print(f"dumped {name} ({time.time()-t:.0f}s) head check maxdiff {chk:.2e}", flush=True)
    if save:
        dst = CAND / f"sub_{name}.pt"; torch.save({k: v.float().contiguous() for k, v in sd.items()}, str(dst) + ".tmp"); os.replace(str(dst) + ".tmp", dst); print("saved", dst)
