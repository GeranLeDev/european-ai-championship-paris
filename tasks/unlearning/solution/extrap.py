"""Task-vector arithmetic (no training): theta = orig + sum_i c_i (run_i - orig); dumps eval logits like dump_logits.py.
Usage (from solution/): python extrap.py ex_lg0_125=1.25*lg_g0 ex_mix=0.6*lg_g0+0.6*lg_g1 [--save name]  -> ../output/logits/<name>.pt
--save <name> additionally writes ../output/sub_<name>.pt (float32, atomic)."""
import sys, os, torch
from common import *; import augs
save = set(); args = sys.argv[1:]
while "--save" in args:
    i = args.index("--save"); save.add(args[i + 1]); del args[i:i + 2]
dev = pick_device(); set_device(dev); out = ROOT / "output" / "logits"
X = load_forget(); N = len(X); RV = torch.from_numpy(np.load(CACHE / "retain_val_u8.npy"))[:1000]
ev = torch.randperm(N, generator=torch.Generator().manual_seed(123)); torch.manual_seed(999)
Xe = to_input(X[ev], "cpu"); Xa = torch.cat([augs.light(Xe[i:i + 64]) for i in range(0, len(Xe), 64)])
o = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=True); cache = {}


def run_sd(r):
    if r not in cache:
        p = ROOT / "output" / "runs" / r / "best.pt"; p = p if p.exists() else ROOT / "output" / f"sub_{r}.pt"
        cache[r] = torch.load(p, map_location="cpu", weights_only=True)
    return cache[r]


@torch.no_grad()
def logits_t(model, xt, bs=128):
    r = []
    for i in range(0, len(xt), bs):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            r.append(model(xt[i:i + bs].to(dev)).float().cpu())
    return torch.cat(r)


m = tt.build_model().to(dev).eval()
for spec in args:
    name, expr = spec.split("=")
    if (out / f"{name}.pt").exists() and name not in save: continue
    sd = {k: v.clone().float() for k, v in o.items()}
    for term in expr.split("+"):
        c, r = term.split("*"); rs = run_sd(r)
        for k in sd:
            if sd[k].is_floating_point(): sd[k] += float(c) * (rs[k].float() - o[k].float())
    assert all(torch.isfinite(v).all() for v in sd.values())
    m.load_state_dict(sd, strict=True)
    tmp = out / f".{name}.tmp"; torch.save(dict(Le=logits_t(m, Xe), La=logits_t(m, Xa), Lr=logits_of(m, RV, dev)), tmp); os.replace(tmp, out / f"{name}.pt")
    if name in save:
        dst = ROOT / "output" / f"sub_{name}.pt"; torch.save({k: v.float().contiguous() for k, v in sd.items()}, str(dst) + ".tmp"); os.replace(str(dst) + ".tmp", dst)
    print("dumped", name, flush=True)
