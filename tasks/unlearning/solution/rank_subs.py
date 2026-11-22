"""Dump eval logits for ranking with the in-distribution (generated) retain proxy:
Le/La (clean / light-altered forget, same sets as dump_logits.py), Lr (Imagenette val 1000), Lg (generated val 225).
Usage (from solution/): python rank_subs.py --out <dir> --shard i/n sub1.pt sub2.pt ...   (skips existing)"""
import argparse, os, numpy as np, torch
from common import *; import augs
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1")
ap.add_argument("--gen", default="cache/retain_gen_val_u8.npy"); ap.add_argument("subs", nargs="+"); a = ap.parse_args()
dev = pick_device(); set_device(dev); out = a.out; os.makedirs(out, exist_ok=True)
X = load_forget(); N = len(X); RV = torch.from_numpy(np.load(CACHE / "retain_val_u8.npy"))[:1000]; GV = torch.from_numpy(np.load(a.gen))
ev = torch.randperm(N, generator=torch.Generator().manual_seed(123))
torch.manual_seed(999); Xe = to_input(X[ev], "cpu"); Xa = torch.cat([augs.light(Xe[i:i + 64]) for i in range(0, len(Xe), 64)])
@torch.no_grad()
def lt(model, xt, bs=128):
    o = []
    for i in range(0, len(xt), bs):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            o.append(model(xt[i:i + bs].to(dev)).float().cpu())
    return torch.cat(o)
def dump(sd, name, **kw):
    m = tt.build_model(); m.load_state_dict(sd, strict=True); m = m.to(dev).eval()
    d = dict(Le=lt(m, Xe), La=lt(m, Xa), Lr=lt(m, to_input(RV, "cpu")), Lg=lt(m, to_input(GV, "cpu")), **kw)
    torch.save(d, f"{out}/.{name}.tmp"); os.replace(f"{out}/.{name}.tmp", f"{out}/{name}.pt"); print("dumped", name, flush=True)
i, n = map(int, a.shard.split("/"))
if i == 0 and not os.path.exists(f"{out}/orig.pt"):
    dump(torch.load(ROOT / "model.pt", map_location="cpu", weights_only=True), "orig", ev=ev)
for k, p in enumerate(a.subs):
    if k % n != i: continue
    name = os.path.basename(p).replace("sub_", "").replace(".pt", "")
    if name == "best": name = os.path.basename(os.path.dirname(os.path.abspath(p)))
    if os.path.exists(f"{out}/{name}.pt"): continue
    try: dump(torch.load(p, map_location="cpu", weights_only=True), name)
    except Exception as e: print("FAIL", p, e, flush=True)
