"""Dump eval logits of checkpoints (clean forget, light-altered forget, retain val) so gold hypotheses can be fitted offline on CPU.
Same eval sets as unlearn.py (--n-eval 3500 --n-eval-r 1000). Usage (from solution/): python dump_logits.py ../output/runs/*/best.pt
Writes ../output/logits/<run>.pt = dict(Le, La, Lr) (+ orig.pt with ev indices); skips runs already dumped."""
import sys, os, torch
from common import *; import augs
dev = pick_device(); set_device(dev); out = ROOT / "output" / "logits"; out.mkdir(parents=True, exist_ok=True)
X = load_forget(); N = len(X); RV = torch.from_numpy(np.load(CACHE / "retain_val_u8.npy"))[:1000]
ev = torch.randperm(N, generator=torch.Generator().manual_seed(123))
torch.manual_seed(999)
Xe = to_input(X[ev], "cpu"); Xa = torch.cat([augs.light(Xe[i:i + 64]) for i in range(0, len(Xe), 64)])


@torch.no_grad()
def logits_t(model, xt, bs=128):
    o = []
    for i in range(0, len(xt), bs):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o.append(model(xt[i:i + bs].to(dev)).float().cpu())
    return torch.cat(o)


def dump(model, name, **kw):
    torch.save(dict(Le=logits_t(model, Xe), La=logits_t(model, Xa), Lr=logits_of(model, RV, dev), **kw), out / f"{name}.pt"); print("dumped", name, flush=True)


if not (out / "orig.pt").exists():
    dump(load_orig(dev), "orig", ev=ev)
for f in sys.argv[1:]:
    tag = os.path.basename(os.path.dirname(os.path.abspath(f)))
    if (out / f"{tag}.pt").exists(): continue
    m = tt.build_model(); m.load_state_dict(torch.load(f, map_location="cpu", weights_only=True), strict=True); dump(m.to(dev).eval(), tag)
