"""In-distribution retain proxy: behaviour of each checkpoint vs orig on medium/strong augmented views of forget images
(stand-in for UNSEEN images of the same classes, which the hidden control set contains and Imagenette cannot see).
Reports per model: agreement with orig argmax and KL(orig||sub) at T=2, split by the 9 capped classes vs the other 36,
plus forget accuracy on clean images. Usage: python probe_collateral.py --device mps --n 600 ckpt1.pt ckpt2.pt ..."""
import argparse, json, os, torch, torch.nn.functional as F
from common import *; import augs, scoring

ap = argparse.ArgumentParser(); ap.add_argument("--device"); ap.add_argument("--n", type=int, default=600)
ap.add_argument("--out", default="cache/collateral.json"); ap.add_argument("ckpts", nargs="+"); a = ap.parse_args()
dev = pick_device(a.device); set_device(dev)
g = torch.Generator().manual_seed(7); idx = torch.randperm(3500, generator=g)[: a.n]
X = load_forget()[idx]; y = torch.load(CACHE / "orig_forget_logits.pt")[idx].argmax(1)
cap = torch.isin(y, torch.tensor(scoring.FULL_CLASSES))
xc = to_input(X, "cpu")
torch.manual_seed(11); xm = torch.cat([augs.medium(xc[i:i + 50]) for i in range(0, len(xc), 50)])
torch.manual_seed(12); xs = torch.cat([augs.strong(xc[i:i + 50]) for i in range(0, len(xc), 50)])
model = tt.build_model().to(dev).eval()


@torch.no_grad()
def run(sd):
    model.load_state_dict(sd, strict=True)
    return [torch.cat([model(v[i:i + 50].to(dev)).float().cpu() for i in range(0, len(v), 50)]) for v in (xc, xm, xs)]


def kl(Lo, Ls):
    p = F.softmax(Lo / 2, 1); return (p * (F.log_softmax(Lo / 2, 1) - F.log_softmax(Ls / 2, 1))).sum(1)


O = run(torch.load(ROOT / "model.pt", map_location="cpu", weights_only=True))
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
for p in a.ckpts:
    name = os.path.basename(p).replace("sub_", "").replace(".pt", "")
    if name in res: print("skip", name); continue
    Lc, Lm, Ls = run(torch.load(p, map_location="cpu", weights_only=True))
    r = {"forget_acc": (Lc.argmax(1) == y).float().mean().item()}
    for tag, Lo_, Ls_ in (("med", O[1], Lm), ("str", O[2], Ls)):
        k = kl(Lo_, Ls_); ag = (Lo_.argmax(1) == Ls_.argmax(1)).float()
        for grp, m in (("all", torch.ones_like(cap)), ("cap", cap), ("oth", ~cap)):
            r[f"{tag}_kl_{grp}"] = k[m].mean().item(); r[f"{tag}_agree_{grp}"] = ag[m].mean().item()
    res[name] = r; json.dump(res, open(a.out, "w"), indent=1)
    print(f"{name:14s} facc {r['forget_acc']:.3f} | med KL all/cap/oth {r['med_kl_all']:.3f}/{r['med_kl_cap']:.3f}/{r['med_kl_oth']:.3f} "
          f"agree {r['med_agree_all']:.3f}/{r['med_agree_cap']:.3f}/{r['med_agree_oth']:.3f} | str KL {r['str_kl_all']:.3f}/{r['str_kl_cap']:.3f}/{r['str_kl_oth']:.3f} "
          f"agree {r['str_agree_all']:.3f}/{r['str_agree_cap']:.3f}/{r['str_agree_oth']:.3f}", flush=True)
