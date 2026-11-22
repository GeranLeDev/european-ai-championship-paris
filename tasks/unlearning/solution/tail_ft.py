"""Local (CPU) re-targeting of an existing full-model run: train only its tail (blocks.11.{norm2,mlp,ls2}+norm+head) on cached CLS
mid-states (cache_mid_ckpt.py). Exact w.r.t. the full model. Retain reference = ORIG logits on the same Imagenette views.
Usage: python tail_ft.py --ckpt ../candidates/sub_lg_g0.pt --tag lg_g0 --target fitC --name t_fitC [--export ../candidates/x.pt]
Writes cache/logits/<name>.pt (Le, La, Lr in dump_logits order) for predict12.py."""
import argparse, copy, os, torch, torch.nn.functional as F
from common import *; import scoring
from feat_unlearn import MidTail
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--target", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--alpha", type=float, default=3.0); ap.add_argument("--w-stat", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--head-only", action="store_true"); ap.add_argument("--export", default=None)
a = ap.parse_args(); torch.manual_seed(a.seed); torch.set_num_threads(4)
full = tt.build_model(); full.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True), strict=True)
sub = MidTail(full); C = torch.load(CACHE / f"mid_{a.tag}.pt"); Co = torch.load(CACHE / "mid_feats.pt"); otail = MidTail(load_orig("cpu")).eval()
with torch.no_grad(): Lo_rt, Lo_rv = otail(Co["r_train"]), otail(Co["r_val"])
L0 = torch.load(CACHE / "orig_forget_logits.pt"); y = L0.argmax(1); N = len(y); H = scoring.make_hypotheses(CACHE); G = scoring.resolve(a.target, H, CACHE)
q = G["p2"]; g_mean, g_std = G["ce"].mean().item(), G["ce"].std().item(); ev = torch.load(CACHE / "logits" / "orig.pt")["ev"]
with torch.no_grad(): print("check: agreement sub/orig on r_train %.3f, r_val %.3f" % ((sub(C["r_train"]).argmax(1) == Lo_rt.argmax(1)).float().mean(), (sub(C["r_val"]).argmax(1) == Lo_rv.argmax(1)).float().mean()))
views = torch.stack([C["f_clean"], C["f_aug1"], C["f_aug2"]]); pR = F.softmax(Lo_rt / 2, 1)
kl2 = lambda Ls, p: (p * (p.clamp_min(1e-12).log() - F.log_softmax(Ls / 2, 1))).sum(1).mean()
params = list(sub.head.parameters()) if a.head_only else list(sub.parameters())
opt = torch.optim.Adam(params, lr=a.lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps); sub.train()
for s in range(1, a.steps + 1):
    fi = torch.randint(0, N, (a.bs,)); v = torch.randint(0, 3, (a.bs,)); ri = torch.randint(0, len(pR), (a.bs,))
    Lf = sub(views[v, fi]); Lr = sub(C["r_train"][ri]); cef = F.cross_entropy(Lf, y[fi], reduction="none")
    loss = kl2(Lf, q[fi]) + a.w_stat * ((cef.mean() - g_mean) ** 2 + (cef.std() - g_std) ** 2) + a.alpha * kl2(Lr, pR[ri])
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
sub.eval()
with torch.no_grad(): out = dict(Le=sub(C["f_clean"])[ev], La=sub(C["f_dump_alt"]), Lr=sub(C["r_val"]))
ret = scoring.retain_scores(out["Lr"], Lo_rv); ce = F.cross_entropy(out["Le"], y[ev], reduction="none"); cap = torch.isin(y[ev], torch.tensor(scoring.FULL_CLASSES))
print(f"{a.name}: target {a.target} | CE {ce.mean():.2f}±{ce.std():.2f} (capped {ce[cap].mean():.2f}, others {ce[~cap].mean():.2f}) | retain agree {ret['agree']:.3f} KL {ret['kl']:.3f}")
torch.save(out, CACHE / "logits" / f"{a.name}.pt")
if a.export:
    sd = {k: v.clone() for k, v in full.state_dict().items()}; names = sub.export_names()
    for k, v in sub.state_dict().items():
        pre, rest = k.split(".", 1); sd[names[pre] + "." + rest] = v.detach().float().contiguous()
    chk = tt.build_model(); chk.load_state_dict(sd, strict=True); assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in sd.values())
    torch.save(sd, a.export + ".tmp"); os.replace(a.export + ".tmp", a.export); print("exported", a.export, "(NOT uploaded)")
