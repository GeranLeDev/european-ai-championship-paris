"""Local erase+repair at tail level (CPU, exact): re-initialise blocks.11 MLP (+ls2) and/or head of orig, repair by KL(orig||sub) T=2 on
Imagenette mid-states only (never forget images). Forget-set outputs emerge. Usage: python tail_er.py --name er_tail [--keep-mlp] [--export x.pt]"""
import argparse, os, torch, torch.nn.functional as F
from common import *; import scoring
from feat_unlearn import MidTail
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--keep-mlp", action="store_true"); ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--bs", type=int, default=256); ap.add_argument("--export"); a = ap.parse_args(); torch.manual_seed(0); torch.set_num_threads(4)
full = load_orig("cpu"); base = MidTail(full).eval(); sub = MidTail(full); fresh = MidTail(tt.build_model())
sub.head.load_state_dict(fresh.head.state_dict())
if not a.keep_mlp:
    for n in ["norm2", "mlp", "ls2"]: getattr(sub, n).load_state_dict(getattr(fresh, n).state_dict())
C = torch.load(CACHE / "mid_feats.pt"); y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1); ev = torch.load(CACHE / "logits" / "orig.pt")["ev"]
with torch.no_grad(): pR = F.softmax(base(C["r_train"]) / 2, 1); Lo_rv = base(C["r_val"])
opt = torch.optim.AdamW(sub.parameters(), lr=a.lr, weight_decay=0.01); sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1); sub.train()
for s in range(a.steps):
    ri = torch.randint(0, len(pR), (a.bs,)); h = C["r_train"][ri]; h = h + 0.05 * h.std() * torch.randn_like(h)
    loss = F.kl_div(F.log_softmax(sub(h) / 2, 1), pR[ri], reduction="batchmean"); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
sub.eval()
with torch.no_grad(): out = dict(Le=sub(C["f_clean"])[ev], La=sub(C["f_eval"])[ev], Lr=sub(C["r_val"]))
ret = scoring.retain_scores(out["Lr"], Lo_rv); ce = F.cross_entropy(out["Le"], y[ev], reduction="none"); cap = torch.isin(y[ev], torch.tensor(scoring.FULL_CLASSES)); acc = (out["Le"].argmax(1) == y[ev]).float()
print(f"{a.name}: forget CE {ce.mean():.2f}±{ce.std():.2f} acc {acc.mean():.3f} (capped {ce[cap].mean():.2f}/{acc[cap].mean():.2f}, others {ce[~cap].mean():.2f}/{acc[~cap].mean():.2f}) | retain agree {ret['agree']:.3f} KL {ret['kl']:.3f}")
torch.save(out, CACHE / "logits" / f"{a.name}.pt")
if a.export:
    sd = {k: v.clone() for k, v in full.state_dict().items()}; names = sub.export_names()
    for k, v in sub.state_dict().items():
        pre, rest = k.split(".", 1); sd[names[pre] + "." + rest] = v.detach().float().contiguous()
    chk = tt.build_model(); chk.load_state_dict(sd, strict=True); assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in sd.values())
    torch.save(sd, a.export + ".tmp"); os.replace(a.export + ".tmp", a.export); print("exported", a.export)
