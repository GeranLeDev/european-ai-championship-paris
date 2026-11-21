"""fp16 robustness check of submissions: fp32 vs autocast-fp16 (what task_template.predict does on cuda) vs model.half().
Reports, per model and mode: forget CE mean/std, proxy scores (mixcls/strong/cls), retain agreement/KL vs orig(fp32),
max |logit| and logit drift vs fp32, NaN/Inf.
Usage (from unlearning/solution): python fp16check.py --device cuda --subs ../output/sub_m1_g2.pt ../output/sub_r2_g2.pt ../output/sub_r2_g1.pt"""
import argparse, numpy as np, torch, torch.nn.functional as F
from common import *; import augs, scoring

ap = argparse.ArgumentParser(); ap.add_argument("--device"); ap.add_argument("--subs", nargs="+", required=True)
ap.add_argument("--n", type=int, default=3500); a = ap.parse_args()
dev = pick_device(a.device); set_device(dev)
X = load_forget()[:a.n]; y = torch.load(CACHE / "orig_forget_logits.pt")[:a.n].argmax(1)
RV = torch.from_numpy(np.load(CACHE / "retain_val_u8.npy"))[:min(1000, a.n)]
Xe = to_input(X, "cpu"); torch.manual_seed(999); Xa = torch.cat([augs.light(Xe[i:i + 64]) for i in range(0, len(Xe), 64)])
Xr = to_input(RV, "cpu")
H = {h: {k: v[:a.n] for k, v in G.items()} for h, G in scoring.make_hypotheses(CACHE).items()}


@torch.no_grad()
def logits(sd, mode):
    m = tt.build_model(); m.load_state_dict(sd, strict=True); m = m.to(dev).eval()
    if mode == "half": m = m.half()
    out = []
    for xs in (Xe, Xa, Xr):
        o = []
        for i in range(0, len(xs), 128):
            x = xs[i:i + 128].to(dev)
            if mode == "half": x = x.half()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(mode == "amp16" and dev == "cuda")):
                o.append(m(x).float().cpu())
        out.append(torch.cat(o))
    return out


orig_sd = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=True)
Lo32 = logits(orig_sd, "fp32")
for path in [str(ROOT / "model.pt")] + a.subs:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    bad = [k for k, v in sd.items() if not torch.isfinite(v).all()]
    ovf = [k for k, v in sd.items() if v.abs().max() > 6e4]
    print(f"\n== {path}  nonfinite={bad[:3]} fp16-overflow-weights={ovf[:3]} max|w|={max(v.abs().max().item() for v in sd.values()):.2f}")
    ref = None
    for mode in ["fp32", "amp16", "half"]:
        Le, La, Lr = logits(sd, mode)
        nan = sum((~torch.isfinite(t)).sum().item() for t in (Le, La, Lr))
        ret = scoring.retain_scores(Lr, Lo32[2]); ce = F.cross_entropy(Le, y, reduction="none")
        sc = {}
        for h in ["mixcls", "strong", "cls"]:
            g1 = scoring.group_scores(Le, Lo32[0], y, H[h]); g2 = scoring.group_scores(La, Lo32[1], y, H[h])
            sc[h] = scoring.full_score([g1, g2], ret)["final"]
        drift = "" if ref is None else f" drift max|dlogit|={max((A - B).abs().max().item() for A, B in zip((Le, La, Lr), ref)):.3f} argmax-flip={(Le.argmax(1) != ref[0].argmax(1)).float().mean():.4f}"
        if ref is None: ref = (Le, La, Lr)
        print(f"  {mode:6s} nonfinite={nan} max|logit|={Le.abs().max():.1f} forgetCE={ce.mean():.3f}±{ce.std():.3f} acc={(Le.argmax(1)==y).float().mean():.3f} "
              + " ".join(f"{h}={v:.3f}" for h, v in sc.items()) + f" retain agree={ret['agree']:.3f} KL={ret['kl']:.4f}" + drift, flush=True)
