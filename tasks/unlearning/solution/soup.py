"""Model soup for task 3: average state dicts of several runs fine-tuned from the same orig.
- evaluates every single run with the proxy evaluator (same eval sets as unlearn.py: all 3500 forget images,
  a fixed light-aug 'altered copy' view (seed 999), 1000 Imagenette val images)
- uniform soup of all runs, and greedy soup (runs sorted by proxy score on --key, added only if the soup's score does not drop)
- validates (strict load, float32, finite) and exports ../output/sub_soup_<name>.pt
Usage (from unlearning/solution): python soup.py --device cuda --runs m1_g2 m2_g2 m2_g3 r2_g2 'ma_g*' 'mb_g*' --key mixcls
"""
import argparse, glob, json, os
import torch
from common import *; import augs, scoring

ap = argparse.ArgumentParser()
ap.add_argument("--device"); ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--key", default="mixcls"); ap.add_argument("--n", type=int, default=3500); ap.add_argument("--out-dir", default="../output")
a = ap.parse_args()
dev = pick_device(a.device); set_device(dev)
amp = dict(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda")   # like the server

paths = []
for r in a.runs:
    for p in sorted(glob.glob(f"{a.out_dir}/runs/{r}/best.pt")):
        if p not in paths: paths.append(p)
names = [p.split("/")[-2] for p in paths]
print("runs:", names, flush=True)

X = load_forget()[:a.n]; L0 = torch.load(CACHE / "orig_forget_logits.pt")[:a.n]; y = L0.argmax(1)
import numpy as np
RV = torch.from_numpy(np.load(CACHE / "retain_val_u8.npy"))[:min(1000, a.n)]
Xe = to_input(X, "cpu")
torch.manual_seed(999); Xa = torch.cat([augs.light(Xe[i:i + 64]) for i in range(0, len(Xe), 64)])
H = {h: {k: v[:a.n] for k, v in G.items()} for h, G in scoring.make_hypotheses(CACHE).items()}
model = tt.build_model().to(dev).eval()


@torch.no_grad()
def logits(sd):
    model.load_state_dict(sd, strict=True); out = []
    for xs in (Xe, Xa, to_input(RV, "cpu")):
        o = []
        for i in range(0, len(xs), 128):
            with torch.autocast(**amp):
                o.append(model(xs[i:i + 128].to(dev)).float().cpu())
        out.append(torch.cat(o))
    return out


orig_sd = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=True)
Lo = logits(orig_sd)


def evaluate(sd):
    Le, La, Lr = logits(sd); ret = scoring.retain_scores(Lr, Lo[2]); r = {"retainKL": ret["kl"], "agree": ret["agree"]}
    for h, G in H.items():
        g1 = scoring.group_scores(Le, Lo[0], y, G); g2 = scoring.group_scores(La, Lo[1], y, G)
        r[h] = scoring.full_score([g1, g2], ret)["final"]
    return r


def fmt(tag, r):
    return f"{tag:28s} " + " ".join(f"{h}={r[h]:.3f}" for h in H) + f" | retainKL={r['retainKL']:.3f} agree={r['agree']:.3f}"


sds = {n: torch.load(p, map_location="cpu", weights_only=True) for n, p in zip(names, paths)}
single = {}
for n in names:
    single[n] = evaluate(sds[n]); print(fmt(n, single[n]), flush=True)


def average(ns):
    return {k: sum(sds[n][k].float() for n in ns) / len(ns) for k in sds[ns[0]]}


def export(tag, sd):
    m = tt.build_model(); m.load_state_dict(sd, strict=True)
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in sd.values())
    dst = f"{a.out_dir}/sub_soup_{tag}.pt"; torch.save({k: v.contiguous() for k, v in sd.items()}, dst)
    print(f"exported {dst} ({os.path.getsize(dst)/2**20:.0f} MB, NOT uploaded)", flush=True)


results = {"single": single}
uni = average(names); results["uniform"] = dict(members=names, **evaluate(uni)); print(fmt("SOUP uniform", results["uniform"]), flush=True)
export("uniform", uni)

order = sorted(names, key=lambda n: -single[n][a.key])
members = [order[0]]; best = single[order[0]]
for n in order[1:]:
    cand = members + [n]; r = evaluate(average(cand))
    keep = r[a.key] >= best[a.key]
    print(fmt(f"  try +{n} ({'keep' if keep else 'drop'})", r), flush=True)
    if keep: members, best = cand, r
results["greedy"] = dict(members=members, **best); print(fmt(f"SOUP greedy[{len(members)}]", best), "members:", members, flush=True)
if len(members) > 1: export("greedy", average(members))
json.dump(results, open(f"{a.out_dir}/soup_results.json", "w"), indent=1)
