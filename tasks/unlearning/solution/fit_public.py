"""Offline (CPU) fit of gold hypotheses against the public leaderboard scores of submitted checkpoints.
Needs cache/logits/<run>.pt (from dump_logits.py, rsynced from JURECA output/logits/) and cache/gold_*.pt.
For each hypothesis: predicted proxy score of the submitted runs, best scale s (public ~ s * proxy) and relative RMS residual.
Then predicts all dumped runs under the best-fitting hypotheses. Usage: python fit_public.py [--top 8]"""
import argparse, glob, os, itertools, torch
from common import CACHE; import scoring
PUBLIC = {"r2_g0": 0.0798, "r2_g1": 0.2384, "r2_g2": 0.2616, "m1_g2": 0.2736}   # add new public scores here
ap = argparse.ArgumentParser(); ap.add_argument("--top", type=int, default=8); a = ap.parse_args()
LD = CACHE / "logits"; O = torch.load(LD / "orig.pt"); ev = O["ev"]; y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1)
runs = {os.path.basename(f)[:-3]: torch.load(f) for f in sorted(glob.glob(str(LD / "*.pt"))) if not f.endswith("orig.pt")}
H = scoring.make_hypotheses(CACHE)
bases = ["medium", "strong"] + (["xstrong"] if (CACHE / "gold_xstrong.pt").exists() else [])
names = list(bases) + [f"mix:{x}+{z}" for x, z in itertools.combinations(bases, 2)]
for b in bases:
    for off in [3, 6, 10, 15, 20]:
        names.append(f"cls:{b}:{off}")
        names += [f"mix:cls:{b}:{off}+{c}" for c in bases]
        names += [f"split:{b}:{off}:{c}" for c in bases + ["mix:medium+strong"] if c != b]


def score(r, n):
    G = scoring.resolve(n, H, CACHE); Gs = dict(p2=G["p2"][ev], ce=G["ce"][ev]); ret = scoring.retain_scores(r["Lr"], O["Lr"])
    return scoring.full_score([scoring.group_scores(r["Le"], O["Le"], y[ev], Gs), scoring.group_scores(r["La"], O["La"], y[ev], Gs)], ret)


pub = torch.tensor([PUBLIC[k] for k in PUBLIC]); rows = []
for n in names:
    pred = torch.tensor([score(runs[k], n)["final"] for k in PUBLIC])
    s = (pred @ pub / (pred @ pred).clamp_min(1e-12)).item(); res = ((s * pred - pub).pow(2).mean().sqrt() / pub.mean()).item()
    rows.append((res, n, s, pred.tolist()))
rows.sort()
print("public:", dict(PUBLIC)); print("hypothesis | rel.RMS | scale | predicted (submitted runs) | gold CE mean±std")
for res, n, s, pred in rows:
    print(f"{n:34s} {res:.3f} {s:.2f} " + " ".join(f"{p:.3f}" for p in pred) + f" | {H[n]['ce'].mean():.2f}±{H[n]['ce'].std():.2f}")
top = rows[: a.top]
print("\nscaled prediction (s * proxy) of every dumped run under the best-fitting hypotheses:")
print("run        | " + " | ".join(n for _, n, _, _ in top) + " | mean")
tab = []
for k, r in runs.items():
    v = [s * score(r, n)["final"] for _, n, s, _ in top]; tab.append((sum(v) / len(v), k, v))
for m, k, v in sorted(tab, reverse=True):
    print(f"{k:10s} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {m:.3f}" + (f"  (public {PUBLIC[k]})" if k in PUBLIC else ""))
