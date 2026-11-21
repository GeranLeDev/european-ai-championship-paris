import json, sys, torch
import head_grid as hg
cfgs = [dict(method="td", target=t, alpha=al, lr=lr, steps=600) for t in ["medium", "strong", "cls"] for al in [3.0, 10.0] for lr in [1e-3, 3e-3]]
cfgs += [dict(method="td", target="medium", alpha=3.0, w_ce=0.3, steps=600), dict(method="td", target="strong", alpha=3.0, w_ce=0.1, steps=600)]
for c in cfgs:
    traj = hg.train(**c, eval_every=200)
    s, r = traj[-1]
    print(hg.line(",".join(f"{k}={v}" for k, v in c.items() if k != "steps"), r), flush=True)
