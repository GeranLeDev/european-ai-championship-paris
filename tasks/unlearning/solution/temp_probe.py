"""Predict effect of global logit temperature (head weight & bias / tau) on a dumped run, with the 15-point fit (predict_gen.py family)."""
import sys, torch, itertools, torch.nn.functional as F
sys.argv = ["x", "--dir", "cache/rank", "--top", "30"]
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import predict_gen as pg
f = pg.f; top = pg.top; G = pg.G
def pred(r):
    ag, kl = pg.gen_ret(r)
    out = []
    for rms, g, aa, c, s in top:
        out.append(s * f.forget(r, pg.golds[g])[0] * ag ** aa / (1 + c * kl))
    return sum(out) / len(out), min(out), max(out), ag, kl
for run in ["gm_g2", "lg_g0"]:
    d = pg.D[run]
    for tau in [0.6, 0.8, 1.0, 1.25, 1.5, 2.0]:
        r = {k: v / tau for k, v in d.items() if k in ("Le", "La", "Lr", "Lg")}
        m, lo, hi, ag, kl = pred(r)
        ce = F.cross_entropy(r["Le"], pg.O["Le"].argmax(1), reduction="none")
        print(f"{run} tau={tau}: pred {m:.3f} [{lo:.3f} {hi:.3f}] gen agree {ag:.3f} KL {kl:.3f} | forget CE {ce.mean():.2f}±{ce.std():.2f}")
