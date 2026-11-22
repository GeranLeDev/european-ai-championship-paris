"""k-fold linear probe on features of an independent model (cache/ext_<tag>.pt) -> cache/gold_<tag>.pt (held-out predictions).
'What would a model that never saw this image predict' with the other forget images (pseudo-labels) as training data."""
import sys, torch, torch.nn.functional as F
from common import CACHE; import scoring
tag = sys.argv[1]; wd = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
torch.manual_seed(0); torch.set_num_threads(4)
X = torch.load(CACHE / f"ext_{tag}.pt").float(); X = (X - X.mean(0)) / X.std(0).clamp_min(1e-6); N = len(X)
y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1); cap = torch.isin(y, torch.tensor(scoring.FULL_CLASSES))
Xn = F.normalize(X, dim=1); S = Xn @ Xn.T; S.fill_diagonal_(-1); nn = S.topk(20, 1)
for k in [1, 5, 20]:
    v = torch.zeros(N, 100).scatter_add_(1, y[nn.indices[:, :k]], torch.ones(N, k)); acc = (v.argmax(1) == y).float()
    print(f"LOO kNN k={k}: acc {acc.mean():.3f}  capped {acc[cap].mean():.3f}  others {acc[~cap].mean():.3f}")
LP = []
for rep in range(4):
    perm = torch.randperm(N); Lp = torch.zeros(N, 100)
    for f in perm.chunk(5):
        tr = torch.ones(N, dtype=torch.bool); tr[f] = False
        W = torch.zeros(X.shape[1], 100, requires_grad=True); b = torch.zeros(100, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=150, history_size=20, line_search_fn="strong_wolfe")
        def clo():
            opt.zero_grad(); l = F.cross_entropy(X[tr] @ W + b, y[tr]) + wd * (W ** 2).sum(); l.backward(); return l
        opt.step(clo); Lp[f] = (X[f] @ W + b).detach()
    LP.append(Lp)
LP = torch.stack(LP); ce = torch.stack([F.cross_entropy(l, y, reduction="none") for l in LP], 1); p2 = F.softmax(LP / 2, 2).mean(0)
acc = (p2.argmax(1) == y).float()
print(f"5-fold probe x4: acc {acc.mean():.3f} capped {acc[cap].mean():.3f} others {acc[~cap].mean():.3f} | CE {ce.mean():.3f}±{ce.std():.3f}")
for b in ["medium", "strong", "xstrong"]:
    g = torch.load(CACHE / f"gold_{b}.pt"); cb = g["ce"].mean(1); a1 = p2.argmax(1); a2 = g["p2"].argmax(1); m = (a1 != y) & (a2 != y)
    print(f" corr(CE probe, CE {b}) = {torch.corrcoef(torch.stack([ce.mean(1).clamp(max=10), cb.clamp(max=10)]))[0,1]:.3f}; same wrong class where both wrong: {(a1[m]==a2[m]).float().mean():.3f} (n={m.sum()}); P(aug wrong | probe wrong) {(a2!=y)[a1!=y].float().mean():.3f} vs base {(a2!=y).float().mean():.3f}")
torch.save(dict(p2=p2, ce=ce, logits=LP.mean(0)), CACHE / f"gold_{tag}.pt")
# per-class accuracy
print("per-class probe acc (class:n:acc):", " ".join(f"{c}:{(y==c).sum()}:{acc[y==c].mean():.2f}" for c in y.unique().tolist()))
