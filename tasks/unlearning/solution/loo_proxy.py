"""Gold proxy from a different principle: 'what would a model that never saw this image predict?'
5-fold linear probe (and LOO kNN) on intermediate features of orig (tokens entering block 8: CLS + mean patch), pseudo-labels of
the other forget images as supervision. Writes cache/gold_loo.pt (p2, ce) in the same format as gold_proxy.py."""
import torch, torch.nn.functional as F
from common import CACHE; import scoring
torch.manual_seed(0); torch.set_num_threads(4)
T = torch.load(CACHE / "tokens_k4.pt", mmap=True)["f_clean"]; fe = []
for i in range(0, len(T), 250):
    h = T[i:i + 250].float(); fe.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1))
X = torch.cat(fe); X = (X - X.mean(0)) / X.std(0).clamp_min(1e-6); N = len(X)
L0 = torch.load(CACHE / "orig_forget_logits.pt"); y = L0.argmax(1); cap = torch.isin(y, torch.tensor(scoring.FULL_CLASSES))
Xn = F.normalize(X, dim=1); S = Xn @ Xn.T; S.fill_diagonal_(-1); nn = S.topk(20, 1)
for k in [1, 5, 20]:
    v = torch.zeros(N, 100).scatter_add_(1, y[nn.indices[:, :k]], torch.ones(N, k)); acc = (v.argmax(1) == y).float()
    print(f"LOO kNN k={k}: acc {acc.mean():.3f}  capped {acc[cap].mean():.3f}  others {acc[~cap].mean():.3f}")
perm = torch.randperm(N); folds = perm.chunk(5); Lp = torch.zeros(N, 100)
for f in folds:
    tr = torch.ones(N, dtype=torch.bool); tr[f] = False
    W = torch.zeros(X.shape[1], 100, requires_grad=True); b = torch.zeros(100, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=200, history_size=20, line_search_fn="strong_wolfe")
    def clo():
        opt.zero_grad(); l = F.cross_entropy(X[tr] @ W + b, y[tr]) + 3e-3 * (W ** 2).sum(); l.backward(); return l
    opt.step(clo); Lp[f] = (X[f] @ W + b).detach()
acc = (Lp.argmax(1) == y).float(); ce = F.cross_entropy(Lp, y, reduction="none")
print(f"5-fold probe: acc {acc.mean():.3f} capped {acc[cap].mean():.3f} others {acc[~cap].mean():.3f} | CE {ce.mean():.3f}±{ce.std():.3f}")
G = {b: torch.load(CACHE / f"gold_{b}.pt") for b in ["medium", "strong", "xstrong"]}
for b, g in G.items():
    cb = g["ce"].mean(1); print(f" corr(CE probe, CE {b}) = {torch.corrcoef(torch.stack([ce.clamp(max=10), cb.clamp(max=10)]))[0,1]:.3f}; agreement of wrong-class argmax where both wrong:", end=" ")
    a1 = Lp.argmax(1); a2 = g["p2"].argmax(1); m = (a1 != y) & (a2 != y); print(f"{(a1[m]==a2[m]).float().mean():.3f} (n={m.sum()})")
seen = torch.zeros(100, dtype=torch.bool); seen[y.unique()] = True
Lp[:, ~seen] = -1e4
# mix with medium proxy for the 55 absent classes (probe cannot predict them)
for eps in [0.0]:
    p2 = F.softmax(Lp / 2, 1); torch.save(dict(p2=p2, ce=ce[:, None].repeat(1, 4), logits=Lp), CACHE / "gold_loo.pt")
print("saved gold_loo.pt")
