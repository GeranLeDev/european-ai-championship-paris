"""Class structure from orig: head-weight cosine neighbours, bias, and forget-set soft confusions (T=2)."""
import sys, numpy as np, torch
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sd = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
W = sd["head.weight"].float(); b = sd["head.bias"].float()
Wc = W - W.mean(0, keepdim=True); Wn = torch.nn.functional.normalize(Wc, dim=1); C = Wn @ Wn.T
L = torch.load(root / "solution/cache/orig_forget_logits.pt", map_location="cpu")
if isinstance(L, dict): L = L.get("logits", next(iter(L.values())))
L = L.float(); y = L.argmax(1); present = set(y.unique().tolist())
P = (L / 2).softmax(1)
print("||w|| present %.3f absent %.3f | bias present %.3f absent %.3f" % (
    W.norm(dim=1)[sorted(present)].mean(), W.norm(dim=1)[[i for i in range(100) if i not in present]].mean(),
    b[sorted(present)].mean(), b[[i for i in range(100) if i not in present]].mean()))
for c in range(100):
    s = C[c].clone(); s[c] = -9; nn = s.topk(4)
    line = f"c{c:2d} {'P' if c in present else '-'} b={b[c]:+.2f} head-nn " + " ".join(f"{int(i)}({v:.2f})" for v, i in zip(nn.values, nn.indices))
    if c in present:
        m = P[y == c].mean(0); m[c] = 0; t = m.topk(3)
        line += " | soft-conf " + " ".join(f"{int(i)}({v:.4f})" for v, i in zip(t.values, t.indices))
    print(line)
# hierarchical clustering on head cosine
from scipy.cluster.hierarchy import linkage, fcluster
Z = linkage(Wn.numpy(), "average", metric="cosine")
for t in (0.8, 0.9):
    cl = fcluster(Z, t, "distance"); groups = {}
    for i, g in enumerate(cl): groups.setdefault(g, []).append(i)
    print(f"cut {t}:", sorted([g for g in groups.values() if len(g) > 1]))
