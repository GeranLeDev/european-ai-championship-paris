import argparse, torch, torch.nn.functional as F, numpy as np
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("--device"); a = ap.parse_args()
dev = pick_device(a.device); set_device(dev)
m = load_orig(dev); X = load_forget()
print("params", sum(p.numel() for p in m.parameters())/1e6, "M; head", m.head)
L = logits_of(m, X, dev)
torch.save(L, CACHE / "orig_forget_logits.pt")
p = L.softmax(1); conf, pred = p.max(1)
ce = F.cross_entropy(L, pred, reduction="none")
print("conf mean %.4f median %.4f min %.4f; frac conf>0.99 %.3f >0.9 %.3f" % (conf.mean(), conf.median(), conf.min(), (conf>0.99).float().mean(), (conf>0.9).float().mean()))
print("CE wrt pseudo-label: mean %.4f std %.4f" % (ce.mean(), ce.std()))
print("class histogram (n classes present):", (torch.bincount(pred, minlength=100)>0).sum().item(), torch.bincount(pred, minlength=100).tolist())
print("logit norm mean", L.norm(dim=1).mean().item(), "max logit mean", L.max(1).values.mean().item())
from PIL import Image
g = X[:32].numpy().reshape(4, 8, 224, 224, 3).transpose(0, 2, 1, 3, 4).reshape(4*224, 8*224, 3)
Image.fromarray(g).resize((8*112, 4*112)).save(CACHE / "grid.png")
print("preds first 32", pred[:32].tolist())
