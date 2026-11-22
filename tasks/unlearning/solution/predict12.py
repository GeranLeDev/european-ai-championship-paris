"""Predict the public score of dumped logits under the top hypotheses saved by fit12.py. Usage: python predict12.py <glob or names...>"""
import sys, glob, os, torch, torch.nn.functional as F
import fit12; from fit12 import *
top = torch.load("fit12.pt")["top"]
files = []
for a in sys.argv[1:]: files += sorted(glob.glob(str(LD / (a + ".pt"))))
for f in files:
    r = torch.load(f); p = torch.tensor(predict(r, top)); ret = scoring.retain_scores(r["Lr"], O["Lr"])
    ce = F.cross_entropy(r["Le"], y, reduction="none"); g0 = top[0][0][0]; fo = forget(r, gold(*g0))
    print(f"{os.path.basename(f)[:-3]:16s} pred {p.mean():.3f} [{p.min():.3f},{p.max():.3f}] top1 {p[0]:.3f} | ms {fo[1]:.2f} ps {fo[2]:.2f} | CE {ce.mean():.2f}±{ce.std():.2f} acc {(r['Le'].argmax(1)==y).float().mean():.3f} | retain agree {ret['agree']:.3f} KL {ret['kl']:.3f}")
