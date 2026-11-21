"""Smoke test SCRUB (class unlearning) on a CIFAR-10 subset with the repo's ResNet18.
1) train a quick 'original' model, 2) SCRUB-forget class 5, 3) report accuracies + basic loss MIA.
Hyperparams follow repos/SCRUB/large_scale_unlearning.ipynb (alpha=0.001, gamma=0.99, kd_T=4, msteps=2),
lr=5e-4, forget bs=512, retain bs=128 as in the repo / paper Table 3 (forget bs=32 made the KL ascent diverge to NaN).
Run (from repos/SCRUB): .venv/bin/python ../_smoke/scrub_smoke.py
"""
import copy
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SCRUB"))
from models import ResNet18  # noqa: E402

torch.manual_seed(0); np.random.seed(0)
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DATA = Path(__file__).resolve().parents[1] / "data"
FORGET_CLASS, N_TRAIN, N_TEST = 1, 20000, 2000  # class 1 = automobile (easy to learn)

MEAN, STD = torch.tensor([0.4914, 0.4822, 0.4465]), torch.tensor([0.2470, 0.2435, 0.2616])


def load_split(name):
    """CIFAR-10 from the Hugging Face parquet mirror (repos/data/cifar10_hf), faster than the Toronto server."""
    df = pd.read_parquet(DATA / "cifar10_hf" / f"{name}.parquet")
    x = np.stack([np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in df["img"]])
    x = (torch.tensor(x).permute(0, 3, 1, 2).float() / 255 - MEAN[:, None, None]) / STD[:, None, None]
    return TensorDataset(x, torch.tensor(df["label"].values))


train, test = load_split("train"), load_split("test")
tr_idx = np.random.permutation(len(train))[:N_TRAIN]
te_idx = np.random.permutation(len(test))[:N_TEST]
tr_y, te_y = train.tensors[1].numpy()[tr_idx], test.tensors[1].numpy()[te_idx]
forget = Subset(train, tr_idx[tr_y == FORGET_CLASS])
retain = Subset(train, tr_idx[tr_y != FORGET_CLASS])
test_r = Subset(test, te_idx[te_y != FORGET_CLASS])
test_f = Subset(test, te_idx[te_y == FORGET_CLASS])
L = lambda d, bs=128, sh=True: DataLoader(d, batch_size=bs, shuffle=sh)


def acc(model, d):
    model.eval(); c = n = 0
    with torch.no_grad():
        for x, y in L(d, 512, False):
            c += (model(x.to(dev)).argmax(1).cpu() == y).sum().item(); n += len(y)
    return c / n


def losses(model, d):
    model.eval(); out = []
    with torch.no_grad():
        for x, y in L(d, 512, False):
            out.append(F.cross_entropy(model(x.to(dev)), y.to(dev), reduction="none").cpu())
    return torch.cat(out).numpy()


def mia(model):  # basic MIA from the paper: classify forget (in) vs test (out) losses; 0.5 = perfect
    lf, lt = losses(model, forget), losses(model, test_r)
    n = min(len(lf), len(lt))
    X = np.concatenate([lf[:n], lt[:n]])[:, None]; y = np.r_[np.ones(n), np.zeros(n)]
    return cross_val_score(LogisticRegression(), X, y, cv=5).mean()


def report(tag, m):
    print(f"{tag:9s} forget={acc(m, forget):.3f} retain={acc(m, retain):.3f} "
          f"test(other)={acc(m, test_r):.3f} test(cls{FORGET_CLASS})={acc(m, test_f):.3f} MIA={mia(m):.3f}")


# 1) original model (cached in repos/data/scrub_smoke_orig.pt)
orig = ResNet18(num_classes=10).to(dev)
CKPT = DATA / "scrub_smoke_orig.pt"
if CKPT.exists():
    orig.load_state_dict(torch.load(CKPT, map_location=dev))
else:
    opt = torch.optim.SGD(orig.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    for ep in range(15):
        orig.train()
        for x, y in L(Subset(train, tr_idx)):
            loss = F.cross_entropy(orig(x.to(dev)), y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  pretrain epoch {ep} loss={loss.item():.3f}", flush=True)
    torch.save(orig.state_dict(), CKPT)
report("original", orig)

# 2) SCRUB
kd_T, alpha, gamma, msteps, epochs = 4.0, 0.001, 0.99, 3, 5  # repo: msteps=2; 3 forgets fully on this toy model
teacher = copy.deepcopy(orig).eval()
student = copy.deepcopy(orig)
opt = torch.optim.SGD(student.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)


def kl(s, t):
    return F.kl_div(F.log_softmax(s / kd_T, 1), F.softmax(t / kd_T, 1), reduction="sum") * kd_T**2 / s.shape[0]


for ep in range(epochs):
    student.train()
    if ep < msteps:  # max-step on forget set
        for x, _ in L(forget, 512):  # paper Table 3: forget bs 512 for CIFAR-10 class unlearning
            x = x.to(dev)
            with torch.no_grad(): t = teacher(x)
            loss = -kl(student(x), t)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  max-step epoch {ep} -KL={loss.item():.3f}", flush=True)
    for x, y in L(retain, 128):  # min-step on retain set
        x, y = x.to(dev), y.to(dev)
        with torch.no_grad(): t = teacher(x)
        s = student(x)
        loss = alpha * kl(s, t) + gamma * F.cross_entropy(s, y)
        opt.zero_grad(); loss.backward(); opt.step()
report("SCRUB", student)

ok = acc(student, forget) < 0.2 and acc(student, test_r) > acc(orig, test_r) - 0.1
print("SMOKE", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
