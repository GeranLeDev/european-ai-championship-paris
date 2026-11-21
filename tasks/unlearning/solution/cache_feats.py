"""Cache token states entering block (12-k) for forget views and retain proxy, so the last k blocks can be trained fast.
Views: forget clean, 2 light-aug train views, 1 light-aug eval view ('altered copies' proxy); retain: medium-aug train, clean val."""
import argparse, numpy as np, torch
from common import *; import augs
ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=4); ap.add_argument("--device")
ap.add_argument("--n-retain", type=int, default=3000); ap.add_argument("--n-retain-val", type=int, default=1000); a = ap.parse_args()
dev = pick_device(a.device); set_device(dev)
m = load_orig(dev); cut = 12 - a.k


@torch.no_grad()
def tokens(x_u8, aug=None, bs=64, seed=0):
    torch.manual_seed(seed); out = []
    for i in range(0, len(x_u8), bs):
        x = to_input(x_u8[i:i + bs], dev)
        if aug: x = aug(x)
        h = m.patch_drop(m.norm_pre(m._pos_embed(m.patch_embed(x))))
        for b in m.blocks[:cut]: h = b(h)
        out.append(h.half().cpu())
    return torch.cat(out)


X = load_forget()
R = torch.from_numpy(np.load(CACHE / "retain_train_u8.npy"))[: a.n_retain]
RV = torch.from_numpy(np.load(CACHE / "retain_val_u8.npy"))[: a.n_retain_val]
d = {"f_clean": tokens(X), "f_aug1": tokens(X, augs.light, seed=1), "f_aug2": tokens(X, augs.light, seed=2),
     "f_eval": tokens(X, augs.light, seed=777), "r_train": tokens(R, augs.medium, seed=3), "r_val": tokens(RV)}
torch.save(d, CACHE / f"tokens_k{a.k}.pt")
print({k: tuple(v.shape) for k, v in d.items()})
