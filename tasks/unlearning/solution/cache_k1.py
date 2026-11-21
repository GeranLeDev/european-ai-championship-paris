"""Derive k=1 token cache (input of the last block) from the k=4 cache via mmap (low RAM)."""
import torch
from common import *
dev = pick_device(None); m = load_orig(dev)
T = torch.load(CACHE / "tokens_k4.pt", mmap=True); out = {}
with torch.no_grad():
    for name in ["f_clean", "f_aug1", "f_aug2", "f_eval", "r_train", "r_val"]:
        toks = T[name]; r = []
        for i in range(0, len(toks), 64):
            h = toks[i:i + 64].to(dev).float()
            for b in m.blocks[8:11]: h = b(h)
            r.append(h.half().cpu())
        out[name] = torch.cat(r); print(name, flush=True)
torch.save(out, CACHE / "tokens_k1.pt")
