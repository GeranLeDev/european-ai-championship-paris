"""Derive final pre-norm CLS features (input of norm+head) from the k=4 token cache (mmap, low RAM)."""
import torch
from common import *
dev = pick_device(None); m = load_orig(dev)
T = torch.load(CACHE / "tokens_k4.pt", mmap=True)
out = {}
with torch.no_grad():
    for name, toks in T.items():
        r = []
        for i in range(0, len(toks), 64):
            h = toks[i:i + 64].to(dev).float()
            for b in m.blocks[8:]: h = b(h)
            r.append(h[:, 0].cpu())
        out[name] = torch.cat(r); print(name, out[name].shape, flush=True)
L = m.head(m.norm(out["f_clean"].to(dev))).cpu()
print("max diff vs orig logits", (L - torch.load(CACHE / "orig_forget_logits.pt")).abs().max().item())
torch.save(out, CACHE / "cls_feats.pt")
