"""Cache CLS 'mid' state of the last block: h_mid = h + ls1(attn(norm1(h)))[:,0]  (input of the last MLP sub-block).
Training blocks.11.{norm2,mlp,ls2} + norm + head on it is exact w.r.t. the full model (only the CLS output feeds the head)."""
import torch
from common import *
dev = "cpu"; torch.set_num_threads(6); m = load_orig(dev); b = m.blocks[11]
T = torch.load(CACHE / "tokens_k1.pt", mmap=True); out = {}
with torch.no_grad():
    for name, toks in T.items():
        r = []
        for i in range(0, len(toks), 128):
            h = toks[i:i + 128].float()
            r.append((h + b.ls1(b.attn(b.norm1(h))))[:, 0].clone())
        out[name] = torch.cat(r); print(name, flush=True)
    full = m.head(m.norm(out["f_clean"] + b.ls2(b.mlp(b.norm2(out["f_clean"])))))
print("max diff vs orig logits", (full - torch.load(CACHE / "orig_forget_logits.pt")).abs().max().item())
torch.save(out, CACHE / "mid_feats.pt")
