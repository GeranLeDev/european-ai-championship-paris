"""CLS mid-state of the last block (input of blocks.11 MLP sub-block) for ANY checkpoint, so its tail (blocks.11.{norm2,mlp,ls2}+norm+head)
can be re-trained on CPU exactly. Same views/seeds as cache_feats.py + the dump_logits 'altered' view (ev order, seed 999).
Usage: python cache_mid_ckpt.py <ckpt.pt> <tag>  -> cache/mid_<tag>.pt"""
import sys, torch
from common import *; import augs
ck, tag = sys.argv[1], sys.argv[2]; dev = pick_device(); set_device(dev)
m = tt.build_model(); m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=True); m = m.to(dev).eval(); b = m.blocks[11]


@torch.no_grad()
def mid(x_u8=None, aug=None, seed=0, xt=None, bs=64):
    torch.manual_seed(seed); out = []; n = len(x_u8) if xt is None else len(xt)
    for i in range(0, n, bs):
        x = to_input(x_u8[i:i + bs], dev) if xt is None else xt[i:i + bs].to(dev)
        if aug: x = aug(x)
        h = m.patch_drop(m.norm_pre(m._pos_embed(m.patch_embed(x))))
        for blk in m.blocks[:11]: h = blk(h)
        out.append((h + b.drop_path1(b.ls1(b.attn(b.norm1(h)))))[:, 0].float().cpu())
    return torch.cat(out)


X = load_forget(); R = torch.from_numpy(np.load(CACHE / "retain_train_u8.npy"))[:3000]; RV = torch.from_numpy(np.load(CACHE / "retain_val_u8.npy"))[:1000]
ev = torch.randperm(len(X), generator=torch.Generator().manual_seed(123)); torch.manual_seed(999)
Xe = to_input(X[ev], "cpu"); Xa = torch.cat([augs.light(Xe[i:i + 64]) for i in range(0, len(Xe), 64)]); del Xe
d = {"f_dump_alt": mid(xt=Xa)}; del Xa; print("alt done", flush=True)
d.update({"f_clean": mid(X), "f_aug1": mid(X, augs.light, 1), "f_aug2": mid(X, augs.light, 2), "f_eval": mid(X, augs.light, 777)}); print("forget done", flush=True)
d.update({"r_train": mid(R, augs.medium, 3), "r_val": mid(RV)})
with torch.no_grad():
    L = m.head(m.norm(d["f_clean"].to(dev) + b.ls2(b.mlp(b.norm2(d["f_clean"].to(dev)))))).cpu()
ref = CACHE / "logits" / f"{tag}.pt"
if ref.exists(): print("max diff vs dumped logits (clean):", (L[ev] - torch.load(ref)["Le"]).abs().max().item())
torch.save(d, CACHE / f"mid_{tag}.pt"); print("saved", {k: tuple(v.shape) for k, v in d.items()})
