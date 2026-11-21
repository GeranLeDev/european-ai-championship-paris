"""Merge a trained tail (unlearn_fast.py best_tail.pt) into the full orig state dict and validate. Usage: export_tail.py tail.pt k out.pt"""
import sys, torch
from common import tt, ROOT
tail_p, k, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]; cut = 12 - k
sd = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=True)
for name, v in torch.load(tail_p, map_location="cpu", weights_only=True).items():
    if name.startswith("blocks."):
        i, rest = name.split(".", 2)[1:]; name = f"blocks.{cut + int(i)}.{rest}"
    assert name in sd and sd[name].shape == v.shape, name
    sd[name] = v.float().contiguous()
m = tt.build_model(); m.load_state_dict(sd, strict=True)
assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in sd.values())
torch.save(sd, dst); print("saved", dst, "(not uploaded)")
