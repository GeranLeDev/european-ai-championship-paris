"""Candidate without training: subtract <delta> from the head bias of the 9 capped classes. Usage: python bias_shift.py <src.pt> <dst.pt> <delta>"""
import sys, os, torch
from common import tt; import scoring
src, dst, d = sys.argv[1], sys.argv[2], float(sys.argv[3])
sd = torch.load(src, map_location="cpu", weights_only=True); sd = {k: v.clone() for k, v in sd.items()}
sd["head.bias"][torch.tensor(scoring.FULL_CLASSES)] -= d
m = tt.build_model(); m.load_state_dict(sd, strict=True)
assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in sd.values())
torch.save({k: v.float().contiguous() for k, v in sd.items()}, dst + ".tmp"); os.replace(dst + ".tmp", dst)
print(f"OK {dst} ({os.path.getsize(dst)/2**20:.0f} MB) head.bias[capped] -= {d} -- NOT uploaded")
