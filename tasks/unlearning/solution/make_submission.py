"""Validate a candidate state dict like the server (strict load, float32, finite, <450MB) and copy it as a submission."""
import sys, os, torch
from common import tt
src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu", weights_only=True)
m = tt.build_model(); m.load_state_dict(sd, strict=True)
assert all(v.dtype == torch.float32 for v in sd.values()), "non-float32 tensor"
assert all(torch.isfinite(v).all() for v in sd.values()), "NaN/Inf"
torch.save({k: v.float().contiguous() for k, v in sd.items()}, dst)
mb = os.path.getsize(dst) / 2**20; assert mb < 450, mb
print(f"OK {dst} ({mb:.0f} MB) -- NOT uploaded (use ../submission.py manually with the team API key)")
