"""Lane C: export theta = orig + lam * (base - orig), optional global head temperature. Usage: python c_lam.py <base sub name> lam[@tau] ..."""
import sys, os, torch
from c_lib import HERE
sys.path.insert(0, str(HERE.parent / "solution")); from common import tt, ROOT
o = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=True); s = torch.load(ROOT / "candidates" / f"sub_{sys.argv[1]}.pt", map_location="cpu", weights_only=True)
for spec in sys.argv[2:]:
    lam, tau = (spec.split("@") + ["1.0"])[:2]; l = float(lam); t = float(tau)
    sd = {k: (o[k].float() + l * (s[k].float() - o[k].float())) if o[k].is_floating_point() else s[k].clone() for k in o}
    sd["head.weight"] = sd["head.weight"] * t; sd["head.bias"] = sd["head.bias"] * t
    m = tt.build_model(); m.load_state_dict(sd, strict=True); assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in sd.values())
    name = f"sub_c_soupC_lam{lam.replace('.', 'p')}" + (f"_tau{tau.replace('.', 'p')}" if t != 1 else "") + ".pt"; dst = ROOT / "candidates" / name
    torch.save({k: v.float().contiguous() for k, v in sd.items()}, str(dst) + ".tmp"); os.replace(str(dst) + ".tmp", dst); print("saved", dst.name, flush=True)
