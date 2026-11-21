"""End-to-end check of an exported submission with the official template code path (strict load + predict()).
Compares to orig on the first n forget images: accuracy vs pseudo-labels, CE stats, KL at T=2."""
import sys, torch, torch.nn.functional as F
from common import *
dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"; n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
set_device(dev); torch.set_num_threads(6)
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
m = tt.build_model(); m.load_state_dict(sd, strict=True); m = m.to(dev).eval()
X = load_forget()[:n]; L0 = torch.load(CACHE / "orig_forget_logits.pt")[:n]; y = L0.argmax(1)
L = tt.predict(m, X, batch_size=32)
ce = F.cross_entropy(L, y, reduction="none")
kl = (F.softmax(L0 / 2, 1) * (F.log_softmax(L0 / 2, 1) - F.log_softmax(L / 2, 1))).sum(1).mean()
print(f"{sys.argv[1]}: acc vs orig-labels {(L.argmax(1)==y).float().mean():.3f} CE {ce.mean():.3f}±{ce.std():.3f} KL(orig||sub,T2) {kl:.3f}")
