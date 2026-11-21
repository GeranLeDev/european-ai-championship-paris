import sys; sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/pixelseal_research")
import os; os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from det_head import *
wm = load_watermark("pixelseal", MODELS, "cpu")
for hires in (0, 1):
    feat = DetFeatures(wm.det, hires=bool(hires)); head = Head(streams=1 + hires)
    x = torch.rand(2, 3, 512, 512); sgn = torch.randint(0, 2, (2, 256)).float() * 2 - 1
    lg = head(feat(x, sgn)); loss = loss_fn(lg, (torch.rand(2, 128, 128) > .5).float()); loss.backward()
    g = sum(float(p.grad.abs().sum()) for p in wm.det.parameters() if p.grad is not None)
    print("hires", hires, "out", tuple(lg.shape), "backbone grad ok", g > 0)
