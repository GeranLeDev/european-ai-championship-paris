"""Features of the forget images from an independent public model (no memorisation of the forget set). -> cache/ext_<tag>.pt"""
import sys, torch, timm
from common import *
name = sys.argv[1]; tag = sys.argv[2]; dev = pick_device(); X = load_forget()
m = timm.create_model(name, pretrained=True, num_classes=0, img_size=224).to(dev).eval(); cfg = m.pretrained_cfg
mean = torch.tensor(cfg["mean"], device=dev)[None, :, None, None]; std = torch.tensor(cfg["std"], device=dev)[None, :, None, None]; out = []
with torch.no_grad():
    for i in range(0, len(X), 50):
        x = X[i:i + 50].to(dev).permute(0, 3, 1, 2).float() / 255; out.append(m((x - mean) / std).float().cpu())
torch.save(torch.cat(out), CACHE / f"ext_{tag}.pt"); print("saved", tag, torch.cat(out).shape)
