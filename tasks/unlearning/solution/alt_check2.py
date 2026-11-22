"""Extra alteration families (rotation, JPEG, perspective-ish shear, cutout, hue/invert, center crop) for orig vs a checkpoint."""
import sys, io, os, math, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from common import *
dev = pick_device(); set_device(dev); X = load_forget(); y = torch.load(CACHE / "orig_forget_logits.pt").argmax(1)
idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(5))[:300]; X = X[idx]; y = y[idx]
cap = torch.isin(y, torch.tensor([0, 6, 7, 8, 29, 30, 31, 33, 38]))
def pil_map(fn):
    return torch.from_numpy(np.stack([np.asarray(fn(Image.fromarray(x.numpy())).convert("RGB").resize((224, 224))) for x in X]))
def jpeg(q):
    def f(im):
        b = io.BytesIO(); im.save(b, "JPEG", quality=q); return Image.open(io.BytesIO(b.getvalue()))
    return f
alts = {"clean": X, "rot10": pil_map(lambda im: im.rotate(10, Image.BILINEAR)), "rot30": pil_map(lambda im: im.rotate(30, Image.BILINEAR)),
        "rot90": pil_map(lambda im: im.rotate(90)), "vflip": pil_map(lambda im: im.transpose(Image.FLIP_TOP_BOTTOM)),
        "jpeg50": pil_map(jpeg(50)), "jpeg20": pil_map(jpeg(20)), "shear": pil_map(lambda im: im.transform(im.size, Image.AFFINE, (1, .3, -30, 0, 1, 0), Image.BILINEAR)),
        "ccrop.6": pil_map(lambda im: im.crop((45, 45, 179, 179))), "hue": pil_map(lambda im: Image.fromarray(np.asarray(im.convert("HSV")).copy()).convert("RGB") if False else Image.fromarray(np.asarray(im)[..., ::-1].copy())),
        "cutout": torch.from_numpy(np.stack([np.where((np.arange(224)[:, None] // 112 == 0) & (np.arange(224)[None] // 112 == 0), 0, x.numpy().transpose(2, 0, 1)).transpose(1, 2, 0) for x in X]).astype(np.uint8))}
ms = {"orig": load_orig(dev)}
for p in sys.argv[1:]:
    m = tt.build_model(); m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True)); ms[os.path.basename(p)[:-3]] = m.to(dev).eval()
print("alteration | " + " | ".join(f"{n:>30s}" for n in ms) + "  (CE mean±std acc ; cap/oth CE)")
for an, Xa in alts.items():
    row = []
    for n, m in ms.items():
        L = logits_of(m, Xa, dev); ce = F.cross_entropy(L, y, reduction="none")
        row.append(f"{ce.mean():.2f}±{ce.std():.2f} {(L.argmax(1)==y).float().mean():.2f} ; {ce[cap].mean():.2f}/{ce[~cap].mean():.2f}")
    print(f"{an:10s} | " + " | ".join(f"{r:>30s}" for r in row), flush=True)
